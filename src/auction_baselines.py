"""Run non-PICLe controls for the Qwen auction-history experiment."""

import argparse
import csv
import random
import re
from pathlib import Path

import torch

from auction_picle import (
    DATA_URL,
    INSTRUCTIONS_URL,
    bid_prices,
    distribution_metrics,
    fetch_if_missing,
    load_groups,
    round_question,
)
from models.qwen import QwenWrapper
from models.scoring import embed_statements, generate_text
from tqdm import tqdm


METHODS = ["no_history", "direct_icl", "random_k", "similarity_k", "recent_k"]


def task_state(row):
    """State text for similarity retrieval, without round ID or participant action."""
    return f"Number of bidders: {int(row['numBidders'])}. Drop-out prices: {bid_prices(row)}."


def system_message(instructions):
    return (
        "You are taking part in a second-price auction experiment. "
        "Use the provided task information and examples to set reserve prices. "
        "Output only lines in the exact form 'round N: integer'; do not add explanations.\n\n"
        + instructions
    )


def make_prompt(wrapper, instructions, demonstrations, target_rows):
    messages = [{"role": "system", "content": system_message(instructions)}]
    for round_id, row in demonstrations:
        messages.extend([
            {"role": "user", "content": round_question(round_id, row)},
            {"role": "assistant", "content": str(int(row["rPrice"]))},
        ])
    query = "\n".join(
        f"Predict the reserve price for round {round_id}: {int(row['numBidders'])} bidders, "
        f"drop-out prices {bid_prices(row)}. Reply exactly 'round {round_id}: integer' "
        "with an integer from 0 to 100 and no explanation."
        for round_id, row in target_rows
    )
    messages.append({"role": "user", "content": query})
    return wrapper.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def parse_one_prediction(text, round_id):
    match = re.search(
        rf"\bround\s*{round_id}\s*[:=-]\s*(\d{{1,3}})\b", text, re.IGNORECASE,
    )
    if match:
        value = int(match.group(1))
        return value if 0 <= value <= 100 else None
    values = re.findall(r"(?<!\d)(?:100|[1-9]?\d)(?!\d)", text)
    if len(values) == 1:
        return int(values[0])
    if text.strip().isdigit() and 0 <= int(text.strip()) <= 100:
        return int(text.strip())
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run no-history, Direct ICL, Random-K, Similarity-K, and Recent-K controls."
    )
    parser.add_argument("--mode", choices=METHODS + ["all"], default="all")
    parser.add_argument("--data_csv", default="data/auction_human_data.csv")
    parser.add_argument("--instructions", default="data/auction_instructions.txt")
    parser.add_argument("--model_dir", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--results_file", default="out/qwen/auction_baseline_predictions.csv")
    parser.add_argument("--metrics_file", default="out/qwen/auction_baseline_metrics.csv")
    parser.add_argument("--picle_predictions", default="out/qwen/auction_picle.csv")
    parser.add_argument("--context_num", type=int, default=30)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_repeats", type=int, default=3)
    parser.add_argument("--max_input_len", type=int, default=16384)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--bidder_groups", nargs="*")
    return parser.parse_args()


def run_group(wrapper, rows, group, instructions, args, methods):
    context = rows[:args.context_num]
    future = list(enumerate(rows[args.context_num:], start=args.context_num + 1))
    context_indexed = list(enumerate(context, start=1))
    selections = {"no_history": [], "direct_icl": context_indexed}

    for method in methods:
        if method.startswith("random_k_seed"):
            run_seed = int(method.removeprefix("random_k_seed"))
            group_seed = run_seed + sum((i + 1) * ord(char) for i, char in enumerate(group))
            selections[method] = random.Random(group_seed).sample(
                context_indexed, min(args.K, len(context_indexed)),
            )
    if "recent_k" in methods:
        selections["recent_k"] = context_indexed[-min(args.K, len(context_indexed)):]

    similarity_indices = None
    if "similarity_k" in methods:
        candidate_vectors = embed_statements(wrapper, [task_state(row) for row in context])
        target_vectors = embed_statements(wrapper, [task_state(row) for _, row in future])
        similarity_indices = (target_vectors @ candidate_vectors.T).topk(
            min(args.K, len(context)), dim=1,
        ).indices

    prediction_rows = []
    metric_rows = []
    human_by_round = {round_id: int(row["rPrice"]) for round_id, row in future}
    predictions_by_method = {}

    for method in methods:
        predictions = {}
        selected_by_round = {}
        if method == "similarity_k":
            targets_to_run = enumerate(future)
        else:
            demos = selections[method]
            targets_to_run = ((index, item) for index, item in enumerate(future))

        for target_index, (round_id, target_row) in tqdm(
            targets_to_run, total=len(future), desc=f"{group} {method}", leave=False,
        ):
            if method == "similarity_k":
                selected = similarity_indices[target_index].flip(0).tolist()
                demos = [(i + 1, context[i]) for i in selected]
            selected_by_round[round_id] = demos
            prompt = make_prompt(wrapper, instructions, demos, [(round_id, target_row)])
            generated = generate_text(wrapper, prompt, max_new_tokens=args.max_new_tokens)
            prediction = parse_one_prediction(generated, round_id)
            if prediction is not None:
                predictions[round_id] = prediction

        predictions_by_method[method] = predictions
        for round_id, row in future:
            prediction_rows.append({
                "bidder_group": group,
                "mode": method,
                "round": round_id,
                "prediction": predictions.get(round_id),
                "human_reserve_price": int(row["rPrice"]),
                "num_bidders": int(row["numBidders"]),
                "selected_rounds": ";".join(
                    str(i) for i, _ in selected_by_round.get(round_id, [])
                ),
                "seed": int(method.removeprefix("random_k_seed"))
                if method.startswith("random_k_seed") else args.seed,
            })

    picle_predictions = load_picle_direct(args.picle_predictions, group)
    common_rounds = set(human_by_round)
    for predictions in predictions_by_method.values():
        common_rounds.intersection_update(predictions)
    if picle_predictions:
        common_rounds.intersection_update(picle_predictions)
    common_rounds = sorted(common_rounds)
    targets = [human_by_round[round_id] for round_id in common_rounds]
    for method in methods:
        values = [predictions_by_method[method][round_id] for round_id in common_rounds]
        w1, ks, n_common = distribution_metrics(values, targets)
        metric_rows.append({
            "bidder_group": group,
            "mode": method,
            "n_common": n_common,
            "w1_reserve_price": w1,
            "ks_distance": ks,
            "seed": int(method.removeprefix("random_k_seed"))
            if method.startswith("random_k_seed") else args.seed,
        })
    if picle_predictions:
        values = [picle_predictions[round_id] for round_id in common_rounds]
        w1, ks, n_common = distribution_metrics(values, targets)
        metric_rows.append({
            "bidder_group": group,
            "mode": "picle_direct",
            "n_common": n_common,
            "w1_reserve_price": w1,
            "ks_distance": ks,
            "seed": args.seed,
        })
    return prediction_rows, metric_rows


def load_picle_direct(path, group):
    path = Path(path)
    if not path.is_file():
        return {}
    predictions = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (row.get("bidder_group") == group and row.get("mode") == "direct"
                    and row.get("prediction")):
                predictions[int(row["round"])] = int(row["prediction"])
    return predictions


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Auction baseline evaluation requires a CUDA GPU.")
    data_path = fetch_if_missing(args.data_csv, DATA_URL)
    instruction_path = fetch_if_missing(args.instructions, INSTRUCTIONS_URL)
    instructions = instruction_path.read_text(encoding="utf-8")
    groups = load_groups(data_path)
    selected_groups = args.bidder_groups or sorted(groups)
    random_methods = [f"random_k_seed{args.seed + i}" for i in range(args.random_repeats)]
    if args.mode == "all":
        methods = ["no_history", "direct_icl"] + random_methods + ["similarity_k", "recent_k"]
    elif args.mode == "random_k":
        methods = random_methods
    else:
        methods = [args.mode]

    wrapper = QwenWrapper(args.model_dir, "bfloat16", False, args.max_input_len)
    result_path = Path(args.results_file)
    metrics_path = Path(args.metrics_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", newline="", encoding="utf-8") as result_stream, \
            metrics_path.open("w", newline="", encoding="utf-8") as metrics_stream:
        prediction_fields = [
            "bidder_group", "mode", "round", "prediction", "human_reserve_price",
            "num_bidders", "selected_rounds", "seed",
        ]
        metric_fields = [
            "bidder_group", "mode", "n_common", "w1_reserve_price", "ks_distance", "seed",
        ]
        prediction_writer = csv.DictWriter(result_stream, fieldnames=prediction_fields)
        metric_writer = csv.DictWriter(metrics_stream, fieldnames=metric_fields)
        prediction_writer.writeheader()
        metric_writer.writeheader()
        for group in selected_groups:
            if group not in groups or len(groups[group]) < args.context_num + 1:
                continue
            rows, metrics = run_group(
                wrapper, groups[group][:60], group, instructions, args, methods,
            )
            prediction_writer.writerows(rows)
            metric_writer.writerows(metrics)
            result_stream.flush()
            metrics_stream.flush()


if __name__ == "__main__":
    main()
