"""Run Qwen-PICLe on the paper's human auction histories, locally."""

import argparse
import csv
import gc
import json
import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

from models.qwen import QwenWrapper
from models.scoring import base_model_context, generate_text, score_features, statement_feature


DATA_URL = (
    "https://raw.githubusercontent.com/diana3135/LLM-Fidelity-in-Decision-Making/"
    "main/human_experiment/auction_human_data.csv"
)
INSTRUCTIONS_URL = (
    "https://raw.githubusercontent.com/diana3135/LLM-Fidelity-in-Decision-Making/"
    "main/src/Auction/execution/experiment_instructions.txt"
)


def fetch_if_missing(path, url):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def load_groups(path):
    groups = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row.get("Bidder Group"):
                groups[row["Bidder Group"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["Stepgroup Loop"]))
    return groups


def bid_prices(row):
    return [int(value) for value in re.findall(r"d:(\d+)", row["newBids"])]


def round_question(round_id, row):
    return (
        f"Round {round_id}. Number of bidders: {int(row['numBidders'])}. "
        f"Drop-out prices: {bid_prices(row)}. What reserve price do you set?"
    )


def round_statement(round_id, row):
    return (
        f"Round {round_id}: The participant set a reserve price of {int(row['rPrice'])} "
        f"with {int(row['numBidders'])} bidders and drop-out prices {bid_prices(row)}, "
        f"earning a profit of {float(row['myProfit']):g}."
    )


def transform_demonstrations(demonstrations, mode, seed, context_num):
    demos = list(demonstrations)
    if mode == "mask":
        # The paper text defines Mask as hiding the round number.
        return [(None, row) for _, row in demos]
    if mode == "reverse":
        demos.reverse()
    elif mode == "shuffle":
        random.Random(seed).shuffle(demos)
    elif mode == "regionshuffle":
        midpoint = context_num // 2
        early = [item for item in demos if item[0] <= midpoint]
        late = [item for item in demos if item[0] > midpoint]
        rng = random.Random(seed)
        rng.shuffle(early)
        rng.shuffle(late)
        demos = early + late
    return demos


def make_masked_question(row):
    return re.sub(r"^Round \d+\.", "Round [MASK].", round_question(0, row))


def train_participant_adapter(wrapper, statements, output_dir, args, seed):
    features = [statement_feature(wrapper, item) for item in statements]
    model = wrapper.huggingface_model
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.0,
        bias="none", target_modules=["q_proj", "v_proj"],
    ))
    training_args = TrainingArguments(
        output_dir=str(output_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=2e-5, weight_decay=0.01, optim="adamw_torch",
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no", save_strategy="no", report_to=[],
        seed=seed, data_seed=seed, remove_unused_columns=False,
    )
    Trainer(
        model=model, args=training_args, train_dataset=features,
        data_collator=DataCollatorForSeq2Seq(wrapper.tokenizer, label_pad_token_id=-100),
    ).train()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    wrapper.tokenizer.save_pretrained(output_dir)
    wrapper.huggingface_model = model.unload()
    wrapper.huggingface_model.config.use_cache = True
    wrapper.huggingface_model.eval()
    gc.collect()
    torch.cuda.empty_cache()


def parse_predictions(text, first_round, last_round):
    values = {}
    for match in re.finditer(r"round\s*(\d+)\s*:\s*(\d+)", text, re.IGNORECASE):
        round_id, value = map(int, match.groups())
        if first_round <= round_id <= last_round and 0 <= value <= 100:
            values[round_id] = value
    return values


def distribution_metrics(predictions, targets):
    """Return empirical 1-Wasserstein distance and two-sample KS statistic.

    Both inputs contain the same held-out rounds for which the model returned
    a valid reserve price. W1 is reported in reserve-price units (0-100).
    """
    if not predictions:
        return None, None, 0
    predicted = sorted(predictions)
    observed = sorted(targets)
    count = len(predicted)
    w1 = sum(abs(left - right) for left, right in zip(predicted, observed)) / count
    support = sorted(set(predicted) | set(observed))
    ks = max(
        abs(
            sum(value <= point for value in predicted) / count
            - sum(value <= point for value in observed) / count
        )
        for point in support
    )
    return w1, ks, count


def run_group(wrapper, rows, group, instructions, args, methods):
    context_num = args.context_num
    context = rows[:context_num]
    future = list(enumerate(rows[context_num:], start=context_num + 1))
    statements = [round_statement(i, row) for i, row in enumerate(context, start=1)]
    features = [statement_feature(wrapper, item) for item in statements]
    adapter_dir = Path(args.output_dir) / "auction" / group.replace("/", "_")
    train_participant_adapter(wrapper, statements, adapter_dir, args, args.seed)

    wrapper.change_lora_adapter(adapter_dir)
    with base_model_context(wrapper):
        base_scores = score_features(wrapper, features, f"{group} Base likelihood")
    sft_scores = score_features(wrapper, features, f"{group} PICLe likelihood")
    delta = sft_scores - base_scores
    selected = torch.topk(delta, min(args.K, len(context))).indices.flip(0).tolist()
    selected_demos = [(index + 1, context[index]) for index in selected]

    output = []
    metric_output = []
    for mode in methods:
        demos = transform_demonstrations(selected_demos, mode, args.seed, context_num)
        messages_demos = []
        for round_id, row in demos:
            question = make_masked_question(row) if mode == "mask" else round_question(round_id, row)
            messages_demos.append((round_id, row, question))

        # Build Qwen chat messages directly so Mask can omit only historical round IDs.
        messages = [{
            "role": "system",
            "content": (
                "You are taking part in a second-price auction experiment. "
                "Use the participant's examples to predict their reserve-price decisions. "
                "Output one line per requested round in exactly the form 'round N: integer'. "
                "Do not add explanations.\n\n" + instructions
            ),
        }]
        for round_id, row, question in messages_demos:
            messages.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": str(int(row["rPrice"]))},
            ])
        future_text = "Predict reserve prices for these rounds:\n" + "\n".join(
            f"round {round_id}: {int(row['numBidders'])} bidders, drop-out prices {bid_prices(row)}"
            for round_id, row in future
        )
        messages.append({"role": "user", "content": future_text})
        prompt = wrapper.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with base_model_context(wrapper):
            generated = generate_text(wrapper, prompt, max_new_tokens=args.max_new_tokens)
        predictions = parse_predictions(generated, context_num + 1, len(rows))
        scored = [
            (predictions[round_id], int(row["rPrice"]))
            for round_id, row in future if round_id in predictions
        ]
        w1, ks, n_scored = distribution_metrics(
            [prediction for prediction, _ in scored],
            [target for _, target in scored],
        )
        metric_output.append({
            "bidder_group": group,
            "mode": mode,
            "n_scored": n_scored,
            "w1_reserve_price": w1,
            "ks_distance": ks,
        })
        for round_id, row in future:
            output.append({
                "bidder_group": group, "mode": mode, "round": round_id,
                "prediction": predictions.get(round_id), "human_reserve_price": int(row["rPrice"]),
                "num_bidders": int(row["numBidders"]), "delta_selected": json.dumps(selected),
            })

    wrapper.huggingface_model = wrapper.huggingface_model.unload()
    wrapper.huggingface_model.eval()
    del features, base_scores, sft_scores, delta
    gc.collect()
    torch.cuda.empty_cache()
    return output, metric_output


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-PICLe auction imitation and history-order ablations.")
    parser.add_argument("--mode", choices=["direct", "mask", "reverse", "shuffle", "regionshuffle", "all"], default="all")
    parser.add_argument("--data_csv", default="data/auction_human_data.csv")
    parser.add_argument("--instructions", default="data/auction_instructions.txt")
    parser.add_argument("--model_dir", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", default="checkpoints/qwen")
    parser.add_argument("--results_file", default="out/qwen/auction_picle.csv")
    parser.add_argument("--metrics_file", default="out/qwen/auction_picle_metrics.csv")
    parser.add_argument("--context_num", type=int, default=30)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_input_len", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--bidder_groups", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen-PICLe auction evaluation requires a CUDA GPU and BF16 support.")
    set_seed(args.seed)
    data_path = fetch_if_missing(args.data_csv, DATA_URL)
    instruction_path = fetch_if_missing(args.instructions, INSTRUCTIONS_URL)
    instructions = instruction_path.read_text(encoding="utf-8")
    groups = load_groups(data_path)
    selected_groups = args.bidder_groups or sorted(groups)
    methods = ["direct", "mask", "reverse", "shuffle", "regionshuffle"] if args.mode == "all" else [args.mode]

    wrapper = QwenWrapper(args.model_dir, "bfloat16", False, args.max_input_len)
    result_path = Path(args.results_file)
    metrics_path = Path(args.metrics_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not result_path.exists()
    write_metrics_header = not metrics_path.exists()
    with result_path.open("a", newline="", encoding="utf-8") as stream:
        fields = ["bidder_group", "mode", "round", "prediction", "human_reserve_price", "num_bidders", "delta_selected"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        if write_header:
            writer.writeheader()
        with metrics_path.open("a", newline="", encoding="utf-8") as metrics_stream:
            metric_fields = ["bidder_group", "mode", "n_scored", "w1_reserve_price", "ks_distance"]
            metrics_writer = csv.DictWriter(metrics_stream, fieldnames=metric_fields)
            if write_metrics_header:
                metrics_writer.writeheader()
            for group in selected_groups:
                if group not in groups or len(groups[group]) < args.context_num + 1:
                    continue
                rows = groups[group][:60]
                records, group_metrics = run_group(wrapper, rows, group, instructions, args, methods)
                writer.writerows(records)
                metrics_writer.writerows(group_metrics)
                stream.flush()
                metrics_stream.flush()


if __name__ == "__main__":
    main()
