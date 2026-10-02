from itertools import zip_longest
from pathlib import Path
import random

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from data.formatting import SYSTEM_MESSAGE, format_query, label_to_int
from models.scoring import generate_text


def concatenate_three_prompts(prompts):
    # Original PICLe statement-language-model objective and cyclic ordering.
    result = []
    for i in range(0, len(prompts) - 2, 3):
        group = prompts[i:i + 3]
        for shift in range(3):
            result.append("".join(text + ".\n" for text in group[shift:] + group[:shift]))
    return result


def concatenate_three_prompts_instruct(prompts):
    result = []
    for i in range(0, len(prompts) - 2, 3):
        group = prompts[i:i + 3]
        for shift in range(3):
            a, b, c = group[shift:] + group[:shift]
            result.append(f"[INST]{a}.[/INST]{b}.[INST]{c}.[/INST]")
    return result


def load_persona(args):
    if args.data_dir:
        path = Path(args.data_dir) / f"{args.target_persona}.jsonl"
    else:
        path = hf_hub_download(
            repo_id="Anthropic/model-written-evals", repo_type="dataset",
            filename=f"persona/{args.target_persona}.jsonl",
        )
    data = load_dataset("json", data_files=str(path), split="train")
    required = {"statement", "question", "answer_matching_behavior"}
    if not required.issubset(data.column_names):
        raise ValueError(f"Missing persona fields: {required - set(data.column_names)}")
    return data.add_column("row_id", list(range(len(data))))


def shuffle_and_split(dataset, ratio=0.7):
    # Same seeds/order as the original on Anthropic's alternating Yes/No rows.
    pos = [i for i, label in enumerate(dataset["answer_matching_behavior"]) if label_to_int(label)]
    neg = [i for i, label in enumerate(dataset["answer_matching_behavior"]) if not label_to_int(label)]
    random.Random(0).shuffle(pos)
    random.Random(1).shuffle(neg)
    indices = [i for pair in zip_longest(pos, neg) for i in pair if i is not None]
    split = int(len(indices) * ratio)
    return dataset[indices[:split]], dataset[indices[split:]]


def persona_split(args, split):
    train, test = shuffle_and_split(load_persona(args))
    data = train if split == "train" else test
    if args.pos_label_sample_only and split == "train":
        indices = [i for i, value in enumerate(data["answer_matching_behavior"]) if label_to_int(value)]
        data = {key: [values[i] for i in indices] for key, values in data.items()}
    return data


def get_sft_data(args, split):
    data = persona_split(args, split)
    if args.inst_delimiter:
        if args.model == "qwen":
            raise ValueError("Qwen uses its chat template; omit --inst_delimiter.")
        return concatenate_three_prompts_instruct(data["statement"])
    return concatenate_three_prompts(data["statement"])


def get_basic_data(args, split, model=None):
    data = persona_split(args, split)
    tokenizer = model.tokenizer if model is not None else None
    return ([format_query(args, q, tokenizer) for q in data["question"]],
            [label_to_int(label) for label in data["answer_matching_behavior"]])


def get_pe_data(args, model, split):
    data = persona_split(args, split)
    persona = args.target_persona
    if args.pe_type == "plain":
        prefix = f'Assume that you have or agree with the persona called "{persona}". '
    else:
        request = f'How would you describe a persona called "{persona}" in one sentence?'
        if args.model == "qwen":
            request = model.tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": request}],
                tokenize=False, add_generation_prompt=True,
            )
        response = generate_text(model, request, max_new_tokens=100)
        prefix = f'The persona called "{persona}" can be described as: {response}. Now assume that you have or agree with this persona. '
    return ([format_query(args, prefix + q, model.tokenizer) for q in data["question"]],
            [label_to_int(label) for label in data["answer_matching_behavior"]])


def get_icl_data(args, icl_mode, K=3, ref_model=None, sft_model=None):
    train, test = shuffle_and_split(load_persona(args))
    if args.pos_label_sample_only:
        indices = [i for i, value in enumerate(train["answer_matching_behavior"]) if label_to_int(value)]
        train = {key: [values[i] for i in indices] for key, values in train.items()}
    if K < 1 or K > len(train["question"]):
        raise ValueError(f"K must be between 1 and the candidate count ({len(train['question'])}).")
    common = (args, test["question"], test["answer_matching_behavior"],
              train["question"], train["answer_matching_behavior"])
    if icl_mode == "random":
        from icl_strategies.select_random import select_random
        return select_random(*common, K=K, ref_model=ref_model)
    if icl_mode == "similarity":
        from icl_strategies.select_similar import select_similar
        return select_similar(*common, K=K, ref_model=ref_model,
                              train_statements=train["statement"], test_statements=test["statement"])
    if icl_mode == "uncertainty":
        from icl_strategies.select_uncertain import select_uncertain
        return select_uncertain(*common, K=K, ref_model=ref_model,
                                choose_uncertain=not args.choose_certain, func=args.uncertainty_func)
    if icl_mode == "likelihood":
        from icl_strategies.select_likely import select_likely
        return select_likely(*common, K=K, ref_model=ref_model, train_statements=train["statement"])
    if icl_mode == "diversity":
        from icl_strategies.select_diverse import select_diverse
        return select_diverse(*common, K=K, ref_model=ref_model, train_statements=train["statement"])
    if icl_mode == "picle":
        from icl_strategies.select_picle import select_picle
        return select_picle(*common, K=K, func=args.likelihood_func, ref_model=ref_model,
                            sft_model=sft_model, train_statements=train["statement"], train_ids=train["row_id"])
    raise ValueError(f"Unknown ICL strategy: {icl_mode}")
