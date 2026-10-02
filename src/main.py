import argparse
import csv
import gc
import json
import math
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

from data.persona import get_basic_data, get_icl_data, get_pe_data, get_sft_data
from models.scoring import base_model_context, generate_text, score_answers, statement_feature


def adapter_directory(args):
    return Path(args.output_dir) / args.target_persona


def train_model(args, wrapper, train_dataset, eval_dataset):
    if not train_dataset:
        raise ValueError("Persona SFT requires at least three training statements.")
    train_features = [statement_feature(wrapper, text) for text in train_dataset]
    eval_features = [statement_feature(wrapper, text) for text in eval_dataset]
    model = wrapper.huggingface_model
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=args.lora_alpha,
        lora_dropout=args.dropout, bias="none",
        target_modules=["q_proj", "v_proj"] if args.model in ("qwen", "llama", "vicuna") else None,
    )
    model = get_peft_model(model, config)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    output_dir = adapter_directory(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr, weight_decay=0.01, optim="adamw_torch",
        bf16=torch.cuda.is_available() and args.dtype == "bfloat16",
        fp16=torch.cuda.is_available() and args.dtype == "float16",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no", save_strategy="epoch", save_total_limit=1,
        logging_steps=10, report_to=[], push_to_hub=False,
        seed=args.seed, data_seed=args.seed, remove_unused_columns=False, label_names=["labels"],
    )
    # Unlike DataCollatorForLanguageModeling, this preserves our content masks.
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_features,
        eval_dataset=eval_features,
        data_collator=DataCollatorForSeq2Seq(wrapper.tokenizer, label_pad_token_id=-100),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    wrapper.tokenizer.save_pretrained(output_dir)
    with (output_dir / "training_config.json").open("w") as file:
        json.dump(vars(args), file, indent=2)
    if eval_features:
        # Fixed final checkpoint only; the test set does not select checkpoints.
        loss = trainer.evaluate()["eval_loss"]
        print(f"Statement evaluation loss: {loss:.4f}; perplexity: {math.exp(min(loss, 80)):.4f}")
    print(f"Saved persona LoRA: {output_dir}")


def map_text_to_action(outputs):
    actions = []
    for output in outputs:
        value = output.strip().lower()
        if value.endswith("."):
            value = value[:-1]
        actions.append({"no": 0, "yes": 1}.get(value, -1))
    return actions


def action_consistency(predictions, labels):
    def accuracy(indices):
        return sum(predictions[i] == labels[i] for i in indices) / len(indices) if indices else 0.0
    return (accuracy(list(range(len(labels)))),
            accuracy([i for i, value in enumerate(labels) if value == 1]),
            accuracy([i for i, value in enumerate(labels) if value == 0]))


def test_model(args, model, test_dataset, doa_test_dataset=None):
    records, predictions, generated_actions = [], [], []
    confidences, binary_confidences, entropies, token_entropies, alterations = [], [], [], [], []
    queries, labels = test_dataset
    if not queries:
        raise ValueError("The test split is empty.")
    for i, (query, label) in enumerate(tqdm(zip(queries, labels), total=len(queries), desc=args.mode)):
        answer_logp, token_logp = score_answers(model, query)
        prediction = int(answer_logp.argmax())
        binary_logp = answer_logp.log_softmax(-1)
        predictions.append(prediction)
        confidences.append(answer_logp[prediction].exp().item())
        binary_confidences.append(binary_logp[prediction].exp().item())
        entropies.append(-(binary_logp.exp() * binary_logp).sum().item())
        token_entropies.append(-(token_logp.exp() * token_logp).sum().item())
        record = {
            "test_index": i, "question": query, "label": label, "prediction": prediction,
            "logp_no": answer_logp[0].item(), "logp_yes": answer_logp[1].item(),
        }
        if args.generate:
            response = generate_text(model, query)
            record["generated_text"] = response
            generated_actions.append(map_text_to_action([response])[0])
        if args.verbose:
            print(query)
            print(f"Prediction: {'Yes' if prediction else 'No'}; expected: {'Yes' if label else 'No'}")
            if args.generate:
                print(f"Generated: {record['generated_text']}")
        if doa_test_dataset is not None:
            _, base_logp = score_answers(model, doa_test_dataset[0][i])
            alterations.append((token_logp.exp() * (token_logp - base_logp)).sum().item())
        records.append(record)
    consistency, positive, negative = action_consistency(predictions, labels)
    mean = lambda values: sum(values) / len(values)
    metrics = {
        "persona": args.target_persona, "method": args.mode,
        "model": args.model_dir, "K": args.K, "seed": args.seed,
        "action_consistency": consistency, "positive_consistency": positive,
        "negative_consistency": negative, "action_confidence": mean(confidences),
        "binary_confidence": mean(binary_confidences), "action_uncertainty": mean(entropies),
        "token_uncertainty": mean(token_entropies),
    }
    if args.generate:
        metrics["generated_consistency"] = action_consistency(generated_actions, labels)[0]
        metrics["valid_ratio"] = sum(value != -1 for value in generated_actions) / len(labels)
    if alterations:
        metrics["degree_of_alteration"] = mean(alterations)
    folder = Path(args.results_dir) / args.target_persona
    folder.mkdir(parents=True, exist_ok=True)
    name = args.exp_name or args.mode
    with (folder / f"{name}_predictions.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (folder / f"{name}_results.json").open("w") as file:
        json.dump({"metrics": metrics, "configuration": vars(args)}, file, indent=2)
    print(json.dumps(metrics, indent=2))
    return consistency, mean(confidences), mean(entropies), mean(token_entropies)


def get_model(args):
    if args.model == "qwen":
        from models.qwen import QwenWrapper
        return QwenWrapper(args.model_dir, args.dtype, args.load_in_4bit, args.max_input_len)
    if args.load_in_4bit:
        raise ValueError("--load_in_4bit is supported by the Qwen wrapper.")
    if args.model == "llama":
        from models.llama import LLaMAWrapper
        cls = LLaMAWrapper
    elif args.model == "vicuna":
        from models.vicuna import VicunaWrapper
        cls = VicunaWrapper
    else:
        from models.gptj import GPTJWrapper
        cls = GPTJWrapper
    model = cls(args.model_dir, memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb)
    model.family = args.model
    model.max_input_len = args.max_input_len
    return model


def resolve_adapter(args):
    path = Path(args.adapter_path) if args.adapter_path else adapter_directory(args)
    if args.likelihood_use_epoch is not None and not args.adapter_path:
        # Backward-compatible epoch lookup without assuming a batch size or steps/epoch.
        saved_config = path / "training_config.json"
        if saved_config.exists() and json.loads(saved_config.read_text()).get("num_epochs") == args.likelihood_use_epoch:
            return path
        for checkpoint in sorted(path.glob("checkpoint-*")):
            state = checkpoint / "trainer_state.json"
            if state.exists() and math.isclose(json.loads(state.read_text()).get("epoch", -1),
                                               args.likelihood_use_epoch, abs_tol=1e-3):
                return checkpoint
        raise FileNotFoundError(f"No saved adapter for epoch {args.likelihood_use_epoch} in {path}; use --adapter_path.")
    if not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"No persona adapter at {path}. Run --mode persona_sft first or set --adapter_path.")
    return path


def main(args):
    set_seed(args.seed)
    adapter_path = resolve_adapter(args) if args.mode == "picle" else None
    model = get_model(args)
    try:
        if args.mode == "persona_sft":
            train_model(args, model, get_sft_data(args, "train"), get_sft_data(args, "test"))
            return None
        if args.mode == "base":
            test_dataset = get_basic_data(args, "test", model)
        elif args.mode == "prompt_engineering":
            test_dataset = get_pe_data(args, model, "test")
        else:
            if args.mode == "picle":
                model.change_lora_adapter(adapter_path)
            test_dataset = get_icl_data(args, args.mode, args.K, ref_model=model)
        doa_dataset = get_basic_data(args, "test", model) if args.eval_doa else None
        # Persona LoRA estimates selection scores only. Final ICL uses the BASE.
        with base_model_context(model):
            return test_model(args, model, test_dataset, doa_dataset)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="PICLe with local Qwen2.5 persona LoRA and ICL.")
    parser.add_argument("--mode", required=True, choices=[
        "base", "prompt_engineering", "random", "similarity", "uncertainty",
        "likelihood", "diversity", "picle", "persona_sft",
    ])
    parser.add_argument("--target_persona", default="risk-averse", help="One persona, or 'all' explicitly.")
    parser.add_argument("--data_dir", help="Optional local directory containing persona JSONL files.")
    parser.add_argument("--model", choices=["qwen", "llama", "vicuna", "gptj"], default="qwen")
    parser.add_argument("--model_dir", help="HF model ID or local model directory.")
    parser.add_argument("--output_dir", help="Parent directory of persona adapters.")
    parser.add_argument("--results_dir", help="Parent directory of evaluation results.")
    parser.add_argument("--adapter_path", help="Saved persona LoRA directory for PICLe.")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--load_in_4bit", action="store_true", help="Optional Qwen NF4 loading; use consistently for all methods.")
    parser.add_argument("--max_input_len", type=int, default=1024, help="Reject longer inputs instead of silently truncating.")
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--pos_label_sample_only", action="store_true", help="PICLe+ label-aware pool, off by default.")
    parser.add_argument("--inst_delimiter", action="store_true", help="Legacy statement formatting, not used for Qwen.")
    parser.add_argument("--midlayer_for_sim", action="store_true")
    parser.add_argument("--penultlayer_for_sim", action="store_true")
    parser.add_argument("--choose_certain", action="store_true")
    parser.add_argument("--uncertainty_func", choices=["bin_entropy", "cat_entropy"], default="bin_entropy")
    parser.add_argument("--likelihood_func", choices=["plain", "diff"], default="diff",
                        help="'diff' is PICLe; 'plain' is the SFT-likelihood ablation.")
    parser.add_argument("--likelihood_use_epoch", type=int, help="Legacy option; prefer --adapter_path.")
    parser.add_argument("--pe_type", choices=["plain", "descriptive"], default="plain")
    parser.add_argument("--eval_doa", action="store_true")
    parser.add_argument("--generate", action="store_true", help="Also save greedy responses and generation validity.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--memory_for_model_activations_in_gb", type=int, default=4, help="Legacy wrappers only.")
    parser.add_argument("--exp_name", default="")
    args = parser.parse_args()
    if args.model_dir is None:
        if args.model != "qwen":
            parser.error("--model_dir is required for legacy backbones.")
        args.model_dir = "Qwen/Qwen2.5-7B-Instruct"
    args.output_dir = args.output_dir or f"checkpoints/{args.model}"
    args.results_dir = args.results_dir or f"out/{args.model}"
    if min(args.batch_size, args.gradient_accumulation_steps, args.num_epochs, args.K, args.max_input_len) < 1:
        parser.error("Batch size, accumulation steps, epochs, K and max input length must be positive.")
    if args.model == "qwen" and args.inst_delimiter:
        parser.error("Qwen uses apply_chat_template; omit --inst_delimiter.")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.target_persona == "all":
        personas = Path(__file__).with_name("personas.txt").read_text().splitlines()
        for persona in personas:
            args.target_persona = persona
            main(args)
    else:
        main(args)
