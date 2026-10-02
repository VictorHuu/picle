"""Local causal-LM scoring and PEFT adapter handling, without vocabulary IDs."""

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from peft import PeftConfig, PeftModel
from tqdm import tqdm

from data.formatting import SYSTEM_MESSAGE


def input_device(wrapper):
    return wrapper.huggingface_model.get_input_embeddings().weight.device


def encode(wrapper, text):
    ids = wrapper.tokenizer(text, add_special_tokens=wrapper.family != "qwen")["input_ids"]
    if len(ids) > wrapper.max_input_len:
        raise ValueError(
            f"Input has {len(ids)} tokens, exceeding --max_input_len={wrapper.max_input_len}. "
            "Increase that option; candidate statements and demonstrations are not truncated."
        )
    return ids


def content_feature(wrapper, prefix, content, suffix=""):
    """Tokenize the full string; supervise only content, including its first token."""
    before = encode(wrapper, prefix)
    through_content = encode(wrapper, prefix + content)
    ids = encode(wrapper, prefix + content + suffix)
    if through_content[:len(before)] != before or ids[:len(through_content)] != through_content:
        raise ValueError("The tokenizer changed tokens across a content boundary.")
    labels = [-100] * len(before) + ids[len(before):len(through_content)]
    labels += [-100] * (len(ids) - len(through_content))
    if not any(label != -100 for label in labels[1:]):
        raise ValueError("The input has no content tokens to score.")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def statement_feature(wrapper, text):
    if wrapper.family == "qwen":
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
        prefix = wrapper.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full = wrapper.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": text}],
            tokenize=False, add_generation_prompt=False,
        )
        if not full.startswith(prefix + text):
            raise ValueError("The chat template does not preserve the statement content.")
        return content_feature(wrapper, prefix, text, full[len(prefix + text):])
    ids = encode(wrapper, text)
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids.copy()}


def model_inputs(wrapper, feature):
    return {key: torch.tensor([feature[key]], device=input_device(wrapper))
            for key in ("input_ids", "attention_mask")}


def content_log_probability(logits, feature):
    labels = torch.tensor(feature["labels"][1:], device=logits.device)
    valid = labels.ne(-100)
    # Selection uses a SUM, not the model's mean training loss or perplexity.
    return -F.cross_entropy(logits[0, :-1][valid].float(), labels[valid], reduction="sum")


@torch.inference_mode()
def score_features(wrapper, features, description="Likelihood"):
    wrapper.huggingface_model.eval()
    scores = []
    for feature in tqdm(features, desc=description):
        result = wrapper.huggingface_model(**model_inputs(wrapper, feature), use_cache=False)
        scores.append(content_log_probability(result.logits, feature).cpu())
        del result
    scores = torch.stack(scores)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Non-finite likelihood; check the model dtype and training loss.")
    return scores


@torch.inference_mode()
def score_answers(wrapper, prompt, answers=None):
    """Return log p(No), log p(Yes), and the first answer-token distribution.

    Each answer can contain multiple tokens. Alternative strings such as ' Yes'
    can be supplied explicitly; there are no assumptions about token IDs.
    """
    if answers is None:
        answers = ("No", "Yes") if wrapper.family == "qwen" else (" No", " Yes")
    wrapper.huggingface_model.eval()
    scores, next_log_probs = [], None
    for answer in answers:
        feature = content_feature(wrapper, prompt, answer)
        result = wrapper.huggingface_model(**model_inputs(wrapper, feature), use_cache=False)
        scores.append(content_log_probability(result.logits, feature).cpu())
        if next_log_probs is None:
            first = next(i for i, label in enumerate(feature["labels"]) if label != -100)
            next_log_probs = result.logits[0, first - 1].float().log_softmax(-1).cpu()
        del result
    scores = torch.stack(scores)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Non-finite answer log-probability.")
    return scores, next_log_probs


@torch.inference_mode()
def generate_text(wrapper, prompt, max_new_tokens=16):
    wrapper.huggingface_model.eval()
    ids = encode(wrapper, prompt)
    inputs = {"input_ids": torch.tensor([ids], device=input_device(wrapper)),
              "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=input_device(wrapper))}
    result = wrapper.huggingface_model.generate(
        **inputs, do_sample=False, max_new_tokens=max_new_tokens,
        pad_token_id=wrapper.tokenizer.pad_token_id,
        eos_token_id=wrapper.tokenizer.eos_token_id,
    )
    return wrapper.tokenizer.decode(result[0, len(ids):], skip_special_tokens=True).strip()


@torch.inference_mode()
def embed_statements(wrapper, statements, layer=-1):
    wrapper.huggingface_model.eval()
    causal_lm = wrapper.huggingface_model
    if isinstance(causal_lm, PeftModel):
        causal_lm = causal_lm.get_base_model()
    embeddings = []
    for statement in tqdm(statements, desc="Statement embeddings"):
        ids = encode(wrapper, statement)
        feature = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        result = causal_lm.base_model(
            **model_inputs(wrapper, feature), output_hidden_states=True, use_cache=False, return_dict=True,
        )
        index = len(result.hidden_states) // 2 if layer == -0.5 else layer
        embeddings.append(result.hidden_states[index][0, -1].float().cpu())
    return torch.stack(embeddings)


def load_adapter(wrapper, path):
    config = PeftConfig.from_pretrained(str(path))
    if config.bias != "none" or config.modules_to_save:
        raise ValueError("PICLe requires a LoRA-only adapter with bias='none' to recover the same base model.")
    model = wrapper.huggingface_model
    if isinstance(model, PeftModel):
        model = model.unload()
    wrapper.huggingface_model = PeftModel.from_pretrained(model, str(path), is_trainable=False)
    wrapper.huggingface_model.eval()


def base_model_context(wrapper):
    model = wrapper.huggingface_model
    return model.disable_adapter() if isinstance(model, PeftModel) else nullcontext()
