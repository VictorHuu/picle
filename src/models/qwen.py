"""Qwen2.5 backbone for PICLe; one set of base weights stays in memory."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from models.scoring import generate_text, load_adapter, score_answers


class QwenWrapper:
    family = "qwen"

    def __init__(self, model_dir="Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16",
                 load_in_4bit=False, max_input_len=1024):
        self.name = model_dir
        self.max_input_len = max_input_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if not self.tokenizer.chat_template:
            raise ValueError("Use Qwen2.5-7B-Instruct, which provides a chat template.")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = getattr(torch, dtype) if device != "cpu" else torch.float32
        kwargs = dict(torch_dtype=torch_dtype, device_map={"": device}, attn_implementation="sdpa")
        if load_in_4bit:
            if device == "cpu":
                raise ValueError("--load_in_4bit requires a CUDA GPU.")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch_dtype,
            )
        self.huggingface_model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
        self.huggingface_model.eval()

    def change_lora_adapter(self, path):
        load_adapter(self, path)

    def generate(self, args, query, return_logits=True, verbose=False):
        response = generate_text(self, query)
        if verbose:
            print(query, response, sep="\n")
        return (response, score_answers(self, query)[0]) if return_logits else response
