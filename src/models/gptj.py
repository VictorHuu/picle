from models.scoring import generate_text, input_device, load_adapter, score_answers

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import get_max_memory
import numpy as np
from peft import PeftModel
import torch
import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, DataCollatorWithPadding, pipeline


# B_INST, E_INST = "[INST]", "[/INST]"
# B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# DEFAULT_SYSTEM_PROMPT = """\
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""



def load_gptj(model_name_or_path, memory_for_model_activations_in_gb=2, peft_path=None):
    max_memory = get_max_memory()
    for k in max_memory.keys():
       max_memory[k] -= memory_for_model_activations_in_gb * (2 ** 30)
    
    config = AutoConfig.from_pretrained(model_name_or_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=config.torch_dtype)
        model = load_checkpoint_and_dispatch(model, model_name_or_path, device_map="auto", max_memory=max_memory,
                                             no_split_module_classes=["GPTJBlock"])
    
    if peft_path is not None:
        model = PeftModel.from_pretrained(model, peft_path, device_map="auto", max_memory=max_memory)
    return model


def gather_last_token(tensor, lengths):
    batch_size = tensor.size(0)
    return tensor[torch.arange(batch_size, device=tensor.device), lengths.to(tensor.device) - 1, :]
    # return tensor[torch.arange(batch_size, device=tensor.device), lengths - 1, :]


class GPTJWrapper(object):
    family = "gptj"
    max_input_len = 1024
    def __init__(self, model_dir, lora_adapter_path=None, memory_for_model_activations_in_gb=2):
        super(GPTJWrapper, self).__init__()
        self.name = model_dir
        self.huggingface_model = load_gptj(model_dir, memory_for_model_activations_in_gb, lora_adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, legacy=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        

        self.generation_pipeline = pipeline(
            "text-generation",
            model=self.huggingface_model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            return_full_text=False,
        )

    def __call__(self, batch, output_log_likelihood=True, output_hidden_states=False, hidden_states_layers_to_output=(-1, -.5), output_only_last_token_hidden_states=False):
        with torch.no_grad():
            input_ids_cuda = batch['input_ids'].to(input_device(self))
            model_output = self.huggingface_model(input_ids=input_ids_cuda, attention_mask=batch['attention_mask'].to(input_device(self)), output_hidden_states=output_hidden_states)
            
            logits_before_softmax = model_output['logits'].float()[:, :-1, :]
            next_token_logit = model_output['logits'].float()[:, -1:, :]
            if output_log_likelihood:
                logits = torch.nn.functional.log_softmax(logits_before_softmax, dim=2)
                tokens_log_likelihood = -1. * torch.nn.functional.nll_loss(logits.permute(0, 2, 1), input_ids_cuda[:, 1:], reduction="none").detach().cpu()
                actual_token_vs_padding_tokens_mask = (batch['attention_mask'][:, 1:] * batch['attention_mask'][:, :-1]).cpu()
                log_likelihood = (tokens_log_likelihood * actual_token_vs_padding_tokens_mask).sum(dim=1)
            else:
                tokens_log_likelihood = None
                log_likelihood = None
            if output_hidden_states:            
                hidden_states_results = []
                for layer_idx in hidden_states_layers_to_output:
                    if layer_idx == -.5:
                        layer_idx = len(model_output.hidden_states) // 2
                    current_layer_hidden_states = model_output.hidden_states[layer_idx].cpu()
                    if output_only_last_token_hidden_states:
                        current_layer_hidden_states = gather_last_token(current_layer_hidden_states, batch['length'])
                    hidden_states_results.append(current_layer_hidden_states)
                hidden_states_results = tuple(hidden_states_results)
            else:
                hidden_states_results = None

            return hidden_states_results, logits_before_softmax.cpu(), tokens_log_likelihood, log_likelihood, next_token_logit

    def _forward_whole_dataset_generator(self, dataset, batch_size, **kwargs):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, collate_fn=DataCollatorWithPadding(self.tokenizer))
        for i, batch in enumerate(tqdm.tqdm(dataloader, desc="batch")):
            yield self.__call__(batch, **kwargs)

    def forward_whole_dataset(self, dataset, batch_size, output_tokens_log_likelihood=False, output_logits_before_softmax=False, **kwargs):
        res = None
        for i, current_res in enumerate(self._forward_whole_dataset_generator(dataset, batch_size, **kwargs)):
            current_hidden_states, current_logits_before_softmax, current_tokens_log_likelihood, current_log_likelihood, _ = current_res
            if res is None:
                res = [None, None, None, None]
                if current_hidden_states is not None:
                    if len(current_hidden_states[0].shape) == 3:
                        hidden_states_shape = (len(dataset), current_hidden_states[0].shape[1], current_hidden_states[0].shape[2])
                    else:
                        hidden_states_shape = (len(dataset), current_hidden_states[0].shape[1])
                    res[0] = tuple([np.zeros(hidden_states_shape, dtype=current_hidden_states[0].numpy().dtype) for _ in range(len(current_hidden_states))])
                if output_logits_before_softmax:
                    res[1] = np.zeros((len(dataset), current_logits_before_softmax.shape[1], current_logits_before_softmax.shape[2]), dtype=current_logits_before_softmax.numpy().dtype)
                if current_tokens_log_likelihood is not None and output_tokens_log_likelihood:
                    res[2] = np.zeros(shape=(len(dataset), current_tokens_log_likelihood.shape[1]), dtype=current_tokens_log_likelihood.numpy().dtype)
                if current_log_likelihood is not None:
                    res[3] = np.zeros(shape=(len(dataset),), dtype=current_log_likelihood.numpy().dtype)
            if current_hidden_states is not None:
                for j in range(len(current_hidden_states)):
                    res[0][j][i*batch_size:(i+1)*batch_size, ...] = current_hidden_states[j].numpy()
            if output_logits_before_softmax:
                res[1][i*batch_size:(i+1)*batch_size, ...] = current_logits_before_softmax.numpy()
            if current_tokens_log_likelihood is not None and output_tokens_log_likelihood:
                res[2][i*batch_size:(i+1)*batch_size, ...] = current_tokens_log_likelihood.numpy()
            if current_log_likelihood is not None:
                res[3][i*batch_size:(i+1)*batch_size] = current_log_likelihood.numpy()
        return tuple(res)

    def change_lora_adapter(self, new_lora_adapter_path):
        load_adapter(self, new_lora_adapter_path)

    def generate(self, args, query, return_logits=True, verbose=False):
        response = generate_text(self, query)
        if verbose:
            print(query, response, sep="\n")
        # Scores are complete No/Yes string log-probabilities, in that order.
        return (response, score_answers(self, query)[0]) if return_logits else response
