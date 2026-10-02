# Persona In-Context Learning (PICLe)

Qwen backbone re-implementation of the ICML 2024 paper,
[PICLe: Eliciting Diverse Behaviors from Large Language Models with Persona In-Context Learning](https://proceedings.mlr.press/v235/choi24e.html),
by Hyeong Kyu Choi and Yixuan Li. The original Llama, Vicuna and GPT-J wrappers remain available.

The default backbone is **Qwen/Qwen2.5-7B-Instruct**. This is a re-implementation on Qwen,
not an exact reproduction of the original paper's backbone/results.

## Install

```bash
conda env create -f environment.yml
conda activate picle
```

Use a CUDA-compatible NVIDIA driver for the pinned PyTorch build. No paid LLM API is used.
Model weights and the Anthropic persona JSONL files are downloaded from Hugging Face.
An optional `HF_TOKEN` environment variable can authenticate those downloads.

## Run one persona

From the repository root:

```bash
bash scripts/qwen/run.sh risk-averse
```

This trains the persona LoRA for four epochs, then evaluates **Base, Persona Prompt,
Random ICL, Similarity ICL and PICLe**, all on the same Qwen checkpoint and 70/30 data split.
It also saves greedy generated answers. It does not run all personas by default.

To run the three requested personas:

```bash
bash scripts/qwen/run.sh risk-averse risk-seeking narcissism
```

The script accepts `MODEL_DIR` (HF ID or local model directory), `OUTPUT_DIR`,
`RESULTS_DIR`, `EPOCHS`, and `SEED` as environment variables.

## Run individual stages

```bash
python src/main.py --mode persona_sft --target_persona risk-averse
python src/main.py --mode base --target_persona risk-averse --generate
python src/main.py --mode prompt_engineering --target_persona risk-averse --generate
python src/main.py --mode random --target_persona risk-averse --generate
python src/main.py --mode similarity --target_persona risk-averse --generate
python src/main.py --mode picle --target_persona risk-averse --generate
```

Defaults: Qwen2.5-7B-Instruct, BF16, LoRA rank 8 / alpha 32 / dropout 0,
q_proj and v_proj, learning rate 2e-5, four epochs, micro-batch 1,
gradient accumulation 8, gradient checkpointing, and K=3.

Adapters are written to `checkpoints/qwen/<persona>/`. PICLe reads that adapter
automatically; `--adapter_path PATH` selects another saved adapter.
`--output_dir` changes the adapter parent directory.
`--data_dir PATH` uses local `<persona>.jsonl` files instead of downloading them.

Results are in `out/qwen/<persona>/`:
- `<method>_results.json`: metrics and arguments.
- `<method>_predictions.csv`: inputs, labels, predictions, answer log-probabilities,
  and generated text when `--generate` is enabled.
- `picle_selection.csv`: candidate Base/SFT log-likelihoods, their difference,
  and selected demonstration positions.

The first selected position is the earliest demonstration; the highest scoring
example is placed closest to the final query.

## PICLe semantics

1. **Persona SFT is statement language modeling.** As in the original implementation,
   three training statements are concatenated with period/newline delimiters and
   cyclically permuted. This is not question-to-Yes/No supervised classification.
2. For Qwen, the statement block is an assistant message under a fixed, neutral
   system prefix, rendered with the tokenizer's own chat template.
   Only statement content contributes to the training loss.
   Candidate scoring uses the same prefix and content token mask. Chat controls,
   the fixed prefix and padding are excluded; the first statement token is included.
   This fixed chat prefix is the Qwen formatting adaptation.
3. Every candidate statement is tokenized once. The same IDs and mask are evaluated
   with the persona adapter disabled and enabled. Selection is
   `delta(x) = sum_logp_SFT(x) - sum_logp_Base(x)`.
   There is no length normalization or probability-space subtraction.
4. Top-K is selected globally per persona. The original reverse-score context
   order is retained.
5. **Final ICL uses Qwen with the persona adapter disabled.** Only one copy of the
   backbone is resident during scoring and inference.

Base likelihoods are kept on CPU during selection. No persistent score cache
needs to be managed.

The default pool includes both labels. `--pos_label_sample_only` enables the
label-aware PICLe+ setting and must be applied to both training and selection.
`--likelihood_func diff` is the default PICLe rule; `plain` explicitly runs the
SFT-likelihood ablation.

All Qwen evaluation prompts and demonstrations use `apply_chat_template()`.
Random ICL retains sampling with replacement. Similarity ICL retains the
original last-token hidden representation and unnormalized dot product; it
uses Qwen's own hidden states and no separate embedding model.

## Evaluation

The main action prediction is the larger complete-string conditional likelihood
of `No` and `Yes`. The scorer handles multi-token strings and never indexes
hardcoded answer token IDs. The Qwen template's assistant prefix fixes answer
spacing; alternate strings can be passed to the scoring function explicitly.

Action consistency is agreement with the persona labels. Action confidence is
the raw probability of the chosen answer string. Binary confidence and action
entropy use probabilities normalized over the two strings. Token uncertainty
is the entropy at the first answer position.

`--generate` additionally saves greedy responses and reports their separate
consistency and valid-answer ratio. Invalid generations count as failures.
These metrics make the change from the old generated-token evaluation explicit;
they should not be presented as an exact numerical reproduction of the paper.

## A single 24GB GPU

The default uses one BF16 backbone, short variable-length inputs, micro-batch 1,
gradient accumulation and gradient checkpointing. Actual memory depends on input
length. `--max_input_len` defaults to 1024 and rejects longer sequences instead
of silently changing candidate statements or dropping demonstrations.

If BF16 training exceeds available memory, use the optional NF4 configuration:

```bash
pip install bitsandbytes==0.45.5
LOAD_IN_4BIT=1 bash scripts/qwen/run.sh risk-averse
```

Use the same precision/quantization setting for training, Base/SFT scoring and
all compared methods. The script passes this setting to every stage.
A 7B GPU training/evaluation run has not been performed in the editing environment.

## Original-code fixes

Separate from Qwen support:
- Load LoRA through PEFT, including alpha/r scaling, rather than manually adding B@A.
- Honor batch size and real epochs; stop deriving checkpoints from fixed 88/44-step counts.
- Correct shifted padding masks and the old wrapper's tuple unpacking.
- Fix the certainty/uncertainty option direction.
- Use complete answer probabilities instead of fixed Llama/GPT-J token IDs.
- Create output directories and use dataset labels explicitly for filtering/splitting.

Legacy backbones require an explicit `--model_dir`. All-persona execution is
opt-in with `--target_persona all`.

## Citation

```bibtex
@inproceedings{choi2024picle,
  title={PICLe: Eliciting Diverse Behaviors from Large Language Models with Persona In-Context Learning},
  author={Hyeong Kyu Choi and Yixuan Li},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```
