# Qwen PICLe

This repository adapts PICLe, the ICML 2024 persona in-context learning method, to
**Qwen2.5-7B-Instruct**. It preserves PICLe's persona LoRA SFT, likelihood-difference
scoring, and Top-K demonstration selection. This is a Qwen re-implementation, not
an exact reproduction of the paper's original backbone results.

## Run PICLe and its five history conditions

Use the AutoDL image with PyTorch 2.8.0, Python 3.12, and CUDA 12.8. Since that image
already includes PyTorch, install the remaining dependencies in its existing Python
environment:

```bash
python -m pip install transformers==4.51.3 peft==0.15.2 accelerate==1.6.0 datasets==3.5.0 huggingface-hub==0.30.2 numpy==1.26.4 scikit-learn==1.6.1 sentencepiece==0.2.0 safetensors==0.5.3 tqdm==4.67.1
```

From the repository root, run the single command below. It performs the PICLe
selection and evaluates all five conditions on the same histories:

```bash
python src/auction_picle.py --mode all
```

The five conditions are **Direct PICLe**, **Mask**, **Reverse**, **Shuffle**, and
**RegionShuffle**. The script uses each participant's first 30 auction rounds for
persona-specific LoRA SFT and candidate selection, then predicts the remaining
rounds with the Qwen base model. It downloads the public auction data and instructions
on first run; no paid API is used. Mask hides round numbers, following the paper text.

To limit the run, pass bidder groups with `--bidder_groups S.1 S.2`. Use `--epochs`,
`--K`, and `--results_file` to change training epochs, selected demonstrations, and
the prediction output path.

Per-round predictions are saved to `out/qwen/auction_picle.csv`. Per-group, per-condition
metrics are saved to `out/qwen/auction_picle_metrics.csv`:

- `w1_reserve_price`: empirical 1-Wasserstein distance between predicted and human
  held-out reserve-price distributions, in reserve-price units.
- `ks_distance`: KS distance, retained for comparison with the paper's metric.
- `n_scored`: number of held-out rounds with valid parsed predictions.

PICLe selects demonstrations by
`delta(x) = log p_SFT(x) - log p_Base(x)` and chooses the Top-K scores. All five
conditions use these selected demonstrations; the four ablations change how their
history is presented. Final inference disables the persona adapter.

## Citation

```bibtex
@inproceedings{choi2024picle,
  title={PICLe: Eliciting Diverse Behaviors from Large Language Models with Persona In-Context Learning},
  author={Hyeong Kyu Choi and Yixuan Li},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```
