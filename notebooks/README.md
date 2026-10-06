# notebooks

## heretic_abliterate_qwen3_4b_thinking.ipynb

Runs [Heretic](https://github.com/p-e-w/heretic) on **Kaggle** (free T4/P100 GPU)
to abliterate the baseline model, then uploads the result to your Hugging Face
account. Chapter 3 of the experiment — the local Mac is too slow/tight for the
Optuna search, so abliteration happens on a cloud GPU and the model comes back
for conversion + eval.

**Why Kaggle, not the Mac:** Heretic's optimisation runs ~100–200 trials, each
generating responses to score refusals and KL-divergence. On a 16 GB M-series
that's hours on CPU (MPS is marginal for a 4B); a T4 does it in a few hours and
is free.

### Use it
1. Upload the `.ipynb` to [kaggle.com/code](https://www.kaggle.com/code) (New Notebook → File → Import).
2. **Accelerator → GPU**, **Internet → On**.
3. **Add-ons → Secrets → `HF_TOKEN`** = a Hugging Face token with **write** access.
4. Run all cells. It uploads to `<your-username>/Qwen3-4B-Thinking-2507-say-no-more`.

### How it stays non-interactive, and why a subprocess
Heretic's post-run menu is a `questionary` TUI, and in 1.4.0 only
`--export-strategy` is a real CLI flag. The notebook writes `run_heretic.py`,
which monkeypatches `questionary` to script every prompt (export=merge,
trial=best, action=upload, repo/visibility/reproducibility preset, token from
the environment) and raises a sentinel to stop after one upload. If a future
Heretic changes its prompt wording, update the `_pick()` matches there.

It runs that script in a **fresh `python` subprocess**, not in the notebook
kernel. Kaggle preloads numpy; installing Heretic pins a different numpy, so an
in-kernel import hits a half-swapped numpy (`_slice` / `_blas_supports_fpe`).
A restart fixes it interactively, but **Save & Run All (Commit)** can't restart.
The subprocess imports everything cleanly from disk, so the notebook runs the
same way interactively or headless. The install also pins `numpy>=2.2,<2.3` and
removes Kaggle's stale `torchao` (peft rejects torchao <0.16 if present).

### Then, back on the Mac
```bash
.venv/bin/hf download <user>/Qwen3-4B-Thinking-2507-say-no-more \
    --local-dir models/Qwen3-4B-Thinking-2507-say-no-more
.venv/bin/mlx_lm.convert --hf-path models/Qwen3-4B-Thinking-2507-say-no-more \
    --mlx-path models/Qwen3-4B-Thinking-2507-say-no-more-mlx-8bit -q --q-bits 8
MODEL=models/Qwen3-4B-Thinking-2507-say-no-more-mlx-8bit scripts/serve.sh
scripts/eval.sh 03-abliterated          # then eval/cyber_refusal_eval.py too
```
