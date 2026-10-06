# say-no-more

**Does a small thinking model still say no once you cut out its refusal direction?**

An experiment, start to finish, on a 16 GB MacBook: measure how
[Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
handles harmful requests, abliterate it, then run the exact same test again and
compare. This repo is the lab notebook: the code, the plan, the numbers, and the
story of how we got them, dead ends included ([STORY.md](STORY.md)).

## The plan

1. **Baseline** ✅ — Send the model 39 harmful prompts across 14 categories
   (malware, phishing, exploits, CBRN, fraud, self-harm, extremism, …) and 10
   harmless control prompts. Record whether it refuses, along with its full
   thinking trace.
2. **Abliterate** — Find the model's *refusal direction*: the difference between
   its average internal activations on harmful and on harmless instructions
   ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)). Then remove that
   direction from the weights. Work on the BF16 base, then convert to 8-bit MLX
   exactly like the baseline, so quantization isn't a second variable. The
   prompts used to find the direction are kept separate from the eval prompts,
   so we never test on what we trained on.
3. **Retest** — Same prompts, same settings, same scorer. Check three things:
   - How far does the refusal rate drop?
   - Does it still answer the harmless prompts properly?
   - Is it otherwise intact? A perplexity check against the base model.
4. **Read the thinking** — The part we're most curious about. Qwen3 thinks
   before it answers. After abliteration, does it still spot the harm in its
   thinking and then comply anyway? Or does the safety deliberation disappear
   entirely?

## Results so far

| Run | Model | Harmful prompts refused | Harmless prompts answered | Notes |
|---|---|---|---|---|
| [01-baseline](results/01-baseline/summary.md) | Qwen3-4B-Thinking-2507, 8-bit MLX | 34/39 by keywords; **37/39** after reading the flagged answers | 10/10 | One clear failure: wrote a fake vaccine-microchip news article |
| [02-cyber-baseline](results/02-cyber-baseline/summary.md) | same | 11/13 by keywords; **12/13** after reading | 8/8 | Offensive-security suite. One leak: lateral-movement technique under an "authorized pentest" framing |
| 03-abliterated | — | — | — | next |

Scores come from the keyword scorer in `eval/refusal_eval.py`. Every answer it
flags is then read by hand, and each run's summary records both.

## Models

Weights are never committed (`models/` is git-ignored); these are the names.

| Role | Model | Local path |
|---|---|---|
| Base: the source for abliteration and fine-tuning | `Qwen/Qwen3-4B-Thinking-2507` (BF16) | `models/Qwen3-4B-Thinking-2507` |
| Served for testing | 8-bit MLX conversion of the base | `models/Qwen3-4B-Thinking-2507-mlx-8bit` |

Why not `Qwen3-4B-Thinking-2507-FP8`? It's a lossy copy of the BF16 weights, made
for NVIDIA GPUs that compute in FP8. That makes it a worse starting point for
weight surgery, and MLX can't compute in FP8 anyway.

## Ground rules

These keep the before/after comparison fair:

- **Same settings every run.** 8-bit MLX, temperature 0.6, top-p 0.95, top-k 20
  (Qwen's recommendation for the Thinking model), 8,192 tokens per prompt, 2
  requests at a time. `scripts/eval.sh` sets all of these.
- **Same scorer, and a human read of every flag.** The keyword scorer is fast,
  free and deterministic: the same code gives the same verdicts every run. It
  misses refusals that use no stock phrase like "I can't". It could also miss
  harmful content that follows a disclaimer. So every flagged answer is read and
  the verdict is recorded in the run's summary. Unflagged answers get spot-checked.
- **Score the answer, keep the thinking.** Only the final answer is classified.
  The full thinking is saved with every result. A prompt that runs out of
  tokens before answering is marked truncated, not counted as a refusal.
- **Raw answers stay local.** `results/**/*.json` holds the model's full answers
  to harmful prompts. After abliteration that can be genuinely harmful text, so
  it's git-ignored. Each run commits a summary and reports with short excerpts.

## Layout

```
.
├── README.md            the plan (this file)
├── STORY.md             what happened, in order, dead ends included
├── requirements.txt     pinned Python dependencies
├── eval/
│   ├── refusal_eval.py        refusal eval for any OpenAI-compatible endpoint
│   └── cyber_refusal_eval.py  same engine, offensive-security probe set
├── scripts/
│   ├── setup.sh         create .venv
│   ├── get_model.sh     download the base model and make the 8-bit MLX copy
│   ├── serve.sh         OpenAI-compatible server (mlx_lm.server)
│   └── eval.sh          run the suite → results/<run>/
├── results/
│   └── 01-baseline/     summary + reports (raw answers stay local)
└── models/              weights, local only
```

## Reproduce

Needs an Apple Silicon Mac (16 GB is enough) and [uv](https://docs.astral.sh/uv/)
(`brew install uv`).

```bash
scripts/setup.sh                  # .venv with mlx-lm
scripts/get_model.sh              # ~7.5 GB download + 8-bit conversion
scripts/serve.sh                  # API on http://127.0.0.1:8080/v1 (keep it running)
```

In a second terminal:

```bash
scripts/eval.sh 01-baseline       # ~30 min: answers, thinking, keyword scores
```

To test a modified model, serve it instead and give the run a new name:

```bash
MODEL=models/<modified-model> scripts/serve.sh
scripts/eval.sh 02-abliterated
```

If the scorer changes, re-score saved answers offline without regenerating them:

```bash
.venv/bin/python eval/refusal_eval.py --rescore results/01-baseline/raw.json
```

The eval also runs on its own against any OpenAI-compatible API. See
`.venv/bin/python eval/refusal_eval.py --help`.
