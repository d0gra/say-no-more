import os, sys, huggingface_hub

MODEL_ID     = os.environ["MODEL_ID"]
MODEL_COMMIT = os.environ.get("MODEL_COMMIT") or None
REPO_ID      = os.environ["REPO_ID"]
PRIVATE      = os.environ.get("PRIVATE", "0") == "1"
N_TRIALS     = os.environ.get("N_TRIALS", "100")
SEED         = os.environ.get("SEED", "0")
CHECKPOINT   = os.environ.get("CHECKPOINT", "/kaggle/working/checkpoints")
HF_TOKEN     = os.environ["HF_TOKEN"]

huggingface_hub.login(token=HF_TOKEN, add_to_git_credential=False)
huggingface_hub.get_token = lambda: HF_TOKEN   # Heretic reads the token from here

import heretic.main as hmain
import heretic.utils as hutils

_counts = {}
def _disp(c):
    if isinstance(c, str):
        return c
    t = getattr(c, "title", None)
    return t if isinstance(t, str) else str(getattr(c, "value", c))
def _val(c):
    if isinstance(c, str):
        return c
    return getattr(c, "value", c)
def _find(choices, needle):
    for c in choices:
        if needle in _disp(c).lower():
            return _val(c)
    return None

def patched_select(message, choices):
    m = str(message).lower()
    _counts[m] = _counts.get(m, 0) + 1
    if "how would you like to proceed" in m:      # checkpoint menu
        return (_find(choices, "show the results") or _find(choices, "continue")
                or _val(choices[0]))
    if "which trial" in m:
        if _counts[m] == 1:
            return _val(choices[0])               # best: fewest refusals
        return _find(choices, "exit") or _val(choices[-1])   # 2nd visit -> exit
    if "what do you want to do" in m:
        return _find(choices, "upload")
    if "export" in m:
        return _find(choices, "merge")
    if "public or private" in m:
        return _find(choices, "private" if PRIVATE else "public")
    if "reproducibility" in m:
        return _find(choices, "full")
    return _val(choices[0])

def patched_text(message, default="", qmark="?", unsafe=False):
    if "repository" in str(message).lower():
        return REPO_ID
    return default

def patched_path(message):
    return "/kaggle/working/out"

def patched_password(message):
    return HF_TOKEN

for mod in (hmain, hutils):
    mod.prompt_select   = patched_select
    mod.prompt_text     = patched_text
    mod.prompt_path     = patched_path
    mod.prompt_password = patched_password

sys.argv = ["heretic", "--model", MODEL_ID,
            "--export-strategy", "merge", "--n-trials", N_TRIALS, "--seed", SEED,
            "--study-checkpoint-dir", CHECKPOINT]
if MODEL_COMMIT:
    sys.argv += ["--model-commit", MODEL_COMMIT]

hmain.run()
print("DONE: abliterated model uploaded to https://huggingface.co/" + REPO_ID)
