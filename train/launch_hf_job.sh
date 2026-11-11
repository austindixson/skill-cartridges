#!/usr/bin/env bash
# Reference launcher for the memory / code / LFM3 cartridges on Hugging Face Jobs. Needs an authenticated `hf` (HF_TOKEN in env or `hf auth login`); never prints it.
# Usage: bash launch_hf_job.sh code|memory|lfm|seed-<code|memory|lfm> 
# Rate rtx-pro-6000 = $2.75/h = $0.0458333/min. Worst case per job = timeout x rate (billing is per minute, --timeout is the hard cap).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
HF=${HF:-hf}
PY=${PY:-python}
NS=${HF_NAMESPACE:?set HF_NAMESPACE to your Hugging Face user or org}
unset HF_TOKEN  # use the stored `hf auth login`; bare --secrets HF_TOKEN resolves it via get_token()
WHAT=${1:?code|memory|lfm|seed-code|seed-memory|seed-lfm}
SEED=0; SUFFIX=1
case "$WHAT" in seed-*) WHAT=${WHAT#seed-}; SEED=2; SUFFIX=seed2 ;; esac
COMMON=(--train_steps 400 --save_steps 100 --log_every 10 --min_steps 100 --probe_steps 5
        --save_reserve_s 180 --save_cost_s 30 --devices cuda:0 --out_dir /tmp/out --seed "$SEED")
case "$WHAT" in
  code)   REPO=$NS/deltune-qwen3.5-9b-code-mask; TIMEOUT=50; RUN=9b-code-$SUFFIX
          ARGS=(--model Qwen/Qwen3.5-9B --task code --eval_batch_size 64 --max_new_tokens 768 --eval_reserve_s 900) ;;
  memory) REPO=$NS/deltune-qwen3.5-9b-memory-mask; TIMEOUT=50; RUN=9b-memory-$SUFFIX
          ARGS=(--model Qwen/Qwen3.5-9B --task memory --max_seq_len 3072 --eval_batch_size 32 --max_new_tokens 64 --eval_reserve_s 600) ;;
  lfm)    REPO=$NS/deltune-lfm3-1.7b-toolalpaca-mask; TIMEOUT=30; RUN=lfm3-1.7b-toolalpaca-$SUFFIX
          ARGS=(--model LiquidAI/LFM3-1.7B --trust_remote_code --task toolalpaca --eval_batch_size 32 --eval_reserve_s 600) ;;
  *) echo "unknown $WHAT" >&2; exit 2 ;;
esac
BUDGET=$(( TIMEOUT * 60 - 180 ))
"$PY" - "$REPO" <<'PY'
import sys, urllib.request, urllib.error
from huggingface_hub import HfApi
repo = sys.argv[1]; api = HfApi()
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
assert api.model_info(repo).private, f"{repo} is not private: refusing"
code = None
try:
    urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=30)
except urllib.error.HTTPError as e:
    code = e.code
if code != 401:
    raise SystemExit(f"{repo}: anonymous API request returned {code}, not 401: refusing")
print(f"repo {repo}: private=True, anon HTTP {code}")
PY
WORST=$("$PY" -c "print(round($TIMEOUT*2.75/60,4))")
echo "launch $RUN timeout ${TIMEOUT}m worst \$$WORST at $(date '+%-I:%M:%S %p %Z')"
"$HF" jobs uv run --flavor rtx-pro-6000 --timeout "${TIMEOUT}m" --name "deltune-${RUN//./_}" \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True -e HF_XET_HIGH_PERFORMANCE=1 -e TOKENIZERS_PARALLELISM=false \
  -e DELTUNE_OUT_BUCKET="$REPO" --secrets HF_TOKEN --detach \
  "$HERE/train_tasks.py" "${ARGS[@]}" "${COMMON[@]}" --wallclock_budget_s "$BUDGET" --run_name "$RUN"
