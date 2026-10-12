#!/usr/bin/env bash
# Free CPU smoke test of the whole kwt path (Qwen3-0.6B, 2 GRPO steps) in the
# locked Linux runtime. Run before every paid TPU launch.
set -euo pipefail
cd "$(dirname "$0")/.."
root=$(pwd -W 2>/dev/null || pwd)
mkdir -p runs/kwt
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$root:/work" -v "$root/.cache/huggingface:/root/.cache/huggingface" \
  -e PYTHONPATH=/work/.cache/kwt-pydeps:/work:/work/.vendor/tunix -e JAX_PLATFORMS=cpu -w /work \
  kws-tunix-dependency-cache:all-prepared \
  /work/.venv-tunix-linux/bin/python -m kwt.grpo --smoke --out /work/runs/kwt/smoke \
  rl_training_config.checkpoint_root_directory=/tmp/kwt/ckpt "$@" 2>&1 | tee runs/kwt/smoke.log
