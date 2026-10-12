"""SFT warm-start -> GRPO -> eval, each in its own process (whole chip each).

    python -m kwt.pipeline --out out [--sft-steps 256] [overrides...]
"""

import argparse
import subprocess
import sys


def run(*args: str) -> None:
  print("[pipeline] " + " ".join(args), flush=True)
  subprocess.run([sys.executable, "-m", *args], check=True)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--sft-steps", default="256")
  ap.add_argument("--sft-lora", help="reuse an existing SFT LoRA (skip SFT + its eval)")
  ap.add_argument("--random-weights", action="store_true")
  ap.add_argument("--eval-limit", default="0")
  ap.add_argument("overrides", nargs="*")
  args, extra = ap.parse_known_args()
  o, rw = args.out, (["--random-weights"] if args.random_weights else [])
  sft = args.sft_lora or f"{o}/sft/lora.npz"
  if not args.sft_lora:
    run("kwt.sft", "--out", f"{o}/sft", "--steps", args.sft_steps, *rw, *args.overrides)
    run("kwt.evaluate", "--lora", sft, "--out", f"{o}/eval_sft",
        "--splits", "val", "--skip-base", "--limit", args.eval_limit, *rw, *args.overrides)
  run("kwt.grpo", "--out", f"{o}/grpo", "--skip-before", "--init-lora", sft, "--eval-rows", "0", *rw, *extra, *args.overrides)
  run("kwt.evaluate", "--lora", f"{o}/grpo/lora.npz", "--out", f"{o}/eval_grpo",
      "--splits", "val", "test", "--skip-base", "--limit", args.eval_limit, *rw,
      *args.overrides)
  print("[pipeline] done", flush=True)


if __name__ == "__main__":
  main()
