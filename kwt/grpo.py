"""Text-mode GRPO with stock Tunix, plus before/after evaluation.

    python -m kwt.grpo --out /tmp/kwt/run                 # full run (v5e-1)
    python -m kwt.grpo --smoke --out /tmp/kwt/smoke       # tiny CPU/TPU check

Writes to --out: eval_before.json, eval_after.json, lora.npz, summary.md.
Tunix checkpoints (resume after spot preemption) go to checkpoint_root_directory.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]

def model_overrides(name: str, hf_id: str) -> list[str]:
  """actor/reference/rollout configs inherit model_config in the Tunix CLI."""
  return [f"model_config.model_name={name}", f"model_config.model_id={hf_id}",
          f"tokenizer_config.tokenizer_path={hf_id}"]


SMOKE = [
    *model_overrides("qwen3-0.6b", "Qwen/Qwen3-0.6B"),
    "batch_size=2",
    "rl_training_config.max_steps=2",
    "grpo_config.num_generations=2",
    "rollout_config.total_generation_steps=48",
]


def _log(msg: str) -> None:
  print(f"[kwt {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_lora(model, path: str) -> int:
  import jax  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  from flax import nnx  # pylint: disable=g-import-not-at-top

  leaves = jax.tree_util.tree_flatten_with_path(nnx.state(model))[0]
  arrays = {jax.tree_util.keystr(p): np.asarray(jax.device_get(v))
            for p, v in leaves if "lora" in jax.tree_util.keystr(p).lower()}
  np.savez(path, **arrays)
  return len(arrays)


def load_lora(model, path: str) -> int:
  import jax  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  from flax import nnx  # pylint: disable=g-import-not-at-top

  saved = dict(np.load(path))
  state = nnx.state(model)
  hits = 0

  def swap(p, v):
    nonlocal hits
    key = jax.tree_util.keystr(p)
    if key in saved:
      hits += 1
      return jax.numpy.asarray(saved[key], dtype=v.dtype)
    return v

  nnx.update(model, jax.tree_util.tree_map_with_path(swap, state))
  if hits != len(saved):
    raise ValueError(f"loaded {hits} of {len(saved)} LoRA tensors")
  return hits


def _public_hf_patch() -> list[str]:
  """Download public models anonymously.

  Tunix's hf_pipeline calls an interactive `hf.login()` whenever HF_TOKEN is
  unset (even for public Qwen), and its CLI rejects hub tokenizer paths without
  a token. Use snapshot_download and hand Tunix local paths instead.
  """
  from huggingface_hub import snapshot_download  # pylint: disable=g-import-not-at-top
  from tunix.oss import utils as oss_utils  # pylint: disable=g-import-not-at-top

  patterns = ["*.json", "*.txt", "*.safetensors", "merges*", "vocab*"]
  oss_utils.hf_pipeline = lambda model_id, _dl: snapshot_download(
      model_id, allow_patterns=patterns)
  return patterns


def _local_tokenizer(overrides: list[str]) -> list[str]:
  key = "tokenizer_config.tokenizer_path="
  hub_id = next((o[len(key):] for o in reversed(overrides) if o.startswith(key)),
                "Qwen/Qwen3-1.7B")
  if hub_id.startswith(("/", "gs://")):
    return []
  from huggingface_hub import snapshot_download  # pylint: disable=g-import-not-at-top

  return [key + snapshot_download(hub_id, allow_patterns=_public_hf_patch())]


def _cpu_mesh_patch() -> None:
  """CPU devices have no chip coords; the pinned allocator needs them."""
  import jax  # pylint: disable=g-import-not-at-top
  from tunix.utils import mesh as mesh_lib  # pylint: disable=g-import-not-at-top

  if jax.default_backend() != "cpu":
    return

  def sequential(requirements, devices=None, **_):
    pool, out = list(devices or jax.devices()), {}
    for name, count in requirements:
      out[name], pool = pool[:count], pool[count:]
    return out

  mesh_lib.allocate_named_mesh_device_slices = sequential


def _random_weights_patch() -> None:
  """Local plumbing test: a tiny random Qwen3 instead of downloaded weights."""
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from tunix.models import automodel, dummy_model_creator  # pylint: disable=g-import-not-at-top
  from tunix.models.qwen3 import model as qwen3  # pylint: disable=g-import-not-at-top

  config = qwen3.ModelConfig(
      num_layers=2, vocab_size=151936, embed_dim=128, hidden_dim=256,
      num_heads=4, head_dim=32, num_kv_heads=2, rope_theta=1_000_000,
      norm_eps=1e-6, use_tied_embedding=True)

  def fake(*_, mesh=None, **__):
    return dummy_model_creator.create_dummy_model(
        qwen3.Qwen3, config, mesh=mesh, dtype=jnp.float32), None

  automodel.AutoModel.from_pretrained = staticmethod(fake)


def build(overrides: list[str]):
  from tunix.cli import grpo_main  # pylint: disable=g-import-not-at-top

  _cpu_mesh_patch()
  overrides = overrides + _local_tokenizer(overrides)
  argv = ["grpo_main", "base_config.yaml",
          f"override_config_file={ROOT / 'configs' / 'kwt_grpo.yaml'}", *overrides]
  pipe = grpo_main.GrpoPipeline(argv)
  # Inherited model_config.mesh would give every role its own device slice;
  # on one chip the reference and rollout models must reuse the actor mesh.
  for key in ("reference_model_config", "rollout_model_config"):
    pipe.config[key].pop("mesh", None)
    pipe.config[key]["same_mesh_as"] = "actor"
  return pipe


def step_metrics(prompts, completions, rewards, advantages, **kwargs):
  """Per-step routing log: keeps the VM hang guard fed, and feeds the blog."""
  del prompts, advantages
  from kwt import format as fmt  # pylint: disable=g-import-not-at-top

  opts = [json.loads(str(o)) for o in kwargs.get("options", [])]
  parsed = [fmt.parse(c, len(o)) for c, o in zip(completions, opts)]
  answers = kwargs.get("answer", [])
  n = max(len(parsed), 1)
  modes = collections.Counter(p.first_mode or "none" for p in parsed)
  row = {
      "reward": float(sum(rewards) / n),
      "valid": sum(p.valid for p in parsed) / n,
      "correct": sum(fmt.is_correct(p, str(a), o)
                     for p, a, o in zip(parsed, answers, opts)) / n,
      "chars": sum(len(c) for c in completions) / n,
      "switched": sum(p.switched for p in parsed) / n,
      **{f"mode_{m}": modes.get(m, 0) / n for m in ("jev", "direct", "cot", "none")},
  }
  print("[step] " + json.dumps(row), flush=True)
  return {}


def _dry_run(pipe, tokenizer, dataset) -> None:
  """Everything create_rl_engine/GrpoLearner configure, minus model weights."""
  from tunix.rl.grpo import grpo_learner  # pylint: disable=g-import-not-at-top

  meshes = pipe.create_role_to_mesh()
  pipe.create_cluster_config(role_to_mesh=meshes,
                             rollout_config=pipe.create_rollout_config(role_to_mesh=meshes))
  grpo_learner.GrpoConfig(**pipe._config_mapping("grpo_config"))  # pylint: disable=protected-access
  batch = next(iter(dataset))
  fns = pipe.obtain_reward_fn()
  n = len(batch["prompts"])
  fake = ["<mode:direct>\n<answer>0</answer>"] * n
  kw = {k: list(v) for k, v in batch.items() if k != "prompts"}
  scores = [fn(prompts=list(batch["prompts"]), completions=fake, **kw) for fn in fns]
  lengths = [len(tokenizer.encode(p)) for p in batch["prompts"]]
  _log(f"dry run ok: meshes={ {r.value: dict(m.shape) for r, m in meshes.items()} } "
       f"reward_fns={[f.__name__ for f in fns]} scores={scores} prompt_tokens={lengths}")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--smoke", action="store_true")
  ap.add_argument("--eval-rows", type=int, default=80, help="val rows per policy")
  ap.add_argument("--split", default="val", help="eval split (test is sealed)")
  ap.add_argument("--skip-before", action="store_true")
  ap.add_argument("--random-weights", action="store_true",
                  help="tiny random Qwen3 (local CPU plumbing test, no download)")
  ap.add_argument("--dry-run", action="store_true",
                  help="build every config, the data and rewards; load no weights")
  ap.add_argument("overrides", nargs="*", help="Tunix key=value overrides")
  args = ap.parse_args()

  from tunix.cli.utils import data as data_lib  # pylint: disable=g-import-not-at-top
  from tunix.rl.grpo import grpo_learner  # pylint: disable=g-import-not-at-top

  from kwt import evaluate, runtime, tasks  # pylint: disable=g-import-not-at-top

  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  overrides = (SMOKE if args.smoke else []) + list(args.overrides)
  if args.random_weights:
    _random_weights_patch()
  pipe = build(overrides)
  eval_rows = tasks.load(args.split, 8 if args.smoke else args.eval_rows)

  _log("loading tokenizer, data, models")
  tokenizer = pipe._get_tokenizer()  # pylint: disable=protected-access
  raw, batch_fn = pipe._load_raw_dataset(tokenizer)  # pylint: disable=protected-access
  pipe.compute_params(raw)
  dataset, _ = data_lib.post_init_dataset(
      raw, tokenizer,
      batch_size=pipe.config.get("batch_size", 1),
      num_batches=pipe.config.get("num_batches"),
      max_prompt_length=pipe._config_mapping("rollout_config").get("max_prompt_length"),  # pylint: disable=protected-access
      num_epochs=pipe.config.get("num_train_epochs", 1),
      custom_batch_fn=batch_fn,
  )
  if args.dry_run:
    _dry_run(pipe, tokenizer, dataset)
    return
  engine = pipe.create_rl_engine(tokenizer)
  actor = engine.actor_trainer.model

  def run_eval(name: str) -> None:
    _log(f"eval {name} on {len(eval_rows)} {args.split} rows")
    rep = evaluate.evaluate(runtime.SwitchingRuntime(actor, tokenizer), eval_rows)
    evaluate.dump(rep, str(out / f"eval_{name}.json"))
    _log("\n" + evaluate.table(rep))

  if not args.skip_before:
    run_eval("before")

  _log("GRPO training")
  learner = grpo_learner.GrpoLearner(
      rl_engine=engine,
      reward_fns=pipe.obtain_reward_fn(),
      metric_fns=[step_metrics],
      algo_config=grpo_learner.GrpoConfig(**pipe._config_mapping("grpo_config")),  # pylint: disable=protected-access
  )
  started = time.time()
  learner.train(dataset)
  _log(f"training done in {time.time() - started:.0f}s")

  n = save_lora(actor, str(out / "lora.npz"))
  _log(f"saved {n} LoRA tensors")
  run_eval("after")
  (out / "DONE").write_text(json.dumps({"overrides": overrides, "seconds": time.time() - started}))


if __name__ == "__main__":
  os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
  main()
