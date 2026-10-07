"""Single fail-closed recovery entry point, cumulative budget, and provenance gates."""
import argparse
import hashlib
import json
from pathlib import Path
from .recovery_budget import RecoveryBudget,validate_config
from .episode_v2 import EpisodeEnvV2
from .evaluation_v2 import select

POLICIES=("native_without_thinking","native_with_thinking","sft_switching","rl_switching",
          "always_direct","always_jev","always_cot","confidence")
REQUIRED_LOCAL=("local-numerical","full-model-parity","local-protocol")


def sha(path):
    from .verified_download import digest
    return digest(path)


def revisions(root):
    root=Path(root)
    paths=sorted((root/"switching").glob("*.py"))+sorted((root/"scripts").glob("*.py"))
    paths+=[root/"requirements-tunix-cpu.lock",root/"requirements-tunix-tpu.lock",root/"configs/recovery.json"]
    return {p.relative_to(root).as_posix():sha(p) for p in paths if p.is_file()}


def audit_data(folder):
    folder=Path(folder);manifest=json.loads((folder/"manifest.json").read_text());seen=set();counts={}
    expected={"train":1536,"val":120,"test":600}
    for split,n in expected.items():
        path=folder/f"{split}.jsonl"
        if sha(path)!=manifest["splits"][split]["sha256"]:raise ValueError("dataset checksum changed")
        rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(rows)!=n:raise ValueError("incorrect v2 dataset count")
        groups={r["group"] for r in rows}
        if seen&groups:raise ValueError("cross-split source overlap")
        seen|=groups;counts[split]=n
        for row in rows:
            if row.get("legacy") or not row["source"]["verified"]:raise ValueError("unverified or exposed episode in v2 dataset")
            if row["source"]["kind"] not in {"deterministic_v2","gsm8k_train"}:raise ValueError("private or unknown source")
            env=EpisodeEnvV2(row)
            for step in row["supervision"]:env.step(step["text"],step.get("choice_id"))
            _,result=env.reward(0)
            if not result["correct"] or not result["grounded"] or result["error"]:raise ValueError("teacher verification failed")
    return {"passed":True,"counts":counts,"manifest_sha256":sha(folder/"manifest.json"),"private_completions_imported":0}


def launch_gates(root,config,data,report_folder):
    root=Path(root);report_folder=Path(report_folder);current=revisions(root);issues=[]
    if not (root/"switching/tunix_experiment.py").is_file():issues.append("the complete Tunix stage runner has not been validated")
    try:
        from scripts.bootstrap_tunix import verify
        verify(root)
    except (FileNotFoundError,ValueError,KeyError) as exc:issues.append(f"pinned source verification failed: {exc}")
    for name in ("requirements-tunix-cpu.lock","requirements-tunix-tpu.lock"):
        path=root/name
        if not path.exists() or "--hash=sha256:" not in path.read_text():issues.append(f"missing dependency hash lock: {name}")
    for name in REQUIRED_LOCAL:
        path=report_folder/f"{name}.json"
        if not path.exists():issues.append(f"missing acceptance report: {name}");continue
        report=json.loads(path.read_text())
        if not report.get("passed"):issues.append(f"failed acceptance report: {name}")
        if not report.get('production_acceptance'):issues.append(f'full locked backend acceptance missing: {name}')
        if report.get("tunix_revision")!=config["tunix_revision"]:issues.append(f"backend revision mismatch: {name}")
        if report.get("revisions")!=current:issues.append(f"acceptance report is stale: {name}")
        if report.get("dataset_manifest_sha256")!=sha(Path(data)/"manifest.json"):issues.append(f"dataset revision mismatch: {name}")
        if name=="full-model-parity" and not report.get("full_model_probability_parity"):issues.append("full-model probability parity remains unverified")
    portable=report_folder/"converted/manifest.json"
    if not portable.exists():issues.append("best-SFT conversion is missing")
    else:
        source=json.loads(portable.read_text())
        if source["source_step"]!=150 or not source["fresh_optimizer"]:issues.append("incorrect recovery source")
        if not (portable.parent/'trainables.npz').is_file() or sha(portable.parent/'trainables.npz')!=source['npz_sha256']:
            issues.append('converted checkpoint failed integrity verification')
    return {"schema_version":2,"passed":not issues,"paid_launch_allowed":not issues,"issues":issues,
            "revisions":current,"dataset_manifest_sha256":sha(Path(data)/"manifest.json")}


def freeze_evaluation(rows,validation_seconds,budget,output,checkpoint_hashes,config):
    # Every available policy needs measured validation timings, including native baselines.
    applicable=[p for p in POLICIES if p!="rl_switching" or "rl" in checkpoint_hashes]
    if set(validation_seconds)!=set(applicable):raise ValueError("all applicable validation timings required before selecting test size")
    selected=None;projection=None
    times=[]
    for values in validation_seconds.values():
        if not values:raise ValueError("warmed validation samples required")
        times.extend([sorted(values)[min(len(values)-1,__import__('math').ceil(.9*len(values))-1)]])
    for count in config["test_sizes"]:
        batches=__import__('math').ceil(count/config.get('evaluation_parallelism',1))
        projection=budget.project("evaluation",[sum(times)],batches,compilation=config.get("evaluation_compilation_seconds",600),checkpoint_cleanup=300)
        if projection["allowed"]:selected=select(rows,count,config["seed"]);break
    if selected is None:raise RuntimeError("even 60 episodes per applicable policy cannot fit protected evaluation allowance")
    plan={"schema_version":2,"policies":applicable,"ids":[r["id"] for r in selected],"count_per_policy":len(selected),
          "checkpoint_hashes":checkpoint_hashes,"projection":projection,"seed":config["seed"],"read_test_results":False,
          "trained_decoding":{"temperature":1.,"top_p":1.,"top_k":None,"sample":True},
          "native_thinking_decoding":{"temperature":.6,"top_p":.95,"top_k":20},
          "native_direct_decoding":{"temperature":.7,"top_p":.8,"top_k":20},"context":max(config["buckets"]),
          "confidence_threshold":config.get('selected_confidence_threshold')}
    target=Path(output)
    if target.exists() and json.loads(target.read_text())!=plan:raise ValueError("sealed evaluation settings already frozen")
    target.write_text(json.dumps(plan,indent=2));return plan


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/recovery.json");p.add_argument("--data",default="data/recovery-v2")
    p.add_argument("--output",default="runs/recovery-v2");p.add_argument("--execute",action="store_true")
    a=p.parse_args();root=Path(__file__).resolve().parents[1];cfg=validate_config(json.loads(Path(a.config).read_text()))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);budget=RecoveryBudget(cfg,out/"budget.json")
    data=audit_data(a.data);gates=launch_gates(root,cfg,a.data,out)
    report={"schema_version":2,"data":data,"gates":gates,"prior_cost_upper_bound":budget.total(),"total_cap":cfg["hard_ceiling"],
            "stage_remaining":{s:budget.remaining(s) for s in ("pilot","sft","rl","evaluation")},
            "status":"ready-for-pilot" if gates["passed"] else "stopped-before-paid-training"}
    (out/"audit.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if a.execute:
        if not gates["passed"]:raise RuntimeError("local acceptance gate failed; no TPU will be created")
        from scripts.recovery_cloud import execute
        execute(cfg,out,a.data)


if __name__=="__main__":main()
