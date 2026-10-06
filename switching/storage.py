"""Atomic local checkpoints and checksummed, completion-marked GCS uploads."""
import concurrent.futures
import hashlib
import json
import os
import random
import time
import uuid
from pathlib import Path

def checksum(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rng_tuple(value):
    """XLA's checkpoint traversal may serialize tuples as lists."""
    return tuple(rng_tuple(item) for item in value) if isinstance(value,(tuple,list)) else value

class Budget:
    def __init__(self, path, rate, stage, limits, start=None,prior_spend=0):
        self.path=Path(path); self.rate=float(rate); self.stage=stage; self.limits=limits
        if self.rate <= 0: raise ValueError("whole-slice hourly rate must be verified and positive")
        self.started=time.time() if start is None else start
        self.previous=json.loads(self.path.read_text()) if self.path.exists() else {"stages":{"pilot":prior_spend},"limit":50}
        self.base_stage=self.previous["stages"].get(stage,0)

    def remaining(self):
        cost=(time.time()-self.started)/3600*self.rate*1.15
        stage_cost=self.base_stage+cost
        total=sum(self.previous["stages"].values())+cost
        return min(self.limits[self.stage]-stage_cost,50-total)

    def check(self, seconds_required=0):
        if self.remaining() <= seconds_required/3600*self.rate*1.15:
            raise RuntimeError("stage cannot complete within the enforced spending limit")

    def finish(self):
        cost=(time.time()-self.started)/3600*self.rate*1.15
        self.previous["stages"][self.stage]=self.base_stage+cost
        self.previous["estimated_total"]=sum(self.previous["stages"].values())
        self.previous["accounting"]="Conservative time/rate estimate, not a billing export; includes 15% margin."
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temporary=self.path.with_suffix(".tmp"); temporary.write_text(json.dumps(self.previous,indent=2)); temporary.replace(self.path)

class Checkpoints:
    def __init__(self, root, gcs=None):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self.gcs=gcs
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=1); self.jobs=[]

    def save(self, tag, model, optimizer, scheduler, progress, rank=0, reference=None):
        import torch
        directory=self.root/f"step-{progress['step']:06d}-{tag}-rank{rank}-{uuid.uuid4().hex[:8]}"; directory.mkdir(exist_ok=False)
        state={"trainable":model.trainable_state(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"progress":progress,"python_rng":random.getstate(),"torch_rng":torch.get_rng_state(),"reference":reference}
        if next(model.parameters()).device.type=="xla":
            import torch_xla.core.xla_model as xm
            state["xla_rng"]=xm.get_rng_state()
            xm.save(state,str(directory/"state.pt"),master_only=False)
        else: torch.save(state,directory/"state.pt")
        if hasattr(model.tokenizer,"save_pretrained"):
            model.tokenizer.save_pretrained(directory/"tokenizer")
        files={str(p.relative_to(directory)).replace("\\","/"):checksum(p) for p in directory.rglob("*") if p.is_file()}
        manifest={"files":files,"progress":progress,"rank":rank,"architecture":model.config,"mode_tokens":["<mode:jev>","<mode:direct>","<mode:cot>"]}
        (directory/"manifest.json").write_text(json.dumps(manifest,indent=2))
        (directory/"COMPLETE").write_text(checksum(directory/"manifest.json"))
        pointer=self.root/f"{tag}-rank{rank}.json"; temp=pointer.with_suffix(".tmp")
        temp.write_text(json.dumps({"directory":directory.name})); temp.replace(pointer)
        if self.gcs: self.jobs.append(self.pool.submit(self.upload,directory))
        return directory

    def upload(self,directory):
        from google.cloud import storage
        bucket_name,prefix=self.gcs.removeprefix("gs://").split("/",1)
        bucket=storage.Client().bucket(bucket_name)
        base=f"{prefix.rstrip('/')}/{directory.name}"
        for path in sorted((p for p in directory.rglob("*") if p.is_file()),key=lambda p:p.name=="COMPLETE"):
            relative=path.relative_to(directory).as_posix()
            bucket.blob(f"{base}/{relative}").upload_from_filename(str(path),if_generation_match=0)

    def load(self,directory,model,optimizer=None,scheduler=None):
        import torch
        directory=Path(directory)
        if (directory/"COMPLETE").read_text()!=checksum(directory/"manifest.json"): raise ValueError("incomplete manifest")
        manifest=json.loads((directory/"manifest.json").read_text())
        for name,digest in manifest["files"].items():
            if checksum(directory/name)!=digest: raise ValueError("checkpoint checksum mismatch")
        state=torch.load(directory/"state.pt",map_location="cpu",weights_only=False)
        self.loaded_reference=state.get("reference")
        model.restore_trainable(state["trainable"])
        if optimizer is not None: optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None: scheduler.load_state_dict(state["scheduler"])
        random.setstate(rng_tuple(state["python_rng"])); torch.set_rng_state(state["torch_rng"])
        if "xla_rng" in state and next(model.parameters()).device.type=="xla":
            import torch_xla.core.xla_model as xm
            xm.set_rng_state(state["xla_rng"])
        return state["progress"]

    def restore_gcs(self,uri,destination):
        from google.cloud import storage
        bucket_name,prefix=uri.removeprefix("gs://").split("/",1)
        bucket=storage.Client().bucket(bucket_name); destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
        for name in ("COMPLETE","manifest.json","state.pt"):
            bucket.blob(f"{prefix.rstrip('/')}/{name}").download_to_filename(str(destination/name))
        manifest=json.loads((destination/"manifest.json").read_text())
        for name in manifest["files"]:
            if name=="state.pt":continue
            target=(destination/name).resolve()
            if not target.is_relative_to(destination.resolve()):raise ValueError("unsafe checkpoint path")
            target.parent.mkdir(parents=True,exist_ok=True)
            bucket.blob(f"{prefix.rstrip('/')}/{name}").download_to_filename(str(target))
        return destination

    def close(self):
        self.pool.shutdown(wait=True)
        for job in self.jobs: job.result()
