"""Verified Orbax bundles: adapters, head, rows, reference, and full recovery."""
import concurrent.futures
import hashlib
import json
import uuid
from pathlib import Path
import orbax.checkpoint as ocp
from .tunix_optim import assert_finite


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CheckpointsV2:
    def __init__(self,root,gcs=None):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.gcs=gcs;self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=1);self.jobs=[]

    def save(self,tag,state,metadata):
        if metadata.get("schema_version")!=2 or metadata.get("backend")!="tunix":raise ValueError("v2 metadata required")
        for key in ("actor","reference","optimizer"):
            if key not in state:raise ValueError(f"missing recoverable state: {key}")
            assert_finite(state[key],f"checkpoint-{key}")
        for key in ("rng","cursor","accepted_updates","scheduler"):
            if key not in state:raise ValueError(f"missing recovery field: {key}")
        folder=self.root/f"step-{int(state['accepted_updates']):06d}-{tag}-{uuid.uuid4().hex[:8]}";folder.mkdir()
        checkpointer=ocp.PyTreeCheckpointer()
        try:checkpointer.save(str(folder/"orbax"),state)
        finally:checkpointer.close()
        files={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob("*") if p.is_file()}
        manifest={**metadata,"files":files,"accepted_updates":int(state["accepted_updates"])}
        (folder/"manifest.json").write_text(json.dumps(manifest,indent=2));(folder/"COMPLETE").write_text(sha(folder/"manifest.json"))
        pointer=self.root/f"{tag}.json";temp=pointer.with_suffix(".tmp");temp.write_text(json.dumps({"directory":folder.name}));temp.replace(pointer)
        if self.gcs:self.jobs.append(self.pool.submit(self.upload,folder))
        return folder

    def load(self,folder,template,expected):
        folder=Path(folder).resolve();manifest=json.loads((folder/"manifest.json").read_text())
        if (folder/"COMPLETE").read_text()!=sha(folder/"manifest.json"):raise ValueError("incomplete checkpoint")
        for key,value in expected.items():
            if manifest.get(key)!=value:raise ValueError(f"checkpoint compatibility mismatch: {key}")
        for name,digest in manifest["files"].items():
            path=(folder/name).resolve()
            if not path.is_relative_to(folder):raise ValueError("unsafe checkpoint path")
            if sha(path)!=digest:raise ValueError("checkpoint checksum mismatch")
        checkpointer=ocp.PyTreeCheckpointer()
        try:
            # Restore onto the caller's topology, never silently reuse saved device IDs.
            restore_args=ocp.checkpoint_utils.construct_restore_args(template)
            state=checkpointer.restore(str(folder/"orbax"),item=template,restore_args=restore_args)
        finally:checkpointer.close()
        for key in ("actor","reference","optimizer"):assert_finite(state[key],f"restore-{key}")
        return state

    def upload(self,folder):
        from google.cloud import storage
        name,prefix=self.gcs.removeprefix("gs://").split("/",1);bucket=storage.Client().bucket(name)
        for path in sorted((p for p in folder.rglob("*") if p.is_file()),key=lambda p:p.name=="COMPLETE"):
            blob=bucket.blob(f"{prefix.rstrip('/')}/{folder.name}/{path.relative_to(folder).as_posix()}")
            blob.metadata={"sha256":sha(path)}
            blob.upload_from_filename(str(path),if_generation_match=0,checksum="auto")
            blob.reload()
            if int(blob.size)!=path.stat().st_size or blob.metadata.get("sha256")!=sha(path):raise RuntimeError("remote checkpoint verification failed")

    def close(self):
        self.pool.shutdown(wait=True)
        for job in self.jobs:job.result()

    def restore_gcs(self,uri,destination,template,expected):
        from .verified_download import restore
        restore(uri,destination)
        return self.load(destination,template,expected)
