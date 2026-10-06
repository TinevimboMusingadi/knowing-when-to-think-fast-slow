"""Recover only completion-marked checkpoints belonging to a named experiment."""
import argparse
import io
import json
import re
import tarfile
from pathlib import Path
from google.cloud import storage
from .storage import Checkpoints

def verify_dataset(current,archived):
    for split in ("train","val","test"):
        if current["splits"][split]["sha256"]!=archived["splits"][split]["sha256"]:
            raise ValueError("recovery dataset differs from the saved experiment")

def recover(run_id,output,bucket_name="keeper-file-storage"):
    if not re.fullmatch(r"\d{8}-\d{6}",run_id):raise ValueError("invalid experiment ID")
    prefix=f"knowing-when-to-switch/{run_id}"
    bucket=storage.Client().bucket(bucket_name)
    blob=bucket.blob(prefix+"/updates/data-update.tar.gz")
    if not blob.exists():blob=bucket.blob(prefix+"/payload.tar.gz")
    with tarfile.open(fileobj=io.BytesIO(blob.download_as_bytes()),mode="r:gz") as archive:
        archived=json.load(archive.extractfile("data/manifest.json"))
    verify_dataset(json.loads(Path("data/manifest.json").read_text()),archived)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manager=Checkpoints(output/"scratch");records=[]
    try:
        for rank in range(4):
            pointer=json.loads(bucket.blob(f"{prefix}/logs/{run_id}/checkpoints/interrupted-rank{rank}.json").download_as_text())
            directory=pointer["directory"]
            if "/" in directory or "\\" in directory or not directory.startswith("step-"):raise ValueError("unsafe checkpoint pointer")
            uri=f"gs://{bucket_name}/{prefix}/checkpoints/{directory}"
            destination=manager.restore_gcs(uri,output/f"rank{rank}")
            manifest=json.loads((destination/"manifest.json").read_text())
            if manifest["rank"]!=rank or manifest["progress"]["phase"]!="sft":raise ValueError("wrong recovery rank or phase")
            records.append({"rank":rank,"uri":uri,"progress":manifest["progress"]})
    finally:manager.close()
    (output/"recovery.json").write_text(json.dumps(records,indent=2))
    return records

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--run-id",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();print(json.dumps(recover(args.run_id,args.output),indent=2))
