"""Extract and verify the pinned source without Windows-incompatible docs paths."""
import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def bootstrap(root):
    root=Path(root).resolve();cfg=json.loads((root/"configs/recovery.json").read_text());pin=cfg["tunix_revision"]
    destination=root/".vendor/tunix";destination.parent.mkdir(parents=True,exist_ok=True)
    if not (destination/".git").exists():subprocess.run(["git","clone","--filter=blob:none","--no-checkout","https://github.com/google/tunix.git",str(destination)],check=True)
    git=["git","-c",f"safe.directory={destination.as_posix()}","-C",str(destination)]
    subprocess.run(git+["fetch","--depth","1","origin",pin],check=True)
    resolved=subprocess.check_output(git+["rev-parse",pin+"^{commit}"],text=True).strip()
    if resolved!=pin:raise ValueError("unexpected source revision")
    # Archive the Python subtree, rather than the root tree containing ':' names.
    archive=destination/"tunix.tar"
    with archive.open("wb") as stream:subprocess.run(git+["archive",pin+":tunix"],stdout=stream,check=True)
    target=destination/"tunix";target.mkdir(exist_ok=True)
    with tarfile.open(archive) as stream:stream.extractall(target,filter="data")
    (destination/'LICENSE').write_bytes(subprocess.check_output(git+['show',pin+':LICENSE']))
    files={p.relative_to(destination).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob("*.py")}
    files['LICENSE']=hashlib.sha256((destination/'LICENSE').read_bytes()).hexdigest()
    report={"revision":pin,"repository":"https://github.com/google/tunix","files":files}
    (destination/"SOURCE.json").write_text(json.dumps(report,indent=2));return report


def verify(root):
    root=Path(root);base=root/".vendor/tunix";report=json.loads((base/"SOURCE.json").read_text())
    if report["revision"]!=json.loads((root/"configs/recovery.json").read_text())["tunix_revision"]:raise ValueError("backend revision changed")
    for name,value in report["files"].items():
        path=(base/name).resolve()
        if not path.is_relative_to(base.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=value:raise ValueError(f"backend source changed: {name}")
    return report["revision"]


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--verify",action="store_true");a=p.parse_args();root=Path(__file__).resolve().parents[1]
    result=verify(root) if a.verify else bootstrap(root);print(result if a.verify else result["revision"])
