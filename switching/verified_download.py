"""Generation-pinned, resumable GCS recovery; markers are checked before use."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path


def digest(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b""):value.update(chunk)
    return value.hexdigest()


def download(blob,target,expected,workers=4,chunk_size=2*1024*1024):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and digest(target)==expected:return
    blob.reload(timeout=60);generation=int(blob.generation);size=int(blob.size)
    parts=target.with_name(target.name+".parts");parts.mkdir(exist_ok=True)
    identity=parts/"identity.json";record={"generation":generation,"size":size,"sha256":expected}
    if identity.exists() and json.loads(identity.read_text())!=record:raise ValueError("partial download generation changed; preserve and investigate")
    identity.write_text(json.dumps(record))
    def fetch(index):
        start=index*chunk_size;end=min(size,start+chunk_size)-1;part=parts/f"{index:06d}"
        if part.exists() and part.stat().st_size==end-start+1:return
        data=blob.download_as_bytes(start=start,end=end,if_generation_match=generation,timeout=(30,180),checksum=None)
        if len(data)!=end-start+1:raise ValueError("incomplete range response")
        temporary=part.with_suffix(".tmp");temporary.write_bytes(data);temporary.replace(part)
    count=(size+chunk_size-1)//chunk_size
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(fetch,range(count)):pass
    assembled=target.with_name(target.name+".verified.tmp")
    with assembled.open("wb") as stream:
        for index in range(count):stream.write((parts/f"{index:06d}").read_bytes())
    if digest(assembled)!=expected:raise ValueError("restored file checksum mismatch")
    assembled.replace(target)


def restore(uri,destination):
    from google.cloud import storage
    name,prefix=uri.removeprefix("gs://").split("/",1);bucket=storage.Client().bucket(name)
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    marker=bucket.blob(prefix+"/COMPLETE").download_as_bytes(timeout=60)
    manifest_bytes=bucket.blob(prefix+"/manifest.json").download_as_bytes(timeout=60)
    if hashlib.sha256(manifest_bytes).hexdigest()!=marker.decode():raise ValueError("remote completion marker mismatch")
    manifest=json.loads(manifest_bytes)
    for name,expected in manifest["files"].items():
        target=(destination/name).resolve()
        if not target.is_relative_to(destination):raise ValueError("unsafe checkpoint path")
        download(bucket.blob(prefix+"/"+name),target,expected)
    (destination/"manifest.json").write_bytes(manifest_bytes);(destination/"COMPLETE").write_bytes(marker)
    return manifest


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--uri",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    result=restore(a.uri,a.output);print(json.dumps({"verified_files":len(result["files"]),"output":a.output}))
