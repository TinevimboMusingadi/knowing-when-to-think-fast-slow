"""Resumable public mirror transport; accept only pinned Hugging Face hashes."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import time
import requests


def fetch_file(info,folder,workers=16,part_size=2*1024*1024):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);final=folder/info['name']
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    if final.exists():
        if final.stat().st_size==info['size'] and digest(final)==info['sha256']:return
        raise ValueError('existing model file does not match the pinned revision')
    parts=folder/(info['sha256']+'.parts');parts.mkdir(exist_ok=True)
    def part(offset):
        end=min(info['size']-1,offset+part_size-1);path=parts/f'{offset:012d}'
        if path.exists() and path.stat().st_size==end-offset+1:return
        url=info.get('url','https://modelscope.cn/models/Qwen/Qwen3-1.7B/resolve/master/'+info['name'])
        for attempt in range(3):
            try:
                with requests.get(url,headers={'Range':f'bytes={offset}-{end}','Cache-Control':'no-cache'},stream=True,timeout=(30,90)) as response:
                    response.raise_for_status()
                    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {offset}-{end}/{info["size"]}':
                        raise ValueError('server did not return the requested pinned-size range')
                    temporary=path.with_suffix('.partial')
                    with temporary.open('wb') as stream:
                        for chunk in response.iter_content(65536):stream.write(chunk)
                    if temporary.stat().st_size!=end-offset+1:raise ValueError('incomplete range')
                    temporary.replace(path);return
            except requests.RequestException:
                if attempt==2:raise
                time.sleep(2**attempt)
    offsets=list(range(0,info['size'],part_size));completed=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(part,offset) for offset in offsets]
        try:
            for job in concurrent.futures.as_completed(jobs):
                job.result();completed+=1
                if completed%32==0:print(json.dumps({'file':info['name'],'verified_size_parts':completed,'total_parts':len(offsets)}),flush=True)
        except BaseException:
            for job in jobs:job.cancel()
            raise
    temporary=final.with_suffix('.assembled')
    with temporary.open('wb') as stream:
        for offset in offsets:
            with (parts/f'{offset:012d}').open('rb') as source:shutil.copyfileobj(source,stream)
    if digest(temporary)!=info['sha256']:raise ValueError('mirror weights differ from pinned Hugging Face checksum; do not use')
    temporary.replace(final)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',default='runs/recovery-v2/public-base-files.json')
    parser.add_argument('--output',default='.cache/public-base');parser.add_argument('--workers',type=int,default=16);args=parser.parse_args()
    metadata=json.loads(Path(args.manifest).read_text());folder=Path(args.output)
    for file in metadata['files']:fetch_file(file,folder,args.workers)
    source=Path('.cache/huggingface/models--Qwen--Qwen3-1.7B/snapshots')/metadata['revision']
    for path in source.iterdir():
        if path.suffix!='.safetensors' and path.is_file():shutil.copyfile(path,folder/path.name)
    (folder/'pinned-manifest.json').write_text(json.dumps(metadata,indent=2))
    Path('runs/recovery-v2/base-model.json').write_text(json.dumps({**metadata,'status':'downloaded-and-sha256-verified',
                      'path':str(folder.resolve()),'transport':'public Qwen ModelScope mirror, pinned Hugging Face content hashes'},indent=2))


if __name__=='__main__':main()
