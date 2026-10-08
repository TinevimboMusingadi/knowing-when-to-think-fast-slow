"""Owned Tunix-only TPU lifecycle; independent deadlines and verified absence."""
import datetime
import base64
import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from scripts.cloud import gcloud,describe
from switching.recovery_budget import RecoveryBudget
from switching.experiment_v2 import launch_gates


HOST_KEY_TYPES=('ssh-ed25519','ecdsa-sha2-nistp256','ecdsa-sha2-nistp384','ecdsa-sha2-nistp521','ssh-rsa')


def observed_host_key(output,ip):
    keys={kind:set() for kind in HOST_KEY_TYPES}
    for line in output.splitlines():
        parts=line.split()
        if len(parts)==3 and parts[0] in {ip,f'[{ip}]:22'} and parts[1] in keys:keys[parts[1]].add(parts[2])
    if any(len(values)>1 for values in keys.values()):raise RuntimeError('ambiguous public host keys for the owned endpoint')
    for kind in HOST_KEY_TYPES:
        if keys[kind]:
            digest=hashlib.sha256(base64.b64decode(next(iter(keys[kind])),validate=True)).digest()
            return kind,'SHA256:'+base64.b64encode(digest).decode().rstrip('=')
    raise RuntimeError('one public host key is required for the owned endpoint')


def observed_host_fingerprint(output,ip):
    return observed_host_key(output,ip)[1]


def probe_host_key(scanner,ip,probes,timeout=90):
    # READY can precede SSH readiness. Probe supported key types without accepting
    # a changed or ambiguous key, and leave the nonce/ownership gates in place.
    deadline=time.monotonic()+timeout
    while True:
        remaining=deadline-time.monotonic()
        if remaining<=0:raise RuntimeError('owned SSH host-key probe did not become ready before its deadline')
        try:
            scanned=subprocess.run([scanner,'-T','10','-t','ed25519,ecdsa,rsa',ip],capture_output=True,text=True,
                                   timeout=min(20,remaining),creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        except subprocess.TimeoutExpired:
            probes.append({'returncode':None,'timed_out':True})
        else:
            probes.append({'returncode':scanned.returncode,'stderr':scanned.stderr.replace(ip,'<owned-endpoint>')[-1000:]})
            # An ambiguous supported key is a trust failure, not boot readiness.
            if any(len(line.split())==3 and line.split()[0] in {ip,f'[{ip}]:22'} and
                   line.split()[1] in HOST_KEY_TYPES for line in scanned.stdout.splitlines()):
                return observed_host_key(scanned.stdout,ip)
        time.sleep(min(2,max(0,deadline-time.monotonic())))


def remote_reply_verified(result,nonce):
    # The TPU gcloud command can report success after PuTTY abandons a prompt.
    # A zero exit status alone must never publish CONNECTION_VERIFIED.
    return result.returncode==0 and nonce in result.stdout.splitlines()


def _verify_remote_access(session,out,report):
    resource=describe(session)
    if resource is None or resource.get('labels',{}).get('kws_run')!=session['run_id'] or resource.get('labels',{}).get('kws_schema')!='v2':
        raise PermissionError('remote access requires verified resource ownership')
    ip=resource['networkEndpoints'][0]['accessConfig']['externalIp']
    identity=['--project',session['project'],'--zone',session['zone'],'--worker','0','--quiet']
    fingerprint=None;flags=[]
    if os.name=='nt':
        scanner=shutil.which('ssh-keyscan.exe')
        if not scanner:raise RuntimeError('Windows OpenSSH public host-key scanner unavailable')
        report['stage']='host-key-probe'
        kind,fingerprint=probe_host_key(scanner,ip,report['host_key_probes'])
        report['host_key_algorithm']=kind
        pinned=Path(out)/'remote-host-keys'/f"{session['run_id']}.json"
        pinned.parent.mkdir(parents=True,exist_ok=True)
        if pinned.exists() and json.loads(pinned.read_text())!={'ip':ip,'fingerprint':fingerprint,'run_id':session['run_id']}:
            raise RuntimeError('owned endpoint host key changed; refusing connection')
        pinned.write_text(json.dumps({'ip':ip,'fingerprint':fingerprint,'run_id':session['run_id']},indent=2))
        flags=['--ssh-flag=-batch','--ssh-flag=-hostkey','--ssh-flag='+fingerprint]
    else:flags=['--ssh-flag=-oStrictHostKeyChecking=accept-new']
    report['stage']='exact-remote-reply'
    nonce='KWS_REMOTE_'+uuid.uuid4().hex
    result=gcloud('compute','tpus','tpu-vm','ssh',session['name'],*identity,*flags,
                  '--command',f'printf "%s\\n" {nonce}',check=False,timeout=60)
    if not remote_reply_verified(result,nonce):raise RuntimeError('remote command response not verified; training gate remains closed')
    after=describe(session)
    if after is None or after.get('labels')!=resource.get('labels') or after['networkEndpoints'][0]['accessConfig']['externalIp']!=ip:
        raise RuntimeError('owned endpoint changed during remote verification')
    report.update(passed=True,checked_at=time.time(),owned_resource_before_after=True,
                  host_key_sha256=fingerprint,exact_remote_reply=True,global_ssh_configuration_modified=False,stage='complete')
    return report


def verify_remote_access(session,out):
    report={'passed':False,'run_id':session['run_id'],'stage':'ownership','host_key_probes':[]}
    try:return _verify_remote_access(session,out,report)
    except Exception as error:
        report['error']={'type':type(error).__name__,'message':str(error)}
        raise
    finally:(Path(out)/'remote-access.json').write_text(json.dumps(report,indent=2))


def delete_and_verify(session,timeout=300):
    started=time.time();last_error=None
    while time.time()-started<timeout:
        try:
            resource=describe(session)
            if resource is None:return time.time()
            labels=resource.get("labels",{})
            if labels.get("kws_run")!=session["run_id"] or labels.get("kws_schema")!="v2":raise PermissionError("resource ownership mismatch; refusing deletion")
            if resource.get('state')!='DELETING':
                gcloud("compute","tpus","tpu-vm","delete",session["name"],"--project",session["project"],"--zone",session["zone"],"--quiet",timeout=60)
        except PermissionError:raise
        except (RuntimeError,subprocess.TimeoutExpired) as exc:last_error=str(exc)
        time.sleep(5)
    raise RuntimeError(f"resource absence remains unverified: {last_error}")


def watchdog(path):
    path=Path(path);session=json.loads(path.read_text());armed=False
    stage_deadline=session["stage_deadline"]
    while time.time()<min(session["deadline"],stage_deadline):
        try:resource=describe(session)
        except (RuntimeError,subprocess.TimeoutExpired):
            time.sleep(5);continue
        if resource is not None:armed=True
        elif armed:return
        try:
            remote=gcloud("storage","cat",session["gcs_prefix"]+"/budget.json",check=False,timeout=15)
            if remote.returncode==0:
                ledger=json.loads(remote.stdout)
                attempt=next((a for a in ledger["attempts"] if a["run_id"]==session["run_id"]),None)
                if attempt is not None:stage_deadline=min(session["deadline"],float(attempt["deadline"]))
        except (RuntimeError,subprocess.TimeoutExpired,ValueError,KeyError):
            # An unreadable ledger cannot extend the last verified deadline.
            pass
        time.sleep(20)
    deleted=delete_and_verify(session)
    config=json.loads(Path(session["config_path"]).read_text());budget=RecoveryBudget(config,session["ledger_path"])
    budget.finish(session["run_id"],deleted,True)
    session.update(state="deleted",deleted_at=deleted,absence_verified=True);path.write_text(json.dumps(session,indent=2))


def preflight(config,run_id):
    from google.cloud import storage
    from google.auth import default
    from google.auth.transport.requests import AuthorizedSession
    # Local ADC may belong to another active project. Scope quota accounting to
    # this experiment without changing global credentials or project settings.
    credentials,_=default(scopes=["https://www.googleapis.com/auth/cloud-platform"],quota_project_id=config['project']);client=AuthorizedSession(credentials)
    number=gcloud("projects","describe",config["project"],"--format=value(projectNumber)").stdout.strip()
    quota=client.get(f"https://serviceusage.googleapis.com/v1beta1/projects/{number}/services/tpu.googleapis.com/consumerQuotaMetrics",params={"view":"FULL","pageSize":200},timeout=60)
    quota.raise_for_status();data=quota.json();metrics=list(data.get('metrics',[]))
    while data.get('nextPageToken'):
        quota=client.get(f"https://serviceusage.googleapis.com/v1beta1/projects/{number}/services/tpu.googleapis.com/consumerQuotaMetrics",
                         params={'view':'FULL','pageSize':200,'pageToken':data['nextPageToken']},timeout=60)
        quota.raise_for_status();data=quota.json();metrics.extend(data.get('metrics',[]))
    qualified=training_quota({'metrics':metrics},config['zone'])
    if not qualified:raise RuntimeError("four interruptible v5e chips are not verified in regional quota")
    identity=["--project",config["project"],"--zone",config["zone"],"--format=json"]
    versions=json.loads(gcloud("compute","tpus","tpu-vm","versions","list",*identity).stdout)
    if not any(v.get("version",v.get("name","")).endswith(config["runtime_version"]) for v in versions):raise RuntimeError("configured TPU runtime is unavailable")
    accelerators=json.loads(gcloud("compute","tpus","tpu-vm","accelerator-types","list",*identity).stdout)
    if not any(a.get("type",a.get("name","")).endswith(config["accelerator"]) for a in accelerators):raise RuntimeError("approved TPU shape is unavailable")
    bucket=storage.Client(project=config["project"],credentials=credentials).bucket(config["bucket"]);bucket.reload(timeout=60)
    if bucket.location not in ("US","US-WEST4"):raise RuntimeError("bucket location needs a new transfer-cost estimate")
    bucket.blob(f"{config['prefix']}/{run_id}/preflight.json").upload_from_string(json.dumps({"schema_version":2,"run_id":run_id,"quota_verified":True}),if_generation_match=0,content_type="application/json")
    price=interruptible_price(client,config)
    return {"schema_version":2,"passed":True,"quota_verified":True,"quota":qualified,"runtime_verified":True,"bucket_location":bucket.location,"price":price,"checked_at":time.time()}


def training_quota(data,zone):
    """Require the v5e training metric and each applicable location limit."""
    result=[];region=zone.rsplit('-',1)[0]
    for entry in data.get('metrics',[]):
        if entry.get('metric')!='tpu.googleapis.com/tpu-v5s-litepod-preemptible':continue
        limits=[];allowed=True
        for limit in entry.get('consumerQuotaLimits',[]):
            scope='zone' if '{zone}' in limit['unit'] else 'region' if '{region}' in limit['unit'] else None
            target=zone if scope=='zone' else region
            if scope is None or target not in limit.get('supportedLocations',[]):allowed=False;break
            buckets=limit.get('quotaBuckets',[])
            selected=[b for b in buckets if b.get('dimensions',{}).get(scope)==target]
            if not selected:selected=[b for b in buckets if not b.get('dimensions')]
            if len(selected)!=1:allowed=False;break
            value=int(selected[0].get('effectiveLimit',0))
            if value!=-1 and value<4:allowed=False;break
            limits.append({'scope':scope,'location':target,'effective_limit':value})
        if allowed and limits:result.append({'metric':entry['metric'],'limits':limits})
    return result


def interruptible_price(client,config):
    services=[];token=None
    while True:
        response=client.get("https://cloudbilling.googleapis.com/v1/services",params={"pageSize":500,**({"pageToken":token} if token else {})},timeout=60)
        response.raise_for_status();data=response.json();services.extend(data.get("services",[]));token=data.get("nextPageToken")
        if not token:break
    # Current v5e VM SKUs are in Compute Engine; Cloud TPU contains legacy
    # accelerator SKUs and subscriptions, not the current regional spot rates.
    service=next((s["name"] for s in services if s.get("displayName")=="Compute Engine"),None)
    if service is None:raise RuntimeError("TPU price catalog unavailable")
    token=None;skus=[]
    while True:
        response=client.get(f"https://cloudbilling.googleapis.com/v1/{service}/skus",params={"currencyCode":"USD","pageSize":5000,**({"pageToken":token} if token else {})},timeout=60)
        response.raise_for_status();data=response.json()
        skus.extend(s for s in data.get('skus',[]) if 'v5e' in re.sub(r'[^a-z0-9]','',s['description'].lower()))
        token=data.get("nextPageToken")
        if not token:break
    return resolve_interruptible_price(skus,config)


def resolve_interruptible_price(skus,config):
    if config['accelerator']!='v5litepod-4':raise ValueError('only the approved four-chip v5e slice is supported')
    region=config['zone'].rsplit('-',1)[0];spot=[];standard=[]
    for sku in skus:
        description=re.sub(r'[^a-z0-9]','',sku['description'].lower());category=sku.get('category',{})
        if region not in sku.get('serviceRegions',[]) or 'tpuv5e' not in description or category.get('resourceGroup')!='TPU':continue
        expressions=sku.get('pricingInfo',[])
        if not expressions:continue
        price=expressions[-1]['pricingExpression'];rates=price.get('tieredRates',[])
        if len(rates)!=1 or price.get('usageUnit')!='h':continue
        unit=rates[0]['unitPrice']
        if unit.get('currencyCode')!='USD':continue
        rate=float(unit.get('units',0))+float(unit.get('nanos',0))/1e9
        row={'sku':sku['name'],'description':sku['description'],'per_chip_hour':rate,'effective_time':expressions[-1].get('effectiveTime')}
        if category.get('usageType')=='Preemptible':spot.append(row)
        if category.get('usageType')=='OnDemand' and description.startswith('tpuv5erunningin'):standard.append(row)
    if len(spot)!=1 or len(standard)!=1:raise RuntimeError(f'current regional v5e rate is ambiguous: spot={spot}, standard={standard}')
    # Google documents v5e prices per chip-hour. Check the matching standard
    # SKU against the published $1.20/chip bound before multiplying spot by four.
    if abs(standard[0]['per_chip_hour']-config['rate_upper_bound']/4)>1e-9:
        raise RuntimeError('same-region standard SKU does not confirm the documented per-chip bound')
    whole=spot[0]['per_chip_hour']*4
    if not 0<whole<=config['rate_upper_bound']:raise RuntimeError('spot slice price exceeds the approved bound')
    return {**spot[0],'usage_unit':'chip-hour','unit_multiplier':4,'whole_slice_rate':whole,
            'unit_source':'https://cloud.google.com/tpu/pricing','standard_sku_check':standard[0]}


def startup(session,uri,config):
    seconds=max(1,int(session["deadline"]-time.time()));run=session["run_id"]
    return f'''#!/bin/bash
set -euo pipefail
mkdir -p /opt/kws-v2
cd /opt/kws-v2
exec > >(tee -a startup.log) 2>&1
cleanup() {{
  gcloud storage cp startup.log {session['gcs_prefix']}/startup.log || true
  OWNER=$(gcloud compute tpus tpu-vm describe {session['name']} --project={session['project']} --zone={session['zone']} --format='value(labels.kws_run)')
  SCHEMA=$(gcloud compute tpus tpu-vm describe {session['name']} --project={session['project']} --zone={session['zone']} --format='value(labels.kws_schema)')
  if [ "$OWNER" = "{run}" ] && [ "$SCHEMA" = "v2" ]; then
    gcloud compute tpus tpu-vm delete {session['name']} --project={session['project']} --zone={session['zone']} --quiet || true
  fi
}}
trap cleanup EXIT
(sleep {seconds}; cleanup) &
gcloud storage cp {uri} payload.tar.gz
tar -xzf payload.tar.gz
python3 -m pip install uv==0.12.23
python3 -m uv python install 3.12.14
python3 -m uv venv --python 3.12.14 .venv
python3 -m uv pip sync --python .venv/bin/python --require-hashes requirements-tunix-tpu.lock
export PYTHONPATH=/opt/kws-v2/.vendor/tunix:/opt/kws-v2
export JAX_COMPILATION_CACHE_DIR=/opt/kws-v2/.cache/jax
export HF_HOME=/opt/kws-v2/.cache/huggingface
CONNECTED=0
for attempt in $(seq 1 60); do
  if gcloud storage cat {session['gcs_prefix']}/CONNECTION_VERIFIED >/dev/null 2>&1; then CONNECTED=1; break; fi
  sleep 5
done
if [ "$CONNECTED" != "1" ]; then echo "Remote-access gate not verified"; exit 1; fi
timeout --signal=TERM --kill-after=120 {max(1,seconds-300)} .venv/bin/python -m switching.tunix_experiment --session session.json
gcloud storage cp -r runs/recovery-v2 {session['gcs_prefix']}/artifacts/
'''


def payload(root,path,out,data,session):
    root=Path(root).resolve();out=Path(out).resolve();data=Path(data).resolve()
    allow=("switching","scripts","licenses","configs/recovery.json","requirements-tunix-cpu.lock","requirements-tunix-tpu.lock",".vendor/tunix/tunix",".vendor/tunix/SOURCE.json",".vendor/tunix/LICENSE")
    with tarfile.open(path,"w:gz") as archive:
        for name in allow:archive.add(root/name,arcname=name,filter=lambda t:None if "__pycache__" in t.name else t)
        for name in ("train.jsonl","val.jsonl","test.jsonl","manifest.json"):archive.add(data/name,arcname="data/recovery-v2/"+name)
        for name in ("converted","source-sft/tokenizer","tiny-reference"):archive.add(out/name,arcname="runs/recovery-v2/"+name)
        for name in ("budget.json","audit.json","local-numerical.json","full-model-parity.json","local-protocol.json"):archive.add(out/name,arcname="runs/recovery-v2/"+name)
        archive.add(out/"session.json",arcname="session.json")


def execute(config,out,data):
    root=Path(__file__).resolve().parents[1];out=Path(out).resolve()
    if not launch_gates(root,config,data,out)["passed"]:raise RuntimeError("local gate changed before launch")
    run=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    check=preflight(config,run);(out/"cloud-preflight.json").write_text(json.dumps(check,indent=2))
    budget=RecoveryBudget(config,out/"budget.json");attempt=budget.begin("pilot",run)
    session={"run_id":run,"name":f"kws-recovery-{run}","project":config["project"],"zone":config["zone"],
             "deadline":time.time()+sum(budget.remaining(s) for s in ("pilot","sft","rl","evaluation"))/(config["rate_upper_bound"]*config["rate_margin"])*3600,
             "created":attempt["created_at"],"stage_deadline":attempt["deadline"],"state":"prepared","config_path":str(root/"configs/recovery.json"),"ledger_path":str(out/"budget.json"),
             "gcs_prefix":f"gs://{config['bucket']}/{config['prefix']}/{run}","price_check":check["price"]}
    path=out/"session.json";path.write_text(json.dumps(session,indent=2));archive=out/"payload.tar.gz"
    try:
        payload(root,archive,out,data,session);uri=session["gcs_prefix"]+"/payload.tar.gz"
        script=out/"startup-v2.sh";script.write_text(startup(session,uri,config),encoding="utf-8",newline="\n")
        gcloud("storage","cp",str(archive),uri,timeout=600)
        options={"stdout":open(out/"watchdog-v2.log","a"),"stderr":subprocess.STDOUT,"stdin":subprocess.DEVNULL}
        if os.name=="nt":options["creationflags"]=subprocess.CREATE_NO_WINDOW
        try:subprocess.Popen([sys.executable,str(root/"scripts/recovery_cloud.py"),"--watchdog",str(path)],**options)
        finally:options["stdout"].close()
        gcloud("compute","tpus","tpu-vm","create",session["name"],"--project",session["project"],"--zone",session["zone"],
               "--accelerator-type",config["accelerator"],"--version",config["runtime_version"],"--preemptible",
               "--labels",f"kws_run={run},kws_schema=v2","--scopes","https://www.googleapis.com/auth/cloud-platform",
               "--metadata-from-file",f"startup-script={script}",timeout=600)
        session["state"]="created";path.write_text(json.dumps(session,indent=2))
        connection_report=verify_remote_access(session,out)
        connection=out/'CONNECTION_VERIFIED';connection.write_text(json.dumps(connection_report))
        gcloud('storage','cp',str(connection),session['gcs_prefix']+'/CONNECTION_VERIFIED',timeout=60)
        while time.time()<session["deadline"]:
            resource=describe(session)
            if resource is None:break
            labels=resource.get("labels",{})
            if labels.get("kws_run")!=run or labels.get("kws_schema")!="v2":raise PermissionError("ownership changed")
            time.sleep(30)
    except Exception as error:
        (out/'launch-failure.json').write_text(json.dumps({'run_id':run,'type':type(error).__name__,
            'message':str(error),'recorded_at':time.time()},indent=2))
        raise
    finally:
        deleted=delete_and_verify(session);budget=RecoveryBudget(config,out/"budget.json")
        remote=gcloud("storage","cp",session["gcs_prefix"]+"/budget.json",str(out/"budget-remote-v2.json"),check=False)
        if remote.returncode==0:
            updated=json.loads((out/"budget-remote-v2.json").read_text())
            if updated["prior_upper_bound"]!=config["prior_spend_upper_bound"]:raise ValueError("worker ledger mismatch")
            budget.ledger=updated
        budget.finish(run,deleted,True);session.update(state="deleted",absence_verified=True,deleted_at=deleted);path.write_text(json.dumps(session,indent=2))
        gcloud("storage","cp",str(out/"budget.json"),session["gcs_prefix"]+"/budget-final.json",check=False)
        gcloud("storage","cp","--recursive",session["gcs_prefix"]+"/artifacts/recovery-v2",str(out/"remote-artifacts"),check=False,timeout=600)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--watchdog",required=True);a=p.parse_args();watchdog(a.watchdog)
