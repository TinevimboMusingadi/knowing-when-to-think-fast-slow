"""Verify pinned upstream source and every locked numerical dependency."""
import importlib.metadata
import re
from pathlib import Path


def fingerprint(root):
    from .experiment_v2 import revisions,sha
    root=Path(root)
    return {'revisions':revisions(root),'dataset_manifest_sha256':sha(root/'data/recovery-v2/manifest.json')}


def require_unchanged(root,report):
    current=fingerprint(root)
    for key,value in current.items():
        if report.get(key)!=value:raise ValueError(f'validation inputs changed during execution: {key}')


def verify(root,backend):
    import tunix
    from scripts.bootstrap_tunix import verify as verify_source
    root=Path(root);verify_source(root)
    if not getattr(tunix,'__file__',None) or Path(tunix.__file__).resolve()!=root/'.vendor/tunix/tunix/__init__.py':
        raise ValueError('production acceptance requires the complete pinned Tunix package import')
    lock=root/('requirements-tunix-tpu.lock' if backend=='tpu' else 'requirements-tunix-cpu.lock')
    versions={};mismatches=[]
    for line in lock.read_text().splitlines():
        match=re.match(r'^([a-zA-Z0-9_.-]+)(?:\[[^]]+\])?==([^ ;\\]+)',line)
        if not match:continue
        package,expected=match.groups()
        try:actual=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:actual=None
        versions[package]=actual
        if actual!=expected:mismatches.append({'package':package,'expected':expected,'actual':actual})
    if mismatches:raise ValueError(f'installed dependencies do not match hash lock: {mismatches}')
    return {'lock':lock.name,'packages':versions,'full_pinned_package_import':True}
