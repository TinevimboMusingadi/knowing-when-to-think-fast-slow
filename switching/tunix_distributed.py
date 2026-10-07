"""Replicated per-device rollouts, FP32 collective reduction, one accepted update."""
import concurrent.futures
import jax
import jax.numpy as jnp
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P,SingleDeviceSharding
import numpy as np
from .tunix_runtime import TunixRuntime
from .tunix_optim import assert_finite


class Replicas:
    def __init__(self,runtime):
        self.devices=jax.local_devices()
        if jax.default_backend()=="tpu" and len(self.devices)!=4:raise ValueError("approved slice must expose four devices")
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=len(self.devices));self.runtimes=[]
        for device in self.devices:
            clone=object.__new__(type(runtime))
            clone.graph=runtime.graph;clone.frozen=jax.device_put(runtime.frozen,SingleDeviceSharding(device))
            clone.actor=jax.device_put(runtime.actor,SingleDeviceSharding(device));clone.tokenizer=runtime.tokenizer
            clone.context=runtime.context;clone.mode_ids=runtime.mode_ids;clone.eos=runtime.eos;clone.pad=runtime.pad
            clone.limits=dict(runtime.limits)
            for name in ("temperature","top_p","top_k","suppress_jev","stop_at_jev","thinking"):
                if hasattr(runtime,name):setattr(clone,name,getattr(runtime,name))
            # Both compiled functions receive frozen and trainable state explicitly.
            # Sharing them retains compilation across phases without a bound-method
            # cycle that keeps a closed replica's full backbone alive.
            clone._call=runtime._call;clone._head=runtime._head
            clone._policy_cache=(runtime.actor,clone.actor);clone._reference_cache=None
            self.runtimes.append(clone)
        self.mesh=Mesh(np.array(self.devices),('data',))
        def average(values):return jax.tree.map(lambda x:jax.lax.pmean(x[0],"data"),values)
        self._average=jax.shard_map(average,mesh=self.mesh,in_specs=P('data'),out_specs=P(),check_vma=False)

    def map(self,operation,actor,reference,batches,keys):
        if len(batches)!=len(self.devices):raise ValueError("one disjoint batch per device required")
        def run(index):
            device=self.devices[index]
            with jax.default_device(device):
                runtime=self.runtimes[index]
                if runtime._policy_cache[0] is not actor:
                    runtime._policy_cache=(actor,jax.device_put(actor,SingleDeviceSharding(device)))
                policy=runtime._policy_cache[1]
                if reference is actor:ref=policy
                else:
                    if runtime._reference_cache is None or runtime._reference_cache[0] is not reference:
                        runtime._reference_cache=(reference,jax.device_put(reference,SingleDeviceSharding(device)))
                    ref=runtime._reference_cache[1]
                return operation(runtime,policy,ref,batches[index],jax.device_put(keys[index],SingleDeviceSharding(device)))
        futures=[self.pool.submit(run,i) for i in range(len(self.devices))]
        # Completion order reveals replica errors instead of waiting on rank zero first.
        results=[None]*len(futures)
        for future in concurrent.futures.as_completed(futures):results[futures.index(future)]=future.result()
        return results

    def mean(self,gradients):
        if len(gradients)!=len(self.devices):raise ValueError("gradient replica count mismatch")
        for gradient in gradients:assert_finite(gradient,"local-gradients")
        sharding=NamedSharding(self.mesh,P('data'))
        def collect(*values):
            pieces=[jax.device_put(v[None],SingleDeviceSharding(d)) for v,d in zip(values,self.devices)]
            return jax.make_array_from_single_device_arrays((len(values),*values[0].shape),sharding,pieces)
        stacked=jax.tree.map(collect,*gradients)
        reduced=self._average(stacked);assert_finite(reduced,"reduced-gradients")
        return jax.device_put(reduced,SingleDeviceSharding(self.devices[0]))

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
        # Release only this replica pool. The original runtime keeps its base
        # and shared compiled functions for the next phase.
        for runtime in self.runtimes:
            runtime._policy_cache=None;runtime._reference_cache=None
            runtime.actor=None;runtime.frozen=None
        self.runtimes.clear()
