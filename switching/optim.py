"""AdamW with device-resident bias correction for stable XLA update graphs."""
import torch

class DeviceAdamW(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01,warmup_steps=0):
        if lr<=0 or eps<=0 or not all(0<=b<1 for b in betas):raise ValueError("invalid AdamW configuration")
        super().__init__(params,dict(lr=lr,base_lr=lr,betas=betas,eps=eps,weight_decay=weight_decay,warmup_steps=warmup_steps))

    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        for group in self.param_groups:
            beta1,beta2=group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:continue
                if parameter.grad.is_sparse:raise ValueError("sparse gradients unsupported")
                state=self.state[parameter]
                if not state:
                    state["step"]=torch.zeros((),dtype=torch.float32,device=parameter.device)
                    state["exp_avg"]=torch.zeros_like(parameter,dtype=torch.float32)
                    state["exp_avg_sq"]=torch.zeros_like(parameter,dtype=torch.float32)
                state["step"].add_(1)
                gradient=parameter.grad.float()
                first,second=state["exp_avg"],state["exp_avg_sq"]
                first.mul_(beta1).add_(gradient,alpha=1-beta1)
                second.mul_(beta2).addcmul_(gradient,gradient,value=1-beta2)
                correction1=1-torch.pow(beta1,state["step"])
                correction2=1-torch.pow(beta2,state["step"])
                # Keep warmup and bias correction on the device instead of
                # embedding a different Python scalar into each update graph.
                if group["warmup_steps"]:
                    rate=group["base_lr"]*torch.clamp(state["step"]/group["warmup_steps"],max=1.0)
                else:rate=group["lr"]
                updated=parameter.float()*(1-rate*group["weight_decay"])
                updated-=rate*(first/correction1)/(torch.sqrt(second/correction2)+group["eps"])
                parameter.copy_(updated.to(parameter.dtype))
        return loss

    def load_state_dict(self,state_dict):
        requested_warmup=[group["warmup_steps"] for group in self.param_groups]
        requested_base=[group["base_lr"] for group in self.param_groups]
        super().load_state_dict(state_dict)
        # Optimizer.load_state_dict casts moments to each parameter's dtype.
        # Restore the original FP32 values directly, avoiding a BF16 round trip.
        original_ids=[identifier for group in state_dict["param_groups"] for identifier in group["params"]]
        parameters=[parameter for group in self.param_groups for parameter in group["params"]]
        for identifier,parameter in zip(original_ids,parameters):
            original=state_dict["state"].get(identifier,{})
            for name in ("step","exp_avg","exp_avg_sq"):
                if name in original:
                    value=original[name]
                    self.state[parameter][name]=torch.as_tensor(value,device=parameter.device,dtype=torch.float32).clone()
        for group,warmup,base in zip(self.param_groups,requested_warmup,requested_base):
            group["warmup_steps"]=warmup;group.setdefault("base_lr",base)
        for parameter,state in self.state.items():
            for name in ("step","exp_avg","exp_avg_sq"):
                if name in state:state[name]=state[name].to(device=parameter.device,dtype=torch.float32)

def numerical_state_valid(model,optimizer):
    """Check updated weights and moments before publishing a recoverable state."""
    checks=[torch.isfinite(p).all() for p in model.parameters() if p.requires_grad]
    for state in optimizer.state.values():
        for name in ("step","exp_avg","exp_avg_sq"):
            value=state.get(name)
            if torch.is_tensor(value):
                checks.append(torch.isfinite(value).all())
                if name=="exp_avg_sq":checks.append((value>=0).all())
    return bool(torch.stack(checks).all().item()) if checks else True
