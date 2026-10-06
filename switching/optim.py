"""AdamW with device-resident bias correction for stable XLA update graphs."""
import torch

class DeviceAdamW(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01):
        if lr<=0 or eps<=0 or not all(0<=b<1 for b in betas):raise ValueError("invalid AdamW configuration")
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay))

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
                # Transfer a tiny scalar as data rather than embedding each warmup
                # learning rate and Python bias-correction value in a new graph.
                rate=torch.tensor(group["lr"],dtype=torch.float32).to(parameter.device)
                updated=parameter.float()*(1-rate*group["weight_decay"])
                updated-=rate*(first/correction1)/(torch.sqrt(second/correction2)+group["eps"])
                parameter.copy_(updated.to(parameter.dtype))
        return loss

    def load_state_dict(self,state_dict):
        super().load_state_dict(state_dict)
        for parameter,state in self.state.items():
            for name in ("step","exp_avg","exp_avg_sq"):
                if name in state:state[name]=state[name].to(device=parameter.device,dtype=torch.float32)
