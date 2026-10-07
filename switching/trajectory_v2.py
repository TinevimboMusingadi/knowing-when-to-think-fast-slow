"""Mixed trajectories separate sampled policy events from public observations."""
from dataclasses import dataclass,field
import math


@dataclass
class TrajectoryV2:
    episode_id:str
    policy_version:str
    events:list=field(default_factory=list)
    reward:float|None=None
    reward_components:dict=field(default_factory=dict)
    termination:str|None=None
    schema_version:int=2

    def tokens(self,prompt,completion,logps,observation):
        if not completion or len(completion)!=len(logps):raise ValueError("sampled token likelihood mismatch")
        self._finite(logps)
        self.events.append({"kind":"tokens","prompt":list(prompt),"completion":list(completion),"old_logps":list(logps),
                            "loss_mask":[True]*len(completion),"observation":observation})

    def decision(self,state,candidates,choice,logp):
        if not 0<=choice<len(candidates):raise ValueError("categorical choice out of range")
        self._finite([logp])
        self.events.append({"kind":"decision","state":state,"candidates":candidates,"choice":choice,"old_logps":[float(logp)],
                            "loss_mask":[True],"action_mask":[True]*len(candidates)})

    @staticmethod
    def _finite(values):
        if not all(math.isfinite(float(x)) for x in values):raise ValueError("nonfinite rollout log probability")

    def finalize(self,reward,components,termination):
        self._finite([reward]);self.reward=float(reward);self.reward_components=dict(components);self.termination=termination

    def public_record(self):
        # Observation strings and reasoning/token arrays stay in private run artifacts.
        return {"schema_version":2,"id":self.episode_id,"policy_version":self.policy_version,"reward":self.reward,
                "reward_components":self.reward_components,"termination":self.termination,
                "events":[{"kind":e["kind"],"sampled_events":len(e["old_logps"])} for e in self.events]}
