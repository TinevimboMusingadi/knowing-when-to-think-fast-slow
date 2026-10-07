"""Public observations and mixed typed/text actions, with a private verifier."""
from copy import deepcopy
import json
import math
import re
from .protocol import MODES, equivalent

SYSTEM = """Choose <mode:jev>, <mode:direct>, or <mode:cot>. JEV emits only its mode token and scores the available typed actions. Direct and CoT emit a mode token and a JSON action: answer(value), ask(field,question), lookup(query), or, in CoT, reason(text). Use observed facts; do not invent missing information. After an unfinished action, choose the next mode yourself. End a text action after its JSON object."""


def public_episode(episode):
    return deepcopy(episode["visible"])


def parse(text):
    if text.strip() == MODES[0]:
        return MODES[0], {"action": "decide"}
    match = re.fullmatch(r"\s*(<mode:(?:direct|cot)>)(\{.*\})\s*", text, re.S)
    if not match:
        raise ValueError("invalid mode/action framing")
    payload = json.loads(match[2])
    if not isinstance(payload, dict):
        raise ValueError("action must be an object")
    kind = payload.get("action")
    allowed = {"answer", "ask", "lookup"} | ({"reason"} if match[1] == MODES[2] else set())
    if kind not in allowed:
        raise ValueError("action incompatible with mode")
    required = {"answer": ("value",), "ask": ("field", "question"), "lookup": ("query",), "reason": ("text",)}[kind]
    if any(key not in payload for key in required):
        raise ValueError("missing action field")
    if kind != "answer" and any(not isinstance(payload[key], str) for key in required):
        raise ValueError("action fields must be text")
    if kind == "answer" and isinstance(payload["value"], float) and not math.isfinite(payload["value"]):
        raise ValueError("answer must be finite")
    return match[1], payload


class EpisodeEnvV2:
    """Only this verifier sees oracle fields; agents receive observations()."""
    def __init__(self, episode, max_transitions=4, max_lookups=2, max_actions=8):
        if episode.get("schema_version") != 2:
            raise ValueError("Episode v2 required; use the explicit legacy adapter")
        self._oracle = deepcopy(episode["oracle"])
        self.visible = public_episode(episode)
        self.id = episode["id"]
        self.messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(self.visible, ensure_ascii=False)}]
        self.modes = []
        self.events = []
        self.answer = None
        self.done = False
        self.error = None
        self.lookups = 0
        self.tool_calls = 0
        self.acquired_fields = set()
        self.max_transitions, self.max_lookups, self.max_actions = max_transitions, max_lookups, max_actions

    def observations(self):
        return deepcopy(self.messages)

    @property
    def candidates(self):
        return deepcopy(self.visible["candidates"])

    @property
    def transitions(self):
        return sum(a != b for a, b in zip(self.modes, self.modes[1:]))

    def state_text(self):
        # Candidate enumeration is excluded from independently encoded branches.
        context = [self.visible["prompt"]]
        for message in self.messages[2:]:
            if message["role"] == "user":
                context.append(message["content"])
            elif message["role"] == "assistant":
                try:
                    _, action = parse(message["content"])
                    if action["action"] == "reason": context.append(action["text"])
                except ValueError:
                    pass
        return "\n".join(context)

    def step(self, text, choice_id=None):
        if self.done: raise ValueError("episode finished")
        try:
            if len(self.events) >= self.max_actions: raise ValueError("action limit")
            mode, action = parse(text)
            changes = self.transitions + int(bool(self.modes) and self.modes[-1] != mode)
            if changes > self.max_transitions: raise ValueError("transition limit")
            self.modes.append(mode)
            self.messages.append({"role": "assistant", "content": text})
            if action["action"] == "decide":
                candidate = next((c for c in self.candidates if c["id"] == choice_id), None)
                if candidate is None: raise ValueError("invalid typed action ID")
                action = {"action": candidate["kind"], **candidate.get("payload", {})}
                if candidate["kind"] == "answer": action["value"] = candidate["value"]
            self.events.append({"mode": mode, "action": deepcopy(action), "choice_id": choice_id})
            kind = action["action"]
            if kind == "answer":
                self.answer, self.done = action["value"], True
                return True
            if kind == "defer":
                observation = "The decision is deferred. The task remains unfinished."
            elif kind == "reason":
                observation = "Continue the unfinished task using the observations so far."
            elif kind == "ask":
                self.tool_calls += 1
                field = action["field"]
                advertised = self.visible.get("clarification_fields", [])
                words = re.sub(r"[^a-z0-9 ]", " ", field.replace(".", " ").lower()).split()
                question = action["question"].lower()
                if field not in advertised or not words or not all(w in question for w in words):
                    observation = "No information supplied: name an available field in the question."
                elif field in self._oracle.get("clarifications", {}):
                    observation = f"Observed {field}: {self._oracle['clarifications'][field]}"
                    self.acquired_fields.add(field)
                else: observation = "No additional information is available."
            elif kind == "lookup":
                self.tool_calls += 1
                self.lookups += 1
                if self.lookups > self.max_lookups: raise ValueError("lookup limit")
                query = action["query"]
                if query not in self.visible.get("lookup_queries", []):
                    observation = "No matching evidence."
                else:
                    result = self._oracle.get("facts", {}).get(query)
                    observation = "No matching evidence." if result is None else f"Lookup evidence: {result['text']}"
                    if result is not None: self.acquired_fields.update(result.get("fields", []))
            else: raise ValueError("unsupported action")
            self.messages.append({"role": "user", "content": observation})
            if len(self.events) >= self.max_actions: raise ValueError("action limit")
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.error, self.done = str(exc), True
        return self.done

    def finish_error(self, reason):
        self.error, self.done = reason, True

    def reward(self, generated_tokens, candidate_encoding_tokens=0):
        required = set(self._oracle.get("required_fields", []))
        grounded = required <= self.acquired_fields
        correct = self.answer is not None and equivalent(self.answer, self._oracle["answer"])
        unsupported = self.answer is not None and not grounded
        acquisition = bool(required & self.acquired_fields)
        cost = generated_tokens + .1 * candidate_encoding_tokens + 32 * self.tool_calls + 8 * self.transitions
        components = {"correctness": float(correct and grounded), "necessary_acquisition": .05 * acquisition,
                      "unsupported": -.5 * unsupported, "invalid": -.25 * bool(self.error),
                      "efficiency": -.1 * min(1., cost / 1024.)}
        return sum(components.values()), {"correct": bool(correct), "grounded": grounded, "error": self.error,
               "acquisition_success": acquisition, "reward_components": components, "cost_proxy": cost}


def adapt_legacy(episode):
    """Explicitly labeled compatibility conversion; never a new held-out record."""
    candidates = [{**c, "kind": "answer"} for c in episode.get("candidates", [])]
    field = "record.value"
    facts = {q: {"text": str(value), "fields": [field]} for q, value in episode.get("facts", {}).items()}
    required = [field] if episode.get("behavior") in {"lookup", "clarification"} else []
    return {"schema_version": 2, "id": episode["id"], "group": episode.get("group", episode["id"]),
            "behavior": episode["behavior"], "legacy": True,
            "visible": {"prompt": episode["prompt"].split(" Candidates: ")[0], "candidates": candidates,
                        "clarification_fields": [field] if "clarification" in episode else [], "lookup_queries": list(facts)},
            "oracle": {"answer": episode["answer"], "required_fields": required, "facts": facts,
                       "clarifications": {field: episode["clarification"]} if "clarification" in episode else {}}}
