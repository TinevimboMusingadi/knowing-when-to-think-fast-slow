"""Public action protocol; no private reasoning vocabulary is imported."""
import json
import re

MODES = ("<mode:jev>", "<mode:direct>", "<mode:cot>")
SYSTEM = """Choose one mode: <mode:jev> for a typed decision, <mode:direct> for a direct answer or context request, <mode:cot> for ordinary reasoning. For JEV emit only <mode:jev>; the runtime immediately scores supplied candidates using observed context. For direct or CoT emit one mode token followed by one JSON object. Actions: answer (value), reason (text), ask (question), lookup (query). Reasoning can establish a state for a later decision. Ask or look up facts that are missing; do not invent them. End each text action after its JSON object."""

def encode_action(mode, action, **fields):
    if mode not in MODES:
        raise ValueError("unknown mode")
    return mode + json.dumps({"action": action, **fields}, ensure_ascii=False)

def parse_action(text):
    if text.strip()==MODES[0]:return MODES[0],{"action":"decide","state":""}
    match = re.fullmatch(r"\s*(<mode:(?:jev|direct|cot)>)(\{.*\})\s*", text, re.S)
    if not match:
        raise ValueError("invalid mode/action framing")
    payload = json.loads(match[2])
    if not isinstance(payload, dict):
        raise ValueError("action must be an object")
    action = payload.get("action")
    allowed = {MODES[0]: {"decide"}, MODES[1]: {"answer", "ask", "lookup"}, MODES[2]: {"reason", "answer"}}
    if action not in allowed[match[1]]:
        raise ValueError("action incompatible with mode")
    required = {"answer": "value", "ask": "question", "lookup": "query", "reason": "text", "decide": "state"}[action]
    if required not in payload:
        raise ValueError("missing action field")
    if action != "answer" and not isinstance(payload[required], str):
        raise ValueError("action field must be text")
    return match[1], payload

def equivalent(actual, expected):
    try:
        return abs(float(actual) - float(expected)) < 1e-6
    except (ValueError, TypeError):
        return str(actual).strip().casefold() == str(expected).strip().casefold()

def decision_state(messages):
    """Only observed context, excluding candidate enumeration and control prompts."""
    pieces=[]
    for message in messages:
        if message["role"]=="system":continue
        text=message["content"]
        if message["role"]=="assistant":
            try:
                _,action=parse_action(text)
                if action["action"]=="reason":pieces.append(action["text"])
            except ValueError:pass
        elif not text.startswith("Continue with"):
            pieces.append(text.split(" Candidates: ")[0])
    return "\n".join(pieces)

class EpisodeEnv:
    """Deterministic training fixture. Public requests never expose hidden answers."""
    def __init__(self, episode, max_transitions=4, max_lookups=2):
        self.episode = episode
        self.messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": episode["prompt"]}]
        self.modes = []
        self.lookups = 0
        self.asked = False
        self.acquired = False
        self.done = False
        self.answer = None
        self.max_transitions = max_transitions
        self.max_lookups = max_lookups
        self.error = None

    def step(self, text, decision=None):
        if self.done:
            raise ValueError("episode is finished")
        try:
            mode, action = parse_action(text)
            transitions = sum(a != b for a, b in zip(self.modes, self.modes[1:]))
            if self.modes and self.modes[-1] != mode:
                transitions += 1
            if transitions > self.max_transitions:
                raise ValueError("transition limit")
            self.modes.append(mode)
            self.messages.append({"role": "assistant", "content": text})
            kind = action["action"]
            if kind == "answer":
                self.answer, self.done = action["value"], True
            elif kind == "decide":
                if decision is None or decision not in [c["value"] for c in self.episode.get("candidates", [])]:
                    raise ValueError("invalid typed decision")
                self.answer, self.done = decision, True
            elif kind == "ask":
                self.asked = True
                reply = self.episode.get("clarification", "No additional information is available.")
                self.acquired |= "clarification" in self.episode
                self.messages.append({"role": "user", "content": reply})
            elif kind == "lookup":
                self.lookups += 1
                if self.lookups > self.max_lookups:
                    raise ValueError("lookup limit")
                facts = self.episode.get("facts", {})
                found = action["query"] in facts
                self.acquired |= found
                self.messages.append({"role": "user", "content": "Lookup result (external evidence): " + str(facts.get(action["query"], "No matching evidence."))})
            else:
                self.messages.append({"role": "user", "content": "Continue with a decision or final answer using the established facts."})
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.error, self.done = str(exc), True
        return self.done

    def reward(self, tokens):
        correct = equivalent(self.answer, self.episode["answer"]) if self.answer is not None else False
        needs = self.episode["behavior"] in {"clarification", "lookup"}
        grounded = not needs or self.acquired
        reward = 1.0 * (correct and grounded) - 0.002 * tokens
        if needs and not self.acquired:
            reward -= 0.5
        if self.error:
            reward -= 0.5
        reward -= 0.05 * max(0, len(self.modes) - self.episode.get("optimal_actions", 1))
        return reward, {"correct": bool(correct), "grounded": grounded, "error": self.error}
