"""Offline reward counterexamples. Scripted trajectories are not model results."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .data_v2 import make
from .episode_v2 import EpisodeEnvV2
from .protocol import MODES, encode_action


def audit():
    checks = []
    scores = {}

    def replay(name, row, actions, tokens=20, candidate_tokens=0):
        env = EpisodeEnvV2(row)
        for text, choice in actions:
            env.step(text, choice)
        value, info = env.reward(tokens, candidate_tokens)
        scores[name] = {"reward": value, **info}
        return value, info

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})

    def answer(row, mode=MODES[1], value=None):
        gold = row["oracle"]["answer"] if value is None else value
        if mode == MODES[0]:
            candidate = next(c for c in row["visible"]["candidates"]
                             if c["kind"] == "answer" and c["value"] == gold)
            return mode, candidate["id"]
        return encode_action(mode, "answer", value=gold), None

    fast = make("val", "fast", 0)
    right, _ = replay("fast_correct", fast, [answer(fast)])
    wrong, _ = replay("fast_wrong", fast, [answer(fast, value=fast["oracle"]["answer"] + 1)])
    check("correct-answer-outranks-wrong-answer", right > wrong)

    missing = make("val", "lookup", 0)
    query = missing["visible"]["lookup_queries"][0]
    lookup = encode_action(MODES[1], "lookup", query=query), None
    unsupported, unsupported_info = replay("unsupported_correct_guess", missing, [answer(missing)])
    grounded, grounded_info = replay("necessary_lookup_then_answer", missing, [lookup, answer(missing)])
    check("correct-guess-without-required-evidence-is-penalized",
          unsupported < 0 and unsupported_info["reward_components"]["correctness"] == 0)
    check("necessary-evidence-enables-grounded-correctness",
          grounded > unsupported and grounded_info["correct"] and grounded_info["grounded"])

    repeated, repeated_info = replay("repeat_lookup_then_answer", missing, [lookup, lookup, answer(missing)])
    check("repeat-lookup-cannot-multiply-acquisition-bonus",
          repeated_info["reward_components"]["necessary_acquisition"] == .05 and repeated < grounded)

    irrelevant = deepcopy(missing)
    irrelevant["oracle"]["facts"][query] = {"text": "The record label uses blue ink.", "fields": []}
    _, irrelevant_info = replay("irrelevant_lookup_then_guess", irrelevant, [lookup, answer(irrelevant)])
    check("irrelevant-evidence-has-no-acquisition-bonus",
          irrelevant_info["reward_components"]["necessary_acquisition"] == 0
          and not irrelevant_info["grounded"])

    present = make("val", "lookup", 1)
    immediate, _ = replay("context_present_answer", present, [answer(present)])
    unnecessary, unnecessary_info = replay("context_present_lookup_then_answer", present,
                                          [lookup, answer(present)])
    check("already-present-evidence-favors-immediate-correct-answer",
          immediate > unnecessary and unnecessary_info["reward_components"]["necessary_acquisition"] == 0)

    clarification = make("val", "clarification", 0)
    ask = encode_action(MODES[1], "ask", field="record.value", question="Hello?"), None
    _, ask_info = replay("arbitrary_question_then_guess", clarification, [ask, answer(clarification)])
    check("arbitrary-question-does-not-unlock-evidence",
          not ask_info["grounded"] and ask_info["reward_components"]["necessary_acquisition"] == 0)

    mode_values = [replay("same_answer_" + mode, fast, [answer(fast, mode)])[0] for mode in MODES]
    check("no-intrinsic-bonus-for-any-mode", max(mode_values) - min(mode_values) < 1e-12)
    reason = encode_action(MODES[2], "reason", text="Consider the observed quantities."), None
    switched, _ = replay("extra_reason_and_switch", fast, [reason, answer(fast)])
    check("gratuitous-switching-is-costly", switched < right)

    _, invalid_info = replay("invalid_action", fast, [("not a valid action", None)])
    check("invalid-execution-is-penalized", invalid_info["reward_components"]["invalid"] == -.25)
    _, exhausted_info = replay("too_many_lookups", missing, [lookup, lookup, lookup])
    check("exhausted-lookup-limit-is-penalized", exhausted_info["reward_components"]["invalid"] == -.25)

    expensive, expensive_info = replay("very_expensive_correct", fast, [answer(fast)], tokens=100000)
    check("efficiency-penalty-is-bounded-and-correctness-dominates",
          expensive_info["reward_components"]["efficiency"] == -.1 and expensive > wrong)
    encoded, _ = replay("candidate_encoding_work", fast, [answer(fast)], candidate_tokens=1000)
    check("candidate-encoding-work-is-charged", encoded < right)

    changed = deepcopy(missing)
    changed["oracle"]["answer"] = "private sentinel"
    check("oracle-answer-is-absent-from-policy-observations",
          EpisodeEnvV2(missing).observations() == EpisodeEnvV2(changed).observations())

    # Document the actual shaping rule; do not silently change historical rewards.
    partial = deepcopy(missing)
    partial["oracle"]["required_fields"].append("record.other")
    _, partial_info = replay("partial_acquisition_then_unsupported_answer", partial, [lookup, answer(partial)])
    return {"schema_version": 1, "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "method": "scripted counterexamples against the current EpisodeEnvV2 verifier",
            "paid_compute": False, "sampled_model_rollouts": False,
            "training_or_performance_claim": False, "passed": all(c["passed"] for c in checks),
            "checks": checks, "scenarios": scores,
            "limitations": [
                "Necessary-acquisition shaping currently grants 0.05 for any required field acquired, even on an unfinished or incorrect task; final correctness still requires all evidence.",
                "This audit does not establish sufficient reward variation in sampled policy groups, stable gradients, or learned switching.",
                "Evidence grounding follows verified fixture metadata; it does not judge the semantic quality of arbitrary reasoning text."],
            "observed_partial_acquisition_bonus": partial_info["reward_components"]["necessary_acquisition"],
            "source_hashes": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("episode_v2.py", "data_v2.py", "audit_reward_v2.py")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]),
                      "sampled_model_rollouts": False, "output": str(args.output)}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
