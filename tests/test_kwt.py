import json

from kwt import format as fmt
from kwt import rewards, tasks


def test_parse_modes():
  p = fmt.parse("<mode:jev>\nB", 4)
  assert p.valid and p.via_jev and p.answer == "B" and not p.switched
  p = fmt.parse("<mode:cot>\n3+4=7\n<mode:jev>\nC", 4)
  assert p.valid and p.switched and p.modes == ("cot", "jev")
  p = fmt.parse("<mode:direct>\n<answer>7</answer>", 0)
  assert p.valid and p.answer == "7" and not p.via_jev


def test_parse_rejects_off_format():
  assert not fmt.parse("<mode:jev>\nB", 0).valid  # no options shown
  assert not fmt.parse("<mode:jev>\nE", 4).valid  # letter out of range
  assert not fmt.parse("<mode:direct>\n<mode:jev>\nA", 4).valid  # only cot may hand off
  assert not fmt.parse("Sure! <mode:direct>\n<answer>7</answer>", 0).valid
  assert not fmt.parse("<mode:cot>\n<answer>1</answer><answer>2</answer>", 0).valid
  assert not fmt.parse("<mode:direct>\n" + "x" * 200 + "<answer>7</answer>", 0).valid


def test_correctness_is_exact():
  opts = ["10", "100", "1000"]
  assert fmt.is_correct(fmt.parse("<mode:jev>\nB", 3), "100", opts)
  assert not fmt.is_correct(fmt.parse("<mode:jev>\nC", 3), "100", opts)
  # no substring credit (the old verifier's loophole)
  assert not fmt.is_correct(fmt.parse("<mode:direct>\n<answer>1000</answer>", 0), "100", [])
  assert fmt.is_correct(fmt.parse("<mode:direct>\n<answer>1,000</answer>", 0), "1000", [])


def test_rewards_prefer_correct_and_short():
  opts = json.dumps(["4", "5"])
  comps = ["<mode:jev>\nB", "<mode:cot>\n" + "think " * 60 + "\n<mode:jev>\nB", "<mode:jev>\nA"]
  total = [sum(x) for x in zip(
      rewards.correctness(None, comps, ["5"] * 3, [opts] * 3),
      rewards.well_formed(None, comps, [opts] * 3),
      rewards.compute_cost(None, comps))]
  assert total[0] > total[1] > total[2]


def test_tasks_render_and_balance():
  rows = tasks.load("val")
  assert {r["behavior"] for r in rows} == set(tasks.BEHAVIORS)
  for r in rows:
    opts = json.loads(r["options"])
    assert "Please solve this task" not in r["question"]
    assert r["prompts"].endswith("<think>\n\n</think>\n\n")
    if r["behavior"] in ("fast", "reason_decide"):
      assert r["answer"] in opts
    if r["behavior"] == "direct":
      assert opts == []
