import json
import tempfile
import unittest
from pathlib import Path
from switching.data import build,make_episode
from switching.protocol import EpisodeEnv,MODES,encode_action,parse_action,decision_state
from switching.storage import Budget

class CoreTests(unittest.TestCase):
    def test_teacher_episodes_are_grounded(self):
        for split in ("train","val","test"):
            for behavior in ("fast","direct","reasoning","reason_decide","clarification","lookup"):
                row=make_episode(split,behavior,11); env=EpisodeEnv(row)
                for step in row["steps"]:env.step(step["completion"],step.get("decision"))
                reward,report=env.reward(0)
                self.assertTrue(report["correct"],(split,behavior,env.error));self.assertTrue(report["grounded"])
                self.assertEqual(reward,1)

    def test_three_decision_types(self):
        for index,kind in enumerate(("choice","yes_no_unknown","bounded_score")):
            row=make_episode("train","fast",index);self.assertEqual(row["question_type"],kind)
            env=EpisodeEnv(row);step=row["steps"][0];env.step(step["completion"],step["decision"])
            self.assertTrue(env.reward(0)[1]["correct"])

    def test_rubrics_are_unique_and_unknown_is_supervised(self):
        rows=[make_episode("train","fast",index) for index in range(120)]
        self.assertEqual(len({r["prompt"] for r in rows}),len(rows))
        labels={r["answer"] for r in rows if r["question_type"]=="yes_no_unknown"}
        self.assertEqual(labels,{"yes","no","unknown"})
        self.assertEqual({r["answer"] for r in rows if r["question_type"]=="bounded_score"},set(range(6)))

    def test_jev_can_continue_into_reasoning(self):
        row=make_episode("train","reasoning",2);env=EpisodeEnv(row)
        first=row["steps"][0];env.step(first["completion"],first["decision"])
        self.assertFalse(env.done)
        for step in row["steps"][1:]:env.step(step["completion"],step.get("decision"))
        self.assertTrue(env.reward(0)[1]["correct"]);self.assertEqual(env.modes[:2],[MODES[0],MODES[2]])

    def test_context_cannot_be_skipped_for_reward(self):
        row=make_episode("test","lookup",0);env=EpisodeEnv(row)
        env.step(encode_action(MODES[1],"answer",value=row["answer"]))
        self.assertFalse(env.reward(0)[1]["grounded"])
        self.assertLess(env.reward(0)[0],0)

    def test_action_validation(self):
        for invalid in ("hello",MODES[0]+'{"action":"answer","value":1}',MODES[1]+'{"action":"lookup","query":3}'):
            with self.assertRaises(ValueError):parse_action(invalid)

    def test_jev_token_is_an_executable_action(self):
        mode,action=parse_action(MODES[0]);self.assertEqual(action["action"],"decide")
        messages=[{"role":"user","content":"Question Candidates: [a,b]"},{"role":"assistant","content":encode_action(MODES[2],"reason",text="Observed reasoning")},{"role":"user","content":"Evidence"}]
        state=decision_state(messages);self.assertIn("Observed reasoning",state);self.assertNotIn("[a,b]",state)

    def test_lookup_limit(self):
        row=make_episode("test","lookup",0);env=EpisodeEnv(row,max_lookups=1)
        text=encode_action(MODES[1],"lookup",query="missing")
        env.step(text);env.step(text);self.assertEqual(env.error,"lookup limit")

    def test_counts_and_group_separation(self):
        with tempfile.TemporaryDirectory() as root:
            manifest=build(root);self.assertEqual(manifest["splits"]["train"]["count"],8000)
            groups=[]
            for split in ("train","val","test"):
                rows=[json.loads(l) for l in (Path(root)/f"{split}.jsonl").read_text().splitlines()]
                groups.append({r["group"] for r in rows})
            self.assertFalse(groups[0]&groups[1]);self.assertFalse(groups[0]&groups[2]);self.assertFalse(groups[1]&groups[2])

    def test_budget_stops_and_persists(self):
        with tempfile.TemporaryDirectory() as root:
            b=Budget(Path(root)/"budget.json",3600,"pilot",{"pilot":5},start=0)
            with self.assertRaises(RuntimeError):b.check()

if __name__=="__main__":unittest.main()
