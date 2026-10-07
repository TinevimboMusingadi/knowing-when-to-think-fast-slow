import unittest
from switching.episode_v2 import EpisodeEnvV2, public_episode
from switching.data_v2 import make
from switching.protocol import MODES


class EpisodeV2Tests(unittest.TestCase):
    def test_no_oracle_or_supervision_in_observations(self):
        row=make("train","lookup",0);env=EpisodeEnvV2(row)
        self.assertNotIn("oracle",public_episode(row));self.assertNotIn("supervision",public_episode(row))
        # The answer is one candidate, but the oracle never identifies that candidate.
        import copy
        changed=copy.deepcopy(row);changed["oracle"]["answer"]="hidden sentinel"
        self.assertEqual(env.observations(),EpisodeEnvV2(changed).observations())
        self.assertNotIn("hidden sentinel",str(env.observations()))

    def test_arbitrary_question_does_not_reveal_answer(self):
        row=make("train","clarification",0);env=EpisodeEnvV2(row)
        env.step(MODES[1]+'{"action":"ask","field":"record.value","question":"Hello?"}')
        self.assertFalse(env.acquired_fields)

    def test_cot_acquires_evidence_and_jev_chooses_answer(self):
        row=make("train","lookup",0);env=EpisodeEnvV2(row)
        query=row["visible"]["lookup_queries"][0]
        env.step(MODES[2]+'{"action":"lookup","query":"'+query+'"}')
        selected=next(c for c in env.candidates if c["kind"]=="answer" and c["value"]==row["oracle"]["answer"])
        env.step(MODES[0],selected["id"])
        _,info=env.reward(19,100)
        self.assertTrue(info["correct"] and info["grounded"]);self.assertEqual(env.transitions,1)

    def test_defer_never_forces_cot_or_ends_task(self):
        env=EpisodeEnvV2(make("train","reasoning",0));env.step(MODES[0],"defer")
        self.assertFalse(env.done);self.assertNotIn("cot",env.messages[-1]["content"])

    def test_unnecessary_acquisition_has_no_bonus(self):
        row=make("train","lookup",1);env=EpisodeEnvV2(row)
        for step in make("train","lookup",0)["supervision"][:1]:
            text=step["text"].replace(make("train","lookup",0)["visible"]["lookup_queries"][0],row["visible"]["lookup_queries"][0]);env.step(text)
        _,info=env.reward(10);self.assertEqual(info["reward_components"]["necessary_acquisition"],0)

    def test_every_generated_teacher_is_valid_and_grounded(self):
        from switching.data_v2 import BEHAVIORS
        for split in ("train","val","test"):
            for behavior in BEHAVIORS:
                for i in range(12):
                    row=make(split,behavior,i);env=EpisodeEnvV2(row)
                    for action in row["supervision"]:env.step(action["text"],action.get("choice_id"))
                    _,info=env.reward(0)
                    self.assertEqual((info["correct"],info["grounded"],info["error"]),(True,True,None))
