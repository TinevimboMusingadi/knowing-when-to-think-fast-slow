from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import restart_rl


class RestartTests(unittest.TestCase):
    def test_insufficient_budget_is_refused_before_provisioning(self):
        self.assertEqual(restart_rl.restart_allowance(43.19,1.8),1.8)
        with self.assertRaises(RuntimeError):
            restart_rl.restart_allowance(44.95,1.8)

    def test_restart_only_trains_rl_and_keeps_fresh_optimizer(self):
        session = {"run_id": "20261007-010000", "name": "kws-20261007-010000", "created": 123}
        script = restart_rl.startup(session, "gs://bucket/payload", "gs://bucket/sft")
        self.assertIn("--phase rl --resume", script)
        self.assertNotIn("--phase sft", script)
        self.assertNotIn("--restore-optimizer", script)
        self.assertIn("--max-steps 3", script)
        self.assertIn("--started 123", script)
        self.assertIn("SECONDS_LEFT-120", script)
        self.assertIn('"runs/20261007-010000/sft"', script)
        self.assertIn("trap cleanup EXIT", script)

    def test_continuation_restores_all_replicas_without_reset(self):
        session = {"run_id": "test", "name": "kws-test", "created": 123}
        sources = [f"gs://bucket/rl-rank{rank}" for rank in range(4)]
        script = restart_rl.startup(session, "gs://bucket/payload", sources)
        for rank, source in enumerate(sources):
            self.assertIn(source, script)
            self.assertIn(f"runs/test/restore/rank{rank}", script)
        self.assertIn('--resume "$RUN/restore/rank{rank}" --restore-optimizer', script)
        self.assertNotIn("--max-steps 3", script)
        self.assertNotIn("--phase sft", script)

    def test_revised_cap_keeps_storage_reserved(self):
        self.assertEqual(restart_rl.restart_allowance(44.95,8.28,60),8.28)
        self.assertAlmostEqual(restart_rl.restart_allowance(44.95,20,60),10.05)
        with self.assertRaises(ValueError):
            restart_rl.restart_allowance(float('nan'),8.28,60)


if __name__ == "__main__":
    unittest.main()
