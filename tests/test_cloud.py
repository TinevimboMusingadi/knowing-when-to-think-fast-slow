import importlib.util
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
from scripts.stage_limit import seconds_remaining
spec=importlib.util.spec_from_file_location("cloud",Path(__file__).parents[1]/"scripts/cloud.py")
cloud=importlib.util.module_from_spec(spec);spec.loader.exec_module(cloud)

class CloudTests(unittest.TestCase):
    def test_cloud_child_has_closed_stdin_and_hidden_window(self):
        result=cloud.subprocess.CompletedProcess([],0,stdout="{}",stderr="")
        with patch.object(cloud.shutil,"which",return_value="gcloud.cmd"),patch.object(cloud,"sdk_command",return_value=["python","gcloud.py"]),patch.object(cloud.subprocess,"run",return_value=result) as run:
            cloud.gcloud("info")
            self.assertEqual(run.call_args.kwargs["stdin"],cloud.subprocess.DEVNULL)
            if cloud.os.name=="nt":self.assertEqual(run.call_args.kwargs["creationflags"],cloud.subprocess.CREATE_NO_WINDOW)

    def test_empty_cloud_errors_include_exit_code(self):
        result=cloud.subprocess.CompletedProcess([],17,stdout="",stderr="")
        with patch.object(cloud.shutil,"which",return_value="gcloud.cmd"),patch.object(cloud,"sdk_command",return_value=["python","gcloud.py"]),patch.object(cloud.subprocess,"run",return_value=result):
            with self.assertRaisesRegex(RuntimeError,"code 17"):cloud.gcloud("info")

    def test_windows_sdk_bypasses_console_batch_wrapper(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);entry=root/"bin/gcloud.cmd";entry.parent.mkdir();entry.touch()
            python=root/"platform/bundledpython/python.exe";python.parent.mkdir(parents=True);python.touch()
            script=root/"lib/gcloud.py";script.parent.mkdir();script.touch()
            command=cloud.sdk_command(str(entry),["--command","x"*10000])
            self.assertEqual(command,[str(python.resolve()),"-S",str(script.resolve()),"--command","x"*10000])

    def test_watchdog_reports_teardown_failure_cause(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"session.json";path.write_text(json.dumps({"deadline":0}))
            with patch.object(cloud,"delete_owned",side_effect=RuntimeError("permission denied")),patch.object(cloud.time,"sleep"),patch("builtins.print") as printed:
                with self.assertRaisesRegex(RuntimeError,"permission denied"):cloud.watchdog(path)
                self.assertEqual(printed.call_count,3)

    def test_independent_stage_timer_includes_margin_and_global_cap(self):
        self.assertEqual(seconds_remaining({},"rl",10,0,3600,{"rl":15}),1095)
        self.assertEqual(seconds_remaining({"stages":{"sft":49}},"rl",10,0,0,{"rl":15}),313)
        self.assertEqual(seconds_remaining({"stages":{"sft":50}},"rl",10,0,0,{"rl":15}),0)

    def test_stage_timer_uses_explicit_revised_cap(self):
        self.assertEqual(seconds_remaining({"stages":{"prior":44.95}},"rl",4.8,0,0,{"rl":8.28},60),5400)
        self.assertEqual(seconds_remaining({"stages":{"prior":60}},"rl",4.8,0,0,{"rl":8.28},60),0)

    def test_fresh_sft_does_not_restore_an_old_dataset(self):
        session={"run_id":"test","name":"kws-test","created":1,"fresh_sft_from_run":"20261006-115020"}
        script=cloud.startup(session,"gs://bucket/payload.tar.gz")
        self.assertNotIn('--phase pilot',script);self.assertNotIn('--restore-optimizer',script)
        self.assertIn('--phase sft',script)
    def test_recovery_keeps_rank_state_and_skips_completed_pilot(self):
        session={"run_id":"test","name":"kws-test","created":1,"resume_sft_run":"20261006-111427"}
        script=cloud.startup(session,"gs://bucket/payload.tar.gz")
        self.assertIn('restore/rank{rank}',script)
        self.assertIn('--restore-optimizer',script)
        self.assertNotIn('--phase pilot',script)
        self.assertIn('timeout --signal=TERM --kill-after=120',script)

    def test_startup_preserves_failure_log(self):
        session={"run_id":"test","name":"kws-test","created":1}
        script=cloud.startup(session,"gs://bucket/payload.tar.gz")
        self.assertIn("startup.log",script);self.assertIn("requirements-lock.txt",script)
        self.assertIn("download.pytorch.org/whl/cpu",script)
    def test_never_delete_other_runs(self):
        session={"name":"kws-one","run_id":"one","project":"p","zone":"z"}
        with patch.object(cloud,"describe",return_value={"labels":{"kws_run":"other"}}),patch.object(cloud,"gcloud") as call:
            with self.assertRaises(RuntimeError):cloud.delete_owned(session)
            call.assert_not_called()

    def test_owned_deletion_only(self):
        session={"name":"kws-one","run_id":"one","project":"p","zone":"z"}
        with patch.object(cloud,"describe",return_value={"labels":{"kws_run":"one"}}),patch.object(cloud,"gcloud") as call:
            self.assertTrue(cloud.delete_owned(session));self.assertIn("kws-one",call.call_args.args)

    def test_auth_errors_are_not_treated_as_deleted(self):
        result=type("Result",(),{"returncode":1,"stderr":"permission denied"})()
        with patch.object(cloud,"gcloud",return_value=result):
            with self.assertRaises(RuntimeError):cloud.describe({"name":"kws-one","project":"p","zone":"z"})

if __name__=="__main__":unittest.main()
