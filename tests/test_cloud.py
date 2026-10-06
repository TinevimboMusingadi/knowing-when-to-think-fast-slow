import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location("cloud",Path(__file__).parents[1]/"scripts/cloud.py")
cloud=importlib.util.module_from_spec(spec);spec.loader.exec_module(cloud)

class CloudTests(unittest.TestCase):
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
