import base64
import hashlib
import subprocess
import unittest
from scripts.recovery_cloud import observed_host_fingerprint,remote_reply_verified


class RemoteAccessTests(unittest.TestCase):
    def test_false_success_after_host_prompt_does_not_open_training_gate(self):
        response=subprocess.CompletedProcess([],0,'SSH: Attempting to connect to worker 0...','Connection abandoned.')
        self.assertFalse(remote_reply_verified(response,'KWS_REMOTE_unique'))
        response.stdout='echo KWS_REMOTE_unique\n'
        self.assertFalse(remote_reply_verified(response,'KWS_REMOTE_unique'))

    def test_exact_response_and_success_are_both_required(self):
        response=subprocess.CompletedProcess([],0,'banner\nKWS_REMOTE_unique\n','')
        self.assertTrue(remote_reply_verified(response,'KWS_REMOTE_unique'))
        response.returncode=255
        self.assertFalse(remote_reply_verified(response,'KWS_REMOTE_unique'))

    def test_key_must_belong_to_the_owned_endpoint_and_be_unambiguous(self):
        key=base64.b64encode(b'public-host-key').decode();ip='192.0.2.1'
        expected='SHA256:'+base64.b64encode(hashlib.sha256(b'public-host-key').digest()).decode().rstrip('=')
        self.assertEqual(observed_host_fingerprint(f'# comment\n{ip} ssh-ed25519 {key}',ip),expected)
        with self.assertRaises(RuntimeError):observed_host_fingerprint(f'192.0.2.2 ssh-ed25519 {key}',ip)
        second=base64.b64encode(b'other-key').decode()
        with self.assertRaises(RuntimeError):observed_host_fingerprint(f'{ip} ssh-ed25519 {key}\n{ip} ssh-ed25519 {second}',ip)
