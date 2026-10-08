import base64
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import recovery_cloud as cloud
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

    def test_supported_keys_are_selected_deterministically(self):
        ip='192.0.2.1';key=base64.b64encode(b'public-host-key').decode()
        for kind in cloud.HOST_KEY_TYPES:
            self.assertEqual(cloud.observed_host_key(f'{ip} {kind} {key}',ip)[0],kind)
        rows=f'{ip} ssh-rsa {key}\n{ip} ecdsa-sha2-nistp256 {key}\n{ip} ssh-ed25519 {key}'
        self.assertEqual(cloud.observed_host_key(rows,ip)[0],'ssh-ed25519')
        with self.assertRaises(RuntimeError):observed_host_fingerprint(f'{ip} unknown {key}',ip)

    def test_boot_readiness_is_retried_without_weakening_key_checks(self):
        ip='192.0.2.1';key=base64.b64encode(b'public-host-key').decode();probes=[]
        responses=[subprocess.CompletedProcess([],1,'','not ready'),
                   subprocess.CompletedProcess([],0,f'{ip} ecdsa-sha2-nistp256 {key}\n','')]
        with patch.object(cloud.subprocess,'run',side_effect=responses) as run,patch.object(cloud.time,'sleep'):
            kind,_=cloud.probe_host_key('scanner',ip,probes)
        self.assertEqual(kind,'ecdsa-sha2-nistp256');self.assertEqual(len(probes),2)
        self.assertIn('ed25519,ecdsa,rsa',run.call_args.args[0])

    def test_ambiguous_keys_are_not_retried(self):
        ip='192.0.2.1';keys=[base64.b64encode(value).decode() for value in (b'one',b'two')]
        response=subprocess.CompletedProcess([],0,'\n'.join(f'{ip} ssh-rsa {key}' for key in keys),'')
        with patch.object(cloud.subprocess,'run',return_value=response) as run,patch.object(cloud.time,'sleep') as sleep:
            with self.assertRaises(RuntimeError):cloud.probe_host_key('scanner',ip,[])
        run.assert_called_once();sleep.assert_not_called()

    def test_failed_readiness_has_a_deadline_and_diagnostics(self):
        response=subprocess.CompletedProcess([],1,'','192.0.2.1 not ready');probes=[]
        with patch.object(cloud.subprocess,'run',return_value=response),patch.object(cloud.time,'sleep'),\
             patch.object(cloud.time,'monotonic',side_effect=[0,0,91,91]):
            with self.assertRaisesRegex(RuntimeError,'deadline'):cloud.probe_host_key('scanner','192.0.2.1',probes)
        self.assertEqual(probes[0]['stderr'],'<owned-endpoint> not ready')

    def test_failed_remote_gate_publishes_failure_not_old_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);(folder/'remote-access.json').write_text('{"passed":true,"run_id":"old"}')
            with patch.object(cloud,'_verify_remote_access',side_effect=RuntimeError('probe failed')):
                with self.assertRaises(RuntimeError):cloud.verify_remote_access({'run_id':'current'},folder)
            import json
            report=json.loads((folder/'remote-access.json').read_text())
            self.assertFalse(report['passed']);self.assertEqual(report['run_id'],'current')
