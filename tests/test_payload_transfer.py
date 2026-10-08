"""Remote payload integrity and no renting after a failed local transfer."""
import base64
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from google.api_core.exceptions import NotFound
import google_crc32c
from scripts import recovery_cloud as cloud


class PayloadTransferTests(unittest.TestCase):
    def fixture(self,data=b'public verified fixture',exists=False,corrupt=False):
        blob=SimpleNamespace(metadata=None,size=None,crc32c=None)
        crc=base64.b64encode(google_crc32c.Checksum(data).digest()).decode()
        def reload(**kwargs):
            if blob.size is None:raise NotFound('not created')
        def upload(filename,**kwargs):
            blob.size=len(data);blob.crc32c='invalid' if corrupt else crc
        blob.reload=Mock(side_effect=reload);blob.upload_from_filename=Mock(side_effect=upload)
        if exists:
            blob.size=len(data);blob.crc32c='invalid' if corrupt else crc
            blob.metadata={'sha256':hashlib.sha256(data).hexdigest()}
        bucket=SimpleNamespace(blob=Mock(return_value=blob))
        client=SimpleNamespace(bucket=Mock(return_value=bucket),_http=SimpleNamespace(request=Mock()))
        return blob,bucket,client

    def test_new_object_requires_create_precondition_and_verified_crc(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'payload';path.write_bytes(b'public verified fixture')
            blob,bucket,client=self.fixture();original=client._http.request
            result=cloud.upload_owned_file({'project':'p','gcs_prefix':'gs://b/owned/run'},path,'payload.tar.gz',client)
            self.assertTrue(result['verified']);self.assertEqual(result['bytes'],path.stat().st_size)
            self.assertEqual(blob.upload_from_filename.call_args.kwargs['if_generation_match'],0)
            self.assertEqual(blob.upload_from_filename.call_args.kwargs['checksum'],'crc32c')
            self.assertEqual(bucket.blob.call_args.args,('owned/run/payload.tar.gz',))
            self.assertEqual(bucket.blob.call_args.kwargs['chunk_size'],256*1024)
            self.assertIs(client._http.request,original)

    def test_corrupt_existing_object_is_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'payload';path.write_bytes(b'public verified fixture')
            blob,_,client=self.fixture(exists=True,corrupt=True);original=client._http.request
            with self.assertRaisesRegex(RuntimeError,'checksum mismatch'):
                cloud.upload_owned_file({'project':'p','gcs_prefix':'gs://b/owned/run'},path,'payload',client)
            blob.upload_from_filename.assert_not_called();self.assertIs(client._http.request,original)

    def test_transfer_deadline_stops_before_another_http_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'payload';path.write_bytes(b'public verified fixture')
            blob,_,client=self.fixture();original=client._http.request
            blob.reload=Mock(side_effect=lambda **kwargs:client._http.request('GET','https://unused.example'))
            with patch.object(cloud.time,'monotonic',side_effect=[0,2]):
                with self.assertRaisesRegex(RuntimeError,'transfer deadline'):
                    cloud.upload_owned_file({'project':'p','gcs_prefix':'gs://b/owned/run'},path,'payload',client,timeout=1)
            original.assert_not_called();self.assertIs(client._http.request,original)

    def test_failed_transfer_cannot_start_cost_clock_or_create_tpu(self):
        config=json.loads(Path('configs/recovery.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);budget=Mock()
            with patch.object(cloud,'launch_gates',return_value={'passed':True}),\
                 patch.object(cloud,'preflight',return_value={'price':{}}),\
                 patch.object(cloud,'RecoveryBudget',return_value=budget),\
                 patch.object(cloud,'payload'),\
                 patch.object(cloud,'upload_owned_file',side_effect=RuntimeError('transport failed')),\
                 patch.object(cloud,'describe',return_value=None),patch.object(cloud,'gcloud') as commands:
                with self.assertRaisesRegex(RuntimeError,'transport failed'):cloud.execute(config,folder,folder)
            budget.begin.assert_not_called();commands.assert_not_called()
            session=json.loads((folder/'session.json').read_text())
            self.assertFalse(session['cost_clock_started']);self.assertFalse(session['provision_requested'])
            self.assertTrue(session['absence_verified'])

    def test_worker_reloads_final_session_after_connection_gate(self):
        session={'deadline':cloud.time.time()+3600,'run_id':'one','name':'owned','project':'p','zone':'z','gcs_prefix':'gs://b/one'}
        script=cloud.startup(session,'gs://b/one/payload.tar.gz',{})
        self.assertLess(script.index('Remote-access gate not verified'),script.index('gs://b/one/session.json session.json'))
        self.assertLess(script.index('gs://b/one/budget.json runs/recovery-v2/budget.json'),script.index('-m switching.tunix_experiment'))
