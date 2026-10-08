import unittest
from unittest.mock import patch
from scripts import recovery_cloud as cloud


class RecoveryCloudTests(unittest.TestCase):
    def session(self):return {"run_id":"one","name":"kws-recovery-one","project":"p","zone":"z","deadline":10000,"stage_deadline":5000,"gcs_prefix":"gs://b/one"}
    def test_deletion_requires_both_ownership_labels(self):
        for labels in ({"kws_run":"one"},{"kws_run":"other","kws_schema":"v2"}):
            with patch.object(cloud,"describe",return_value={"labels":labels}),patch.object(cloud,"gcloud") as command:
                with self.assertRaises(PermissionError):cloud.delete_and_verify(self.session())
                command.assert_not_called()
    def test_delete_success_requires_a_following_absence_check(self):
        with patch.object(cloud,"describe",side_effect=[{"labels":{"kws_run":"one","kws_schema":"v2"}},None]) as read,patch.object(cloud,"gcloud") as command,patch.object(cloud.time,"sleep"):
            cloud.delete_and_verify(self.session());self.assertEqual(read.call_count,2);command.assert_called_once()
    def test_pending_deletion_is_observed_without_reissuing_delete(self):
        pending={'labels':{'kws_run':'one','kws_schema':'v2'},'state':'DELETING'}
        with patch.object(cloud,'describe',side_effect=[pending,None]),patch.object(cloud,'gcloud') as command,patch.object(cloud.time,'sleep'):
            cloud.delete_and_verify(self.session());command.assert_not_called()
    def test_startup_uses_only_the_locked_tunix_environment(self):
        script=cloud.startup(self.session(),"gs://b/one/payload.tar.gz",{})
        self.assertIn("requirements-tunix-tpu.lock",script);self.assertIn("--require-hashes",script)
        self.assertNotIn("torch_xla",script);self.assertIn('"$SCHEMA" = "v2"',script)

    def test_quota_cannot_use_another_zone_or_serving_capacity(self):
        import copy
        metric={'metric':'tpu.googleapis.com/tpu-v5s-litepod-preemptible','consumerQuotaLimits':[
          {'unit':'1/{project}/{zone}','supportedLocations':['us-west4-a','us-west4-b'],
           'quotaBuckets':[{'effectiveLimit':'1536'},{'dimensions':{'zone':'us-west4-a'},'effectiveLimit':'0'},
                           {'dimensions':{'zone':'us-west4-b'},'effectiveLimit':'1536'}]},
          {'unit':'1/{project}/{region}','supportedLocations':['us-west4'],
           'quotaBuckets':[{'dimensions':{'region':'us-west4'},'effectiveLimit':'-1'}]}]}
        self.assertEqual(cloud.training_quota({'metrics':[metric]},'us-west4-a'),[])
        metric['consumerQuotaLimits'][0]['quotaBuckets'][1]['effectiveLimit']='4'
        self.assertTrue(cloud.training_quota({'metrics':[metric]},'us-west4-a'))
        serving=copy.deepcopy(metric);serving['metric']='tpu.googleapis.com/tpu-v5s-litepod-serving-preemptible'
        self.assertEqual(cloud.training_quota({'metrics':[serving]},'us-west4-a'),[])

    def test_v5e_price_uses_four_chip_hours_and_rejects_ambiguous_skus(self):
        def sku(name,usage,rate):
            return {'name':name,'description':'TpuV5e '+('attached to Spot Preemptible VMs running in Las Vegas' if usage=='Preemptible' else 'running in Las Vegas'),
                    'category':{'usageType':usage,'resourceGroup':'TPU'},'serviceRegions':['us-west4'],
                    'pricingInfo':[{'pricingExpression':{'usageUnit':'h','tieredRates':[{'unitPrice':{'currencyCode':'USD','units':str(int(rate)),'nanos':round((rate-int(rate))*1e9)}}]}}]}
        rows=[sku('spot','Preemptible',.494237),sku('standard','OnDemand',1.2)]
        config={'zone':'us-west4-a','accelerator':'v5litepod-4','rate_upper_bound':4.8}
        result=cloud.resolve_interruptible_price(rows,config)
        self.assertAlmostEqual(result['whole_slice_rate'],1.976948);self.assertEqual(result['unit_multiplier'],4)
        with self.assertRaises(RuntimeError):cloud.resolve_interruptible_price(rows+[rows[0]],config)
        with self.assertRaises(RuntimeError):cloud.resolve_interruptible_price([rows[0],sku('wrong','OnDemand',4.8)],config)
