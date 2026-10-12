# Budget killer (kill switch 5)

This is the cloud-side backstop. It works even if the VM's own guard never ran. When project spend reaches the budget, it deletes every TPU VM labelled `kwt=1`.

You only need to set it up once. Run these commands in Cloud Shell:

```bash
PROJECT=serious-unison-452122-i2
gcloud pubsub topics create kwt-budget --project=$PROJECT
gcloud functions deploy kwt-budget-killer --gen2 --project=$PROJECT --region=us-west4 \
  --runtime=python312 --source=scripts/budget_killer --entry-point=kill_tpus \
  --trigger-topic=kwt-budget --set-env-vars=PROJECT=$PROJECT
```

The function runs as the default compute service account. That account needs the TPU Admin role (`roles/tpu.admin`).

Then create the budget in Console → Billing → Budgets & alerts:
- scope it to this project;
- set the amount to $50;
- add threshold rules at 50%, 90% and 100%;
- under "Connect a Pub/Sub topic", choose `kwt-budget`.

GCP billing data lags by a few hours. So this is a backstop, not the main control. The VM's own deadline and heartbeat stop it within minutes.
