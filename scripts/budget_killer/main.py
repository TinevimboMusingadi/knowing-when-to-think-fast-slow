"""Kill switch 5: a GCP budget alert deletes every TPU VM labelled kwt=1.

Wired as: Billing budget -> Pub/Sub topic -> this Cloud Run function.
It acts only when cost >= budget, so the 50%/90% alerts are just emails.
Deploy steps: scripts/budget_killer/README.md
"""

import base64
import json
import os

import functions_framework
from googleapiclient import discovery

PROJECT = os.environ["PROJECT"]


@functions_framework.cloud_event
def kill_tpus(event):
  data = json.loads(base64.b64decode(event.data["message"]["data"]))
  cost, budget = data.get("costAmount", 0), data.get("budgetAmount", 0)
  print(f"budget alert: cost={cost} budget={budget}")
  if budget <= 0 or cost < budget:
    return
  tpu = discovery.build("tpu", "v2", cache_discovery=False)
  nodes = tpu.projects().locations().nodes()
  parent = f"projects/{PROJECT}/locations/-"
  for node in nodes.list(parent=parent).execute().get("nodes", []):
    if node.get("labels", {}).get("kwt") == "1":
      print("deleting", node["name"])
      nodes.delete(name=node["name"]).execute()
