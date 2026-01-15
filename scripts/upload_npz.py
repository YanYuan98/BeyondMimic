import os
import wandb
import numpy as np

REGISTRY_NAME = "motions"
COLLECTION_NAME = "82_08_stageii_mod_edit"

run = wandb.init(project="csv_to_npz", name=COLLECTION_NAME)

npz_path = "./logs/tmp/82_08_stageii/motion.npz"
if os.path.exists(npz_path):
    print("npz_path exits!")
    logged_artifact = run.log_artifact(artifact_or_path=npz_path, 
                                    name=COLLECTION_NAME, type=REGISTRY_NAME)

    run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}")
