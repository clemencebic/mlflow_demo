# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

client = dataiku.api_client()
project = client.get_default_project()

# Get or create SavedModel
sm_name = "catboost-uci-bank"
sm_id = None
for sm in project.list_saved_models():
    if sm_name != sm["name"]:
        continue
    else:
        sm_id = sm["id"]
        print("Found SavedModel {} with id {}".format(sm_name, sm_id))
        break
if sm_id:
    sm = project.get_saved_model(sm_id)
else:
    sm = project.create_mlflow_pyfunc_model(name=sm_name,
                                            prediction_type=DSSPredictionMLTaskSettings.PredictionTypes.BINARY)
    sm_id = sm.id
    print("SavedModel not found, created new one with id {}".format(sm_id))
    
version_id = "v01" # Change this to iterate to a new version

# Create version in SavedModel
for v in sm.list_versions():
    if v["id"] == version_id:
        raise Exception("SavedModel version already exists! Choose a new version name.")

sm_version = sm.import_mlflow_version_from_path(version_id=version_id,
                                                path=model_dir,
                                                code_env_name="mlflow_catboost")