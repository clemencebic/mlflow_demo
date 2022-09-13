# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from dataikuapi.dss.ml import DSSPredictionMLTaskSettings

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
    





# Write recipe outputs
test_folder = dataiku.Folder("i3nlEc79")
test_folder_info = test_folder.get_info()
