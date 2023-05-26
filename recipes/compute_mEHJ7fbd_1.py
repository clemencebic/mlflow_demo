# -*- coding: utf-8 -*-
import dataiku
import dataikuapi
import os

from dataikuapi.dss.ml import DSSPredictionMLTaskSettings

import pandas as pd
import mlflow.catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from datetime import datetime

# add new comments for demo

# Read recipe inputs
training_data = dataiku.Dataset("training_data")
df = training_data.get_dataframe()

cat_cols= ["job", "marital", "education", "default", "housing","loan", "month"]
cont_cols= ["age", "balance", "day", "duration", "campaign"]
target= ["y"]

# Train a catboost model on training data
cat_col_idx = [df.columns.get_loc(c) for c in cat_cols]
X = df.drop(target, axis=1)
y = LabelBinarizer().fit_transform(df[target[0]])

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7, random_state=1337)
model = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=15, eval_metric="AUC")
model.fit(X_train, y_train,
          cat_features=cat_col_idx,
          eval_set=(X_val, y_val))

# Save the model to a managed folder
catboost_models_folder = dataiku.Folder("catboost_models")
catboost_models_folder_dir = catboost_models_folder.get_path()
version=1
model_dir = "{}/catboost-uci-bank-V{}".format(catboost_models_folder_dir,version)

mlflow.catboost.save_model(model, model_dir)
print("Model saved at {} !".format(os.path.abspath(model_dir)))
