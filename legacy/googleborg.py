# -*- coding: utf-8 -*-

import kagglehub

# Download latest version
path = kagglehub.dataset_download("derrickmwiti/google-2019-cluster-sample")

print("Path to dataset files:", path)

import pandas as pd

df  = pd.read_csv("/root/.cache/kagglehub/datasets/derrickmwiti/google-2019-cluster-sample/versions/1/borg_traces_data.csv")
df.head()

df.columns

df = df[['start_time', 'end_time', 'average_usage', 'maximum_usage','cpu_usage_distribution','cluster']]

df= df.sort_values(by = ["cluster","start_time"])
df.drop_duplicates(subset="start_time",keep="first", inplace=True)
df.shape
df