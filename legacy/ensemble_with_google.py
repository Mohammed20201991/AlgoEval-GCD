# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings("ignore")
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve,auc, roc_curve, classification_report
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import f1_score
import shap
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import kagglehub

# Download latest version
path = kagglehub.dataset_download("derrickmwiti/google-2019-cluster-sample")

print("Path to dataset files:", path)

df_full = pd.read_csv("/root/.cache/kagglehub/datasets/derrickmwiti/google-2019-cluster-sample/versions/1/borg_traces_data.csv", index_col=0)
df_full.head()

df_full.info()

df_full["failed"].value_counts()

"""The full dataset of 405,894 rows causes the notebook to crash from too much memory use, so I have to use a subset of the data. I am randomly sampling data from both the failed and not-failed rows to maintain the fail/no-fail proportions in the resultant dataset."""

scale = 5
df_0 = df_full[df_full["failed"] == 0].sample(int(df_full["failed"].value_counts()[0]) // scale)
df_1 = df_full[df_full["failed"] == 1].sample(int(df_full["failed"].value_counts()[1]) // scale)
df = pd.concat([df_0, df_1], ignore_index=True)
df["failed"].value_counts()

df[["time", "vertical_scaling", "start_time", "end_time", "assigned_memory", "page_cache_memory",
    "cycles_per_instruction", "memory_accesses_per_instruction", "sample_rate"]].describe()

df_categorical = ["scheduler", "instance_events_type", "collection_id", "scheduling_class",
                  "collection_type", "priority", "alloc_collection_id", "instance_index", "machine_id",
                  "resource_request", "constraint", "collections_events_type", "user", "collection_name",
                  "collection_logical_name", "start_after_collection_ids", "average_usage",
                  "maximum_usage", "random_sample_usage", "cpu_usage_distribution",
                  "tail_cpu_usage_distribution", "cluster", "event"]

for i in df_categorical:
    df[i] = df[i].astype("object")

df[["scheduler", "instance_events_type", "collection_id", "scheduling_class", "collection_type",
    "priority", "alloc_collection_id", "instance_index", "machine_id", "resource_request",
   ]].describe(include="O")

df[["constraint", "collections_events_type", "user", "collection_name", "collection_logical_name",
    "start_after_collection_ids", "average_usage", "maximum_usage", "random_sample_usage",
    "cpu_usage_distribution", "tail_cpu_usage_distribution", "cluster", "event"]].describe(include="O")

df.columns

df.dropna(subset=["resource_request"], axis="rows", inplace=True)
df.reset_index(drop=True, inplace=True)

def split_cell(data):
    number = re.compile(r"[0-9]")
    if type(data) != float:
        data = data.split(": ")
        cpu = data[1]
        cpu = cpu.split(",")[0]
        cpu = float(cpu)

        memory = data[2].split("}")
        if number.match(memory[0]):
            memory = float(memory[0])
        else:
            memory = np.nan
    else:
        cpu, memory = np.nan, np.nan

    return [cpu, memory]

dff = df.drop(["resource_request", "vertical_scaling", "scheduler", "cycles_per_instruction",
               "start_after_collection_ids", "average_usage", "maximum_usage", "random_sample_usage",
               "cpu_usage_distribution", "tail_cpu_usage_distribution", "resource_request",
               "memory_accesses_per_instruction", "constraint", "event",
               "collection_type", "collections_events_type", "instance_events_type"], axis=1)

dff["rr_cpu"] = df["resource_request"].apply(lambda x: split_cell(x)[0])
dff["rr_memory"] = df["resource_request"].apply(lambda x: split_cell(x)[1])

dff["au_cpu"] = df["average_usage"].apply(lambda x: split_cell(x)[0])
dff["au_memory"] = df["average_usage"].apply(lambda x: split_cell(x)[1])

dff["rsu_cpu"] = df["random_sample_usage"].apply(lambda x: split_cell(x)[0])

dff["mu_cpu"] = df["maximum_usage"].apply(lambda x: split_cell(x)[0])
dff["mu_memory"] = df["maximum_usage"].apply(lambda x: split_cell(x)[1])

del df

dff.dropna(inplace=True)
dff.reset_index(drop=True, inplace=True)

categorical_columns = ["user", "collection_name", "collection_logical_name",
                       "scheduling_class", "cluster"]

dffX = dff.drop(["collection_id","machine_id", "alloc_collection_id","failed"], axis=1)

y = dff["failed"]

del dff

X = pd.get_dummies(dffX, columns=categorical_columns, sparse=True)

del dffX

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X.sample()

"""## Logistic Regression"""

start = time()
lad = AdaBoostClassifier(
    LogisticRegression(solver='liblinear',class_weight='balanced'),
    n_estimators=500,
    learning_rate=1.5,
    algorithm="SAMME",
)

lad.fit(X_train, y_train)
end = time()
duration = round((end - start), 2)

lad_predictions = lad.predict(X_test)

print(classification_report(y_test, lad_predictions), "\n", duration, "seconds")

# plot no skill roc curve
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# calculate roc curve for model
fpr, tpr, _ = roc_curve(y_test, lad_predictions)
# plot model roc curve
plt.plot(fpr, tpr, marker='.', label='Logistic')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_pred_proba = lad.predict_proba(X_test)[::,1]
auc = roc_auc_score(y_test, y_pred_proba)
auc

"""Making a copy of the indepentent variables so I can use a section of that for the explainable portion. I pick just the first 1000 rows to use, mostly because of how long it takes to calculate the shap values. Times to calculate values for each of the three models are shown below each calculation."""

X_copy = X.copy()
for i in X_copy.columns:
    X_copy[i] = X_copy[i].astype("float32")
X_copy.info()

explainer_lr = shap.Explainer(lad.predict, X_copy.values[:1000])

start = time()
shap_values_lr = explainer_lr(X_copy.values[:1000])
end = time()
duration = round((end - start), 2)

print(f"It took {duration} seconds to calculate the shap values for the Logistic Regression model.")

shap.summary_plot(shap_values_lr, X_copy[:1000], max_display=10, feature_names=X_copy.columns)

"""I get the prediction of some of the values so that I can use one of each fail/no-fail prediction from it to show the sample waterfall charts following. The output will change with each running of the notebook from top to bottom since I randomly sample the data to pick just some of it due to memory constraints on using the entire dataset."""

lad.predict(X_copy[:10])

shap.plots.waterfall(shap_values_lr[0], max_display=4)

shap.plots.waterfall(shap_values_lr[1], max_display=4)

"""## Random Forest"""

start = time()
rad = AdaBoostClassifier(
    RandomForestClassifier(max_depth=1, class_weight="balanced"),
    n_estimators=500,
    learning_rate=1.5,
    algorithm="SAMME",
)

rad.fit(X_train, y_train)
end = time()
duration = round((end - start), 2)

rad_predictions = rad.predict(X_test)

print(classification_report(y_test, rad_predictions), "\n", duration, "seconds")

# plot no skill roc curve
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# calculate roc curve for model
fpr, tpr, _ = roc_curve(y_test, rad_predictions)
# plot model roc curve
plt.plot(fpr, tpr, marker='.', label='Logistic')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_pred_proba = rad.predict_proba(X_test)[::,1]
auc = roc_auc_score(y_test, y_pred_proba)
auc

explainer_rf = shap.Explainer(rad.predict, X_copy.values[:1000])

start = time()
shap_values_rf = explainer_rf(X_copy.values[:1000])
end = time()
duration = round((end - start), 2)

print(f"It took {duration} seconds to calculate the shap values for the Random Forest model.")

shap.summary_plot(shap_values_rf, X_copy[:1000], max_display=10, feature_names=X_copy.columns)

"""I get the prediction of some of the values so that I can use one of each fail/no-fail prediction from it to show the sample waterfall charts following. The output will change with each running of the notebook from top to bottom since I randomly sample the data to pick just some of it due to memory constraints on using the entire dataset."""

rad.predict(X_copy[:10])

shap.plots.waterfall(shap_values_rf[0], max_display=4)

shap.plots.waterfall(shap_values_rf[3], max_display=4)

"""## Decision Tree"""

start = time()
dad = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1, class_weight="balanced"),
    n_estimators=500,
    learning_rate=1.5,
    algorithm="SAMME",
)

dad.fit(X_train, y_train)
end = time()
duration = round((end - start), 2)

dad_predictions = dad.predict(X_test)

print(classification_report(y_test, dad_predictions), "\n", duration, "seconds")

# plot no skill roc curve
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# calculate roc curve for model
fpr, tpr, _ = roc_curve(y_test, dad_predictions)
# plot model roc curve
plt.plot(fpr, tpr, marker='.', label='Logistic')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

y_pred_proba = dad.predict_proba(X_test)[::,1]
auc = roc_auc_score(y_test, y_pred_proba)
auc

explainer_dt = shap.Explainer(dad.predict, X_copy.values[:1000])

start = time()
shap_values_dt = explainer_dt(X_copy.values[:1000])
end = time()
duration = round((end - start), 2)

print(f"It took {duration} seconds to calculate the shap values for the Decision Tree model.")

shap.summary_plot(shap_values_dt, X_copy[:1000], max_display=10, feature_names=X_copy.columns)

"""I get the prediction of some of the values so that I can use one of each fail/no-fail prediction from it to show the sample waterfall charts following. The output will change with each running of the notebook from top to bottom since I randomly sample the data to pick just some of it due to memory constraints on using the entire dataset."""

dad.predict(X_copy[:10])

shap.plots.waterfall(shap_values_dt[0], max_display=4)

shap.plots.waterfall(shap_values_dt[3], max_display=4)