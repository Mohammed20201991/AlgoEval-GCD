# -*- coding: utf-8 -*-

# Install Kaggle API
!pip install -q kaggle

import kagglehub

# Download latest version
path = kagglehub.dataset_download("derrickmwiti/google-2019-cluster-sample")

print("Path to dataset files:", path)

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.utils import plot_model
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

"""https://research.google/resources/datasets/?search=google+cluster+data&"""

data = pd.read_csv('/root/.cache/kagglehub/datasets/derrickmwiti/google-2019-cluster-sample/versions/1/borg_traces_data.csv')
data.head()

data.info()

data.isna().sum()

data.drop('Unnamed: 0', axis=1, inplace=True)
data = data.select_dtypes(exclude=['object'])
data.dropna(inplace=True)
data = data[data.columns.drop(list(data.filter(like='_id')))]

data.shape

data.info()

# can we give figuer different color for comparisaion reason
data.hist(figsize=(20,20))

data.head()

def check_outlier(DataFrame):
    # DataFrame.hist(figsize=(10,10))
    dic = {}
    for i in DataFrame:
        Q1 = DataFrame[i].quantile(0.25)
        Q3 = DataFrame[i].quantile(0.75)
        IQR = Q3-Q1
        up = Q3 + 1.5*IQR
        low = Q1 - 1.5*IQR

        if DataFrame[(DataFrame[i] > up) | (DataFrame[i] < low)].any(axis=None):
            dic[i] = 1
        else:
            dic[i] = 0
    return pd.Series(dic)

check_outlier(data)

def outlier_solver(DataFrame):
    for i in DataFrame:
        Q1 = DataFrame[i].quantile(0.25)
        Q3 = DataFrame[i].quantile(0.75)
        IQR = Q3 - Q1
        up_lim  = Q3 + 1.5 * IQR
        low_lim = Q1 - 1.5 * IQR
        DataFrame.loc[DataFrame[i] > up_lim,i]  = up_lim
        DataFrame.loc[DataFrame[i] < low_lim,i] = low_lim
    return check_outlier(DataFrame)

outlier_solver(data)

data.hist(figsize=(20,20))

def Correlation_figure(DataFrame):
    plt.figure(figsize=(12,12))
    sns.heatmap(DataFrame.corr(),annot=True,linewidths=0.6)

Correlation_figure(data)

data.drop(['collection_type','sample_rate'], axis=1, inplace=True)
Correlation_figure(data)

sns.countplot(data, x="cluster", hue="failed")

sns.barplot(data, x="cluster", y="time", hue="failed")

so.Plot(data, "start_time", "end_time").add(so.Line())

X = data.drop(['cluster','failed'], axis=1, inplace=False)
y = keras.utils.to_categorical(data.loc[:,'cluster'])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, random_state=42, shuffle=True)

input_shape  = [X.shape[1]]
output_shape = y.shape[1]

model = keras.Sequential([
    # input layer
    layers.BatchNormalization(input_shape=input_shape),

    # hidden layer
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.15),

    # hidden layer
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),

    # hidden layer
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    # hidden layer
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),

    # hidden layer
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.15),

    # hidden layer
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.1),

    # output layer
    layers.Dense(output_shape, activation='softmax')
])

plot_model(model, show_shapes = True)

def Model_Evaluation(model, X_train, y_train, X_valid, y_valid):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = keras.callbacks.EarlyStopping(patience=5, min_delta=0.001, restore_best_weights=True,)

    history = model.fit(
        X_train, y_train,
        validation_data = (X_valid, y_valid),
        batch_size= 10000,
        epochs = 50,
        callbacks=[early_stopping],
    )

    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss']].plot(title="Loss Graph")
    history_df.loc[:, ['accuracy', 'val_accuracy']].plot(title="Accuracy Graph")

Model_Evaluation(model, X_train, y_train, X_valid, y_valid)

def ConfM(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()

y_log = data.loc[:,'failed']

X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.1, random_state=42, shuffle=True)

def ClassificationModel(model, X_train, X_test, y_train, y_test):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    ConfM(y_test, y_pred)

LogisticRegressionModel = LogisticRegression(penalty='l2', solver='liblinear', C=1, random_state=42, max_iter=10000)
ClassificationModel(LogisticRegressionModel, X_train, X_test, y_train, y_test)

SGDClassifierModel = SGDClassifier(penalty='l2', random_state=42, max_iter=10000, loss='perceptron', learning_rate='optimal')
ClassificationModel(SGDClassifierModel, X_train, X_test, y_train, y_test)

MLPClassifierModel = MLPClassifier(hidden_layer_sizes=(100,3),
                      activation='tanh',
                      solver='adam',
                      alpha=0.0001,
                      learning_rate_init=0.001,
                      max_iter=200,
                      random_state=42,
                      early_stopping = False
                     )
ClassificationModel(MLPClassifierModel, X_train, X_test, y_train, y_test)