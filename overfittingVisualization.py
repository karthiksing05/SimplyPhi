import numpy as np
import tensorflow as tf
import keras
tf.config.run_functions_eagerly(True)

import os

os.environ["PYPHI_WELCOME_OFF"] = "yes"

import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

import pyphi
pyphi.config.PROGRESS_BARS = False # may need to comment this one out for bigger networks, but this is fine for now
pyphi.config.VALIDATE_SUBSYSTEM_STATES = False

import itertools
import pickle

# Creating the dataset
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# my pivot based import!
from helper.converter import Converter

# visualizing the model
from helper.visualize import visualize_graph

### NOTE CHATGPT DATA WOOO
import numpy as np
from scipy.stats import skew, kurtosis, entropy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def capped_relu(x):
    return tf.keras.activations.relu(x, max_value=1)

tf.keras.utils.get_custom_objects().update({'capped_relu': capped_relu})

folder_path = "overfittingAnalysisPaper/overfittingAnalysisPickles"

columns = [
    "Filename",
    "Train",
    "Val",
    "Variance",
    "Skewness",
    "Kurtosis",
    "L2 Norm",
    "Gini Coef",
    "Entropy",
    "Phi"
]

df = pd.DataFrame(columns=columns)

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".pickle"):  # Check if the file is a pickle file
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            
            train_loss = data[0]
            val_loss = data[1]
            sia_metrics = data[3]
            phi_vals = data[2]

            for i in range(len(train_loss)):
                entry = [
                    filename.split("Analysis")[-1].split(".")[0],
                    train_loss[i].numpy(),
                    val_loss[i].numpy(),
                    sia_metrics[i]["Variance"],
                    sia_metrics[i]["Skewness"],
                    sia_metrics[i]["Kurtosis"],
                    sia_metrics[i]["L2 Norm"],
                    sia_metrics[i]["Gini Coef"],
                    sia_metrics[i]["Entropy"],
                    phi_vals[i]
                ]
                df.loc[len(df)] = entry
                print(i)

print(df.values.shape) # 15 epochs x 28 files
print(df.head())

df.to_csv("overfittingAnalysisPaper/compilation.csv")