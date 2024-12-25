import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

os.environ["PYPHI_WELCOME_OFF"] = "yes"

import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

import pyphi
from pyphi import visualize
pyphi.config.PROGRESS_BARS = True # may need to comment this one out for bigger networks, but this is fine for now
pyphi.config.VALIDATE_SUBSYSTEM_STATES = False
pyphi.config.PARTITION_TYPE = "BI"

import itertools
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

def capped_relu(x):
    return tf.keras.activations.relu(x, max_value=1)

tf.keras.utils.get_custom_objects().update({'capped_relu': capped_relu})


# with open("iitOverfittingNoAdaptL2H3.pickle", "rb") as f:
#     datalst = pickle.load(f)

#     phi_train = datalst[0]
#     phi_val = datalst[1]
#     reg_train = datalst[2]
#     reg_val = datalst[3]

# index = range(len(phi_train))

# ## UNCOMMENT THESE FOR TRAINING VERIFICATIONS!
# phi_train_losses_scaled = (tf.convert_to_tensor(phi_train) - tf.reduce_min(phi_train)) / (tf.reduce_max(phi_train) - tf.reduce_min(phi_train))
# reg_train_losses_scaled = (tf.convert_to_tensor(reg_train) - tf.reduce_min(reg_train)) / (tf.reduce_max(reg_train) - tf.reduce_min(reg_train))

# # print(phi_train)
# # exit()
# # print(phi_train_losses_scaled, reg_train_losses_scaled)

# # print(phi_val, reg_val)

# plt.plot(index, phi_val, label='Phi Values', marker='o')
# plt.plot(index, reg_val, label='Reg Values', marker='s')
# # plt.plot(index, phi_train_losses_scaled, label='Phi Values', marker='o')
# # plt.plot(index, reg_train_losses_scaled, label='Reg Values', marker='s')

# plt.title("Phi Values and Regular Values")
# plt.xlabel("Index")
# plt.ylabel("Values")

# plt.legend()

# plt.grid(True)
# plt.show()


# OVERFITTING ANALYSIS
with open("overfittingAnalysis26.pickle", "rb") as f:
    datalst = pickle.load(f)

    phi_train = datalst[0]
    phi_val = datalst[1]
    phis = datalst[2]
    siaMetrics = datalst[3]

index = range(len(phi_train))

## UNCOMMENT THESE FOR TRAINING VERIFICATIONS!
phi_train_scaled = (tf.convert_to_tensor(phi_train) - tf.reduce_min(phi_train)) / (tf.reduce_max(phi_train) - tf.reduce_min(phi_train))
phi_val_scaled = (tf.convert_to_tensor(phi_val) - tf.reduce_min(phi_val)) / (tf.reduce_max(phi_val) - tf.reduce_min(phi_val))
phis_scaled = (tf.convert_to_tensor(phis) - tf.reduce_min(phis)) / (tf.reduce_max(phis) - tf.reduce_min(phis))

plt.plot(index, phi_train_scaled, label='Training Loss Values', marker='o')
plt.plot(index, phi_val_scaled, label='Validation Loss Values', marker='o')
plt.plot(index, phis_scaled, label='Phi Values', marker='s')
# important_keys = ["Entropy"]
important_keys = siaMetrics[0].keys()
for name in important_keys:
    metric = [siaMetrics[i][name] for i in range(len(siaMetrics))]
    print(metric)
    metric_scaled = (tf.convert_to_tensor(metric) - tf.reduce_min(metric)) / (tf.reduce_max(metric) - tf.reduce_min(metric))
    plt.plot(index, metric_scaled, label=f'SIA Values {name}', marker='s')

plt.title("Loss Values and Phis")
plt.xlabel("Index")
plt.ylabel("Values")

plt.legend()

plt.grid(True)
plt.show()