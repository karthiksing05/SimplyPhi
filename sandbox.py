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

import itertools
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

def capped_relu(x):
    return tf.keras.activations.relu(x, max_value=1)

tf.keras.utils.get_custom_objects().update({'capped_relu': capped_relu})


# LINEARADAPTP GOOD TEST
with open("iitOverfittingLinearAdaptH0.pickle", "rb") as f:
    datalst = pickle.load(f)

    phi_train = datalst[0]
    phi_val = datalst[1]
    reg_train = datalst[2]
    reg_val = datalst[3]

index = range(len(phi_train))

## UNCOMMENT THESE FOR TRAINING VERIFICATIONS!
phi_train_losses_scaled = (tf.convert_to_tensor(phi_train) - tf.reduce_min(phi_train)) / (tf.reduce_max(phi_train) - tf.reduce_min(phi_train))
reg_train_losses_scaled = (tf.convert_to_tensor(reg_train) - tf.reduce_min(reg_train)) / (tf.reduce_max(reg_train) - tf.reduce_min(reg_train))

print(phi_train)
# exit()
print(phi_train_losses_scaled, reg_train_losses_scaled)

# print(phi_val, reg_val)

plt.plot(index, phi_val, label='Phi Values', marker='o')
plt.plot(index, reg_val, label='Reg Values', marker='s')

plt.title("Phi Values and Regular Values")
plt.xlabel("Index")
plt.ylabel("Values")

plt.legend()

plt.grid(True)
plt.show()