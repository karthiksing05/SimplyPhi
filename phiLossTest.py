import numpy as np
import pyphi.new_big_phi
import tensorflow as tf

import os

os.environ["PYPHI_WELCOME_OFF"] = "yes"

import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

import pyphi
pyphi.config.PROGRESS_BARS = True # may need to comment this one out for bigger networks, but this is fine for now
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
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

categories = ['A', 'B', 'C']
n_samples = 100
categorical_data = np.random.choice(categories, n_samples)

category_map = {'A': 0, 'B': 1, 'C': 2}
categorical_data_numeric = np.vectorize(category_map.get)(categorical_data)

numerical_data1 = np.random.randn(n_samples)
numerical_data2 = np.random.randn(n_samples)

data = pd.DataFrame({
    'Category': categorical_data_numeric,
    'Numerical1': numerical_data1,
    'Numerical2': numerical_data2
})

X = data[['Category', 'Numerical1', 'Numerical2']].values

coefficients = np.array([1.5, -2.0, 3.0])
y = X.dot(coefficients) + np.random.randn(n_samples) * 0.5

### MY STUFF STARTS HERE

try:
    y[0][0]
except IndexError:
    y = y.reshape((-1, 1))

# print(X[0:5])
# print(y[0:5])

inputVars = [('cat', 3), ('num', 2), ('num', 2)]
outputVars = [('num', 7)]

inputPreprocessors = []
outputPreprocessors = []

# for each X-variable, if it's numerical, scale it, and if it's categorical, onehotencode it
newX = np.array([])
for i, var in enumerate(inputVars):
    if var[0] == 'num':
        scaler = MinMaxScaler()
        transformed = scaler.fit_transform(X[:, i].reshape(-1, 1))
        inputPreprocessors.append(scaler)
    elif var[0] == 'cat':
        encoder = OneHotEncoder(sparse_output=False)
        transformed = encoder.fit_transform(X[:, i].reshape(-1, 1))
        inputPreprocessors.append(encoder)
    else:
        raise Exception("Wrong datatype specification for one of the variables in 'inputVars'.")
    if not newX.any():
        newX = transformed
    else:
        newX = np.hstack((newX, transformed))

X = newX

# same for each y-variable
newY = np.array([])
for i, var in enumerate(outputVars):
    if var[0] == 'num':
        scaler = MinMaxScaler()
        transformed = scaler.fit_transform(y[:, i].reshape(-1, 1))
        inputPreprocessors.append(scaler)
    elif var[0] == 'cat':
        encoder = OneHotEncoder(sparse_output=False)
        transformed = encoder.fit_transform(y[:, i].reshape(-1, 1))
        inputPreprocessors.append(encoder)
    else:
        raise Exception("Wrong datatype specification for one of the variables in 'inputVars'.")
    if not newY.any():
        newY = transformed
    else:
        newY = np.hstack((newY, transformed))

y = newY

# MAGIC BIT WOOO
converter = Converter(inputVars, outputVars)

# Global Phi-calc related constants
all_states = list(itertools.product([0, 1], repeat=converter.totalNodes))
cm = np.ones((converter.totalNodes, converter.totalNodes))

# the only reason we are doing this "preprocessing" is to simulate the granularity that will be
# present in actual data encoding for the prediction pipeline!
preprocessed_X = np.array([converter.nodes_to_input(converter.input_to_nodes(sample)) for sample in X])
preprocessed_y = np.array([converter.nodes_to_output(converter.output_to_nodes(sample)) for sample in y])

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(preprocessed_X, preprocessed_y, test_size=0.20)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation='relu', name='hidden1'),
    tf.keras.layers.Dense(converter.totalNodes, activation='relu', name='TPMOutput'),
    tf.keras.layers.Dense(converter.numOutputSpaces, activation='relu', name='userOutput')
])

# for some reason, need to specify the input like this?
model(tf.keras.Input(shape=(converter.numInputSpaces,), name="input"))

# Quick note on loss function and metric: this is weird because of our data being multiple outputs BUT it's all the same LOL this is making me laugh.
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0075),
    metrics=["mean_squared_error"]
)

NUM_EPOCHS = 3

# for j in range(NUM_EPOCHS):
#     print(f"EPOCH {j}:")
#     history = model.fit(X_train, y_train, epochs=1, verbose=1)

# print(model.evaluate(X_test, y_test))

# exit()

def phi_loss_func(model):
    """
    THIS IS THE EVALUATION BUT AS A LOSS FUNCTION!!!
    """

    tpm = []

    print(f"Completing {len(all_states)} iters to calculate TPM:")
    interval = 0.1
    percent_to_complete = interval

    for i, state in enumerate(all_states):
        npState = np.array([converter.nodes_to_input(state)]).reshape(1, -1)
        # activations = converter.get_TPM_activations(model, npState)
        activations = converter.output_to_nodes(model.predict(npState, verbose=0), regularization=0.01)
        tpm.append(activations)
        if i / len(all_states) >= percent_to_complete:
            print(f"Completed {i} iters (~{round(percent_to_complete, 2) * 100}%) so far!")
            percent_to_complete += interval

    print(f"Completed {i + 1} iters (~{round(percent_to_complete, 2) * 100}%) so far!")

    tpm = np.array(tpm)

    labels = tuple([f"Node_{i}" for i in range(converter.totalNodes)])

    substrate = pyphi.Network(tpm, cm=cm, node_labels=labels)

    big_phi_avg = 0

    for state in all_states:
        fc_sia = pyphi.new_big_phi.maximal_complex(substrate, state)
        if type(fc_sia) != pyphi.new_big_phi.NullPhiStructure:
            fc_structure = pyphi.new_big_phi.phi_structure(pyphi.Subsystem(substrate, state, nodes=fc_sia.node_indices))
            big_phi_avg += fc_structure.big_phi

    big_phi_avg /= len(all_states)
    big_phi_avg *= -1 # NEGATING THE LOSS TO MAKE THE MODEL BETTER
    return big_phi_avg

def evaluate_tpm4(tpm):
    """
    A library-based evaluation using IIT 4.0's built-in functions
    """

    labels = tuple([f"Node_{i}" for i in range(converter.totalNodes)])

    substrate = pyphi.Network(tpm, cm=cm, node_labels=labels)

    phi_avg = 0
    big_phi_avg = 0
    sias = []
    structs = []

    for state in all_states:
        fc_sia = pyphi.new_big_phi.maximal_complex(substrate, state)
        if type(fc_sia) == pyphi.new_big_phi.NullPhiStructure:
            fc = None
            fc_structure = None
        else:
            fc = pyphi.Subsystem(substrate, state, nodes=fc_sia.node_indices)
            fc_structure = pyphi.new_big_phi.phi_structure(fc)
            phi_avg += fc_structure.phi
            big_phi_avg += fc_structure.big_phi

        sias.append(fc_sia)
        structs.append(fc_structure)

    phi_avg /= len(all_states)
    big_phi_avg /= len(all_states)
    return phi_avg, big_phi_avg, sias, structs

iterations = []
phi_avgs = []
big_phi_avgs = []
all_sias = []
all_structs = []

for j in range(NUM_EPOCHS):
    print(f"EPOCH {j}:")
    history = model.fit(preprocessed_X, preprocessed_y, epochs=1, verbose=1)

    iterations.append(history.history)

    tpm = []

    print(f"Completing {len(all_states)} iters to calculate TPM:")
    interval = 0.1
    percent_to_complete = interval

    for i, state in enumerate(all_states):
        npState = np.array([converter.nodes_to_input(state)]).reshape(1, -1)
        # activations = converter.get_TPM_activations(model, npState)
        activations = converter.output_to_nodes(model.predict(npState, verbose=0), regularization=0.01)
        tpm.append(activations)
        if i / len(all_states) >= percent_to_complete:
            print(f"Completed {i} iters (~{round(percent_to_complete, 2) * 100}%) so far!")
            percent_to_complete += interval

    print(f"Completed {i + 1} iters (~{round(percent_to_complete, 2) * 100}%) so far!")

    tpm = np.array(tpm)

    phi_avg, big_phi_avg, phi_sias, phi_structures = evaluate_tpm4(tpm)
    phi_avgs.append(phi_avg)
    big_phi_avgs.append(big_phi_avg)
    all_sias.append(phi_sias)
    all_structs.append(phi_structures)
    print(f"Phi Avg for EPOCH {j}: {phi_avg}")
    print(f"Big Phi Avg for EPOCH {j}: {big_phi_avg}")

with open("regressionTest.pickle", "wb") as f:
    pickle.dump([[iterations, phi_avgs, big_phi_avgs, all_sias, all_structs], model, converter, X, y], f)