import numpy as np
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
pyphi.config.PROGRESS_BARS = False # may need to comment this one out for bigger networks, but this is fine for now
pyphi.config.VALIDATE_SUBSYSTEM_STATES = False

import itertools
import pickle

# Creating the dataset
from sklearn.datasets import make_regression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# my pivot based import!
from helper.converter import Converter

# visualizing the model
from helper.visualize import visualize_graph

NUM_INPUTS = 3
NUM_OUTPUTS = 1

X, y = make_regression(
    n_samples=125,
    n_features=NUM_INPUTS,
    noise=5
)

try:
    y[0][0]
except IndexError:
    y = y.reshape((-1, 1))

inputVars = [('num', 2), ('num', 2), ('num', 2)]
outputVars = [('num', 6)]

inputPreprocessors = []
outputPreprocessors = []

# for each X-variable, if it's numerical, scale it, and if it's categorical, onehotencode it
for i, var in enumerate(inputVars):
    if var[0] == 'num':
        scaler = MinMaxScaler()
        X[:, i] = (scaler.fit_transform(X[:, i].reshape(-1, 1))).reshape(-1)
        inputPreprocessors.append(scaler)
    elif var[0] == 'cat':
        # TODO figure this out in the future, how to inplace-replace one categorical column
        # with however many one-hot-encoding columns are needed
        pass
    else:
        raise Exception("Wrong datatype specification for one of the variables in 'inputVars'.")

# same for each y-variable
for i, var in enumerate(outputVars):
    if var[0] == 'num':
        scaler = MinMaxScaler()
        y[:, i] = (scaler.fit_transform(y[:, i].reshape(-1, 1))).reshape(-1)
        outputPreprocessors.append(scaler)
    elif var[0] == 'cat':
        # TODO figure this out in the future, how to inplace-replace one categorical column
        # with however many one-hot-encoding columns are needed
        pass
    else:
        raise Exception("Wrong datatype specification for one of the variables in 'outputVars'.")

# MAGIC BIT WOOO
converter = Converter(inputVars, outputVars)

# the only reason we are doing this "preprocessing" is to simulate the granularity that will be
# present in actual data encoding for the prediction pipeline!
preprocessed_X = np.array([converter.nodes_to_input(converter.input_to_nodes(sample)) for sample in X])
preprocessed_y = np.array([converter.nodes_to_output(converter.output_to_nodes(sample)) for sample in y])

def capped_relu(x):
    return tf.keras.activations.relu(x, max_value=1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(converter.totalNodes, activation=capped_relu, name='TPMOutput'),
    tf.keras.layers.Dense(NUM_OUTPUTS, activation='relu', name='userOutput')
])

# for some reason, need to specify the input like this?
model(tf.keras.Input(shape=(NUM_INPUTS,), name="input"))

# Quick note on loss function and metric: this is weird because of our data being multiple outputs BUT it's all the same LOL this is making me laugh.
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0075),
    metrics=["mean_squared_error"]
)

# visualizing what the model looks like!
# visualize_graph(model)

NUM_EPOCHS = 10

# for j in range(NUM_EPOCHS):
#     print(f"EPOCH {j}:")
#     history = model.fit(preprocessed_X, preprocessed_y, epochs=1, verbose=1)

# exit()

all_states = list(itertools.product([0, 1], repeat=converter.totalNodes))

cm = np.ones((converter.totalNodes, converter.totalNodes))

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
        try:
            fc = pyphi.Subsystem(substrate, state, nodes=fc_sia.node_indices)
            fc_structure = pyphi.new_big_phi.phi_structure(fc)

            phi_avg += fc_structure.phi
            big_phi_avg += fc_structure.big_phi
            sias.append(fc_sia)
            structs.append(fc_structure)
        except AttributeError:
            sias.append(None)
            structs.append(None)

    phi_avg /= len(all_states)
    big_phi_avg /= len(all_states)
    return phi_avg, big_phi_avg, sias, structs

trainLosses = []
valLosses = []
phi_avgs = []
big_phi_avgs = []
all_sias = []
all_structs = []
tpms = []

X_train, X_test, y_train, y_test = train_test_split(preprocessed_X, preprocessed_y, test_size=0.2, shuffle=True)

for j in range(NUM_EPOCHS):
    print(f"EPOCH {j}:")

    history = model.fit(X_train, y_train, epochs=1, verbose=1)
    trainLosses.append(history.history)

    validation = model.evaluate(X_test, y_test)
    valLosses.append(validation[0])

    tpm = []

    print(f"Completing {len(all_states)} iters to calculate TPM:")
    interval = 0.1
    percent_to_complete = interval

    for i, state in enumerate(all_states):
        npState = np.array([converter.nodes_to_input(state)]).reshape(1, -1)
        # activations = converter.get_TPM_activations(model, npState)
        activations = converter.output_to_nodes(model.predict(npState, verbose=0), regularization=0.05)
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
    tpms.append(tpm)
    print(f"Phi Avg for EPOCH {j}: {phi_avg}")
    print(f"Big Phi Avg for EPOCH {j}: {big_phi_avg}")

with open("rawRegressionTest.pickle", "wb") as f:
    pickle.dump([[trainLosses, valLosses, phi_avgs, big_phi_avgs, all_sias, all_structs, tpms], model, X, y], f)