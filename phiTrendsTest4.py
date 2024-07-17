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
pyphi.config.PROGRESS_BARS = True # may need to comment this one out for bigger networks, but this is fine for now
pyphi.config.VALIDATE_SUBSYSTEM_STATES = False

import itertools
import pickle

# Creating the dataset
samples = 1000
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import MinMaxScaler

X, y = make_blobs(
    samples,
    n_features=2,
    centers=2
)

# X, y = make_circles(
#     samples,
#     noise=0.03
# )

X = MinMaxScaler().fit_transform(X)

X_bins = 3
y_bins = 3
TOTAL_NODES = X_bins + y_bins

def preprocess_input(sample):
    X_coord = sample[0] - 0.000001
    y_coord = sample[1] - 0.000001

    preprocessed = np.zeros(TOTAL_NODES)
    preprocessed[int(np.floor(X_coord * X_bins))] = 1
    preprocessed[int(X_bins + np.floor(y_coord * y_bins))] = 1

    return preprocessed

def preprocess_output(sample): # defining output in terms of what we read from the neural network
    if type(sample) == np.ndarray:
        sample = sample[0]
    sample = float(sample)
    return np.array([sample] * TOTAL_NODES)

def postprocess_output(output): # interpreting readings of the neural network in terms of the sample
    # actually we have a couple different options here: can go for an average or a majority rules
    # will try average first but TODO this
    return int((sum(output) / len(output)) > 0.5)

preprocessed_X = np.array([preprocess_input(sample) for sample in X])
preprocessed_y = np.array([preprocess_output(sample) for sample in y])

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=((TOTAL_NODES),), name="t0"),
    tf.keras.layers.Dense(TOTAL_NODES, activation='sigmoid', name="t1")
])

# Quick note on loss function and metric: this is weird because of our data being multiple outputs BUT it's all the same LOL this is making me laugh.
model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0075),
    metrics=["binary_accuracy"]
)

NUM_EPOCHS = 10

all_states = list(itertools.product([0, 1], repeat=TOTAL_NODES))

all_test_states = []
for i in range(X_bins):
    for j in range(y_bins):
        XLst = list(np.zeros(X_bins))
        yLst = list(np.zeros(y_bins))
        XLst[i] = 1
        yLst[j] = 1
        state = tuple(XLst + yLst)
        state = tuple([int(x) for x in state])
        all_test_states.append(state)

cm = np.ones((TOTAL_NODES, TOTAL_NODES))

def evaluate_tpm4(tpm):
    """
    A library-based evaluation using IIT 4.0's built-in functions
    """

    labels = tuple([f"Node{i}" for i in range(TOTAL_NODES)])

    substrate = pyphi.Network(tpm, cm=cm, node_labels=labels)

    phi_avg = 0
    big_phi_avg = 0
    sias = []
    structs = []

    for test_state in all_test_states:
        fc_sia = pyphi.new_big_phi.maximal_complex(substrate, test_state)
        fc = pyphi.Subsystem(substrate, test_state, nodes=fc_sia.node_indices)
        fc_structure = pyphi.new_big_phi.phi_structure(fc)

        phi_avg += fc_structure.phi
        big_phi_avg += fc_structure.big_phi
        sias.append(fc_sia)
        structs.append(fc_structure)

    phi_avg /= len(all_test_states)
    big_phi_avg /= len(all_test_states)
    return phi_avg, big_phi_avg, sias, structs

phi_avgs = []
big_phi_avgs = []
all_sias = []
all_structs = []

for j in range(NUM_EPOCHS):
    print(f"EPOCH {j}:")
    model.fit(preprocessed_X, preprocessed_y, epochs=1, verbose=1)

    tpm = []

    print(f"Completing {len(all_states)} iters to calculate TPM:")
    interval = 0.1
    percent_to_complete = interval

    for i, state in enumerate(all_states):
        npState = np.array([state]).reshape(1, -1)
        preds = model.predict(npState, verbose=0)
        tpm.append(preds[0])
        if i / len(all_states) >= percent_to_complete:
            print(f"Completed {i} iters (~{round(percent_to_complete, 2) * 100}%) so far!")
            percent_to_complete += interval

    tpm = np.array(tpm)

    # phi_avg, MCs = evaluate_tpm(tpm)
    phi_avg, big_phi_avg, phi_sias, phi_structures = evaluate_tpm4(tpm)
    phi_avgs.append(phi_avg)
    big_phi_avgs.append(big_phi_avg)
    all_sias.append(phi_sias)
    all_structs.append(phi_structures)
    print(f"Phi Avg for EPOCH {j}: {phi_avg}")
    print(f"Big Phi Avg for EPOCH {j}: {big_phi_avg}")

with open("phiTrendsTest4.pickle", "wb") as f:
    pickle.dump([[phi_avgs, big_phi_avgs, all_sias, all_structs], model, X, y], f)

