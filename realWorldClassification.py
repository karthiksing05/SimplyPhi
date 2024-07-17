"""
A working example of how the neural network pivot analysis can be done for a
real-world classification dataset.
"""

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

import numpy as np
import tensorflow as tf

# Creating the dataset
samples = 1000
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

iris_data = load_iris() # load the iris dataset

X = iris_data.data
y = iris_data.target.reshape(-1, 1) # Convert data to a single column

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# One Hot encode the class labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# NOTE: the output preprocessing for BitCnts relies on being a multiple of the number of output
# classes if the data is classification
# NOTE: long-term create a platform that does this for you! modifying variables over and over in a script
# isn't enough

# TODO find a good way to process and encode/decode outputs in general

bitCnts = [3, 3, 3, 3]
binCnts = [2 ** x for x in bitCnts]

NUM_INPUTS = len(X[0])
TOTAL_NODES = sum(bitCnts)
NUM_OUTPUTS = len(y[0])

def get_indices_of_ones(n):
    # Convert the integer to binary and remove the '0b' prefix
    binary_representation = bin(n)[2:]
    binary_representation = reversed(binary_representation)

    # Collect the indices of all '1's
    indices = [i for i, bit in enumerate(binary_representation) if bit == '1']

    return indices

def input_NN_to_TPM(sample):

    preprocessed = np.zeros(TOTAL_NODES)

    numBits = 0

    for i, coord in enumerate(sample):
        coord -= 1e-6

        idxs = [numBits + x for x in get_indices_of_ones(int(np.floor(coord * binCnts[i])))]

        for idx in idxs:
            preprocessed[idx] = 1

        numBits += bitCnts[i]

    return preprocessed

# prepares the bin mode for transitional input (this is lowk our input layer)
def input_TPM_to_NN(sample):

    inp = np.zeros((NUM_INPUTS,))

    numBits = 0

    for i, bitCnt in enumerate(bitCnts):

        for b in range(bitCnt):
            inp[i] += sample[numBits + b] * (2 ** b)

        numBits += bitCnt
        inp[i] /= bitCnt

    return inp

# converts the classes to the X nodes stuff
def output_NN_to_TPM(sample): # defining output in terms of what we read from the neural network
    if type(sample) == np.ndarray:
        sample = sample[0]
    sample = int(sample)
    portion = np.zeros(NUM_OUTPUTS)
    portion[sample] = 1
    output = list(portion) * int(TOTAL_NODES / NUM_OUTPUTS)
    return output

# converts the nodes to classes
def output_TPM_to_NN(output): # interpreting readings of the neural network in terms of the sample
    # actually we have a couple different options here: can go for an average or a majority rules
    # will try average first but TODO this
    cnts = [0] * NUM_OUTPUTS
    for i in range(int(TOTAL_NODES / NUM_OUTPUTS)):
        for j in range(NUM_OUTPUTS):
            cnts[j] += output[NUM_OUTPUTS * i + j]

    decoded_output = [0] * NUM_OUTPUTS
    decoded_output[np.argmax(cnts)] = 1

    return decoded_output

# the only reason we are doing this "preprocessing" is to simulate the granularity that will be
# present in actual data encoding for the prediction pipeline!
preprocessed_X = np.array([input_TPM_to_NN(input_NN_to_TPM(sample)) for sample in X])
preprocessed_y = np.array([output_TPM_to_NN(output_NN_to_TPM(sample)) for sample in y])

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(preprocessed_X, preprocessed_y, test_size=0.20)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=((NUM_INPUTS),), name="transform"),
    tf.keras.layers.Dense(10, activation='relu', name='fc1'),
    tf.keras.layers.Dense(10, activation='relu', name='fc2'),
    tf.keras.layers.Dense(NUM_OUTPUTS, activation='softmax', name='output')
])

# Quick note on loss function and metric: this is weird because of our data being multiple outputs BUT it's all the same LOL this is making me laugh.
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0075),
    metrics=["accuracy"]
)

NUM_EPOCHS = 5

all_states = list(itertools.product([0, 1], repeat=TOTAL_NODES))

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

    for state in all_states:
        fc_sia = pyphi.new_big_phi.maximal_complex(substrate, state)
        fc = pyphi.Subsystem(substrate, state, nodes=fc_sia.node_indices)
        fc_structure = pyphi.new_big_phi.phi_structure(fc)

        phi_avg += fc_structure.phi
        big_phi_avg += fc_structure.big_phi
        sias.append(fc_sia)
        structs.append(fc_structure)

    phi_avg /= len(all_states)
    big_phi_avg /= len(all_states)
    return phi_avg, big_phi_avg, sias, structs

losses = []
accuracies = []
phi_avgs = []
big_phi_avgs = []
all_sias = []
all_structs = []

for j in range(NUM_EPOCHS):
    print(f"EPOCH {j}:")
    history = model.fit(preprocessed_X, preprocessed_y, epochs=1, verbose=1)

    losses.append(history.history['loss'])
    accuracies.append(history.history['accuracy'])

    tpm = []

    print(f"Completing {len(all_states)} iters to calculate TPM:")
    interval = 0.1
    percent_to_complete = interval

    for i, state in enumerate(all_states):
        npState = np.array([input_TPM_to_NN(state)]).reshape(1, -1)
        preds = model.predict(npState, verbose=0)
        tpm.append(output_NN_to_TPM(preds[0]))
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

with open("irisPhiTest.pickle", "wb") as f:
    pickle.dump([[losses, accuracies, phi_avgs, big_phi_avgs, all_sias, all_structs], model, [input_NN_to_TPM, output_NN_to_TPM, input_TPM_to_NN, output_TPM_to_NN], X, y], f)