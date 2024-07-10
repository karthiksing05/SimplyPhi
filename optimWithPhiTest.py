"""
THIS IS THE MAIN TEST WE'VE BEEN TALKING ABOUT!!!
OPTIMIZING WITH PHI THROUGH BACK-PROPAGATION (this is going to probably require a 
complete design from scratch of our neurons + neural network so begin this last 
since it can probably overlap with GHP stuff!)

Possible ways to go about optimization:
- IIT first and then accuracy?
- accuracy first and then IIT?
- random values, IIT, accuracy?

reason for the last two is that IIT may not be able to produce values of interest
within the first lil bit of data but also it could!

possible ideas for a loss function:
- loss = C - phi
- negative of consciousness (if the goal is to minimize a loss function, then a
negative value makes sense!)
- remember: we need a derivative too (maybe make the derivative phi and ignore the
actual loss since it doesn't matter in the scope of the training?) (*)

start this with 2 bins for each of X and y!
"""

import numpy as np
import matplotlib.pyplot as plt

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
pyphi.config.USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE = False

# Creating the dataset
samples = 1000
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import MinMaxScaler

X, y = make_blobs(
    samples,
    n_features=2,
    centers=2
)

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

from helper import things

network = [
    things.Dense(TOTAL_NODES, TOTAL_NODES),
    things.Sigmoid()
]

def predict(network, inp):
    output = inp
    for layer in network:
        output = layer.forward(output)
    return output

num_epochs = 10
learning_rate = 0.008
verbose = True

preprocessed_X = np.array([preprocess_input(sample) for sample in X])
preprocessed_y = np.array([preprocess_output(sample) for sample in y])

adam_optimizer = things.AdamOptimizer(network, lr=learning_rate)

for e in range(num_epochs):
    total_loss = 0
    avg_acc = 0
    for x_sample, y_sample in zip(preprocessed_X, preprocessed_y):
        x_sample = x_sample.reshape(-1, 1)
        y_sample = y_sample.reshape(-1, 1)

        # Forward pass
        y_pred = x_sample
        for layer in network:
            y_pred = layer.forward(y_pred)

        # Calculate binary cross-entropy loss
        loss = np.mean(things.binary_cross_entropy(y_sample, y_pred))
        total_loss += loss
        avg_acc += np.mean((y_pred > 0.5) == y_sample)

        # Backward pass
        grad = things.binary_cross_entropy_prime(y_sample, y_pred) # Gradient of binary cross-entropy loss
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate=adam_optimizer.lr)

        # Update gradients in Adam optimizer
        for layer in network:
            if isinstance(layer, things.Dense):
                layer.grad_weights = things.clip_gradients(layer.grad_weights)
                layer.grad_bias = things.clip_gradients(layer.grad_bias)
                grad = np.dot(layer.weights.T, grad)

        adam_optimizer.step()

    avg_loss = total_loss / len(preprocessed_X)
    avg_acc /= len(preprocessed_X)
    print(f"Epoch {e + 1}/{num_epochs}, Avg Loss: {avg_loss}, Avg Accuracy: {avg_acc}")

