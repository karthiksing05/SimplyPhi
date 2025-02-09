"""
In general, the main goal for this test is to involve 
"""

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

categories = ['A', 'B']
n_samples = 100
categorical_data = np.random.choice(categories, n_samples)

category_map = {'A': 0, 'B': 1}
categorical_data_numeric = np.vectorize(category_map.get)(categorical_data)

numerical_data1 = np.random.randn(n_samples)
numerical_data2 = np.random.randn(n_samples)

data = pd.DataFrame({
    'Category': categorical_data_numeric,
    'Numerical1': numerical_data1,
    'Numerical2': numerical_data2
})

X = data[['Category', 'Numerical1', 'Numerical2']].values

coefficients = np.random.randn(3)
y = X.dot(coefficients) + np.random.randn(n_samples) * 0.5

### MY STUFF STARTS HERE

try:
    y[0][0]
except IndexError:
    y = y.reshape((-1, 1))

inputVars = [('cat', 2), ('num', 2), ('num', 2)]
outputVars = [('num', 6)]

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

# Actual loss function for validation
def actual_loss(y_true, y_pred):
    # print(y_true, y_pred)
    y_true = tf.cast(y_true, tf.float32)
    loss = tf.reduce_mean(tf.square(y_true - y_pred))
    # print(loss)
    return loss

def calculate_skewness(array):
    """Calculates the skewness of the 2D array."""
    return skew(array.flatten())

def calculate_kurtosis(array):
    """Calculates the kurtosis of the 2D array."""
    return kurtosis(array.flatten())

def phi_loss_func(model):
    """
    THIS IS THE EVALUATION BUT AS A LOSS FUNCTION!!! Need to write it as a function
    within a function because of the way that Keras auto-accepts loss functions
    """

    @tf.function
    def loss(y_true, y_pred):

        tpm = []

        print(f"Completing {len(all_states)} iters to calculate TPM:")
        interval = 0.1
        percent_to_complete = interval

        for i, state in enumerate(all_states):
            npState = np.array([converter.nodes_to_input(state)]).reshape(1, -1)
            activations = converter.get_TPM_activations(model, npState)
            # activations = converter.output_to_nodes(model.predict(npState, verbose=0), regularization=0.01)
            tpm.append(activations)
            if i / len(all_states) >= percent_to_complete:
                print(f"Completed {i} iters (~{round(percent_to_complete, 2) * 100}%) so far!")
                percent_to_complete += interval

        print(f"Completed {i + 1} iters (~{round(percent_to_complete, 2) * 100}%) so far!")

        tpm = np.array(tpm)

        labels = tuple([f"Node_{i}" for i in range(converter.totalNodes)])

        substrate = pyphi.Network(tpm, cm=cm, node_labels=labels)

        sias = []

        substrate = pyphi.Network(tpm, cm=cm, node_labels=labels)
        subsets = [itertools.combinations(range(converter.totalNodes), r) for r in range(1, converter.totalNodes + 1)]
        subsets = [list(subset) for r in subsets for subset in r]

        # Calculate and print the phi value for each subsystem
        for i in range(len(all_states)):
            sias.append([])
            for subset in subsets:
                subsystem = pyphi.Subsystem(substrate, all_states[i], subset)
                sia = pyphi.new_big_phi.sia(subsystem)
                sias[i].append(sia)

        #### HEATMAP CALCULATIONS!!!
        # Can apply a logarithmic scale!

        phiSums = [[0 for _ in range(converter.totalNodes)] for _ in range(converter.totalNodes)]

        flattenedSias = list(itertools.chain.from_iterable(sias))

        for i in range(len(flattenedSias)):
            causeRIA = flattenedSias[i].cause
            effectRIA = flattenedSias[i].effect

            if not causeRIA or not effectRIA:
                continue

            for m in list(effectRIA.mechanism):
                for p in list(effectRIA.purview):
                    phiSums[m][p] += (causeRIA.phi * causeRIA.selectivity + effectRIA.phi * effectRIA.selectivity)

        phiSums = np.array(phiSums)

        heatmapEvaluation = calculate_kurtosis(phiSums) + calculate_skewness(phiSums)

        accuracy_dependency = actual_loss(y_pred, tf.zeros_like(y_pred)) * 1e-6
        heatmapEvaluation += accuracy_dependency

        heatmapEvaluation *= -1 # NEGATING THE LOSS TO MAKE IT WORK WITH THE MODEL
        print("HEATMAP EVALUATION: ", heatmapEvaluation)
        return tf.constant(heatmapEvaluation, dtype=tf.float32)

    return loss

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(preprocessed_X, preprocessed_y, test_size=0.20)
NUM_EPOCHS = 10

def capped_relu(x):
    return tf.keras.activations.relu(x, max_value=1)

# Define a simple model for demonstration purposes
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(converter.totalNodes, activation=capped_relu, name='TPMOutput', kernel_initializer=keras.initializers.RandomNormal(stddev=0.1), bias_initializer=keras.initializers.Zeros()),
        tf.keras.layers.Dense(converter.numOutputSpaces, activation=capped_relu, name='userOutput', kernel_initializer=keras.initializers.RandomNormal(stddev=0.1), bias_initializer=keras.initializers.Zeros())
    ])
    return model

# Training loop using the pseudo-constant loss for training and actual loss for validation
def train_model(model, train_dataset, val_dataset, epochs, learning_rate, doSplit):
    """
    Note that split is the fractional amount of phi in the weighted average!
    """

    train_losses = []
    val_losses = []

    # Initialize the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Metrics to keep track of loss
    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # split = (epoch + 1) / epochs - 1e-6
        if doSplit:
            split = 0.99999
        else:
            split = 0.0
        # NOTE THIS IS HELLA IMPORTANT!! PROGRAMMED SPLIT TO DECREASE AS TIME WENT ON!!!
        
        # Training step
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = actual_loss(y_batch_train, logits)
                loss_value *= (1 - split)
                if split > 0:
                    loss_value += (split * phi_loss_func(model)(y_batch_train, logits))

            grads = tape.gradient(loss_value, model.trainable_weights)
            # print(grads, type(grads)) # TODO HERE FIGURE OUT HOW TO REPLACE THESE GRADIENTS WITH SIA ANALYSES
            
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_loss_metric.update_state(loss_value)
        
        # Validation step
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            val_loss_value = actual_loss(y_batch_val, val_logits)
            val_loss_metric.update_state(val_loss_value)
        
        # Print metrics at the end of each epoch
        train_loss = train_loss_metric.result()
        val_loss = val_loss_metric.result()
        print(f"Training loss: {train_loss:.4f} - Validation loss: {val_loss:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Reset metrics at the end of each epoch
        train_loss_metric.reset_state()
        val_loss_metric.reset_state()

    return train_losses, val_losses

if __name__ == "__main__":

    # Convert the data to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

    # Create the model
    phi_model = create_model()

    # Train the model
    phi_train_losses, phi_val_losses = train_model(phi_model, train_dataset, val_dataset, epochs=NUM_EPOCHS, learning_rate=0.01, doSplit=True)

    # Create the model (NO PHI CALCS)
    model = create_model()

    # Train the model (NO PHI CALCS)
    train_losses, val_losses = train_model(model, train_dataset, val_dataset, epochs=NUM_EPOCHS, learning_rate=0.01, doSplit=False)
    # print(train_losses, val_losses)

    with open("phiQualitativeBP.pickle", "wb") as f:
        pickle.dump([phi_train_losses, phi_val_losses, train_losses, val_losses, model], f)