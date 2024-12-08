
"""
No adapt test using the custom value from Heatmaps!
No means that the split relationship will not be adjusted.
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

FILENAME = "iitOverfittingLinearAdaptH0.pickle"

with open(FILENAME, "rb") as f:
    datalst = pickle.load(f)
    X = datalst[6]
    y = datalst[7]

inputVars = [('cat', 2), ('num', 2), ('num', 2)]
outputVars = [('num', 6)]

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

def phi_loss_func(epochNum):
    """
    Instead of a loss function, this is going to return a noisy number of sorts
    that starts around -10 and slowly decreases to -30, the average magnitude
    of phis after a certain point.
    """

    @tf.function
    def loss(y_true, y_pred):
        return tf.constant(np.random.rand() * -5 - 20, dtype=tf.float32)

    return loss

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(preprocessed_X, preprocessed_y, test_size=0.20)
NUM_EPOCHS = 25

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

        if doSplit:
            split = 1.0 - 1e-6 # TODO LOGARITHM TEST MULITPLE SPLIT DIFFERENCES
        else:
            split = 0.0
        # NOTE THIS IS HELLA IMPORTANT!! PROGRAMMED SPLIT TO DECREASE AS TIME WENT ON!!!
        
        # Training step
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = actual_loss(y_batch_train, logits)
                loss_value *= (1 - split)
                if split != 0.0:
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

    with open(f"iitOverfittingPlaceboAdapt[{FILENAME.split("ing")[1].split("Adapt")[0]}H].pickle", "wb") as f:
        pickle.dump([phi_train_losses, phi_val_losses, train_losses, val_losses, phi_model, model, X, y], f)
