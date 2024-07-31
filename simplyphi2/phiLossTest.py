"""
Expanding universal test to apply phi-values to neural network optimization routine!

Takes native account of phi for optimization!
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

coefficients = np.array([1.5, -2.0, 3.0])
y = X.dot(coefficients) + np.random.randn(n_samples) * 0.5

### MY STUFF STARTS HERE

try:
    y[0][0]
except IndexError:
    y = y.reshape((-1, 1))

# print(X[0:5])
# print(y[0:5])

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

# TODO TURN THIS INTO A GRADIENTTTT
def phi_loss_func(model):
    """
    THIS IS THE EVALUATION BUT AS A LOSS FUNCTION!!! Need to write it as a function
    within a function because of the way that Keras auto-accepts loss functions

    Solution: still need to add a small denomination of loss to associate with
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

        big_phi_avg = 0

        for state in all_states:
            try:
                fc_sia = pyphi.new_big_phi.maximal_complex(substrate, state)
            except Exception as e:
                print("ERROR THROWN: " + e)
            if type(fc_sia) != pyphi.new_big_phi.NullPhiStructure:
                fc_structure = pyphi.new_big_phi.phi_structure(pyphi.Subsystem(substrate, state, nodes=fc_sia.node_indices))
                big_phi_avg += fc_structure.big_phi

        big_phi_avg /= len(all_states)

        small_dependency = tf.reduce_mean(tf.square(y_pred - tf.zeros_like(y_pred))) * 1e-6
        big_phi_avg += small_dependency

        big_phi_avg *= -1 # NEGATING THE LOSS TO MAKE IT WORK WITH THE MODEL
        print("BIG PHI AVG: ", big_phi_avg)
        return tf.constant(big_phi_avg, dtype=tf.float32)

    return loss

# loss!
# model.compile(
#     loss=phi_loss_func(model),
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0075),
#     metrics=["mean_squared_error"],
#     run_eagerly=True
# )

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

# Actual loss function for validation
def actual_loss(y_true, y_pred):
    # print(y_true, y_pred)
    y_true = tf.cast(y_true, tf.float32)
    loss = tf.reduce_mean(tf.square(y_true - y_pred))
    # print(loss)
    return loss

# Training loop using the pseudo-constant loss for training and actual loss for validation
def train_model(model, train_dataset, val_dataset, epochs, learning_rate):

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
        
        # Training step
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = phi_loss_func(model)(y_batch_train, logits)
            
            grads = tape.gradient(loss_value, model.trainable_weights)
            
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
    model = create_model()

    # Train the model
    train_losses, val_losses = train_model(model, train_dataset, val_dataset, epochs=NUM_EPOCHS, learning_rate=0.01)
    print(train_losses, val_losses)

    with open("phiLossTest.pickle", "wb") as f:
        pickle.dump([train_losses, val_losses], f)

"""
Takeaways from initial test:
- it worked...just not as well as we wanted it to :/
- how can we customize back-propagation to better the system? mess around with learning rate and scaling of the loss?
"""