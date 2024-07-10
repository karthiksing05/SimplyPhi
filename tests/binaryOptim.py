"""
binning thing with the input layers (optimizing binary encoding)

this is done thru a "transition layer" that turns the input neurons into usable values
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

bitCnts = [3, 3]
binCnts = [2 ** x for x in bitCnts]

TOTAL_VARS = len(bitCnts)
TOTAL_NODES = sum(bitCnts)

def get_indices_of_ones(n):
    # Convert the integer to binary and remove the '0b' prefix
    binary_representation = bin(n)[2:]
    binary_representation = reversed(binary_representation)

    # Collect the indices of all '1's
    indices = [i for i, bit in enumerate(binary_representation) if bit == '1']

    return indices

# prepares the bin mode for transitional input
def bins_to_transition(sample):

    inp = np.zeros((TOTAL_VARS,))

    numBits = 0

    for i, bitCnt in enumerate(bitCnts):

        for b in range(bitCnt):
            inp[i] += sample[numBits + b] * (2 ** b)

        numBits += bitCnt
        inp[i] /= bitCnt

    return inp

def preprocess_input(sample):

    preprocessed = np.zeros(TOTAL_NODES)

    numBits = 0

    for i, coord in enumerate(sample):
        coord -= 1e-6

        idxs = [numBits + x for x in get_indices_of_ones(int(np.floor(coord * binCnts[i])))]

        for idx in idxs:
            preprocessed[idx] = 1

        numBits += bitCnts[i]

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

preprocessed_X = np.array([bins_to_transition(preprocess_input(sample)) for sample in X])
preprocessed_y = np.array([preprocess_output(sample) for sample in y])

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=((TOTAL_VARS),), name='transition'),
    tf.keras.layers.Dense(TOTAL_NODES, activation='sigmoid', name="t1")
])

# Quick note on loss function and metric: this is weird because of our data being multiple outputs BUT it's all the same LOL this is making me laugh.
model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.008),
    metrics=["binary_accuracy"]
)

NUM_EPOCHS = 10

for i in range(NUM_EPOCHS):
    print(f"EPOCH {i}:")
    model.fit(preprocessed_X, preprocessed_y, epochs=1, verbose=1)

def plot_decision_boundary(model, X, y):
    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    # Create X values (we're going to predict on all of these)
    x_in = np.c_[xx.ravel(), yy.ravel()]
    x_in = MinMaxScaler().fit_transform(x_in)
    # Make predictions using the trained model
    unprocessed_y_pred = model.predict(np.array([bins_to_transition(preprocess_input(sample)) for sample in x_in]))
    y_pred = np.array([[postprocess_output(sample)] for sample in unprocessed_y_pred])
    # Check for multi-class
    if len(y_pred[0]) > 1:
        print("doing multiclass classification...")
        # We have to reshape our predictions to get them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classifcation...")
        y_pred = np.round(y_pred).reshape(xx.shape)
    # Plot decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

plot_decision_boundary(model, X, y)