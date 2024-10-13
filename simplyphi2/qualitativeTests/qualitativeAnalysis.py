import pyphi.data_structures
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
pyphi.config.PROGRESS_BARS = False # may need to comment this one out for bigger networks, but this is fine for now
pyphi.config.VALIDATE_SUBSYSTEM_STATES = False

import itertools
import pickle

from scipy.stats import skew, kurtosis, entropy

import cv2

"""
CODE SNIPPET FOR SHOWING PHI-STRUCTURES!
first_complex = pyphi.Subsystem(
        pyphi.Network(tpms[epochNum], cm=cm, node_labels=labels),
        all_states[stateCandidate],
        nodes=sias[epochNum][stateCandidate].node_indices
    )

phiStructureCandidate = phiStructures[epochCandidate][stateCandidate]

graphic = visualize.phi_structure.plot_phi_structure(
    phi_structure=phiStructureCandidate, 
    subsystem=subsystemCandidate
)
graphic.show()
"""

def capped_relu(x):
    return tf.keras.activations.relu(x, max_value=1)

tf.keras.utils.get_custom_objects().update({'capped_relu': capped_relu})

datalst = []

with open("phiTrendsTest4PhiStructs.pickle", "rb") as f:
    datalst = pickle.load(f)

sias = datalst[0][4]
phiStructures = datalst[0][5]
# for SIAs and phiStructures, there's an array NUM_EPOCHS x NUM_STATES
tpms = datalst[0][6] # there's NUM_EPOCHS TPMs, each of them have NUM_STATES rows

NUM_EPOCHS = len(sias)
NUM_STATES = len(sias[0])

TOTAL_NODES = 6
all_states = list(itertools.product([0, 1], repeat=6))
cm = np.ones((TOTAL_NODES, TOTAL_NODES))
labels = tuple([f"Node{i}" for i in range(TOTAL_NODES)])

def siaDeepDive(epoch, verbose=False):

    sias = []

    tpm = tpms[epoch]

    substrate = pyphi.Network(tpm, cm=cm, node_labels=labels)
    subsets = [itertools.combinations(range(TOTAL_NODES), r) for r in range(1, TOTAL_NODES + 1)]
    subsets = [list(subset) for r in subsets for subset in r]

    # Calculate and print the phi value for each subsystem
    for i in range(NUM_STATES):
        sias.append([])
        for subset in subsets:
            subsystem = pyphi.Subsystem(substrate, all_states[i], subset)
            sia = pyphi.new_big_phi.sia(subsystem)
            sias[i].append(sia)
            if verbose:
                print(f"Subsystem: {subsystem}, Φ: {sia.phi}")

    #### HEATMAP CALCULATIONS!!!
    # Can apply a logarithmic scale!

    phiSums = [[0 for _ in range(TOTAL_NODES)] for _ in range(TOTAL_NODES)]
    causePhiSums = [[0 for _ in range(TOTAL_NODES)] for _ in range(TOTAL_NODES)]
    effectPhiSums = [[0 for _ in range(TOTAL_NODES)] for _ in range(TOTAL_NODES)]

    flattenedSias = list(itertools.chain.from_iterable(sias))

    for i in range(len(flattenedSias)):
        causeRIA = flattenedSias[i].cause
        effectRIA = flattenedSias[i].effect

        if not causeRIA or not effectRIA:
            continue

        for m in list(causeRIA.mechanism):
            for p in list(causeRIA.purview):
                causePhiSums[m][p] += causeRIA.phi * causeRIA.selectivity

        for m in list(effectRIA.mechanism):
            for p in list(effectRIA.purview):
                effectPhiSums[m][p] += effectRIA.phi * effectRIA.selectivity

        for m in list(effectRIA.mechanism):
            for p in list(effectRIA.purview):
                phiSums[m][p] += (causeRIA.phi * causeRIA.selectivity + effectRIA.phi * effectRIA.selectivity)

    # from pprint import pprint
    # pprint(phiSums)
    # print("CAUSE")
    # [print(x) for x in causePhiSums]
    # print("EFFECT")
    # [print(x) for x in effectPhiSums]
    return np.array(phiSums)

def heatmapVideo(data_list, output_file='epochInfoDist.mp4'):

    # Video file properties
    frame_rate = 4  # 0.25 seconds per frame (4 frames per second)
    frame_size = (640, 480)  # Frame size for the video

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)

    for data in data_list:
        # Create heatmap plot using matplotlib
        fig, ax = plt.subplots()
        heatmap = ax.imshow(data, cmap='YlOrRd', interpolation='nearest')
        plt.colorbar(heatmap)

        # Add the data values to the heatmap, rounded to 3 decimal places
        for i in range(TOTAL_NODES):
            for j in range(TOTAL_NODES):
                value = round(data[i][j], 3)
                ax.text(j, i, str(value), ha='center', va='center', color='black', fontsize=8)

        # Remove axes for cleaner video frame
        ax.set_xticks([])
        ax.set_yticks([])

        # Save the plot as an image in memory
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Resize image to match the video frame size
        img_resized = cv2.resize(img, frame_size)

        # Convert the image from RGB (matplotlib) to BGR (OpenCV)
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

        # Write the frame to the video
        out.write(img_bgr)

        # Close the plot to free memory
        plt.close(fig)

    # Release the video writer
    out.release()

    print(f'Video saved as {output_file}')

def calculate_variance(array):
    """Calculates the variance of the 2D array."""
    return np.var(array)

def calculate_skewness(array):
    """Calculates the skewness of the 2D array."""
    return skew(array.flatten())

def calculate_kurtosis(array):
    """Calculates the kurtosis of the 2D array."""
    return kurtosis(array.flatten())

def calculate_entropy(array):
    """Calculates the entropy of the 2D array."""
    # Normalize the array values to get a probability distribution
    flattened_array = array.flatten()
    value_counts = np.bincount(flattened_array.astype(int))  # Count occurrences of each value
    probabilities = value_counts / len(flattened_array)  # Convert to probabilities
    return entropy(probabilities)

def calculate_gini_coef(array):
    """Calculates the Gini coefficient for the 2D array."""
    # Flatten the array and sort it
    flattened_array = array.flatten()
    sorted_array = np.sort(flattened_array)
    n = len(sorted_array)
    cumulative_sum = np.cumsum(sorted_array)
    return (2 * np.sum(cumulative_sum) / (n * np.sum(sorted_array))) - (n + 1) / n

def calculate_l2_norm(array):
    """Calculates the L2 norm of the 2D array."""
    return np.linalg.norm(array)

def scaleHeatmap(heatmap):
    return np.log2(heatmap + 1)

def extrapolateHeatmap(heatmap):
    return calculate_variance(heatmap)

if __name__ == "__main__":

    # heatmaps = [siaDeepDive(i) for i in range(NUM_EPOCHS)]

    # with open("actReg30_siaEpochs2.pickle", "wb") as f:
    #     pickle.dump(heatmaps, f)

    with open("actReg30_siaEpochs2.pickle", "rb") as f:
        heatmaps = pickle.load(f)

    with open("phiTrendsTest4PhiStructs.pickle", "rb") as f:
        regData = pickle.load(f)

    heatmapVideo(heatmaps)

    phis = regData[0][3]
    extpols = [extrapolateHeatmap(heatmap) for heatmap in heatmaps]

    # Plotting the numbers
    plt.plot(phis, extpols, marker='o')

    # Adding titles and labels
    plt.title('Extrapolation of Heatmap Values')
    plt.xlabel('Phi')
    plt.ylabel('Variance')

    # Display the plot
    plt.show()
