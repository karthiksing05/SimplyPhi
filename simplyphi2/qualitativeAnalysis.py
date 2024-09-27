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
pyphi.config.PROGRESS_BARS = True # may need to comment this one out for bigger networks, but this is fine for now
pyphi.config.VALIDATE_SUBSYSTEM_STATES = False

import itertools
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

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

with open("activationRegressionTest30.pickle", "rb") as f:

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

def compileSIAs(epoch):

    """
    phiSums is a matrix that stores the connections between each input and output node

    phiSums[Node1][Node5] = 4.5 signifies that the connection betweeen Node1 and Node5 
    has a phi-strength of 4.5

    calculating separately for causes and effects! BUT IMPORTANT NOTE: cause and effect RIAs
    are the same transition explanations over different time intervals so we can just sum
    weighted with selectivity and assume stuff
    """
    phiSums = [[0 for _ in range(TOTAL_NODES)] for _ in range(TOTAL_NODES)]
    causePhiSums = [[0 for _ in range(TOTAL_NODES)] for _ in range(TOTAL_NODES)]
    effectPhiSums = [[0 for _ in range(TOTAL_NODES)] for _ in range(TOTAL_NODES)]

    epochSias = sias[epoch]

    """
    CAUSE STUFF:
    - grab the mechanism and purview and use that to adjust edges!
    - let's average both phis out across the cause and effect, weighted sum via selectivity!
    - (phiCause * selectivityCause + phiEffect * selectivityEffect) / (selectivityCause + selectivityEffect)
    """

    for i in range(len(all_states)):
        causeRIA = epochSias[i].cause
        effectRIA = epochSias[i].effect

        for m in list(causeRIA.mechanism):
            for p in list(causeRIA.purview):
                causePhiSums[m][p] += causeRIA.phi * causeRIA.selectivity

        for m in list(effectRIA.mechanism):
            for p in list(effectRIA.purview):
                effectPhiSums[m][p] += effectRIA.phi * effectRIA.selectivity

        for m in list(effectRIA.mechanism):
            for p in list(effectRIA.purview):
                phiSums[m][p] += (causeRIA.phi * causeRIA.selectivity + effectRIA.phi * effectRIA.selectivity)
    
    return phiSums

def heatmapVideo(numEpochs, output_file='epochInfoDist.mp4'):

    data_list = [compileSIAs(epoch) for epoch in range(numEpochs)]

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

# OMG CAN WE CREATE A VALUE OF THE RELATIVE SPREAD OF THE DATA AS OUR SIA ANALYSIS TOO OMG OMG OMG
# still need to figure out how to apply but should be chillin