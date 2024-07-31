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

def capped_relu(x):
    return tf.keras.activations.relu(x, max_value=1)

tf.keras.utils.get_custom_objects().update({'capped_relu': capped_relu})

with open("phiTrendsTest4.pickle", "rb") as f:
    datalst = pickle.load(f)

    sias = datalst[0][4]
    phiStructures = datalst[0][5]
    tpms = datalst[0][6]

    fig, ax = plt.subplots()

    # print(sias[0][0].network)
    # exit()

    TOTAL_NODES = 6
    cm = np.ones((TOTAL_NODES, TOTAL_NODES))
    labels = tuple([f"Node{i}" for i in range(TOTAL_NODES)])

    subsystems0 = [pyphi.Subsystem(pyphi.Network(tpms[i], cm=cm, node_labels=labels), siaLst[0].current_state, nodes=siaLst[0].node_indices) for i, siaLst in enumerate(sias)]
    phiStructures0 = [x[0] for x in phiStructures]
    
    for i, struct in enumerate(phiStructures0):
        graphic = visualize.phi_structure.plot_phi_structure(
            phi_structure=struct, 
            subsystem=subsystems0[i]
        )
        graphic.show()

# with open("longerRegressionTest.pickle", "rb") as f:
#     datalst = pickle.load(f)

#     losses = [x['loss'][0] for x in datalst[0][0]]
#     lossDerivatives = np.gradient(losses)
#     valLosses = datalst[0][1]
#     phis = datalst[0][3]
#     phiDerivatives = np.gradient(phis)

#     data = pd.DataFrame(columns=["Loss", "Val-Loss", "Phi", "Phi-Gradient"])
#     data["Loss"] = losses
#     data["Val-Loss"] = valLosses
#     data["Phi"] = phis
#     data["Phi-Gradient"] = phiDerivatives

#     data.to_csv("longerRegressionTest.csv")

#     fig, ax = plt.subplots()

#     ax.plot(MinMaxScaler().fit_transform(np.array(losses).reshape(-1, 1)), label='Loss')
#     # ax.plot(MinMaxScaler().fit_transform(np.array(valLosses).reshape(-1, 1)), label='Val-Loss')
#     ax.plot(MinMaxScaler().fit_transform(np.array(phis).reshape(-1, 1)), label='Phi')
#     ax.plot(MinMaxScaler().fit_transform(np.array(phiDerivatives).reshape(-1, 1)).reshape(-1, 1), label='Change in Phi')

#     ax.set_title('Epochs for Regression Modeling')
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Value')

#     ax.legend()

#     plt.show()

# with open("phiPlusLossTest.pickle", "rb") as f:
#     datalst = pickle.load(f)

#     phis = [tensor.numpy() for tensor in datalst[0]] # REMEMBER THAT PHIS ARE NEGATIVEEEE
#     losses = [tensor.numpy() for tensor in datalst[2]]
#     print(phis, losses)
#     # phiDerivatives = np.gradient(phis)

#     fig, ax = plt.subplots()

#     ax.plot(MinMaxScaler().fit_transform(np.array(losses).reshape(-1, 1)), label='Loss')
#     ax.plot(MinMaxScaler().fit_transform(np.array(phis).reshape(-1, 1)), label='Phi')
#     # ax.plot(MinMaxScaler().fit_transform(np.array(phiDerivatives).reshape(-1, 1)).reshape(-1, 1), label='Change in Phi')

#     ax.set_title('Epochs for Phi Loss Test')
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Value')

#     ax.legend()

#     plt.show()
