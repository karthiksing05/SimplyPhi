import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from graphGIIT import compute_graph_based_phi

from scipy.spatial import ConvexHull
from scipy.stats import entropy

import itertools
import pickle

import os
os.environ["PYPHI_WELCOME_OFF"] = "yes"

import pyphi

import time

import collections.abc
# Fix deprecated collections
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

# PyPhi settings
pyphi.config.PROGRESS_BARS = False
pyphi.config.VALIDATE_SUBSYSTEM_STATES = False

"""
This lowk sucks for the following reasons:
- not really correlated to phi-values which is what we want
- similar time-complexity to IIT 4.0
"""

def compute_geometric_phi_init(tpm):
    """
    Compute integrated information (Phi) for a state-by-node transition probability matrix.

    Args:
        tpm (np.ndarray):   A 2^n x n matrix where each row represents the output probabilities
                            for a specific input state (sorted in little-endian order).

    Returns:
        float: The Geometric Phi value for the system.
    """
    # Validate matrix shape
    num_states, num_nodes = tpm.shape
    if num_states != 2**num_nodes:
        raise ValueError("The number of rows in the matrix must be 2^n for n nodes.")

    # Calculate the whole-system distribution
    joint_distribution = np.prod(tpm, axis=1)

    # Normalize to ensure it forms a valid probability distribution
    joint_distribution /= np.sum(joint_distribution)

    # Compute marginal distributions for subsets of nodes
    phi_values = []
    for k in range(1, num_nodes):  # Subsystems of size 1 to n-1
        for subset in itertools.combinations(range(num_nodes), k):
            complement = [i for i in range(num_nodes) if i not in subset]

            # Marginal distributions for subset and complement
            subset_probs = np.prod(tpm[:, subset], axis=1)
            complement_probs = np.prod(tpm[:, complement], axis=1)

            # Normalize
            subset_probs /= np.sum(subset_probs)
            complement_probs /= np.sum(complement_probs)

            # Compute mismatch (KL divergence)
            kl_div = np.sum(joint_distribution * (np.log(joint_distribution + 1e-9) - \
                                                 (np.log(subset_probs + 1e-9) + \
                                                  np.log(complement_probs + 1e-9))))

            phi_values.append(kl_div)

    # Integrated information is the minimum mismatch across all partitions
    return min(phi_values)

def compute_geometric_phi_hull(tpm):

    # Normalize TPM rows
    tpm = tpm / np.sum(tpm, axis=1, keepdims=True)

    # Add small perturbation to avoid flat simplex issues
    epsilon = 1e-10
    tpm += np.random.uniform(-epsilon, epsilon, tpm.shape)

    # Check for degeneracy
    if np.linalg.matrix_rank(tpm) < tpm.shape[1]:
        tpm += np.random.uniform(-epsilon, epsilon, tpm.shape)

    # Whole-system geometry
    try:
        whole_hull = ConvexHull(tpm, qhull_options="QJ QbB")
        whole_volume = whole_hull.volume
    except Exception as e:
        print(f"ConvexHull error: {e}")
        return None

    # Subsystem measures
    num_nodes = tpm.shape[1]
    subsystem_volumes = 0

    for i in range(num_nodes):
        subsystem_tpm = np.delete(tpm, i, axis=1)
        if subsystem_tpm.shape[1] > 1:
            try:
                subsystem_hull = ConvexHull(subsystem_tpm, qhull_options="QJ QbB")
                subsystem_volumes += subsystem_hull.volume
            except Exception as e:
                print(f"ConvexHull error for subsystem: {e}")
                return None

    # Compute geometric Phi
    phi = whole_volume - subsystem_volumes
    return phi

def convert_to_state_tpm(node_tpm):
    """
    Convert a TPM where each row corresponds to an input state
    and each column corresponds to a node's output probabilities
    into a state-by-state TPM where each row corresponds to the
    input state and each column represents a full output state.

    Args:
        node_tpm (ndarray): Input TPM with shape (2^n, n),
                            where each row corresponds to an input state
                            and each column to node probabilities.

    Returns:
        ndarray: State-by-state TPM with shape (2^n, 2^n).
    """
    num_states = node_tpm.shape[0]  # Number of input states
    num_nodes = node_tpm.shape[1]  # Number of nodes
    num_output_states = 2**num_nodes

    # Generate all possible output states (e.g., [0, 0], [0, 1], ..., [1, 1])
    output_states = list(itertools.product([0, 1], repeat=num_nodes))

    # Initialize the new TPM
    state_tpm = np.zeros((num_states, num_output_states))

    # Compute the joint probabilities for each input state
    for input_state in range(num_states):
        for output_index, output_state in enumerate(output_states):
            prob = 1.0
            for node, node_state in enumerate(output_state):
                prob *= node_tpm[input_state, node] if node_state == 1 else (1 - node_tpm[input_state, node])
            state_tpm[input_state, output_index] = prob

    return state_tpm

def compute_geometric_phi_distance(tpm):
    """
    Compute Geometric IIT Phi using KL divergence without explicit partitions.
    Args:
        tpm (ndarray): State-by-state Transition Probability Matrix.

    Returns:
        float: Geometric IIT Phi value.

    Based on the implementation in this, complete with KL divergence:
    https://www.pnas.org/doi/epdf/10.1073/pnas.1603583113
    """

    tpm = convert_to_state_tpm(tpm)

    # Normalize the TPM (each row sums to 1)
    tpm = tpm / np.sum(tpm, axis=1, keepdims=True)  # Avoid division by zero with keepdims=True

    # Compute the whole-system distribution (mean of all TPM rows)
    whole_distribution = np.mean(tpm, axis=0)  # Average across all input states

    # Compute the independent distribution assuming node independence
    num_states = tpm.shape[1]  # Total number of output states (2^n)
    num_nodes = int(np.log2(num_states))  # Number of nodes in the system
    node_distributions = []

    # Precompute masks for each node
    state_indices = np.arange(num_states)
    node_masks = [(state_indices & (1 << node)) > 0 for node in range(num_nodes)]

    # Extract marginal probabilities for each node
    for node, mask in enumerate(node_masks):
        # Calculate the probability of the node being ON or OFF
        on_prob = np.sum(tpm[:, mask], axis=1)
        off_prob = 1 - on_prob
        node_distributions.append(np.stack([off_prob, on_prob], axis=1))

    # Compute the independent joint probabilities
    independent_distribution = np.zeros(num_states)
    for state in range(num_states):
        prob = 1.0
        for node, node_dist in enumerate(node_distributions):
            bit = (state >> node) & 1  # Extract the bit value for this node
            prob *= node_dist[:, bit].mean()  # Average over all input states
        independent_distribution[state] = prob

    # Ensure distributions sum to 1 (add small epsilon for stability)
    whole_distribution /= np.sum(whole_distribution) + 1e-10
    independent_distribution /= np.sum(independent_distribution) + 1e-10

    # Compute divergence (KL Divergence)
    phi = entropy(whole_distribution, independent_distribution)  # KL Divergence

    # Optional: Normalize phi (scaled between 0 and 1)
    max_phi = entropy(whole_distribution, np.ones_like(whole_distribution) / len(whole_distribution))
    phi_normalized = phi / max_phi if max_phi > 0 else 0

    return phi_normalized  # Return normalized phi

def capped_relu(x):
    return tf.keras.activations.relu(x, max_value=1)

tf.keras.utils.get_custom_objects().update({'capped_relu': capped_relu})

if __name__ == "__main__":

    for i in range(0, 20):

        with open(f"simplyphi1/universalTests/clf/universalTestClf{i}.pickle", "rb") as f:
            data = pickle.load(f)

        train_loss = [x[0] for x in data[0][0]]
        val_loss = [x[0] for x in data[0][1]]

        # with open(f"simplyphi1/universalTests/reg/universalTestReg{i}.pickle", "rb") as f:
        #     data = pickle.load(f)

        # with open(f"universalTestReg2.pickle", "rb") as f:
        #     data = pickle.load(f)

        # train_loss = [x["loss"][0] for x in data[0][0]]
        # val_loss = data[0][1]

        phi_avgs = data[0][2]

        tpms = data[0][6]

        giit_phis = []
        giit_hull_phis = []
        giit_distance_phis = []

        giit_times = []
        giit_hull_times = []
        giit_distance_times = []

        for tpm in tpms:

            pt = time.time()
            giit_phis.append(compute_geometric_phi_init(tpm))
            ct = time.time()
            giit_times.append(ct - pt)

            pt = time.time()
            giit_hull_phis.append(compute_geometric_phi_hull(tpm))
            ct = time.time()
            giit_hull_times.append(ct - pt)

            pt = time.time()
            giit_distance_phis.append(compute_geometric_phi_distance(tpm))
            ct = time.time()
            giit_distance_times.append(ct - pt)

        print("Geo IIT Avg Time:", sum(giit_times) / len(giit_times))
        print("Geo IIT Max Time:", max(giit_times))
        print("Geo IIT Hull Avg Time:", sum(giit_hull_times) / len(giit_hull_times))
        print("Geo IIT Hull Max Time:", max(giit_hull_times))
        print("Geo IIT Distance Avg Time:", sum(giit_distance_times) / len(giit_distance_times))
        print("Geo IIT Distance Max Time:", max(giit_distance_times))

        df = pd.DataFrame({
            "Train Losses": train_loss,
            "Val Losses": val_loss,
            "Geometric ΦG": giit_phis,
            "Geometric Hull-based ΦG": giit_hull_phis,
            "Geometric Distance-based ΦG": giit_distance_phis,
            "Φ (IIT 4.0)": phi_avgs
        })

        print(df.head())

        # Create pairwise scatter plot grid
        # sns.pairplot(df, kind="scatter", plot_kws={'alpha': 0.7})
        # plt.suptitle("Pairwise Comparisons of Φ Metrics", y=1.02)
        # plt.show()

        # Create line graphs for all of them
        from sklearn.preprocessing import MinMaxScaler

        for col in df.columns:  
            plt.plot(list(range(len(train_loss))), MinMaxScaler().fit_transform(np.array(list(df[col])).reshape(-1, 1)), label=col)

        plt.legend()
        plt.show()

        correlation_matrix = df.corr()

        print(correlation_matrix)

"""
NOTES SO FAR:

TODO ADD TPM SAMPLING AND SEE IF PRESERVED THROUGH TESTS!

- only good for neural network stuff but that's all that matters!!

- pretty solid for universalTestClf
- SO FAR SO GOOD FOR 

DISTANCE BASED PHI LOWK THE MOVE OMG OMG OMG LOCKED IN HERE YAYAYAYAY
--> really good for classification, to be determined for regressions
- increases for classification but decreases for regression?
"""