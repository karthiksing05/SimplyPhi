import numpy as np
import pandas as pd
from geometricIIT import compute_geometric_phi_init, compute_geometric_phi_distance, compute_geometric_phi_hull
import os
os.environ["PYPHI_WELCOME_OFF"] = "yes"x

import pyphi
import time
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

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

def phi_iit_4(tpm):
    """Evaluate phi using IIT 4.0's built-in functions."""
    labels = tuple([f"Node{i}" for i in range(len(tpm[0]))])
    cm = np.ones((len(tpm[0]), len(tpm[0])))
    substrate = pyphi.Network(tpm, cm=cm, node_labels=labels)

    phi_avg = 0
    big_phi_avg = 0
    all_states = list(itertools.product([0, 1], repeat=len(tpm[0])))

    for test_state in all_states:
        fc_sia = pyphi.new_big_phi.maximal_complex(substrate, test_state)
        fc = pyphi.Subsystem(substrate, test_state, nodes=fc_sia.node_indices)
        fc_structure = pyphi.new_big_phi.phi_structure(fc)

        phi_avg += fc_structure.phi
        big_phi_avg += fc_structure.big_phi

    phi_avg /= len(all_states)
    big_phi_avg /= len(all_states)
    return phi_avg, big_phi_avg

# Function to generate random state-by-node TPM
def generate_random_tpm(num_nodes=3):
    num_states = 2 ** num_nodes
    tpm = np.random.rand(num_states, num_nodes)
    tpm = tpm / tpm.sum(axis=0)  # Normalize so that the columns sum to 1
    return tpm

def calculate_phi_values(num_tpm=100, num_nodes=2):
    """
    Calculate phi values for multiple TPMs, including geometric phi, IIT phi, Fisher geometric phi, 
    and other strategies based on manifold geometry.
    Args:
        num_tpm (int): Number of random TPMs to generate.
        num_nodes (int): Number of nodes in each TPM.

    Returns:
        Tuple: Lists of phi values, average and max times for each method.
    """

    BENCHMARK = 20

    phi_g_values = []
    phi_g_hull_values = []
    phi_g_dist_values = []
    phi_values = []
    big_phi_values = []

    times_phi_iit = []

    # Iterate over the number of TPMs
    for i in range(num_tpm):
        tpm = generate_random_tpm(num_nodes)

        # Time geometric integrated information (geometric phi)
        phi_g = compute_geometric_phi_init(tpm)  # Using your method
        phi_g_values.append(phi_g)

        phi_g_hull = compute_geometric_phi_hull(tpm)  # Using your method
        phi_g_hull_values.append(phi_g_hull)

        phi_g_dist = compute_geometric_phi_distance(tpm)  # Using your method
        phi_g_dist_values.append(phi_g_dist)

        # Time phi_iit_4 (IIT metric)
        start_time = time.time()
        phi_iit, big_phi_iit = phi_iit_4(tpm)  # Using your method
        end_time = time.time()
        times_phi_iit.append(end_time - start_time)
        phi_values.append(phi_iit)
        big_phi_values.append(big_phi_iit)

        if (i + 1) % (num_tpm // BENCHMARK) == 0 or i == (num_tpm - 1):
            print(f"Completed {i + 1} iterations!")

    # Print results
    print(f"Average time for IIT phi: {np.mean(times_phi_iit):.4f}s")
    print(f"Max time for IIT phi: {np.max(times_phi_iit):.4f}s")

    return (phi_g_values, phi_g_hull_values, phi_g_dist_values, phi_values, big_phi_values)

# Generating and saving values:
phi_g_values, hull_g_values, dist_g_values, phi_values, big_phi_values = calculate_phi_values(num_tpm=100, num_nodes=3)

# Save the results to files for later loading
np.save("phi_g_values.npy", phi_g_values)
np.save("hull_g_values.npy", hull_g_values)
np.save("dist_g_values.npy", dist_g_values)
np.save("phi_values.npy", phi_values)
np.save("big_phi_values.npy", big_phi_values)

# Loading values:
phi_g_values = np.load("phi_g_values.npy")
hull_g_values = np.load("hull_g_values.npy")
dist_g_values = np.load("dist_g_values.npy")
phi_values = np.load("phi_values.npy")
big_phi_values = np.load("big_phi_values.npy")

# Creating a DataFrame to store the values
df = pd.DataFrame({
    "Geometric ΦG": phi_g_values,
    "Geometric (Hull) ΦG": hull_g_values,
    "Geometric (Distance) ΦG": dist_g_values,
    "Φ (IIT 4.0)": phi_values,
    "Big Φ (IIT 4.0)": big_phi_values
})

# Create pairwise scatter plot grid
sns.pairplot(df, kind="scatter", plot_kws={'alpha': 0.7})
plt.suptitle("Pairwise Comparisons of Φ Metrics", y=1.02)
plt.show()

# Calculate correlation matrix
correlation_matrix = df.corr()

print(correlation_matrix)

# # Create heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
# plt.title("Correlation Heatmap of Φ Metrics")
# plt.show()
