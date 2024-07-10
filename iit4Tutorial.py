"""
just a 4.0 migration of the currently-ran 3.0 experiment so we get comfortable with
working with phi-structures, distinctions, relations
"""

import pyphi
import numpy as np
import itertools

pyphi.config.PROGRESS_BARS = False
pyphi.config.PARALLEL = False
pyphi.config.SHORTCIRCUIT_SIA = False
pyphi.config.VALIDATE_SUBSYSTEM_STATES = False

# Define units
unit_labels = ["A", "B", "C"]
n_units = len(unit_labels) # 3 units

# Define activation functions (all units have the same function)
unit_activation_function = pyphi.network_generator.ising.probability
k = 4 # determines the slope of the sigmoid

# Define weighted connectivity among units
weights = np.array(
    [
        [-.2, 0.7, 0.2], # outgoing connections from A
        [0.7, -.2, 0.0], # outgoing connections from B
        [0.0, -.8, 0.2], # outgoing connections from C
    ]
)

# Generate the substrate model
substrate = pyphi.network_generator.build_network(
    [unit_activation_function] * n_units,
    weights,
    temperature=1 / k,
)

# Print the state-by-node, forward TPM characterizing the substrate
print('Substrate TPM: \n(input state) : probability that units turn ON')
for input_state, transition_probability in zip(pyphi.utils.all_states(3), pyphi.convert.to_2d(substrate.tpm.round(2))):
  print(f'{input_state} : {transition_probability}')