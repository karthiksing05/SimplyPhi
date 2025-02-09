for k in range(5):
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

            geo_phi = compute_geometric_phi_distance(tpm)

            accuracy_dependency = actual_loss(y_pred, tf.zeros_like(y_pred)) * 1e-6

            return geo_phi + accuracy_dependency

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
                split = 0.5 + 0.5 * (epoch + 1) / epochs - 1e-6 # TODO LOGARITHM TEST MULITPLE SPLIT DIFFERENCES
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

        with open(f"GIITLinearAdapt{k}.pickle", "wb") as f:
            pickle.dump([phi_train_losses, phi_val_losses, train_losses, val_losses, phi_model, model, X, y], f)
