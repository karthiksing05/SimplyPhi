"""
Given data, a tool that generates the optimal neural network as well as functions to
encode and decode both inputs and outputs, returned as a pickle file with granularity!
"""

import tensorflow as tf
import numpy as np

def _get_indices_of_ones(n):
    # Convert the integer to binary and remove the '0b' prefix
    binary_representation = bin(n)[2:]
    binary_representation = reversed(binary_representation)

    # Collect the indices of all '1's
    indices = [i for i, bit in enumerate(binary_representation) if bit == '1']

    return indices

class Converter(object):

    """
    This object is an easy wrapper for all the variables that are processed by the set
    of nodes. 

    each variables-list is a list of tuples that stores the amount of bits each variable
    needs for encoding as well as whether the variable is categorial / numerical. The
    lists should be passed in with the same format of the input, and the output should
    follow suit.

    Example:
    inputVars = [('cat', 3), ('num', 2), ('num', 3)]
    --> a categorical variable with 3 categories, a numerical variable with 2^2 bins, and
        a numerical variable with 2^3 bins
    outputVars = [('num', 8)]
    --> a numerical variable with 2^8 bins
    """

    inputVars = []
    outputVars = []

    def __init__(self, inputVars:list, outputVars:list):
        self.inputVars = inputVars
        self.outputVars = outputVars

        self.numInputSpaces = sum([x[1] if x[0] == "cat" else 1 for x in self.inputVars])
        self.numOutputSpaces = sum([x[1] if x[0] == "cat" else 1 for x in self.outputVars])

        self.inputNodes = sum([x[1] for x in self.inputVars])
        self.outputNodes = sum([x[1] for x in self.outputVars])

        self.totalNodes = max(self.inputNodes, self.outputNodes)

    def input_to_nodes(self, sample):

        preprocessed = np.zeros(self.totalNodes)

        numBits = 0

        i = 0
        varIter = 0
        while varIter < len(self.inputVars):

            if self.inputVars[varIter][0] == "cat": # TODO VALIDATE THAT THIS WORKS
                idxs = [numBits + x for x in range(self.inputVars[varIter][1]) if sample[i + x]]
                i += self.inputVars[varIter][1]

            elif self.inputVars[varIter][0] == "num":
                coord = sample[i]
                coord -= 1e-6
                idxs = [numBits + x for x in _get_indices_of_ones(int(np.floor(coord * (2 ** self.inputVars[varIter][1]))))]
                i += 1

            for idx in idxs:
                preprocessed[idx] = 1

            numBits += self.inputVars[varIter][1]
            varIter += 1

        return preprocessed

    # prepares the bin mode for transitional input (this is lowk our input layer)
    def nodes_to_input(self, sample):

        inp = np.zeros((self.numInputSpaces,))

        numBits = 0

        currPlace = 0

        for var in self.inputVars:

            if var[0] == "num":
                for b in range(var[1]):
                    inp[currPlace] += sample[numBits + b] * (2 ** b)
                inp[currPlace] /= (2 ** var[1])
                currPlace += 1

            elif var[0] == "cat":
                for j in range(var[1]):
                    inp[currPlace + j] = sample[numBits + j]
                currPlace += var[1]

            numBits += var[1]

        return inp

    # converts the classes to the X nodes stuff
    def output_to_nodes(self, sample, regularization=0.0):
        preprocessed = np.zeros(self.totalNodes)

        numBits = 0

        i = 0
        varIter = 0
        while varIter < len(self.outputVars):

            if self.outputVars[varIter][0] == "cat": # TODO VALIDATE THAT THIS WORKS
                idxs = [numBits + x for x in range(self.outputVars[varIter][1]) if sample[i + x]]
                i += self.outputVars[varIter][1]

            elif self.outputVars[varIter][0] == "num":
                coord = sample[i]
                coord -= 1e-6
                idxs = [numBits + x for x in _get_indices_of_ones(int(np.floor(coord * (2 ** self.outputVars[varIter][1]))))]
                i += 1

            for idx in idxs:
                preprocessed[idx] = 1

            numBits += self.outputVars[varIter][1]
            varIter += 1

        preprocessed = np.where(preprocessed == 0, regularization, preprocessed)

        return preprocessed

    # converts the nodes to classes
    def nodes_to_output(self, sample):

        out = np.zeros((self.numInputSpaces,))

        numBits = 0

        currPlace = 0

        for var in self.outputVars:

            if var[0] == "num":
                for b in range(var[1]):
                    out[currPlace] += sample[numBits + b] * (2 ** b)
                out[currPlace] /= (2 ** var[1])
                currPlace += 1

            elif var[0] == "cat":
                for j in range(var[1]):
                    out[currPlace + j] = sample[numBits + j]
                currPlace += var[1]

            numBits += var[1]

        return out
    
    def get_TPM_activations(self, model, X, regularization=0.01):
        """
        Helper method to potentially solve the probability issue! Essentially, we create a
        layer before our actual output layer with enough nodes to represent the binning and
        then use that layer's activations to generate our TPM!

        Long term, a goal might be to integrate this philosophy on both sides and study
        outputs, but this is a pretty easy solution while triggering a network using activations 
        halfway through may be kinda weird...TBD!
        """
        activations = [X]
        input_data = X

        for layer in model.layers:
            layer_output = layer(input_data)
            activations.append(tf.keras.backend.eval(layer_output))
            input_data = layer_output

        # grabs the second-to-last layer, which should be our transforming layer
        relevantActivations = activations[-2][0]

        for activation in relevantActivations:
            if activation > 1:
                print("ACTIVATION ENERGY BIGGER THAN ONE WTF")
                print(relevantActivations)
                print(X)

        # NOTE implementing noising natively here to see if it resolves that problem too
        relevantActivations = np.where(relevantActivations == 0, regularization, relevantActivations)

        return relevantActivations