import numpy as np

class Layer:
    def __init__(self):
        self.inp = None
        self.output = None

    def forward(self, inp):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return inp gradient
        pass

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, inp):
        self.inp = inp
        return self.activation(self.inp)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_prime(self.inp)