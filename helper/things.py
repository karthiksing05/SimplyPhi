"""
Native Implementation of a densely connected neural network
(so that we can mess around with back-propagation and stuff later)

Following this tutorial: https://www.youtube.com/watch?v=pauPCy_s0OkAD (just need to make sure to
implement my own loss function stuffs and optimize bc binary)
"""

import numpy as np
from helper.structs import *

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)  # He initialization
        self.bias = np.zeros((output_size, 1))  # Zero initialization for biases

    def forward(self, inp):
        self.inp = inp
        return np.dot(self.weights, self.inp) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.inp.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.grad_weights = weights_gradient
        self.grad_bias = output_gradient
        return input_gradient

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

def binary_cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return y_pred - y_true

def binary_accuracy(y_true, y_pred):
    return (np.count_nonzero(y_true == np.matrix.round(y_pred)) / np.size(y_true))

def clip_gradients(grad, clip_value=1.0):
    return np.clip(grad, -clip_value, clip_value)

class AdamOptimizer:
    def __init__(self, network, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.network = network
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []
        self.v = []
        self.t = 0

        for layer in network:
            if isinstance(layer, Dense):
                self.m.append([np.zeros_like(layer.weights), np.zeros_like(layer.bias)])
                self.v.append([np.zeros_like(layer.weights), np.zeros_like(layer.bias)])

    def step(self):
        self.t += 1
        lr_t = self.lr * (np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t))

        for i, layer in enumerate(self.network):
            if isinstance(layer, Dense):
                # Update weights
                grad_w = layer.grad_weights
                grad_b = layer.grad_bias

                self.m[i][0] = self.beta1 * self.m[i][0] + (1 - self.beta1) * grad_w
                self.m[i][1] = self.beta1 * self.m[i][1] + (1 - self.beta1) * grad_b
                self.v[i][0] = self.beta2 * self.v[i][0] + (1 - self.beta2) * (grad_w ** 2)
                self.v[i][1] = self.beta2 * self.v[i][1] + (1 - self.beta2) * (grad_b ** 2)

                m_hat_w = self.m[i][0] / (1 - self.beta1 ** self.t)
                m_hat_b = self.m[i][1] / (1 - self.beta1 ** self.t)
                v_hat_w = self.v[i][0] / (1 - self.beta2 ** self.t)
                v_hat_b = self.v[i][1] / (1 - self.beta2 ** self.t)

                layer.weights -= lr_t * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                layer.bias -= lr_t * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

