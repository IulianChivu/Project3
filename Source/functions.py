import numpy as np

EPS = 10*-18

def tanh(z):
    return  ( np.exp(z) - np.exp(-z) ) / ( np.exp(z) + np.exp(-z) )

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return 1 * (z > 0)

def tanh_derivative(z):
    return 1 - tanh(z)**2

def sigmoid(z):
    return 1 / ( 1 + np.exp(-z) )

def sigmoid_derivative(z):
    return sigmoid(z) * ( 1 - sigmoid(z) )

def loss(y_hat, y):
    return -( y * np.log2(y_hat ) + (1-y) * np.log2(1-y_hat ) )

def cost(m , Loss):
    return 1/m * np.sum(Loss)

