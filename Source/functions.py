import numpy as np

def tanh(z):
    return  ( np.exp(z) - np.exp(-z) ) / ( np.exp(z) + np.exp(-z) ) 

def tanh_derivative(z):
    return 1 - tanh(z)**2

def sigmoid(z):
    return 1 / ( 1 + np.exp(-z) )

def sigmoid_derivative(z):
    return sigmoid(z) * ( 1 - sigmoid(z) )

def loss(y_hat, y):
    return -( y * np.log2(y_hat) + (1-y) * np.log2(1-y_hat) )

