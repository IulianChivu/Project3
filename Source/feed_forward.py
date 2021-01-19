import numpy as np
import os
import time
import json
from functions import *

def feed_forward():
    
    start = time.time()
    
    cwdPath = os.getcwd()
    wPath = cwdPath.replace("Source", "Weights")
    bPath = cwdPath.replace("Source", "Biases")
    dPath = cwdPath.replace("Source", "NN_Data")
    iPath = cwdPath.replace("Source", "NN_Info")
    
    with open(iPath + '/nn_info') as f: 
        info = f.read() 
    
    info_dict = json.loads(info)
    #number of layers
    L = int(info_dict["L"])
    
    
    #initially A = X = the entire data set
    #trying full vectorization
    A = np.loadtxt(dPath + "/nn_data", dtype='f', delimiter=',')
    
    
    #Z must be cahced, needed in backpropagation
    Z_list = list()
    #A must be cahced, needed in backpropagation
    A_list = list()
    
    for i in range(L-1):
        W = np.loadtxt(wPath + "/w" + str(i), dtype='f', delimiter=',')
        b = np.loadtxt(bPath + "/b" + str(i), dtype='f', delimiter=',')
        try:
            b = b.reshape(b.shape[0], 1)
        except:
            pass
        Z = np.dot(W, A) + b
        Z_list.append(Z)
        if i != L-2:
            A = tanh(Z)
            A_list.append(A)
        else:
            A = sigmoid(Z)
            A_list.append(A)
    
    end = time.time()
    print("Time spent in forward propagation [s]: " + str(end- start))
    
    return (Z_list, A_list)