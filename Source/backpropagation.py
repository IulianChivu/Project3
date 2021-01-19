import numpy as np
import os
import time
from functions import *

def back_propagation(Z_list, A_list):

    start = time.time()
    
    cwdPath = os.getcwd()
    lPath = cwdPath.replace("Source", "Labels")
    wPath = cwdPath.replace("Source", "Weights")
    
    Y = np.loadtxt(lPath + "/labels", dtype='f', delimiter=',')
    
    L = len(Z_list)
    
    #number of training examples
    m = Y.shape[0]
    
    dW_list = list()
    db_list = list()
    dZ_old = 0
    
    for i in range(L-1, -1, -1):
        if i == L-1:
            dZ = A_list[i] - Y
            dW = 1/m * np.dot(dZ, A_list[i-1].T)
            db = 1/m * np.sum(dZ, axis = 0, keepdims=True)
            
            print(dW.shape)
            print(i)
            #Appending derivatives
            dW_list = dW_list.append(dW)
            db_list = db_list.append(db)
            dZ_old = dZ
        else:
            W = np.loadtxt(wPath + "/w" + str(i), dtype='f', delimiter=',')
            print(i)
            dZ = np.dot(W.T, dZ_old) * tanh_derivative(Z_list[i])
            dW = 1/m * np.dot(dZ, A_list[i-1].T)
            db = 1/m * np.sum(dZ, axis = 0, keepdims=True)
            dZ_old = dZ
            
            print(dW.shape)
            #Appending derivatives
            dW_list = dW_list.append(dW)
            db_list = db_list.append(db)
    
    return (dW, db)
    
    end = time.time()
    print("Elapsed time [s]: " + str(end- start))