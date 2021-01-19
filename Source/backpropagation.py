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
    #Y = Y.reshape(Y.shape[0], 1)
    print(Y.shape)
    
    L = len(Z_list) + 1
    
    #number of training examples
    m = Y.shape[0]
    
    dW_list = list()
    db_list = list()
    dZ_old = 0
    
    #start from L-2 because weights start at 0

    for i in range(L-2, -1, -1):
        #first index is start - 1
        print(str(i) + "for")
        
        #last layer special case, w starts at 0
        if i == L - 2:
            print("if" + str(i))
            dZ = A_list[i] - Y
            dW = 1/m * np.dot(dZ, A_list[i-1].T)
            try:
                db = 1/m * np.sum(dZ, keepdims=True )
            except:
                dZ = dZ.reshape(dZ.shape[0], 1)
                db = 1/m * np.sum(dZ, keepdims=True )
            
            #first element is last weight
            dW_list.append(dW)
            db_list.append(db)
            
            #dZ superscript[i+1]
            dZ_old = dZ
            
        else:
            print(i)
            W = np.loadtxt(wPath + "/w" + str(i), dtype='f', delimiter=',')
            try:
                dZ = np.dot(W.T, dZ_old) * tanh_derivative(Z_list[i])
            except:
                dZ_old = dZ_old.reshape(dZ_old.shape[0], 1)
                dZ = np.multiply(np.dot(W.T, dZ_old) , tanh_derivative(Z_list[i]) ) 
                dW = 1/m * np.dot(dZ, A_list[i-1].T) 
                db = 1/m * np.sum(dZ, axis = 1, keepdims=True )
        
                #first element is last weight
                dW_list.append(dW)
                db_list.append(db)
                
                #dZ superscript[i+1]
                dZ_old = dZ
            
    
    return (dW_list, db_list)
    
    end = time.time()
    print("Elapsed time [s]: " + str(end- start))