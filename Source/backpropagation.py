import numpy as np
import os
import time
import csv
from functions import *

def back_propagation(Z_list, A_list):

    start = time.time()
    
    cwdPath = os.getcwd()
    lPath = cwdPath.replace("Source", "Labels")
    wPath = cwdPath.replace("Source", "Weights")
    
    Y_filename = lPath + "/labels.csv"
    Y_raw_data = open(Y_filename, 'rt')
    
    Y_reader = csv.reader(Y_raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    Y_temp = list(Y_reader)
    Y = np.array(Y_temp).astype('float')
    
    #print(Y.shape)
    
    L = len(Z_list) + 1
    
    #number of training examples
    m = Y.shape[0]
    
    dW_list = list()
    db_list = list()
    dZ_old = 0
    
    #start from L-2 because weights start at 0

    for i in range(L-2, -1, -1):
        #first index is start - 1?
        #print(str(i))
        
        #last layer special case, w starts at 0
        if i == L - 2:
            

            dZ = A_list[i+1] - Y
            
            #print("multiplying "+str(dZ.shape)+" with "+str(A_list[i+1].T.shape))
            
            dW = 1/m * np.dot(dZ, A_list[i].T)
            db = 1/m * np.sum(dZ, axis = 1, keepdims=True )
            
            #print("resulting "+str(dW.shape))
            
            #first element is last weight
            dW_list.append(dW)
            db_list.append(db)
            
            #dZ superscript[i+1]
            dZ_old = dZ
            
        else:
            

            W_filename = wPath + "/w" + str(i+1) + ".csv"
            W_raw_data = open(W_filename, 'rt')
            W_reader = csv.reader(W_raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
            W_temp = list(W_reader)
            W = np.array(W_temp).astype('float')

            #print("multiplying "+str(W.T.shape)+" with "+str(dZ_old.shape))

            dZ = np.multiply(np.dot(W.T, dZ_old) , relu_derivative(Z_list[i]) ) 
            
            #print("resulting "+str(dZ.shape))
            
            #print("multiplying "+str(dZ.shape)+" with "+str(A_list[i].T.shape))
            
            dW = 1/m * np.dot(dZ, A_list[i].T) 
            db = 1/m * np.sum(dZ, axis = 1, keepdims=True )
            
            #print("resulting "+str(dW.shape))
            
            
            #first element is last weight
            dW_list.append(dW)
            db_list.append(db)
                
            #dZ superscript[i+1]
            dZ_old = dZ
            
    
    end = time.time()
    #print("Elapsed time [s]: " + str(end- start))
    
    return (dW_list, db_list)
    