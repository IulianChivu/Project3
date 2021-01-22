import numpy as np
import os
import time
import json
import csv
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
    A = np.loadtxt(dPath + "/nn_data.csv", dtype='f', delimiter=',')
    
    
    #Z must be cahced, needed in backpropagation
    Z_list = list()
    #A must be cahced, needed in backpropagation
    A_list = list()
    A_list.append(A)
    
    for i in range(L-1):
        
        W_filename = wPath + "/w" + str(i) + ".csv"
        b_filename = bPath + "/b" + str(i) + ".csv"

        W_raw_data = open(W_filename, 'rt')
        b_raw_data = open(b_filename, 'rt')
        
        W_reader = csv.reader(W_raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
        b_reader = csv.reader(b_raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
        
        W_temp = list(W_reader)
        b_temp = list(b_reader)
        
        W = np.array(W_temp).astype('float')
        b = np.array(b_temp).astype('float')

        #print("multiplying "+str(W.shape)+" with "+str(A.shape))
        Z = np.dot(W, A) + b
        #print("resulting "+str(Z.shape))
        Z_list.append(Z)
        
        if i != L-2:
            A = relu(Z)
            A_list.append(A)
        else:
            A = sigmoid(Z)
            A_list.append(A)
    
    end = time.time()
    #print("Time spent in forward propagation [s]: " + str(end- start))
    
    return (Z_list, A_list)