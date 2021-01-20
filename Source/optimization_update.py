import numpy as np
import os
import time
import csv
from functions import *

def optimize_and_update(dW, db, learning_rate):
    
    #dW = list containing derivatives for weights
    
    start = time.time()
    
    cwdPath = os.getcwd()
    wPath = cwdPath.replace("Source", "Weights")
    bPath = cwdPath.replace("Source", "Biases")
    
    L = len(dW) + 1
    
    for i in range(L - 1):
        
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
        
        #Updating parameters
        W = W - learning_rate * dW[L - 2 - i]
        b = b - learning_rate * db[L - 2 - i]
        
        #Saving new values to file
        
        file_name = '../Weights/w' + str(i) + ".csv"
        np.savetxt(file_name, W, delimiter=',')
        
        file_name = '../Biases/b' + str(i) + ".csv"
        np.savetxt(file_name, b, delimiter=',')
        
    end = time.time()
    #print("Time spent in optimization and updating[s]: " + str(end- start))
    
