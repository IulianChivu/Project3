import numpy as np
import os
import time
import json

def initialize_weights_biases():

    start = time.time()
    
    cwdPath = os.getcwd()
    cwdPath = cwdPath.replace("Source", "NN_Info")
    
    with open(cwdPath + '/nn_info') as f: 
        info = f.read() 
    
    info_dict = json.loads(info) 
    
    #number of layers
    L = int(info_dict["L"])
    
    #number of sets of weights matrices
    WEIGHTS_SETS = L - 1
    
    cwdPath = os.getcwd()
    cwdPath = cwdPath.replace("Source", "Weights")
    
    #list of weights matrices dimensions
    #make sure you have enough dimensions set in the file
    WEIGHTS_DIMS = list(np.loadtxt(cwdPath + "/weights_dims", dtype='int', delimiter=','))
    
    np.random.seed(1)
    
    for i in range(WEIGHTS_SETS):
        w = np.random.rand(WEIGHTS_DIMS[i][0],WEIGHTS_DIMS[i][1])
        file_name = '../Weights/w' + str(i) + ".csv"
        np.savetxt(file_name, w, delimiter=',')
        
    
    cwdPath = os.getcwd()
    cwdPath = cwdPath.replace("Source", "Biases")
    
    #initializing biases
    for i in range(WEIGHTS_SETS):
        #b = np.random.rand(WEIGHTS_DIMS[i][0], 1)
        b = np.zeros( (WEIGHTS_DIMS[i][0], 1) )
        file_name = '../Biases/b' + str(i) + ".csv"
        np.savetxt(file_name, b, delimiter=',')
    
    
    end = time.time()
    #print("Elapsed time [s]: " + str(end- start))
    
    
