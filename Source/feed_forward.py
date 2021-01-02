import numpy as np
import os
import time
import json

start = time.time()

cwdPath = os.getcwd()
cwdPath = cwdPath.replace("Source", "NN_Data")

#initially A = X = the entire data set
#trying full vectorization
A = np.loadtxt(cwdPath + "/nn_data", dtype='f', delimiter=',')


cwdPath = os.getcwd()
cwdPath = cwdPath.replace("Source", "NN_Info")
with open(cwdPath + '/nn_info') as f: 
    info = f.read() 

info_dict = json.loads(info)
#number of layers
L = int(info_dict["L"])

cwdPath = os.getcwd()
cwdPath = cwdPath.replace("Source", "Weights")
cwdPath = cwdPath + "/w0"

W_list = list()
W = np.loadtxt(cwdPath, dtype='f', delimiter=',')


#Z must be cahced, needed in backpropagation
Z_list = list()

#L-1-1; 1 for weights sets and 1 because w0 has already been appended
for i in range(1, L - 2):
    



end = time.time()
print("Elapsed time [s]: " + str(end- start))