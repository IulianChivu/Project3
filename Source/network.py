import numpy as np
import os
import time
import json
import csv
import matplotlib.pyplot as plt
from functions import *
from backpropagation import *
from feed_forward import *
from optimization_update import *
from initialize_weights_biases import *

start = time.time()

cwdPath = os.getcwd()
lPath = cwdPath.replace("Source", "Labels")

Y_filename = lPath + "/labels.csv"
Y_raw_data = open(Y_filename, 'rt')
    
Y_reader = csv.reader(Y_raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
Y_temp = list(Y_reader)
Y = np.array(Y_temp).astype('float')

initialize_weights_biases()

LEARNING_RATE = 0.05

cost_list = list()

for i in range(20):
    
    ffw = feed_forward()
    prediction = ffw[1][ len(ffw[1]) - 1]
    Loss = loss(prediction , Y)
    Cost = cost(Y.shape[1], Loss)
    bp = back_propagation(*ffw)
    optimize_and_update(*bp, LEARNING_RATE)
    print("Cost: " +str(Cost))
    cost_list.append(Cost)


end = time.time()
print("Time spent training [s]: " + str(end- start))

plt.plot(cost_list)
plt.show()