import mne
import os
import matplotlib.pyplot as plt
import numpy as np


y = np.ones((1,100))

MEAN_LEN = 2
window_len = 10
WINDOW_NUMBER = 7
OVERLAP = 0
START_SAMPLE = 0
SEQUENCE_LENGTH = 10

mean_matrix = np.repeat(np.identity(window_len//MEAN_LEN), repeats = MEAN_LEN, axis=0)
obs_line = list()

for k in range(WINDOW_NUMBER):
    print(k)
    print("y: " + str(y.shape))
    
    window = y[:, START_SAMPLE : START_SAMPLE + SEQUENCE_LENGTH]
    window_mean = np.dot(window, mean_matrix) / MEAN_LEN
    
    print("multiply" + str(window.shape) + "with " + str(mean_matrix.shape))
    print("results: " + str(window_mean.shape))
    
    obs_line.append(window_mean)
    START_SAMPLE += window_len - OVERLAP

obs_line = np.hstack(obs_line)


plt.plot(y[0, :])

plt.figure()
plt.plot(window_mean.T)