import mne
import os
from our_pca import our_pca
import matplotlib.pyplot as plt
import numpy as np
import time
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


start = time.time()

cwdPath = os.getcwd()
cwdPath = cwdPath.replace("Source", "Data")

#store all data names from cwd in a list
all_signals_list = os.listdir(cwdPath)

#add "/" before the names of data for concatenation with cwdPath
for i in range(len(all_signals_list)):
    all_signals_list[i] = "/" + all_signals_list[i]


#we have no P9, P10, AF7, AFZ, AF8 gonzales, P9 you find in filmele np.reverse(am)+trix
#currently 40 channels
channels = "C5 C3 C1 CZ C2 C4 C6 CP5 CP3 CP1 CPZ \
CP2 CP4 CP6 P7 P5 P3 PZ P2 P4 P6 P8 FC5 FC3 FC1 FCZ FC2 FC4 FC6 \
F7 F5 F3 F1 FZ F2 F4 F6 F8 AF3 AF4"

channels = channels.split(' ')

tags = ["tag2", "tag3"]


MEAN_LEN = 10
window_len = 1000
WINDOW_NUMBER = 1
OVERLAP = 500
START_SAMPLE = 500
SEQUENCE_LENGTH = 1000

selected_tags0 = [all_signals_list[i] for i in range(len(all_signals_list)) if all_signals_list[i].find(tags[0]) >= 0]
selected_tags1 = [all_signals_list[i] for i in range(len(all_signals_list)) if all_signals_list[i].find(tags[1]) >= 0]

selected_tags = (selected_tags0, selected_tags1)

mean_matrix = np.repeat(np.identity(window_len//MEAN_LEN), repeats = MEAN_LEN, axis=0)

our_data = list()
#for all tags
for i in range(len(selected_tags)):
    
    #for all signals with tag i
    for j in range(len(selected_tags[i])):
    
        f_name = cwdPath + selected_tags[i][j]
        raw = mne.io.read_raw_fif(f_name)
        obs_line = list()
        
        #for selected channels in signal j with tag i
        for channel_name in channels:
            #selecting elemnt 0 of touple = samples
            samples = raw[channel_name][0]
            START_SAMPLE = 500
            for k in range(WINDOW_NUMBER):
                window = samples[:, START_SAMPLE : START_SAMPLE + SEQUENCE_LENGTH]
                window_mean = np.dot(window, mean_matrix) / MEAN_LEN
                obs_line.append(window_mean)
                START_SAMPLE += OVERLAP
            
        obs_line = np.hstack(obs_line)
        our_data.append(obs_line)
    
    
our_data = np.vstack(our_data)
our_data = our_pca(our_data, 0.95)

cwdPath = cwdPath.replace("Data", "NN_Data")
cwdPath = cwdPath + "/nn_data"
np.savetxt(cwdPath, our_data, delimiter=',')


end = time.time()
print("Elapsed time [s]: " + str(end- start))
