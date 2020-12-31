import mne
import os
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


our_file = open('originaldata', 'w')
our_file.write("\n")
our_file.close()

our_file = open('originaldata', 'w')

#we have no P9, P10, AF7, AFZ, AF8 gonzales, P9 you find in filmele np.reverse(am)+trix
channels = "C5 C3 C1 CZ C2 C4 C6 CP5 CP3 CP1 CPZ \
CP2 CP4 CP6 P7 P5 P3 PZ P2 P4 P6 P8 FC5 FC3 FC1 FCZ FC2 FC4 FC6 \
F7 F5 F3 F1 FZ F2 F4 F6 F8 AF3 AF4"

channels = channels.split(' ')

tags = ["tag2", "tag3"]


mean_len = 10
window_len = 1000

selected_tags0 = [all_signals_list[i] for i in range(len(all_signals_list)) if all_signals_list[i].find(tags[0]) >= 0]
selected_tags1 = [all_signals_list[i] for i in range(len(all_signals_list)) if all_signals_list[i].find(tags[1]) >= 0]

selected_tags = (selected_tags0, selected_tags1)

mean_matrix = np.repeat(np.identity(window_len//mean_len), repeats = mean_len, axis=0)

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
            window = samples[:, 500:1500]
            window_mean = np.dot(window, mean_matrix) / mean_len
            obs_line.append(window_mean)
            
        obs_line = np.hstack(obs_line)
        our_data.append(obs_line)
    
    
our_data = np.vstack(our_data)

our_file.write(str(our_data))

our_file.close()



end = time.time()
print("Elapsed time [s]: " + str(end- start))
