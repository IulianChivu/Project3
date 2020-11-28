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

#number of EEG channels present in data
n_eeg = 64

#add "/" before the names of data for concatenation with cwdPath
for i in range(len(all_signals_list)):
    all_signals_list[i] = "/" + all_signals_list[i]


our_file = open('originaldata', 'w')
our_file.write("\n")
our_file.close()

our_file = open('originaldata', 'a')

our_data = list()

for i in range(len(all_signals_list) - 991):
    
    f_name = cwdPath + all_signals_list[i]
    raw = mne.io.read_raw_fif(f_name)
    obs_line = list()
    
    for channel_name in raw.ch_names[:n_eeg]:
        samples = raw[channel_name][0]
        window = samples[:, 500:501]
        obs_line.append(window)
        
    obs_line = np.hstack(obs_line)
    our_data.append(obs_line)
    
    
our_data = np.vstack(our_data)

our_file.close()

end = time.time()
print("Elapsed time [s]: " + str(end- start))
