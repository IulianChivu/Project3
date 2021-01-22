import mne
import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt

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

selected_tags0 = [all_signals_list[i] for i in range(len(all_signals_list)) if all_signals_list[i].find(tags[0]) >= 0]
selected_tags1 = [all_signals_list[i] for i in range(len(all_signals_list)) if all_signals_list[i].find(tags[1]) >= 0]

selected_tags = (selected_tags0, selected_tags1)
selected_tags_all = selected_tags0 + selected_tags1

#distribution of data examples across data sets
TRAIN_SET_PERCENTAGE = 0.6
DEV_SET_PEERCENTAGE = 0.2
#make them sum up to 0.99
TEST_SET_PERCENTAGE = 0.19

#number of total examples
data_set_size = len(selected_tags0) + len(selected_tags1)


#defining training set size
train_set_size = round(data_set_size * TRAIN_SET_PERCENTAGE)


#defining dev set size
dev_set_size = round(data_set_size * DEV_SET_PEERCENTAGE)

#defining test set size
test_set_size = round(data_set_size * TEST_SET_PERCENTAGE)

total = train_set_size + dev_set_size + test_set_size

if total != data_set_size:
    train_set_size += data_set_size - total

#shuffling the data set
np.random.shuffle(selected_tags_all)

#selecting examples for each data set
train_set = selected_tags_all[0 : train_set_size]

dev_set = selected_tags_all[train_set_size : train_set_size + dev_set_size]

test_set = selected_tags_all[train_set_size + dev_set_size : data_set_size]

MEAN_LEN = 10
WINDOW_LEN = 4000
START_SAMPLE = 500

mean_matrix = np.repeat(np.identity(WINDOW_LEN//MEAN_LEN), repeats = MEAN_LEN, axis=0)

train_data = list()

for i in range(len(train_set)):
    
    f_name = cwdPath + train_set[i]
    raw = mne.io.read_raw_fif(f_name)
    obs_line = list()
    
        
        
    for channel_name in channels:
        #selecting elemnt 0 of touple = samples
        samples = raw[channel_name][0]
            
        window = samples[:, START_SAMPLE : START_SAMPLE + WINDOW_LEN]
        window_mean = np.dot(window, mean_matrix) / MEAN_LEN
        obs_line.append(window_mean)

    obs_line = np.hstack(obs_line) 
            
    
    train_data.append(obs_line)
    

train_data = np.vstack(train_data)

train_data = train_data.T

#saving
cwdPath = cwdPath.replace("Data", "NN_Data")
cwdPath = cwdPath + "/nn_data.csv"
np.savetxt(cwdPath, train_data, delimiter=',')

#extracting labels works only for 2 classes
train_labels = list()
for i in train_set:
    if i.find("tag2") > 0:
        train_labels.append(0)
    else:
        train_labels.append(1)

train_labels = np.array(train_labels)
train_labels = train_labels.reshape(1, train_labels.shape[0])

#saving labels
cwdPath = os.getcwd()
cwdPath = cwdPath.replace("Source", "Labels")
cwdPath = cwdPath + "/labels.csv"
np.savetxt(cwdPath, train_labels, delimiter=',', fmt='% 1d')


end = time.time()
print("Elapsed time [s]: " + str(end- start))

