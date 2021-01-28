import numpy as np
import os
import time
import json
import csv

start = time.time()

cwdPath = os.getcwd()
lPath = cwdPath.replace("Source", "New_data")

data_filename = lPath + "/data.csv"
data_raw_data = open(data_filename, 'rt')
    
data_reader = csv.reader(data_raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
data_temp = list(data_reader)
data = np.array(data_temp).astype('float')






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


#DATA HAS SAME ORDER AS SELECTED TAGS

indexes = np.arange(182)
sorted_indexes = list()

for i in range(len(selected_tags_all)):
    if selected_tags_all[i].find(tags[0]) >= 0:
        sorted_indexes.append( (indexes[i], 0) )
    else:
        sorted_indexes.append( (indexes[i], 1) )


#shuffling the data set
np.random.shuffle(sorted_indexes)

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


#selecting examples for each data set
train_set_indexes = sorted_indexes[0 : train_set_size]

dev_set_indexes = sorted_indexes[train_set_size : train_set_size + dev_set_size]

test_set_indexes = sorted_indexes[train_set_size + dev_set_size : data_set_size]

train_set = list()
train_labels = list()

for i in range(len(train_set_indexes)):
    
    train_labels.append(train_set_indexes[i][1])
    
    train_set.append(data[train_set_indexes[i][0]])

train_set = np.vstack(train_set)
train_set = train_set.T

#normalizing the data
max_value = np.max(train_set)
min_value = np.min(train_set)
train_set = (1 / (max_value - min_value) ) * (train_set - min_value)


train_labels = np.hstack(train_labels)
train_labels = train_labels.reshape(1, train_set_size)

#saving labels
cwdPath = os.getcwd()
cwdPath = cwdPath.replace("Source", "Labels")
cwdPath = cwdPath + "/labels.csv"
np.savetxt(cwdPath, train_labels, delimiter=',')


#saving training set
cwdPath = os.getcwd()
cwdPath = cwdPath.replace("Source", "NN_Data")
cwdPath = cwdPath + "/nn_data.csv"
np.savetxt(cwdPath, train_set, delimiter=',')


