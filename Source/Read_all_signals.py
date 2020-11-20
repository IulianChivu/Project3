import mne
import os
import matplotlib.pyplot as plt
import numpy as np

cwdPath = os.getcwd()
cwdPath = cwdPath.replace("Source", "Data")

#store all data names from cwd in a list
all_signals_list = os.listdir(cwdPath)

#add "/" before the names of data for concatenation with cwdPath
for i in range(len(all_signals_list)):
    all_signals_list[i] = "/" + all_signals_list[i]

# =============================================================================
# #example of reading all raw data
# for i in range(len(allSignals_list)):
#     fName = cwdPath + allSignals_list[i]
#     raw = mne.io.read_raw_fif(fName)
# =============================================================================

#print(all)signals_list)

#example of reading 3 raw data
#store the data from the files in a list
raw = []
for i in range(3):
    fName = cwdPath + all_signals_list[i]
    raw.append(mne.io.read_raw_fif(fName))

print("lungimea setului de date:", len(raw))

samples = raw[0]


samples_list = list()
window_size = 1000
offset = 500
for cn in samples.ch_names:
    for index in range(0, 7):
        samples_list.append(samples[cn,  offset: offset+window_size][0])
        offset += 500
    


#number of samples
for i in range(3):
    print("nr of samples in the", i+1, "file =", len(raw[i]))
    
#print(type(raw[i]))

#extracting the samples of all signals as a list of tuples
raw_selection = []
for i in range(3):
    raw_selection.append(raw[i]["CB2", 0:len(raw[i])])



print(type(raw_selection[0])) #tuple
print(raw_selection[0][1]) #samples

#verify the number of samples
for i in range(3):
    print("nr of samples in the", i+1, "file =", len(raw_selection[i][1]))
    
print(type(raw_selection[0][1]))
print(raw_selection[0][1].shape)

#raw_selection[0][1] = np.delete(raw_slelection[0][1], [0:500])