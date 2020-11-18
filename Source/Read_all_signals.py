import mne
import os
import matplotlib.pyplot as plt

cwdPath = os.getcwd()
cwdPath = cwdPath.replace("Source", "Data")

#store all data names from cwd in a list
allSignals_list = os.listdir(cwdPath)

#add "/" before the names of data for concatenation with cwdPath
for i in range(len(allSignals_list)):
    allSignals_list[i] = "/" + allSignals_list[i]

# =============================================================================
# #example of reading all raw data
# for i in range(len(allSignals_list)):
#     fName = cwdPath + allSignals_list[i]
#     raw = mne.io.read_raw_fif(fName)
# =============================================================================

print(allSignals_list)

#crearea bazei de date
class EEG_signal:
    chanel_name = 'empty'
    Delta = 0
    Theta = 0
    Alpha = 0
    Beta = 0
    Gamma = 0
    
class Data_set:
    name = 'empty'
    signal = []
