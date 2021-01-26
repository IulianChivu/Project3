import mne
import os
import matplotlib.pyplot as plt
import numpy as np

cwdPath = os.getcwd()
cwdPath = cwdPath.replace("Source", "Data")
fileItem = "/imagined_speech_MM05_10_tag3.raw.fif"
fName = cwdPath + fileItem
raw = mne.io.read_raw_fif(fName)

#plotting without "block=true" beacuse it blocks execution in Linux
raw.plot(duration=5, n_channels=30)
print(raw.info)

#number of samples
print(len(raw))

#extracting the samples of a certain signal as a tuple
raw_selection = raw["CB2", 0:len(raw)]


print(type(raw_selection))
#print(raw_selection[1])


x = raw_selection[1]
#samples array must be transposed due to the way it is stored 
y = raw_selection[0]

MEAN_LEN = 10
window_len = 1000
WINDOW_NUMBER = 7
OVERLAP = 500
START_SAMPLE = 0
SEQUENCE_LENGTH = 1000

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

y = raw_selection[0].T
#plottin one of the signals
plt.figure()
plt.title("CB2")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [V]")
plt.plot(x, y)

plt.figure()
plt.plot(obs_line.T)