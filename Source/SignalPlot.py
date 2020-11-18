import mne
import os
import matplotlib.pyplot as plt

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
print(raw_selection[1])

#plottin one of the signals
plt.figure()
plt.title("One of the signals")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [V]")
x = raw_selection[1]
#samples array must be transposed due to the way it is stored 
y = raw_selection[0].T
plt.plot(x, y)
