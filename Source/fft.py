import mne
import os
import matplotlib.pyplot as plt
import numpy as np

cwdPath = os.getcwd()
cwdPath = cwdPath.replace("Source", "Data")
allSignalsList = os.listdir(cwdPath)
#Add "/" before the names of data and concatenatenate with cwdPath
for i in range(len(allSignalsList)):
    allSignalsList[i] = "/" + allSignalsList[i]
    
fileItem = allSignalsList[1]
fName = cwdPath + fileItem
raw = mne.io.read_raw_fif(fName)

#Store all data names from cwd in a list


#print(raw.info)
#number of samples
#print(len(raw))

#extracting the frist channel
channelName = raw.ch_names[0]

#extracting the samples of a certain signal as a tuple
raw_selection = raw[channelName, 0:len(raw)]

#The sampling frequency
Fs = 10**3

#plottin one of the signals
plt.figure()
plt.title(channelName)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [V]")
time = raw_selection[1]
#samples array must be transposed due to the way it is stored 
amplitude = raw_selection[0].T
plt.plot(time, amplitude)

#Extracting the signal's samples
samples = raw_selection[0]

#Performing FFT for the entire signal
yf = np.fft.fft(samples)
yf = yf.T
N = samples.size

#Genrating the frequency bins
freqRes = Fs/N
freqs = np.arange(0, Fs/2, freqRes)

#Plotting the entire spectrum of the signal
plt.figure()
plt.title("Spectrum of " + channelName)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude (Energy) [J]")
plt.plot(freqs, np.abs(yf[0:N//2]))


windowN = 1000
yshort = samples[:, 0:windowN]
print(type(yshort))
yfShort = np.fft.fft(yshort)
yfShort = yfShort.T
xfShort = np.arange(0, 1000, 1)
plt.figure()
plt.plot(np.abs(yfShort))
