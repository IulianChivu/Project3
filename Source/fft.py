import mne
import os
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import numpy as np

cwdPath = os.getcwd()
cwdPath = cwdPath.replace("Source", "Data")
fileItem = "/imagined_speech_MM05_10_tag3.raw.fif"
fName = cwdPath + fileItem
raw = mne.io.read_raw_fif(fName)


#print(raw.info)

#number of samples
#print(len(raw))

#extracting the samples of a certain signal as a tuple
raw_selection = raw["CB2", 0:len(raw)]


#print(type(raw_selection))
#print(raw_selection[1])

#plottin one of the signals
plt.figure()
plt.title("One of the signals")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [V]")
time = raw_selection[1]
#samples array must be transposed due to the way it is stored 
amplitude = raw_selection[0].T
plt.plot(time, amplitude)


yf = np.fft.fft(raw_selection[0])
yf = yf.T
print("yf shape = " + str(yf.shape))
print("yf type = " + str(type(yf)))
N = raw_selection[0].size
print("type N = " + str(type(N)))


freqs = np.linspace(0, 500, N//2)
print("freqs type = " + str(type(freqs)))

plt.figure()
plt.plot(freqs, np.abs(yf[0:N//2]))


yshort = raw_selection[0].T[0:100].T
print(type(yshort))
yfShort = np.fft.fft(yshort)
yfShort = yfShort.T
print(np.abs(yfShort))
xfShort = np.arange(0, 1000, 10)
plt.figure()
plt.plot(xfShort, np.abs(yfShort))
