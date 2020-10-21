import mne
import os

cwdPath = os.getcwd()
cwdPath = cwdPath.replace("Source", "Data")
fileItem = "/imagined_speech_MM05_00_tag1.raw.fif"
fName = cwdPath + fileItem
raw = mne.io.read_raw_fif(fName)
raw.plot(duration=5, n_channels=30, block=True)
print(raw)

