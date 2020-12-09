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

#add "/" before the names of data for concatenation with cwdPath
for i in range(len(all_signals_list)):
    all_signals_list[i] = "/" + all_signals_list[i]


our_file = open('outlabels', 'w')
our_file.write("\n")
our_file.close()

our_file = open('outlabels', 'a')

tags = set()

for our_str in all_signals_list:
    
    tag = int(our_str.split('_')[len(our_str.split('_')) -1].split('.')[0].replace('tag', ''))
    tags.add(tag)
    our_file.write(str(tag) + '\n')

our_file.close()

print(tags)

end = time.time()
print("Elapsed time [s]: " + str(end- start))