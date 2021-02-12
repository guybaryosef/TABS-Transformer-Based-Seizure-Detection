#
# getSamplingFrequencies.py - Create a map of {samplingFrequencies : amount}
# from a set of pickled data in the hard-coded 'data' variable directory.
#


import glob
import pyedflib
import numpy as np
import pickle
from tqdm import tqdm
import os
import sys


data = glob.glob("/mnt/pedomeister/eeg/contestData/trainPickles_v2/*.pkl")

freq_vals = {}
bar = tqdm(data)
for ele in bar:
	pkl = pickle.load(open(ele, 'rb'))

	cur = pkl["channel_sampling_frequencies"][0]	

	if cur in freq_vals:
		freq_vals[ cur ] += 1
	else:
		freq_vals[cur] = 1

print(freq_vals)

pickle.dump(freq_vals, open("total_freq_values", "wb"))

