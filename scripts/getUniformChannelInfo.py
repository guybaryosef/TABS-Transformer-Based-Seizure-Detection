# getUniformChannelInfo - Takes in a directory filled with
# our pickle files and returns a python dictionary containing
# the numer of times each channel appeared across all the pickle
# files. This is a precursor step necessary for us to construct
# the uniform channels.
#
# To run:
# 	python  getUniformChannelInfo.py  PICKLE_DIRECTORY
#


import sys
import pickle
from tqdm import tqdm
import numpy as np
import glob


PICKLE_DIR = sys.argv[1]

pickles = glob.glob(PICKLE_DIR + "/*")

num_channels = {}
channel_names = {}


bar = tqdm(range(len(pickles)))
for i in bar:

	pickle_name = pickles[i].split("/")[-1].split(".")[0]

	try:
		cur_pickle = pickle.load(open(pickles[i], 'rb'))

		# if the sampling frequency is not uniform, just ignore the file
		labels = cur_pickle['channel_labels']

		num_channels[len(labels)] = num_channels.get(len(labels), 0) + 1
		
		if sum([1 for s in labels if "EKG" in s]) == 0:
			print(pickles[i])
		
		for l in labels:
			channel_names[l] = channel_names.get(l, 0) + 1
	except:
		pass

	bar.set_description(f"current file {i+1}: {pickles[i]}")

sorted(num_channels.items(), key=lambda s: s[0])
sorted(channel_names.items(), key=lambda s: s[0])

print(f"num_channels:")
for key,val in num_channels.items():
	print(f"{key}: {val}")

print(f"\n\n\nchannel_names:")
for key,val in channel_names.items():
	print(f"{key}: {val}")

o = open(PICKLE_DIR+"_CHANNEL_DATA_INFO.txt", 'w')
o.write("num_channnels:\n")
for key,val in num_channels.items():
	o.write(f"{key}: {val}\n")

o.write(f"\n\n\nchannel_names:\n")
for key,val in channel_names.items():
	o.write(f"{key}: {val}\n")

