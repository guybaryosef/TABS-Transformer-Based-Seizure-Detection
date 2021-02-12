# updatedCreateUniformChannelsResampled.py - Takes as input a directory filled with our
# patient pickle files, and adds to each of these pickles the following fields:
#   - uniform_channel_labels - list of 19 uniform channel names.
#   - uniform_channel_values - the data from the 19 uniform channels.
#
# This script also resamples all the uniform channels to 250hz.
#
# To run:
# 	python updatedCreateUniformChannelsResampled.py PICKLE_DIR OUTPUT_DIR
#


import sys
import pickle
from tqdm import tqdm
import numpy as np
import glob
from scipy.signal import resample

PICKLE_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]


UNIFORM_LABELS = [ ['EEG FP1-REF','EEG FP1-LE'], \
		   ['EEG FP2-REF','EEG FP2-LE'], \
                   ['EEG F3-REF', 'EEG F3-LE'],  \
	    	   ['EEG F4-REF', 'EEG F4-LE'],  \
		   ['EEG C3-REF', 'EEG C3-LE'],  \
		   ['EEG C4-REF', 'EEG C4-LE'],  \
		   ['EEG P3-REF', 'EEG P3-LE'],  \
		   ['EEG P4-REF', 'EEG P4-LE'],  \
		   ['EEG O1-REF', 'EEG O1-LE'],  \
		   ['EEG O2-REF', 'EEG O2-LE'],  \
		   ['EEG F7-REF', 'EEG F7-LE'],  \
		   ['EEG F8-REF', 'EEG F8-LE'],  \
		   ['EEG T3-REF', 'EEG T3-LE'],  \
		   ['EEG T4-REF', 'EEG T4-LE'],  \
		   ['EEG T5-REF', 'EEG T5-LE'],  \
		   ['EEG T6-REF', 'EEG T6-LE'],  \
		   ['EEG FZ-REF', 'EEG FZ-LE'],  \
		   ['EEG CZ-REF', 'EEG CZ-LE'],  \
		   ['EEG PZ-REF', 'EEG PZ-LE'] ]

UNIFORM_RESAMPLE_FREQ = 250

pickles = glob.glob(PICKLE_DIR + "/*.pkl")

bar = tqdm(range(len(pickles)))
for i in bar:
	bar.set_description(f"current file: {pickles[i]}")

	cur_pickle = pickle.load(open(pickles[i], 'rb'))

	# create the uniform channel value matrix
	labels = cur_pickle['channel_labels']
	label_set_indicies = []

	ZERO_EKG = False
	for channel in UNIFORM_LABELS:
		for l in channel:
			if l in labels:
				label_set_indicies.append( labels.index(l) )
				continue

	if len(label_set_indicies) != 19:
		printf("ERROR: missing labels in file: {pickles[i]}!")
	
	cur_pickle["uniform_channel_labels"] = UNIFORM_LABELS		
	cur_pickle["uniform_channel_values"] = cur_pickle['channel_values'][label_set_indicies, :]	

	# if the sampling frequency is not uniform, just ignore the file
	freqs = cur_pickle['channel_sampling_frequencies']
	freqs = [freqs[i] for i in label_set_indicies]

	if len(freqs)*int(freqs[0]) != sum([int(i) for i in freqs]):
		print(f"ERROR: {pickles[i]} did not have a uniform sampling frequency")
		continue

	sampling_freq = float( freqs[0] )


	###### resampling ######
	seconds_in_file =  cur_pickle["uniform_channel_values"].shape[1] / sampling_freq
	new_vec_len 	=  int(UNIFORM_RESAMPLE_FREQ * seconds_in_file)

		
	cur_pickle["uniform_channel_values"] = resample(cur_pickle["uniform_channel_values"], new_vec_len, axis=1)

	cur_pickle["channel_sampling_frequencies"] = [UNIFORM_RESAMPLE_FREQ]*cur_pickle["channel_sampling_frequencies"][0]
	sampling_freq = float(cur_pickle["channel_sampling_frequencies"][0] )
	###### resampling ######

	# create labels vector
	annotations = cur_pickle['annotations']

	labels_vector = np.zeros(cur_pickle["uniform_channel_values"].shape[1])	
	for cur_ann in annotations:
		start = int( float(cur_ann['start'])*sampling_freq + 0.5 )
		end   = int( float(cur_ann['end'])*sampling_freq   + 0.5 )
		
		if cur_ann['label'] == 'bckg':
			labels_vector[start:end] = 0
		elif 'sq' in cur_ann['label'] or 'seiz' in cur_ann['label']:
			labels_vector[start:end] = 1
		else:
			print(f"ERROR: annotation label is: {cur_ann['label']}. ")

	cur_pickle["labels_vector"] = labels_vector

	if labels_vector.shape[0] != cur_pickle["uniform_channel_values"].shape[1]:
		print(f"ERROR: Invalide x,y values: x shape: {cur_pickle['uniform_channel_values'].shape} | y shape: {labels_vector.shape}.")

	pickle_name = pickles[i].split("/")[-1].split(".")[0]
	pickle.dump(cur_pickle, open("{}/{}.pkl".format(OUTPUT_DIR, pickle_name), "wb"))

