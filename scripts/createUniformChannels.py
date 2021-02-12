#
# createUniformChannels.py - Takes as input our patient pickle files
# and returns a copy of these files in the output directory, with 
# another array, 'uniform_channel_values', filled with a subset of the
# eeg data that has a uniform structure (specifically the amount and 
# order of channels) across all th patient pickle files. 
#
# The channels we are using in the UNIFORM_LABELS variable, was decided
# upon based on analysis of the getUniformChannelInfo.py script's output.
#
# NOTE: This script was replaced by updatedCreateUniformChannelsResampled.py
#


import sys
import pickle
from tqdm import tqdm
import numpy as np
import glob


PICKLE_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]


UNIFORM_LABELS = [ 	['EEG FP1-REF','EEG FP1-LE'], \
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
					['EEG PZ-REF', 'EEG PZ-LE'],  \
					['EEG EKG-REF', 'EEG EKG1-REF', 'EEG EKG-LE'] ]


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
		inserted = 0
		for l in channel:
			if l in labels:
				label_set_indicies.append( labels.index(l) )
				inserted = 1
				continue

		if inserted == 0:
			if "EKG" in channel[0]:
				ZERO_EKG = True
				label_set_indicies.append(0)
			else:
				print(f"In {pickles[i]}, couldn't find {channel}.")
			
	cur_pickle["uniform_channel_labels"] = UNIFORM_LABELS
		
	cur_pickle["uniform_channel_values"] = cur_pickle['channel_values'][label_set_indicies, :]
	
	if ZERO_EKG:
		cur_pickle["uniform_channel_values"][-1, :] = np.zeros(cur_pickle["uniform_channel_values"].shape[1])

	# if the sampling frequency is not uniform, just ignore the file
	freqs = cur_pickle['channel_sampling_frequencies']
	freqs = [freqs[i] for i in label_set_indicies]

	if len(freqs)*int(freqs[0]) != sum([int(i) for i in freqs]):
		print(f"ERROR: {pickles[i]} did not have a uniform sampling frequency")
		continue

	sampling_freq = float( freqs[0] )

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

