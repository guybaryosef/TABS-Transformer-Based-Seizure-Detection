#
# makeIntoPkl.py - Take the data from the TUH EEG corpus and merge it
# into pickled files. This merges a patient's (sometimes multiple)
# EEG data and doctors notes into a single binary file.
#
# This file is for non-contest data.
#


import glob
import pyedflib
import numpy as np
import pickle
from tqdm import tqdm
import os
import sys


print(f"input dir: {sys.argv[1]} | output_dir: {sys.argv[2]}")
FILE_DIRECTORY = sys.argv[1]
OUTPUT_DIR     = sys.argv[2]

edf_files        = sorted(glob.glob(FILE_DIRECTORY + "/*.edf"))
annotation_files = sorted(glob.glob(FILE_DIRECTORY + "/*.tse*"))
text_files       = sorted(glob.glob(FILE_DIRECTORY + "/*.txt"))

file_pointer = 0

for i in tqdm(range(len(text_files))):
	
	text_file_path = text_files[i]	

	text_name = text_file_path.split("/")[-1].split(".")[0]

	while file_pointer < len(edf_files):
		current_edf_file = edf_files[file_pointer]
		full_file_name = current_edf_file.split("/")[-1].split(".")[0]
		current_file_name = "_".join(current_edf_file.split("/")[-1].split("_")[:-1])
		if current_file_name != text_name:
			break
		
		f = pyedflib.EdfReader(current_edf_file)
		n = f.signals_in_file
		signal_labels = f.getSignalLabels()
		
		sigbufs = np.zeros((n, f.getNSamples()[0]))
		sampling_freq = [] 
		for j in np.arange(n):
			sig = f.readSignal(j)
			sampling_freq.append(f.getSampleFrequency(j))
			sigbufs[j, 0:sig.size] = sig 

		# parse annotation
		f = open(annotation_files[file_pointer]).read()
		f = f.split("\n")[2:]	
		annotation_list = []

		for i, name in enumerate(f):
			splits = name.split(" ")
			if len(splits) <4:
				continue
			annotation_dic = {}
			annotation_dic["start"]       = splits[0]
			annotation_dic["end"]         = splits[1]
			annotation_dic["label"]       = splits[2]
			annotation_dic["probability"] = splits[3]
			annotation_list.append(annotation_dic)

		# create pickled object
		session_object = {}

		session_object["doctors_notes"] = open(text_file_path, encoding='ISO-8859-1').read()
		
		session_object["channel_labels"] = signal_labels
		session_object["channel_values"] = sigbufs
		session_object["annotations"] = annotation_list
		session_object["channel_sampling_frequencies"] = sampling_freq

		pickle.dump(session_object, open("{}/{}.pkl".format(OUTPUT_DIR,full_file_name), "wb"))

		file_pointer +=1
		
