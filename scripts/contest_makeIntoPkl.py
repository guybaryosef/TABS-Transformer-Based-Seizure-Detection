#
# contest_makeIntoPkl.py - Take the data from the TUH EEG corpus and merge it
# into pickled files. This merges a patient's (sometimes multiple)
# EEG data and doctors notes into a single binary file.
#
# This file is for contest data.
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
ANNOTATION_FILE= "/mnt/pedomeister/eeg/newTuh_eeg_1.5.1/_DOCS/ref_dev.txt"

edf_files        = sorted(glob.glob(FILE_DIRECTORY + "/*.edf"))
text_files       = sorted(glob.glob(FILE_DIRECTORY + "/*.txt"))


with open(ANNOTATION_FILE, 'r+') as annotation_file:
    lines = annotation_file.readlines()

    i = 0
    while i < len(lines):

        cur_line = lines[i]
        i += 1
        parsed_line = cur_line.split(" ")

        cur_file_name = parsed_line[0]
           
        print(f"File {i}: {cur_file_name}") 

        # get annotations    
        annotation_list = []
        
        annotation_dic1 = {}
        annotation_dic1["start"]       = parsed_line[1]
        annotation_dic1["end"]         = parsed_line[2]
        annotation_dic1["label"]       = parsed_line[3]
        annotation_dic1["probability"] = parsed_line[4]
        annotation_list.append(annotation_dic1)
        
        while i < len(lines):    
            next_line = lines[i].split(" ")
            if next_line[0] == cur_file_name:
               i += 1
               annotation_dic1 = {}
               annotation_dic1["start"]       = next_line[1]
               annotation_dic1["end"]         = next_line[2]
               annotation_dic1["label"]       = next_line[3]
               annotation_dic1["probability"] = next_line[4]
               annotation_list.append(annotation_dic1)
            else:
               break

        # get text file
        text_name = "_".join(cur_file_name.split("_")[:-1])
        
        text_file_path = [s for s in text_files if text_name in s]    
        text_file_path = text_file_path[0]


        # get edf file
        current_edf_file = [s for s in edf_files if cur_file_name in s]
        current_edf_file = current_edf_file[0]
        
        f = pyedflib.EdfReader(current_edf_file)
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()
        sigbufs = np.zeros((n, f.getNSamples()[0]))
        sampling_freq = [] 
        for j in np.arange(n):
            sig = f.readSignal(j)
            sampling_freq.append(f.getSampleFrequency(j))
            sigbufs[j, 0:sig.size] = sig 

        # create pickled object
        session_object = {}

        session_object["doctors_notes"] = open(text_file_path, encoding='ISO-8859-1').read()
        
        session_object["channel_labels"] = signal_labels
        session_object["channel_values"] = sigbufs
        session_object["annotations"] = annotation_list
        session_object["channel_sampling_frequencies"] = sampling_freq

        pickle.dump(session_object, open("{}/{}.pkl".format(OUTPUT_DIR,cur_file_name), "wb"))

        
