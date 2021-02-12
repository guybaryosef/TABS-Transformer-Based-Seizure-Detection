# contest_runPostProcessing.py - Runs the contest version of the runPostProcessing.py script.
# It follows the same postprocessing steps as runPostProcessing.py, but the output is in a
# different format- requiered for compatability with the newer v3.3.2 evaluation script.
#
# To run:
# 	- python contest_runPostProcessing.py REFERENCE_FILE MODEL_OUTPUT_PATH POST_PROCESSING_DIRECTORY [THRESHOLD] 
#
# NOTE:
#   - If no threshold is inputted, defaults to 0.8.
#


import torch
from postprocessing import PostProcessing 
import pickle
import sys
import numpy as np
import os
import glob


if __name__ == '__main__':
	# process program arguments
	ref_file	  = open(sys.argv[1], 'r')
	model_output_path = sys.argv[2]
	pp_output_dir     = sys.argv[3]
	if len(sys.argv) > 4:
		threshold = float(sys.argv[4])
	else:
		threshold = 0.8
	print(f"Running posprocessing script with params: refence_file: {sys.argv[1]} | input_dir:{model_output_path} | output_dir:{pp_output_dir} | thresh:{threshold}")

	# create output file
	output_file = open(pp_output_dir + "/hyp.txt", 'w+')

	# go through reference file, opening relevant pickle from input dir each time
	unidentified_files_list = []
	cur_patient_name = None
	for line in ref_file:
		parsed_line = line.split(" ")
		parsed_name = parsed_line[0]
		if parsed_name == cur_patient_name:	# already evaluated this patient file
			continue
		else:
			cur_patient_name = parsed_name

		# get the patient file's sampling frequency	
		pickle_name = "/mnt/pedomeister/eeg/contestData/validatePickles_v2/" + cur_patient_name + ".pkl"
		freq = pickle.load(open(pickle_name, "rb"))['channel_sampling_frequencies'][0]
	
		# load the model outputs	
		try:
			file_name = glob.glob(model_output_path + "/" + cur_patient_name + ".*")[0]
		except:
			unidentified_files_list.append(cur_patient_name)
			continue	
		print(f"current file name: {file_name}")
		cur_model_output = torch.load(file_name)



		softmax_vals     = PostProcessing.softmaxing(cur_model_output[1:3,:])	
		new_vec          = PostProcessing.smoothBinary(softmax_vals[1,:], softmax_vals.shape[-1], 1)
		output_vec = PostProcessing.justThresh(new_vec, threshold)
		output_vec = PostProcessing.smoothBinaryPost(output_vec, 1000, 1)
		output_vec = PostProcessing.smoothBinaryPost(output_vec, 50, 1)

		seiz_arr = PostProcessing.contest_createTseFile(output_vec, freq)
		
		# write the tse seizure contents into the output file
		for seiz in seiz_arr:
			output_file.write(parsed_name + " " + str(seiz["begin"]) + " " + str(seiz["end"]) + " " + str(seiz["confidence"]) + "\n")

	print(f"Unidentified patient files: {unidentified_files_list}")
