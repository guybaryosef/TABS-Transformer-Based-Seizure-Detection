# runPostProcessing.py - Our helper script used to postprocess a model output file. 
#
# This script is not used for the contest data, which uses the newer version (v3.3.2)
# of the evaluation script. This script uses the older version of the evaluation script that
# we used originally for training and model building.
# 
# To run:
# 	python runPostProcessing.py MODEL_OUTPUT_FILE OUTPUT_DIRECTORY THRESHOLD
#


import torch
from postprocessing import PostProcessing 
import pickle
import sys
import numpy as np
import os


if __name__ == '__main__':
	print(f"Running posprocessing script with params: input_file:{sys.argv[1]} | output_dir:{sys.argv[2]} | thresh:{sys.argv[3]}")
	vec = torch.load(sys.argv[1])

	pp_output_dir = sys.argv[2]

	if len(sys.argv) > 3:
		threshold = float(sys.argv[3])
	else:
		threshold = 0.8

	pickle_file_name = sys.argv[1].split("/")[-1].split(".pt")[0]
	
	pickle_name = "/zooper1/jped/eeg/pickledEdfsValidationData/" + pickle_file_name
	freq = pickle.load(open(pickle_name, "rb"))['channel_sampling_frequencies'][0]

	try:
		pred_dir = pp_output_dir+"/prediction"
		os.mkdir(pred_dir)
	except:
		pass
	
	
	softmax_vals     = PostProcessing.softmaxing(vec[1:3,:])	
	new_vec          = PostProcessing.smoothBinary(softmax_vals[1,:], softmax_vals.shape[-1], 1)
	#PostProcessing.smoothBinary(thresholded_vec, 5000, 1)
	output_vec = PostProcessing.justThresh(new_vec, threshold)
	#thresholded_vec = PostProcessing.thresholding(vec[1:3,:], threshold)

	PostProcessing.createTseFile(output_vec, pickle_file_name, pred_dir, freq)
		
