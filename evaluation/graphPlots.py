# graphPlots.py - A short script to run the graphing function
# on one of our model outputs.
#
# To run:
#	python  graphPlots.py  PATIENT_FILE  THRESHOLD  WINDOW_SIZE  VOTE_COUNT
# 

from postprocessing import PostProcessing
import sys
import torch


pt_file=torch.load(sys.argv[1])

PostProcessing.keeneGraph(pt_file[0,:], pt_file[1:3,:], float(sys.argv[2]), sys.argv[1], int(sys.argv[3]), int(sys.argv[4]))
