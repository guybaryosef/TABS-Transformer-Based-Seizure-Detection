# testModel.py - Runs a model on a set of data, saving each input file's
# output as a 2D matrix with the width being the length of the input file, and 3 rows:
# The first row is the ground truth and the 2nd and 3rd row are the
# model's output (2nd row - seizure confidence, 3rd row - background confidence).
#
# To run: 
#	python3 testModel.py MODEL_FILE MODEL_WEIGHTS MODEL_OUTPUT_DIR DEVICE_NUMBER
# 
# The test_list variable provides the directory of files to test.
#


import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import pickle
import numpy as np
import sys
import os
from enum import Enum, auto
from importlib import import_module

from dataLoaders import OvlpEvalDataLoader,ChannelSelection,testDataLoader
from postprocessing import PostProcessing


class Y_Placement(Enum):
        MIDDLE = auto()
        END    = auto()

WINDOW_SIZE      = 300
THRESHOLD_WINDOW = 10
BATCH_SIZE = 512
device = sys.argv[4]
print("device ", device)
#TEST_LIST   = "/zooper1/jped/eeg/models/dataLists/validation_smallest.txt"
TEST_LIST   = "/zooper1/jped/eeg/models/dataLists/contestValidationPicklesList_resampled.txt"

class EvalBuilder():
        def __init__(self, test_generator, window_size=WINDOW_SIZE):
                self.data = test_generator        
                
                self.window_size = window_size
                self.half_window = window_size//2


        def setUp(self, model, y_placement=Y_Placement.MIDDLE):
                
                # iterate through each patient file
                
                for i,l in enumerate(self.data):
                        x    = l[0].cuda(DEVICE).squeeze(-2)
                        tmp_x = torch.zeros((x.shape[0], x.shape[1], x.shape[2] + 300))
                        tmp_x[:, :, 150:x.shape[2]+150]  = x
                        x = tmp_x
                        # shape of x:  20,time
                        # change it to: batch_size, 20
                        y    = l[1].cuda(DEVICE)
                        freq = l[2].cuda(DEVICE).item()
                        file_name = './'+ sys.argv[3] + '/' + l[3][0].split('/')[-1] + '.pt'                 
                        print(f"Evaluating file number: {i} | with y shape: {y.shape} | x_shape: {x.shape} | {file_name}")
        
                        batch_model_input = torch.zeros(BATCH_SIZE, x.shape[1], self.window_size).cuda(DEVICE) 
                        pred_vec = torch.zeros(2, y.shape[1])                       
                        
                        if y_placement == Y_Placement.MIDDLE:
                                # get our predictions vector
                                for x_ind in range(self.half_window, x.shape[-1]-self.half_window):
                                        if (x_ind-self.half_window) % BATCH_SIZE == 0 and x_ind != self.half_window:
                                                val = model(batch_model_input)
                                                batch_model_input = torch.zeros(BATCH_SIZE, x.shape[1], self.window_size).cuda(DEVICE) 
                                                pred_vec[:, x_ind-BATCH_SIZE-self.half_window:x_ind-self.half_window] = torch.t(val).detach()      
                                        batch_model_input[x_ind % BATCH_SIZE,:,:] = x[:,:, x_ind-self.half_window: (x_ind+self.half_window)]
                                
                                # have to get the leftovers
                                if torch.sum(batch_model_input).item() != 0:
                                        leftover_batch = (x_ind - self.half_window) % BATCH_SIZE
                                        val = model(batch_model_input[:leftover_batch + 1, :])
                                        pred_vec[:, x_ind-leftover_batch-self.half_window:x_ind-self.half_window + 1] = torch.t(val).detach()      


                        elif y_placement == Y_Placement.END:
                                # get our predictions vector
                                for x_ind in range(self.window_size,x.shape[-1]):
                                        if (x_ind - self.half_window) % BATCH_SIZE == 0 and x_ind != self.half_window:
                                                val = model(batch_model_input)
                                                pred_vec[:, x_ind-BATCH_SIZE:x_ind] = torch.t(val)        
                                        batch_model_input[x_ind % BATCH_SIZE,:,:] = x[:,:, x_ind-self.window_size:x_ind]
                                
                        pred_vec = pred_vec.detach()        
                        save_vec = torch.zeros(3 ,y.shape[1])        
                        save_vec[0,:]    = y
                        save_vec[1:3, :] = pred_vec        
                        torch.save(save_vec, file_name)

# driver
if __name__=='__main__':
        try:
                os.mkdir(sys.argv[3])
        except:
                pass
        DEVICE = torch.device("cuda:{}".format(device))

        # load in some model
        print("Loading Model checkpoint...", end=' ')
        Model = getattr(import_module(sys.argv[1]), 'EnsembleModel')
        model = Model()
        model.cuda(DEVICE)
        model.load_state_dict( torch.load(sys.argv[2], map_location=DEVICE) )
        model.eval()

        # test its score        
        print("done.\nLoading test generator...", end=" ")
        test_set = OvlpEvalDataLoader(TEST_LIST, 20)
        test_generator = data.DataLoader(test_set, batch_size=1)

        evalBuilder = EvalBuilder(test_generator)
        evalBuilder.setUp(model, Y_Placement.MIDDLE)



