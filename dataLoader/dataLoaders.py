#
# A file containing the different DataLoader objects used throughout our project:
#  - eegDataLoader 	(original - not used anymore)
#  - fastDataLoader	(for training)
#  - testDataLoader	(for validation)
#  - OvlpEvalDataLoader (for evaluation)
#


import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import pickle
import glob
import math
import tqdm
import numpy as np
import sys

UNIFORM_CHANNELS = True

class eegDataLoader(data.Dataset):
    def __init__(self, text_file, window_size, old_loader=True):
        self.patient_list = open(text_file, "r").read().split("\n")
        self.patient_list.pop()
        self.window_size = window_size
        self.half_window = int(self.window_size/2)
        self.cumSum = []
        self.old_loader = old_loader
        sizes = []
        print("starting cumsum")
        for patient in tqdm.tqdm(self.patient_list):
            sizes.append(pickle.load(open(patient, 'rb'))['uniform_channel_values'].shape[-1]-self.window_size)
        self.cumSum = np.cumsum(sizes)
        print("done with cumsum")
    
    def __len__(self):
        return self.cumSum[-1]

    def __getitem__(self, index):
        def un_cum_sum(index):
            for i, border in enumerate(self.cumSum):
                if border>index:
                    if i == 0:
                        return i, index
                    return i, (index-self.cumSum[i-1])
        patient_index, start = un_cum_sum(index)
        pkl_obj = pickle.load(open(self.patient_list[patient_index], 'rb'))
        if self.old_loader:
            mid_point = start + self.half_window
            input_tensor = torch.tensor(pkl_obj['uniform_channel_values'][0:19, (mid_point-self.half_window):(mid_point+self.half_window)])
            output_classification = torch.tensor(pkl_obj['labels_vector'][0,mid_point]) 
        else:
            start += self.window_size
            input_tensor = torch.tensor(pkl_obj['uniform_channel_values'])[0:25, start-self.window_size:start]
            output_classification = torch.tensor(pkl_obj['annotations'][start])
        return input_tensor, output_classification


class fastDataLoader(data.Dataset):
        def __init__(self, text_file, window_size, num_of_files, epoch_size=0, old_loader=True):
                self.patient_list = open(text_file, 'r').read().split('\n')
                self.patient_list.pop()    # remove last empty entry
                self.patient_len  = len(self.patient_list)
                if old_loader:
                    self.parser = self.parsePickle
                else:
                    self.parser = self.parseNewPickle

                self.window_size = window_size
                self.half_window = int(self.window_size/2)

                # get first collection of files
                self.f_data    = [0 for _ in range(num_of_files)]  # a list of the open pickles
                self.f_max     = [0 for _ in range(num_of_files)]  # a list with the max number of samples to get from each open pickle
                self.f_sampled = [0 for _ in range(num_of_files)]  # a list of the current amount of samples aquired from each open pickle

                if epoch_size == 0:
                        self.epoch_size=self.patient_len
                else:
                        self.epoch_size = epoch_size
                self.n_files   = num_of_files
                self.cur_files_ind = torch.randint(0, self.patient_len, (self.n_files,1))
                for i,cur_file in enumerate(self.cur_files_ind):
                        self.loadFile(i, cur_file)

        def __len__(self):
                return self.epoch_size

        def __getitem__(self, index):           
                """
                Gets the x,y pair from the specific open pickle file specified by index, and then
                does open files upkeep (checks if we sampled the max amount from the pickle and 
                if so opens a new pickle instead.
                """
                index = torch.randint(0,self.n_files, (1,)).item()
                x,y = self.parser(index)

                ### open pickles upkeep
                self.f_sampled[index] += 1
                if self.f_sampled[index] == self.f_max[index]:
                        # get new patient file index (not from current sample)
                        new_patient_pkl = self.cur_files_ind[0]
                        while new_patient_pkl in self.cur_files_ind:
                                new_patient_pkl = torch.randint(0,self.patient_len, (1,)).item()

                        self.loadFile(index, new_patient_pkl)

                return x,y

        def parsePickle(self, index):        
                """
                Gets the x,y pair from the patient pickle file with the specified index in the patient_list.
                """
                cur_pkl = self.f_data[index]
                rand_samp = torch.randint(self.half_window, cur_pkl['uniform_channel_values'].shape[1]-self.half_window, (1,)).item()
                if UNIFORM_CHANNELS:
                        x = torch.tensor(cur_pkl['uniform_channel_values'][:, (rand_samp-self.half_window):(rand_samp+self.half_window) ])
                else:
                        x = torch.tensor(cur_pkl['channel_values'][:, (rand_samp-self.half_window):(rand_samp+self.half_window) ])
                        
                        # for comparison's sake, for the not uniform channels we will randomly
                        # choose the same amount of channels as the uniform list. Keep those 
                        # channels in the same order though
                        n_chans = cur_pkl['uniform_channel_values'].shape[0]
                        n_chans_total = cur_pkl['channel_values'].shape[0]      
                        chans = torch.randint(0, n_chans_total, (n_chans,1))
                        chans.sort()
                        x = x[chans,:]  
                y = torch.tensor(cur_pkl['labels_vector'].squeeze()[rand_samp]) 
                return x.float(),y
        
        def parseNewPickle(self, index):
                cur_pkl = self.f_data[index]
                start = torch.randint(self.window_size, cur_pkl['uniform_channel_values'].shape[1], (1,))
                input_tensor = torch.tensor(cur_pkl['uniform_channel_values'])[0:25, start-self.window_size:start]
                output_classification = torch.tensor(cur_pkl['annotations'][start])
                return input_tensor, output_classification

        def loadFile(self, arr_index, patient_index):
                pkl_file = pickle.load(open(self.patient_list[patient_index],'rb')) 
                self.f_data[arr_index]    = pkl_file
                self.f_max[arr_index]     = pkl_file['channel_values'].shape[1]//5      # The MAX number of elements to sample from a patient file
                self.f_sampled[arr_index] = 0


class testDataLoader(data.Dataset):
     def __init__(self, text_file, window_size, old_loader=True):
                self.patient_list = open(text_file, 'r').read().split('\n')
                self.patient_list.pop()    # remove last empty entry
                self.patient_len  = len(self.patient_list)
                self.window_size = window_size
                self.half_window = int(self.window_size/2)
                self.old_loader = old_loader
                self.totalSize = 0
                print("starting size")
                for patient in tqdm.tqdm(self.patient_list):
                    self.totalSize += (pickle.load(open(patient, 'rb'))['uniform_channel_values'].shape[-1]-self.window_size)
                print("done getting size")
                if old_loader:
                    self.parser = self.parsePickle
                else:
                    self.parser = self.parseNewPickle

                self.window_size = window_size
                self.half_window = int(self.window_size/2)

                self.cur_file_num = 0
                self.num_workers = 1
                self.cur_index = self.window_size
                self.cur_pkl, self.cur_pkl_size = self.loadFile(self.cur_file_num)

     def __len__(self):
                return self.totalSize

     def __getitem__(self, index):           
         """
           sequentially walk through the files
         """
         
         # are we past the size of this file? 
         if self.cur_index+1  >= self.cur_pkl_size:
              #reset and load in a new file
              self.cur_index = self.window_size
              self.cur_file_num = (self.num_workers + self.cur_file_num) % self.patient_len
              self.cur_pkl, self.cur_pkl_size = self.loadFile(self.cur_file_num)
         x,y = self.parser()
         tmp_x = torch.zeros(x.shape[0], self.window_size)
         tmp_x[:, :x.shape[1]] = x
         self.cur_index +=1
         return tmp_x,y

     def parsePickle(self):        
                """
                Gets the x,y pair from the patient pickle file with the specified index in the patient_list.
                """
                start = self.cur_index - self.window_size
                n_chans = self.cur_pkl['uniform_channel_values'].shape[0]
                #x = torch.zeros(n_chans, self.window_size)
                #y = torch.zeros(self.window_size)
                if UNIFORM_CHANNELS:
                        #tmp =  torch.tensor(self.cur_pkl['uniform_channel_values'][:, (start):(start+self.window_size) ])
                        x =  torch.tensor(self.cur_pkl['uniform_channel_values'][:, (start):(start+self.window_size) ])
                        #x[:,:tmp.shape[1]] = tmp               
                else:
                        x = torch.tensor(self.cur_pkl['channel_values'][:, (start):(start+self.window_size)])
                        
                        # for comparison's sake, for the not uniform channels we will randomly
                        # choose the same amount of channels as the uniform list. Keep those 
                        # channels in the same order though
                        n_chans_total = self.cur_pkl['channel_values'].shape[0]      
                        chans = torch.randint(0, n_chans_total, (n_chans,1))
                        chans.sort()
                        x = x[chans,:]
                #tmp_y = torch.tensor(self.cur_pkl['labels_vector'].squeeze()[start+self.half_window]) 
                labels_vector = torch.tensor(self.cur_pkl['labels_vector'].squeeze())
                y = torch.tensor(0)
                if (start+self.half_window) < labels_vector.shape[0]:
                    y = labels_vector[start+self.half_window]
                #print(tmp_y.size(), tmp_y)
                #y[:tmp_y] = tmp_y               
                return x.float(),y
 
     def parseNewPickle(self):
                input_tensor = torch.tensor(self.cur_pkl['uniform_channel_values'])[0:25, self.cur_index-self.window_size:self.cur_index]
                #output_classification = torch.tensor(self.cur_pkl['annotations'][self.cur_index])
                #return input_tensor, output_classification
                return input_tensor,torch.tensor([])
       
     def loadFile(self, patient_index):
                pkl_file = pickle.load(open(self.patient_list[patient_index],'rb')) 
                return pkl_file, pkl_file['channel_values'].shape[-1]





from enum import Enum, auto
class ChannelSelection(Enum):
        UNIFORM_CHANNEL   = auto()
        RANDOM_SELECTION  = auto()
        FIRST_X_CHANNELS  = auto()


class OvlpEvalDataLoader(data.Dataset):
        def __init__(self, text_file, channel_selection_var_=0):
                self.patient_list = open(text_file, 'r').read().split('\n')
                self.patient_list.pop()    # remove last empty entry
                self.patient_len  = len(self.patient_list)
                self.cur_file     = 0
                self.channel_sel_var = channel_selection_var_

        def __len__(self):
                return self.patient_len

        def __getitem__(self, index):           
                """
                Gets the x,y pair from the specific open pickle file specified by index, and then
                does open files upkeep (checks if we sampled the max amount from the pickle and 
                if so opens a new pickle instead.
                """
                if self.cur_file >= self.patient_len:
                        return 0
                x,y,freq,name = self.parsePickle()
                self.cur_file += 1

                return x, y, freq, name

        def parsePickle(self):        
                """
                Gets the x,y pair from the patient pickle file with the specified index in the patient_list.
                """
                f       = open(self.patient_list[self.cur_file],'rb')
                cur_pkl = pickle.load(f) 

                if  UNIFORM_CHANNELS:
                        x = torch.tensor(cur_pkl['uniform_channel_values'])
                else:
                        x = torch.tensor(cur_pkl['channel_values'])
                        n_chans_total = cur_pkl['channel_values'].shape[0]      
                        chans = torch.randint(0, n_chans_total, (self.channel_sel_var,1))
                        chans.sort()
                        x    = x[chans,:]  
                        
                y    = torch.tensor(cur_pkl['labels_vector']).view(-1)
                freq = cur_pkl['channel_sampling_frequencies'][0]
                return x.float(), y, freq, self.patient_list[self.cur_file]
        
