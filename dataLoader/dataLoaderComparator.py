#
# dataLoaderComparator.py - Compares the speed of our different data loaders.
#


import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torchaudio
import pickle
import glob
import numpy as np
import math
import tqdm
import random
import time


UNIFORM_CHANNELS = False

class DataLoader1(data.Dataset):
	def __init__(self, text_file, window_size):
		self.patient_list = open(text_file, 'r').read().split('\n')
                # remove last empty entry
		self.patient_list.pop()
		self.window_size = window_size
		self.half_window = int(self.window_size/2)

	def __len__(self):
		return len(self.patient_list)

	def __getitem__(self, index):		
		pkl_obj = pickle.load(open(self.patient_list[index], 'rb'))

		rand_samp = int(np.round(np.random.uniform(self.half_window, pkl_obj['channel_values'].shape[1]-self.half_window)))
		
		if UNIFORM_CHANNELS:
			x = torch.tensor(pkl_obj['uniform_channel_values'][:,rand_samp-self.half_window:rand_samp+self.half_window])
		else:
			x = torch.tensor(pkl_obj['channel_values'][:,rand_samp-self.half_window:rand_samp+self.half_window])
			
			# for comparison's sake, for the not uniform channels we will randomly
			# choose the same amount of channels as the uniform list. Keep those 
			# channels in the same order though
			n_chans = pkl_obj['uniform_channel_values'].shape[0]
			n_chans_total = pkl_obj['channel_values'].shape[0]	
			chans = random.sample(range(n_chans_total), n_chans)
			chans.sort()
			x = x[chans,:]
	
		y = torch.tensor(pkl_obj['labels_vector'][0,rand_samp])	

		return x.float(),y


class DataLoader2(data.Dataset):
	def __init__(self, text_file, window_size, num_of_files, epoch_size=0):
		self.patient_list = open(text_file, 'r').read().split('\n')
		self.patient_list.pop()	   # remove last empty entry
		self.patient_len  = len(self.patient_list)
		
		self.window_size = window_size
		self.half_window = int(self.window_size/2)

		# get first collection of files
		self.f_data    = [0 for _ in range(num_of_files)]  # a list of the open pickles
		self.f_max     = [0 for _ in range(num_of_files)]  # a list with the max number of samples to get from each open pickle
		self.f_sampled = [0 for _ in range(num_of_files)]  # a list of the current amount of samples aquired from each open pickle

		if epoch_size == 0:
			self.epoch_size=self.patient_len
		else:
			self.epoch_size=epoch_size

		self.n_files   = num_of_files
		self.cur_files_ind = random.sample(range(self.patient_len), self.n_files)
		for i,cur_file in enumerate(self.cur_files_ind):
			self.loadFile(i,cur_file)

	def __len__(self):
		return self.epoch_size

	def __getitem__(self, index):		
		"""
		Gets the x,y pair from the specific open pickle file specified by index, and then
		does open files upkeep (checks if we sampled the max amount from the pickle and 
		if so opens a new pickle instead.
		"""
		index = np.random.randint(0,self.n_files)
		x,y = self.parsePickle(index)

		### open pickles upkeep
		self.f_sampled[index] += 1
		if self.f_sampled[index] == self.f_max[index]:
			# get new patient file index (not from current sample)
			new_patient_pkl = self.cur_file_ind[0]
			while new_patient_pkl in self.cur_file_ind:
				new_patient_pkl = random.sample(range(self.patient_len),1)			

			self.loadFile(index, new_patient_pkl)

		return x,y

	def parsePickle(self, index):	
		"""
		Gets the x,y pair from the patient pickle file with the specified index in the patient_list.
		"""
		
		cur_pkl = self.f_data[index]

		rand_samp = np.random.randint(self.half_window, cur_pkl['channel_values'].shape[1]-self.half_window)
		
		if UNIFORM_CHANNELS:
			x = torch.tensor(cur_pkl['uniform_channel_values'][:, (rand_samp-self.half_window):(rand_samp+self.half_window) ])
		else:
			x = torch.tensor(cur_pkl['channel_values'][:, (rand_samp-self.half_window):(rand_samp+self.half_window) ])
			
			# for comparison's sake, for the not uniform channels we will randomly
			# choose the same amount of channels as the uniform list. Keep those 
			# channels in the same order though
			n_chans = cur_pkl['uniform_channel_values'].shape[0]
			n_chans_total = cur_pkl['channel_values'].shape[0]	
			chans = random.sample(range(n_chans_total), n_chans)
			chans.sort()
			x = x[chans,:]
	
		y = torch.tensor(cur_pkl['labels_vector'][0,rand_samp])	
		return x.float(),y

	def loadFile(self, arr_index, patient_index):
		pkl_file = pickle.load(open(self.patient_list[patient_index],'rb'))
	
		self.f_data[arr_index] 	  = pkl_file
		self.f_max[arr_index]     = pkl_file['channel_values'].shape[1]//3	
		self.f_sampled[arr_index] = 0


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_SIZE = 200
LR = 1e-3
BATCH_SIZE = 1
NUM_OF_FILES = 15

dl1 = DataLoader1("../datas/newLists/trainDataWithSeizure_withUniform.txt", WINDOW_SIZE)
dl2 = DataLoader2("../datas/newLists/trainDataWithSeizure_withUniform.txt", WINDOW_SIZE,NUM_OF_FILES)

gen1 = data.DataLoader(dl1, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
gen2 = data.DataLoader(dl2, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)


counts1 = 0
counts2 = 0

print("Starting:")
	

start2 = time.time()	
for i in range(15):
	for x,y in gen2:
		x = x.cuda(DEVICE)
		y = y.cuda(DEVICE)
		counts2 += 1
end2 = time.time()
print("In middle")

start1 = time.time()	
for i in range(15):	
	for x,y in gen1:
		x = x.cuda(DEVICE)
		y = y.cuda(DEVICE)	
		counts1 += 1
end1 = time.time()

print(f"Elapsed Times:\nPlain dataloader: {end1-start1} | New dataloader: {end2-start2}")
print(f"Counts 1: {counts1} | Counts 2: {counts2}")
	




