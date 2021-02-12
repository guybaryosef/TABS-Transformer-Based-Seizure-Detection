#
# create50_50seizureScript.py - Takes as input a directory filled with our 
# patient pickle files. Iterates through them, creating a subset of them
#  that has a total seizure/not-seizure ratio of roughly 50-50.
#
# The list with this pickle file subset names has the same name
# as the input directory, only with an added _50.txt at the end
# of it.
#
# To run:
# 	python  create50_50seizureScript.py  INPUT_DIRECTORY


import sys
import numpy as np
import glob
import pickle
from tqdm import tqdm

INPUT_DIR = sys.argv[1]

pickles = glob.glob(INPUT_DIR + "/*.pkl")

equal_list = []

# form list of all files that contain a seziure
bar = tqdm(range(len(pickles)))
for i in bar:
	
	pkl = pickle.load(open(pickles[i], 'rb'))
	y =  pkl['labels_vector']
	y = y.squeeze()

	if sum(y) > 0.0:
		equal_list.append(pickles[i])


# find the precentage of values that are seizure
total = 0
count = 0

bar = tqdm(range(len(equal_list)))
for i in bar:
	pkl = pickle.load(open(equal_list[i], 'rb'))

	y = pkl['labels_vector']

	total += y.shape[-1]
	count += sum(y)	

print(f"Percent seizure in new list is: {100*count/total}%")

# get the precentage higher
LOWEST_PERCENT = 0.25

newer_list = []

for i in bar:
	pkl = pickle.load(open(equal_list[i], 'rb'))
	y = pkl['labels_vector']
	
	cur_percent = sum(y) / y.shape[-1]

	if cur_percent >= LOWEST_PERCENT:
		newer_list.append(equal_list[i])


# get the percentage of the values that are seizure in NEWER list
total = 0
count = 0

bar = tqdm(range(len(newer_list)))
for i in bar:
	pkl = pickle.load(open(newer_list[i], 'rb'))

	y = pkl['labels_vector']

	total += y.shape[-1]
	count += sum(y)	

print(f"Percent seizure in newer list is: {100*count/total}%")


output = open(INPUT_DIR + "_50.txt", 'w')
for l in newer_list:
	output.write(l + "\n")
