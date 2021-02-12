#!/bin/bash

# A bash script to create symbolic links to all the relevant contest data 
# from the seizures dataset and place it into 3 general folders, 
# differentiated by the EEG references (ar, le, and ar_a).
# 
# The data will be in the collectedSeizureData directory located
# at the current working directory.
#
# To run:
# 	data_gatherer.sh  full/path/to/seizure/dataset  full/path/to/output/directory


input_dir=$1
output_dir=$2

for f in $(find ${input_dir} -name '*.edf' -or -name '*.txt' -or -name '*.tse_bi' -or -name '*.rec'); do
	ln -s -t ${output_dir} $f
done
