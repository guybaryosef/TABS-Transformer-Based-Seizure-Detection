## fullEvalScript.sh - runs through the hoops to evaluate our model's outputs.
##
## Assuming that the model was trained and that its weights were saved. 
## Also assuming that testModel was run using this trained model and that the model
## outputs were saved to a directory.
##
## Note: do not use this for contest data- if evaluating data for the contest, use
## contest_fullEvalScript.sh.
##
## To run:
## 	- fullEvalScript.sh  MODEL_FILE MODEL_WEIGHTS MODEL_OUTPUT_DIR DEVICE_NUMBER POST_PROCESSING_DIR THRESH_VAL
##


model_file=$1
model_weights=$2
model_output_dir=$3
device_num=$4
pp_output_dir_name=$5
threshold=$6


source /zooper1/jped/eeg/eegvenv/bin/activate
python3 testModel.py ${model_file} ${model_weights} ${model_output_dir} ${device_num}
deactivate

./evaluateModelOutputsScript.sh ${model_output_dir} ${pp_output_dir_name} ${threshold}

