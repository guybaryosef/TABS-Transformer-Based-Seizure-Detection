
## evaluateModelOutputsScript.sh - runs through the hoops to evaluate a model's outputs.
##
## Assuming that the model was trained and that its weights were saved. 
## Also assuming that testModel was run using this trained model and that the model
## outputs were saved to a directory.
##
## To run:
## 	- contest_evaluateModelOutputsScript.sh MODEL_OUTPUTS_DIRECTORY POST_PROCESSING_OUTPUT_DIRECTORY_NAME THRESHOLD_VALUE
##

model_output_dir_path=$1
pp_output_dir_name=$2
threshold=$3
ref_file="/mnt/pedomeister/eeg/newTuh_eeg_1.5.1/_DOCS/ref_dev.txt"

pp_dir="$(pwd)/../model_outputs_postprocessing/${pp_output_dir_name}"  # full path to postprocessing directory
echo "Full path to postprocessing directory: ${pp_dir}"
mkdir ${pp_dir}

echo "executes runPostProcessing on the model output files found at ${model_output_dir_path}"
source /zooper1/jped/eeg/eegvenv/bin/activate
python3 contest_runPostProcessing.py ${ref_file} ${model_output_dir_path} ${pp_dir} ${threshold}
deactivate

# run the OVLP eval script
ovlp_output_dir="${pp_dir}/evalOutput"
echo "Evaluating OVLP script in ${ovlp_output_dir} directory.\nHypothesis file: ${pp_dir}/hyp.txt\nReference file: ${ref_file}"
mkdir ${ovlp_output_dir}
cd ${ovlp_output_dir}
/zooper1/jped/eeg/scripts/contestEvalScript/v3.3.1/scripts/nedc_eval_eeg.py ${ref_file} "${pp_dir}/hyp.txt"
