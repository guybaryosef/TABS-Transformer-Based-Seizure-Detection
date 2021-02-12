## evaluateModelOutputsScript.sh - runs through the hoops to evaluate a model's outputs.
##
## Assuming that the model was trained and that its weights were saved. 
## Also assuming that testModel was run using this trained model and that the model
## outputs were saved to a directory.
##
## To run:
## 	- evaluateModelOutputsScript.sh MODEL_OUTPUTS_DIRECTORY POST_PROCESSING_OUTPUT_DIRECTORY_NAME THRESHOLD_VALUE
##


model_output_dir_path=$1
pp_output_dir_name=$2
threshold=$3

source /zooper1/jped/eeg/eegvenv/bin/activate

pp_dir="$(pwd)/../model_outputs_postprocessing/${pp_output_dir_name}"  # full path to postprocessing directory
echo "Full path to postprocessing directory: ${pp_dir}"
mkdir ${pp_dir}

echo "executes runPostProcessing on each model output file found at ${model_output_dir_path}"
find ${model_output_dir_path} | xargs -I {} -P20 python3 runPostProcessing.py {} ${pp_dir} ${threshold}

pp_txt_file="${pp_dir}/${pp_output_dir_name}.txt"
echo "Create ${pp_txt_file}, a .txt file of the full path of all the .tse files created:"
find "${pp_dir}/prediction" > ${pp_txt_file}

echo "Sort the created txt file:"
sort "${pp_txt_file}" > tmp.txt
tail -n +2 tmp.txt > ${pp_txt_file}
rm tmp.txt

# make the target list the same length as the prediction list
pred_list_len=$(wc -l ${pp_txt_file} | awk '{print $1;}')

targ_list="/zooper1/jped/eeg/models/ground_truth/newTarget.list"

# run the OVLP eval script
ovlp_output_dir="${pp_dir}/evalOutput"
echo "Evaluating OVLP script in ${ovlp_output_dir} directory:"
mkdir ${ovlp_output_dir}

deactivate
export PATH=$PATH:/zooper1/jped/eeg/scripts/evalScript/crap/F4DE/bin

python2.7 /zooper1/jped/eeg/scripts/evalScript/v1.3.0/src/nedc_eval_eeg.py -odir ${ovlp_output_dir}  -parameters /zooper1/jped/eeg/scripts/evalScript/v1.3.0/seniorProjectsParams.txt ${targ_list} ${pp_txt_file}

