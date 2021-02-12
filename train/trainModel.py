#
# trainModel.py - Our model training script.
#
# To run:
# 	python trainModel.py --model_file MODEL_FILE --device DEVICE_NUMBER --checkpoint_name CHECKPOINT_NAME --train_file  TRAINING_PICKLES_LIST_FILE --validation_file VALIDATION_PICKLES_LIST_FILE
#

import argparse 
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import pickle
import glob
from dataLoaders import fastDataLoader, testDataLoader
import numpy as np
import math
import tqdm
import sys
from importlib import import_module


parser = argparse.ArgumentParser()
parser.add_argument("--model_file", type=str, help="pass in the file containing the model class you wish to train")
parser.add_argument("--device", type=int, default=0, help="which gpu do you want to operate on")
parser.add_argument("--checkpoint_name", type=str, required=True, help="where you want to save the checkpoint")
parser.add_argument("--train_file", type=str, default="../datas/small_windows_50.txt", help="the text file with training example pickles")
parser.add_argument("--validation_file", type=str, default="../data/validation_subset2.txt", help="the text file with the validation example pickles")
parser.add_argument("--load_file", type=str, help="pass this in if you want to load the model from a previous checkpoint")
parser.add_argument("--epoch_size", type=int, default=1000000, help="The number of smaples for the training fast data loader")


def calculateStatistics(true_positive, false_negative, true_negative, false_positive, data_set, loss, max_accuracy, validation=False):
    if ((true_positive + false_negative) == 0):
        sensitivity = 0
    else:
        sensitivity = true_positive/(true_positive + false_negative)
    if ((true_negative + false_positive) == 0):
        specificity = 0
    else:
        specificity = true_negative/(true_negative + false_positive)
    false_p_per_hour = (false_positive/len(data_set)) *  256 * 3600
    accuracy = (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative)
    split = "validation" if validation else "training"
    tqdm.tqdm.write("{} loss: {}, sensitivity: {}, specificity: {}, accuracy: {}, max_acc: {}, fpph: {}".format(split,round(loss, 3), round(sensitivity, 3), round(specificity, 3),round(accuracy, 3), round(max_accuracy, 3), round(false_p_per_hour,3)))
    return accuracy, specificity, sensitivity, loss


args = parser.parse_args()

Model = getattr(import_module(args.model_file), 'EnsembleModel')
model = Model()


DEVICE = args.device
print("device ", DEVICE)
WINDOW_SIZE = 300
HIDDEN_DIM = WINDOW_SIZE
LR = 1e-4


def setValidationWorker(worker_id):
        worker_info =data.get_worker_info()
        if worker_info:
                dataset = worker_info.dataset
                dataset.num_workers = worker_info.num_workers
                dataset.cur_file_num = worker_info.id % dataset.patient_len
                dataset.cur_pkl, dataset.cur_pkl_shape = dataset.loadFile(dataset.cur_file_num)

print("finished setting up validation workers.")
training_set = fastDataLoader(args.train_file, WINDOW_SIZE, 20, epoch_size=args.epoch_size, old_loader=True)
validation_set = testDataLoader(args.validation_file, WINDOW_SIZE, old_loader=True)
training_generator = data.DataLoader(training_set, batch_size=128, shuffle=True, num_workers=24)
validation_generator = data.DataLoader(validation_set, batch_size=1,num_workers=1, worker_init_fn=setValidationWorker)
print("Loaded up the train and validation data loaders.")


print("Initialized model.")
if args.load_file:
    print("loading checkpoint ...")
    model.load_state_dict(torch.load(args.load_file, map_location=torch.device("cuda:{}".format(DEVICE))))
    print("done loading checkpoint")


model.cuda(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.1)


true_negative  = 0
true_positive  = 0
false_positive = 0
false_negative = 0
max_accuracy_train = 0
max_accuracy       = 0
max_sensitivity    = 0
max_specificity    = 0
min_loss 	   = 1000000000

loss  = 0
ALPHA = 0.6


print("Starting to train...")
for j in range(1000):
    train_loss = 0
    i = 0
    num_positive = 0
    num_negative = 0
    print("training...")
    for x_sample, y_label in tqdm.tqdm(training_generator):
        i+=1         
        model.train().cuda(DEVICE)
        x_sample = x_sample.cuda(DEVICE).squeeze()
        y_label = y_label.cuda(DEVICE)
        y_hat = model(x_sample.to(torch.float))
        ## Do the mix up here: 
        # step 1 make a copy of the batch that is rolled.
        roll_factor =  torch.randint(0, x_sample.shape[0], (1,)).item()
        rolled_x_sample = torch.roll(x_sample, roll_factor, dims=0)        
        rolled_y_label = torch.roll(y_label, roll_factor, dims=0)        
        # step 2 make a tensor of lambdas sampled from the beta distribution
        lambdas = np.random.beta(ALPHA, ALPHA, x_sample.shape[0])
        # trick from here https://forums.fast.ai/t/mixup-data-augmentation/22764
        lambdas = torch.reshape(torch.tensor(np.maximum(lambdas, 1-lambdas)), (-1,1,1)).cuda(DEVICE)
        # step 3 interpolate
        mixed_x_sample = lambdas*x_sample + (1-lambdas)*rolled_x_sample
        y_hat = model(mixed_x_sample.to(torch.float))
        # step 4 edit the loss
        output = lambdas.squeeze()*F.cross_entropy(y_hat, y_label.long()) + (1-lambdas.squeeze())*F.cross_entropy(y_hat, rolled_y_label.long())
        output = output.sum()
        train_loss += output.item()
        optimizer.zero_grad()
        output.backward()
        optimizer.step()
        pred = y_hat.max(dim=-1).indices 
        confusion = (pred/y_label.to(torch.float))
        true_positive += torch.sum(confusion == 1).item()
        false_negative += torch.sum(confusion == 0).item()
        true_negative += torch.sum(torch.isnan(confusion)).item()
        false_positive += torch.sum(confusion == float('inf')).item()
    [accuracy_train, specificity, sensitivity, loss] = calculateStatistics(true_positive, false_negative, true_negative, false_positive, training_set, train_loss, max_accuracy_train)
    if accuracy_train > max_accuracy_train: 
            tqdm.tqdm.write("checkpointing train model acc")
            torch.save(model.state_dict(), "{}_maxTrainAccuracy".format(args.checkpoint_name))
            max_accuracy_train = accuracy_train
    tqdm.tqdm.write("checkpointing train model")
    torch.save(model.state_dict(), "{}_epoch_{}_{}_{}".format(args.checkpoint_name, j, specificity, sensitivity)) 
    train_loss = 0
    #if True:
    if (j+1) % 5 == 0:
        true_negative = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0
        loss = 0
        print("validating...")
        for x_sample_valid, y_label_valid in tqdm.tqdm(validation_generator):
            model.eval().cuda(DEVICE)
            x_sample_valid = x_sample_valid.cuda(DEVICE).to(torch.float)
            y_label_valid = y_label_valid.cuda(DEVICE)
            y_hat = model(x_sample_valid)
            output = F.cross_entropy(y_hat, y_label_valid.long())
            loss += output.item()
            pred = y_hat.max(dim=-1).indices 
            confusion = (pred/y_label_valid.to(torch.float))
            true_positive += torch.sum(confusion == 1).item()
            false_negative += torch.sum(confusion == 0).item()
            true_negative += torch.sum(torch.isnan(confusion)).item()
            false_positive += torch.sum(confusion == float('inf')).item()
        [accuracy, specificity, sensitivity, loss]  = calculateStatistics(true_positive, false_negative, true_negative, false_positive, validation_set, loss, max_accuracy, validation=True)

	# checkpoint values
        if accuracy > max_accuracy: 
            tqdm.tqdm.write("checkpointing validation model - max accuracy")
            torch.save(model.state_dict(), "{}_maxAccuracy".format(args.checkpoint_name))
            max_accuracy = accuracy

        if sensitivity > max_sensitivity: 
            tqdm.tqdm.write("checkpointing validation model - max sensitivity")
            torch.save(model.state_dict(), "{}_maxSensitivity".format(args.checkpoint_name))
            max_sensitivity = sensitivity

        if specificity > max_specificity:
            tqdm.tqdm.write("checkpointing validation model - max specificity")
            torch.save(model.state_dict(), "{}_maxSpecificity".format(args.checkpoint_name))
            max_specificity = specificity

        if loss < min_loss:
            tqdm.tqdm.write("checkpointing validation model - minimum loss")
            torch.save(model.state_dict(), "{}_minLoss".format(args.checkpoint_name))
            min_loss = loss

