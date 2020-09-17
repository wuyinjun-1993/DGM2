'''
Created on Jun 17, 2020

'''

import os
import logging
import pickle

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math 
import glob
import re
from shutil import copyfile
import sklearn as sk
import subprocess
import datetime
from torch.utils.data import Dataset, DataLoader
# from data.generate_time_series import climate_data_name, kddcup_data_name



data_dir = 'dataset_dir/'
synthetic_sub_data_dir = 'synthetic_data/' 
mimic3_data_dir = 'mimic3/'
beijing_data_dir = 'KDDCUP'
climate_data_dir = 'climate/tensor/'
physionet_data_dir = 'physionet/'
output_dir = 'output/'
GRUI_train_dir = data_dir + 'GRUI/imputation_train_results/WGAN_no_mask/'

GRUI_test_dir = data_dir + '/GRUI/imputation_test_results/WGAN_no_mask/'



GRUD_method = 'GRUD'

LODE_method = 'L_ODE'

cluster_ODE_method = 'DGM2_O'

cluster_method = 'DGM2_L'


climate_data_name = 'USHCN'

kddcup_data_name = 'KDDCUP'

mimic_data_name = 'MIMIC3'

l_ODE_method = 'L_ODE'

climate_data_train_len = 80

mimic3_data_train_len = 48

mimic3_data_len = 72

physionet_data_train_len = 40

beijing_data_train_len = 24

train_ratio = 5.0/6

class MyDataset(Dataset):
    def __init__(self, dataset, mask, origin_mask, new_random_mask, lens, time_stamps, delta_time_stamps):
        
        self.data = dataset
        
        self.origin_data = dataset.clone()
        
        self.mask = mask
        
        self.origin_mask = origin_mask
        
        self.new_random_mask = new_random_mask
        
        self.time_stamps = time_stamps
        
        self.delta_time_stamps = delta_time_stamps
        
        self.lens = lens
        
    def __getitem__(self, index):
        data, mask = self.data[index], self.mask[index]
        
        origin_data = self.origin_data[index]
        
        curr_origin_mask = self.origin_mask[index].clone()
        
        curr_new_random_mask = self.new_random_mask[index].clone()
        
        curr_lens = self.lens[index].clone()
        
        curr_time_staps = self.time_stamps[index].clone()
        
        curr_delta_time_stamps = self.delta_time_stamps[index].clone()
        
#         assert torch.sum(mask) == torch.sum(1-np.isnan(data))
        
        return data, mask, origin_data, curr_origin_mask,curr_new_random_mask, curr_lens, curr_time_staps, curr_delta_time_stamps, index

    def __len__(self):
        return len(self.data)



def remove_none_observations(train_set, train_mask, train_time_stamps, train_lens):
    
    
    tr_train_set = torch.zeros_like(train_set)
    
    tr_train_set[:] = np.nan
    
    tr_train_mask = torch.zeros_like(train_mask)
    
    tr_train_lens = torch.zeros_like(train_lens)
    
    tr_train_time_stamps = torch.zeros_like(train_time_stamps)
    
    tr_train_time_stamps[:] = train_lens.max() + 1
    
    for i in range(train_set.shape[0]):
#         if i >= 24018:
#             print('here')
            
        
        ids = torch.sum(np.isnan(train_set[i]), 1) < train_set.shape[2]
        
        curr_train_set = train_set[i, ids]
        tr_train_set[i, 0:curr_train_set.shape[0]] = curr_train_set
        
        tr_train_mask[i, 0:curr_train_set.shape[0]] = train_mask[i, ids]
        
        tr_train_lens[i] = curr_train_set.shape[0]
        
        tr_train_time_stamps[i, 0:curr_train_set.shape[0]] = train_time_stamps[i, ids]
        
    return tr_train_set, tr_train_mask, tr_train_time_stamps, tr_train_lens

def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device

def shift_test_samples(data, r):
    
    rows, column_indices = np.ogrid[:data.shape[0], :data.shape[1]]

    # Use always a negative shift, so that column_indices are valid.
    # (could also use module operation)
    r[r < 0] += data.shape[1]
    column_indices2 = (torch.from_numpy(column_indices) - r[:, np.newaxis]).type(torch.LongTensor)
    
    result = data[rows, column_indices2]
    
    return result

def check_shift_correctness(train_part, forecast_part, n_observed_tps, n_predict_tps):
    
    
    for i in range(len(n_observed_tps)):
        assert(torch.norm(train_part[i][n_observed_tps[i]:n_predict_tps[i] + n_observed_tps[i]] - forecast_part[i][0:n_predict_tps[i]]) == 0)
    


def get_delta_time_stamps_all_dims(time_stamps):
    all_ids = torch.tensor(list(range(time_stamps.shape[1]-1)), device = time_stamps.device)
    
    delta_time_stamps = torch.zeros_like(time_stamps, device = time_stamps.device)
    
    delta_time_stamps[:, all_ids + 1] = time_stamps[:,all_ids + 1] - time_stamps[:,all_ids]
    
    delta_time_stamps[delta_time_stamps < 0] = 0
    
    return delta_time_stamps


def get_delta_time_stamps(masks, time_stamps):
    
    all_ids = torch.tensor(list(range(time_stamps.shape[1]-1)), device = masks.device)
    
    delta_time_stamps = torch.zeros_like(time_stamps, device = masks.device)
    
    delta_time_stamps[:, all_ids + 1] = time_stamps[:,all_ids + 1] - time_stamps[:,all_ids]
    
    
#     print(delta_time_stamps)
    
    res_delta_time_stamps = torch.zeros_like(masks, dtype = torch.float, device = masks.device)
 
#     for j in range(masks.shape[0]):
#          
# #         curr_delta_time_stamps = delta_time_stamps[j]
#          
#          
#         time_gap_tensors = torch.zeros(masks.shape[2], dtype = torch.float)
#          
#         for k in range(masks.shape[1]):
#              
#             res_delta_time_stamps[j,k] = time_gap_tensors + delta_time_stamps[j,k]
#              
#             time_gap_tensors= (1 - masks[j,k])*time_gap_tensors + (1 - masks[j,k])*delta_time_stamps[j,k]
            
    time_gap_tensors = torch.zeros([masks.shape[0], masks.shape[2]], dtype = torch.float, device = masks.device)
         
    for k in range(masks.shape[1]):
         
        res_delta_time_stamps[:,k] = time_gap_tensors + delta_time_stamps[:,k].view(masks.shape[0], 1)
         
        time_gap_tensors= (1 - masks[:,k])*time_gap_tensors + (1 - masks[:,k])*delta_time_stamps[:,k].view(masks.shape[0], 1)      
#             curr_masks = masks[j,k]
#              
#             print('here')
    
    
    return res_delta_time_stamps


def init_network_weights(net, std = 0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


def remove_empty_time_steps(x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, x_time_stamps, x_to_predict_time_stamps):
        non_missing_time_steps = (torch.sum(x_mask, (0,2)) != 0)
        
        print('non empty time count::', torch.sum(non_missing_time_steps))
        
        x = x[:, non_missing_time_steps]
        
        origin_x = origin_x[:, non_missing_time_steps]
        
        x_mask = x_mask[:, non_missing_time_steps]
               
        origin_x_mask = origin_x_mask[:, non_missing_time_steps]
        
        new_x_mask = new_x_mask[:, non_missing_time_steps]
        
        x_time_stamps = x_time_stamps[:,non_missing_time_steps]
        
        x_lens[:] = torch.sum(non_missing_time_steps) 
        
        non_missing_time_steps = (torch.sum(x_to_predict_mask, (0,2)) != 0)
        
        x_to_predict = x_to_predict[:, non_missing_time_steps]
        
        origin_x_to_pred = origin_x_to_pred[:, non_missing_time_steps]
        
        x_to_predict_mask = x_to_predict_mask[:, non_missing_time_steps]
        
        x_to_predict_origin_mask = x_to_predict_origin_mask[:, non_missing_time_steps]
        
        x_to_predict_new_mask = x_to_predict_new_mask[:, non_missing_time_steps]
        
        x_to_predict_lens[:] = torch.sum(non_missing_time_steps)
        
        x_to_predict_time_stamps = x_to_predict_time_stamps[:,non_missing_time_steps] 
        
        return x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, x_time_stamps, x_to_predict_time_stamps

def split_data_extrap(data_dict, dataset = ""):
    device = get_device(data_dict["data"])


    n_observed_tps = data_dict["lens"]*5//6
    if dataset.startswith("mimic3"):
        
        n_observed_tps[:] = mimic3_data_train_len#data_dict["time_stamps"][:,0:48]#torch.sum(data_dict["time_stamps"] < 48, 1)
    
    if dataset == 'mimic3_17_5':
        n_observed_tps[:] = 48
        
    if dataset.startswith(climate_data_name):
        n_observed_tps[:] = climate_data_train_len
        
    if dataset.startswith(kddcup_data_name):
        n_observed_tps[:] = beijing_data_train_len
#     if dataset.startswith("mimic3"):
#         
#         n_observed_tps[:] = 48
#         print('here')
    
    T_max_train = n_observed_tps.max().item()
    
    n_predicted_tps = data_dict["lens"] - n_observed_tps
    
    T_max_test =  n_predicted_tps.max().item()

#     n_observed_tp = data_dict["data"].size(1)*5 // 6
#     if dataset == "mimic3":
#         n_observed_tp = data_dict["data"].size(1) // 3


    forecast_part = shift_test_samples(data_dict["data"], -n_observed_tps)
    
    forecast_time_stamps = shift_test_samples(data_dict["time_stamps"], -n_observed_tps)
    
    forecast_delta_time_stamps = shift_test_samples(data_dict["delta_time_stamps"], -n_observed_tps)
    
    forecast_mask_part = shift_test_samples(data_dict["mask"], -n_observed_tps)
    
    forecast_origin_mask_part = shift_test_samples(data_dict["origin_mask"], -n_observed_tps)
    
    forecast_new_rand_mask_part = shift_test_samples(data_dict["new_random_mask"], -n_observed_tps)

    check_shift_correctness(data_dict["data"], forecast_part, n_observed_tps, n_predicted_tps)

#     split_dict = {"observed_data": data_dict["data"][:,:n_observed_tp,:].clone(),
#                 "observed_lens": n_observed_tps.clone(),
# #                 "observed_tp": data_dict["time_steps"][:n_observed_tp].clone(),
#                 "origin_observed_data": data_dict["origin_data"][:,:n_observed_tp,:].clone(),
#                 
#                 "data_to_predict": data_dict["data"][:,n_observed_tp:,:].clone(),
#                 "lens_to_predict": n_predicted_tps.clone(),
# #                 "tp_to_predict": data_dict["time_steps"][n_observed_tp:].clone(),
#                 "origin_data_to_predict": data_dict["origin_data"][:,n_observed_tp:,:].clone()}
    
    split_dict = {"observed_data": data_dict["data"][:,0:T_max_train,:].clone(),
                "observed_lens": n_observed_tps.clone(),
                "time_stamps": data_dict["time_stamps"][:,0:T_max_train].clone(),
                "delta_time_stamps": data_dict["delta_time_stamps"][:,0:T_max_train].clone(),
#                 "observed_tp": data_dict["time_steps"][:n_observed_tp].clone(),
                "origin_observed_data": data_dict["origin_data"][:,0:T_max_train,:].clone(),
                
                "data_to_predict": forecast_part[:,0:T_max_test,:].clone(),
                "lens_to_predict": n_predicted_tps.clone(),
                "time_stamps_to_predict": forecast_time_stamps[:,0:T_max_test].clone(),
                "delta_time_stamps_to_predict":forecast_delta_time_stamps[:,0:T_max_test].clone(),
#                 "tp_to_predict": data_dict["time_steps"][n_observed_tp:].clone(),
                "origin_data_to_predict": forecast_part[:,0:T_max_test,:].clone()}
    
    
    

    split_dict["observed_mask"] = None 
    split_dict["mask_predicted_data"] = None 
    split_dict["labels"] = None 

    if ("mask" in data_dict) and (data_dict["mask"] is not None):
        split_dict["observed_mask"] = data_dict["mask"][:,0:T_max_train,:].clone()
        split_dict["mask_predicted_data"] = forecast_mask_part[:,0:T_max_test,:].clone()
        
        split_dict["observed_origin_mask"] = data_dict["origin_mask"][:,0:T_max_train,:].clone()
        split_dict["origin_mask_predicted_data"] = forecast_origin_mask_part[:,0:T_max_test,:].clone()

        split_dict["observed_new_mask"] = data_dict["new_random_mask"][:,0:T_max_train,:].clone()
        split_dict["new_mask_predicted_data"] = forecast_new_rand_mask_part[:,0:T_max_test,:].clone()

#     assert torch.sum(data_dict["mask"]) == torch.sum(1-np.isnan(data_dict["data"]))
# 
#     assert torch.sum(split_dict["observed_mask"]) == torch.sum(1-np.isnan(split_dict["observed_data"]))
#     
#     assert torch.sum(split_dict["mask_predicted_data"]) == torch.sum(1-np.isnan(split_dict["data_to_predict"]))
        
    if ("labels" in data_dict) and (data_dict["labels"] is not None):
        split_dict["labels"] = data_dict["labels"].clone()

    split_dict["mode"] = "extrap"
    split_dict["ids"] = data_dict["ids"]
    

#     curr_new_mask = data_dict["origin_mask"]*(1-data_dict["new_random_mask"])
#             
#     assert (torch.nonzero(~(curr_new_mask ==  (1-data_dict["new_random_mask"])))).shape[0] == 0
#     
#     curr_new_mask = split_dict["observed_origin_mask"]*(1-split_dict["observed_new_mask"])
#             
#     assert (torch.nonzero(~(curr_new_mask ==  (1-split_dict["observed_new_mask"])))).shape[0] == 0
#     
#     curr_new_mask = split_dict["origin_mask_predicted_data"]*(1-split_dict["new_mask_predicted_data"])
#             
#     assert (torch.nonzero(~(curr_new_mask ==  (1-split_dict["new_mask_predicted_data"])))).shape[0] == 0
    
    return split_dict


def compute_gaussian_probs(x, mean, std, mask):
    
    
#         gaussian = Independent(Normal(loc = mean[0][0], scale = std[0][0]), 0)
#         log_prob = gaussian.log_prob(x[0][0])
    
    prob = -0.5*((x - mean)/std)**2 - torch.log((std*np.sqrt(2*np.pi)))
    
#         print(torch.norm(prob[0][0] - log_prob))
    
    return -prob*mask


def compute_gaussian_probs0(x, mean, logvar, mask):
    
    
#         gaussian = Independent(Normal(loc = mean[0][0], scale = std[0][0]), 0)
#         log_prob = gaussian.log_prob(x[0][0])
    
    std = torch.exp(0.5 * logvar)
    
    prob = 0.5*(((x - mean)/std)**2 + logvar + 2*np.log(np.sqrt(2*np.pi)))# + torch.log((std*np.sqrt(2*np.pi)))
    
#         print(torch.norm(prob[0][0] - log_prob))
    
    return prob*mask


def linspace_vector(start, end, n_points):
    # start is either one value or a vector
    size = np.prod(start.size())

    assert(start.size() == end.size())
    if size == 1:
        # start and end are 1d-tensors
        res = torch.linspace(start, end, n_points)
    else:
        # start and end are vectors
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res, 
                torch.linspace(start[i], end[i], n_points)),0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res

def create_net(n_inputs, n_outputs, n_layers = 0, 
    n_units =10, nonlinear = nn.Tanh, add_softmax = False, dropout = 0.0):
    if n_layers >= 0:
        layers = [nn.Linear(n_inputs, n_units)]
        for i in range(n_layers):
            layers.append(nonlinear())
            layers.append(nn.Linear(n_units, n_units))
            layers.append(nn.Dropout(p = dropout))
    
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_outputs))
        if add_softmax:
            layers.append(nn.Softmax(dim=-1))
    
    else:
        layers = [nn.Linear(n_inputs, n_outputs)]
        
        if add_softmax:
            layers.append(nn.Softmax(dim=-1))
    
    return nn.Sequential(*layers)


def sample_standard_gaussian(mu, sigma):
    device = get_device(mu)

    d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()

def check_mask(data, mask):
    #check that "mask" argument indeed contains a mask for data
    n_zeros = torch.sum(mask == 0.).cpu().numpy()
    n_ones = torch.sum(mask == 1.).cpu().numpy()

    # mask should contain only zeros and ones
    assert((n_zeros + n_ones) == np.prod(list(mask.size())))

    # all masked out elements should be zeros
    assert(torch.sum(data[mask == 0.] != 0.) == 0)

def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim//2

    if len(data.size()) == 3:
        res = data[:,:,:last_dim], data[:,:,last_dim:]

    if len(data.size()) == 2:
        res = data[:,:last_dim], data[:,last_dim:]
    return res

def split_train_test(data, train_fraq = 0.8):
    n_samples = data.size(0)
    data_train = data[:int(n_samples * train_fraq)]
    
#     masks_train = masks[:int(n_samples * train_fraq)]
    
    data_test = data[int(n_samples * train_fraq):]
    
#     masks_test = masks[int(n_samples * train_fraq):]
    
    
    return data_train, data_test


def split_data_interp(data_dict):
    device = get_device(data_dict["data"])

    split_dict = {"observed_data": data_dict["data"].clone(),
                "observed_tp": data_dict["time_steps"].clone(),
                "data_to_predict": data_dict["data"].clone(),
                "tp_to_predict": data_dict["time_steps"].clone()}

    split_dict["observed_mask"] = None 
    split_dict["mask_predicted_data"] = None 
    split_dict["labels"] = None 

    if "mask" in data_dict and data_dict["mask"] is not None:
        split_dict["observed_mask"] = data_dict["mask"].clone()
        split_dict["mask_predicted_data"] = data_dict["mask"].clone()

    if ("labels" in data_dict) and (data_dict["labels"] is not None):
        split_dict["labels"] = data_dict["labels"].clone()

    split_dict["mode"] = "interp"
    return split_dict


def get_dict_template():
    return {"observed_data": None,
            "observed_tp": None,
            "data_to_predict": None,
            "tp_to_predict": None,
            "observed_mask": None,
            "mask_predicted_data": None,
            "labels": None
            }
    
    

def get_next_batch(data_dict):
    # Make the union of all time points and perform normalization across the whole dataset
#     data_dict = dataloader.__next__()

    batch_dict = get_dict_template()

    # remove the time points where there are no observations in this batch
    non_missing_tp = torch.sum(data_dict["observed_data"],(0,2)) != 0.
    batch_dict["observed_data"] = data_dict["observed_data"][:, non_missing_tp]
    batch_dict["observed_tp"] = data_dict["observed_tp"][non_missing_tp]

    # print("observed data")
    # print(batch_dict["observed_data"].size())

    if ("observed_mask" in data_dict) and (data_dict["observed_mask"] is not None):
        batch_dict["observed_mask"] = data_dict["observed_mask"][:, non_missing_tp]

    batch_dict[ "data_to_predict"] = data_dict["data_to_predict"]
    batch_dict["tp_to_predict"] = data_dict["tp_to_predict"]

    non_missing_tp = torch.sum(data_dict["data_to_predict"],(0,2)) != 0.
    batch_dict["data_to_predict"] = data_dict["data_to_predict"][:, non_missing_tp]
    batch_dict["tp_to_predict"] = data_dict["tp_to_predict"][non_missing_tp]

    # print("data_to_predict")
    # print(batch_dict["data_to_predict"].size())

    if ("mask_predicted_data" in data_dict) and (data_dict["mask_predicted_data"] is not None):
        batch_dict["mask_predicted_data"] = data_dict["mask_predicted_data"][:, non_missing_tp]

    if ("labels" in data_dict) and (data_dict["labels"] is not None):
        batch_dict["labels"] = data_dict["labels"]

    batch_dict["mode"] = data_dict["mode"]
    return batch_dict


def add_mask(data_dict):
    data = data_dict["observed_data"]
    mask = data_dict["observed_mask"]

    if mask is None:
        mask = torch.ones_like(data).to(get_device(data))

    data_dict["observed_mask"] = mask
    return data_dict


def subsample_timepoints(data, time_steps, mask, n_tp_to_sample = None):
    # n_tp_to_sample: number of time points to subsample. If not None, sample exactly n_tp_to_sample points
    if n_tp_to_sample is None:
        return data, time_steps, mask
    n_tp_in_batch = len(time_steps)


    if n_tp_to_sample > 1:
        # Subsample exact number of points
        assert(n_tp_to_sample <= n_tp_in_batch)
        n_tp_to_sample = int(n_tp_to_sample)

        for i in range(data.size(0)):
            missing_idx = sorted(np.random.choice(np.arange(n_tp_in_batch), n_tp_in_batch - n_tp_to_sample, replace = False))

            data[i, missing_idx] = 0.
            if mask is not None:
                mask[i, missing_idx] = 0.
    
    elif (n_tp_to_sample <= 1) and (n_tp_to_sample > 0):
        # Subsample percentage of points from each time series
        percentage_tp_to_sample = n_tp_to_sample
        for i in range(data.size(0)):
            # take mask for current training sample and sum over all features -- figure out which time points don't have any measurements at all in this batch
            current_mask = mask[i].sum(-1).cpu()
            non_missing_tp = np.where(current_mask > 0)[0]
            n_tp_current = len(non_missing_tp)
            n_to_sample = int(n_tp_current * percentage_tp_to_sample)
            subsampled_idx = sorted(np.random.choice(non_missing_tp, n_to_sample, replace = False))
            tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)

            data[i, tp_to_set_to_zero] = 0.
            if mask is not None:
                mask[i, tp_to_set_to_zero] = 0.

    return data, time_steps, mask



def cut_out_timepoints(data, time_steps, mask, n_points_to_cut = None):
    # n_points_to_cut: number of consecutive time points to cut out
    if n_points_to_cut is None:
        return data, time_steps, mask
    n_tp_in_batch = len(time_steps)

    if n_points_to_cut < 1:
        raise Exception("Number of time points to cut out must be > 1")

    assert(n_points_to_cut <= n_tp_in_batch)
    n_points_to_cut = int(n_points_to_cut)

    for i in range(data.size(0)):
        start = np.random.choice(np.arange(5, n_tp_in_batch - n_points_to_cut-5), replace = False)

        data[i, start : (start + n_points_to_cut)] = 0.
        if mask is not None:
            mask[i, start : (start + n_points_to_cut)] = 0.

    return data, time_steps, mask




def subsample_observed_data(data_dict, n_tp_to_sample = None, n_points_to_cut = None):
    # n_tp_to_sample -- if not None, randomly subsample the time points. The resulting timeline has n_tp_to_sample points
    # n_points_to_cut -- if not None, cut out consecutive points on the timeline.  The resulting timeline has (N - n_points_to_cut) points

    if n_tp_to_sample is not None:
        # Randomly subsample time points
        data, time_steps, mask = subsample_timepoints(
            data_dict["observed_data"].clone(), 
            time_steps = data_dict["observed_tp"].clone(), 
            mask = (data_dict["observed_mask"].clone() if data_dict["observed_mask"] is not None else None),
            n_tp_to_sample = n_tp_to_sample)

    if n_points_to_cut is not None:
        # Remove consecutive time points
        data, time_steps, mask = cut_out_timepoints(
            data_dict["observed_data"].clone(), 
            time_steps = data_dict["observed_tp"].clone(), 
            mask = (data_dict["observed_mask"].clone() if data_dict["observed_mask"] is not None else None),
            n_points_to_cut = n_points_to_cut)

    new_data_dict = {}
    for key in data_dict.keys():
        new_data_dict[key] = data_dict[key]

    new_data_dict["observed_data"] = data.clone()
    new_data_dict["observed_tp"] = time_steps.clone()
    new_data_dict["observed_mask"] = mask.clone()

    if n_points_to_cut is not None:
        # Cut the section in the data to predict as well
        # Used only for the demo on the periodic function
        new_data_dict["data_to_predict"] = data.clone()
        new_data_dict["tp_to_predict"] = time_steps.clone()
        new_data_dict["mask_predicted_data"] = mask.clone()

    return new_data_dict


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def add_random_missing_values(dataset, masks, missing_ratio, train_time_len):
    
#     masks = torch.ones_like(dataset)

#     dataset_view = dataset.view(dataset.shape[0]*dataset.shape[1], dataset.shape[2])
    
    inference_masks = masks[:, 0: train_time_len]
    
    test_masks = masks[:, train_time_len:]
        
    masks_view = inference_masks.reshape(inference_masks.shape[0]*inference_masks.shape[1], inference_masks.shape[2])

    random_removal_masks = torch.ones_like(masks_view)
#     for i in range(dataset.shape[0]):
    for j in range(masks_view.shape[1]):
        
        non_masks_ids = torch.nonzero(masks_view[:,j] != 0) 
#         dataset[:, ]
#         
#         dataset[i,masks[i,:,j]!=0,j]
        
        random_ids = torch.randperm(non_masks_ids.shape[0])
        
        new_masks_random_ids = non_masks_ids[random_ids[0:int(non_masks_ids.shape[0]*missing_ratio)]]
        
#         missing_random_ids = random_ids[]
        
        masks_view[new_masks_random_ids,j] = 0 
        
        random_removal_masks[new_masks_random_ids,j] = 0 
             
    
    print('training part missing ratio::', 1 - torch.mean(masks_view))
    
    print('forecasting part missing ratio::', 1 - torch.mean(test_masks))
    
    
    masks =  masks_view.view(inference_masks.shape[0], inference_masks.shape[1], inference_masks.shape[2])
    
    masks = torch.cat([masks, test_masks], 1)
    
    random_removal_masks = random_removal_masks.view(inference_masks.shape[0], inference_masks.shape[1], inference_masks.shape[2])
    
    random_removal_masks = torch.cat([random_removal_masks, torch.ones_like(test_masks)], 1)
    
    print('full missing ratio::', 1 - torch.mean(masks))
    
    return masks, random_removal_masks



def split_and_subsample_batch(data_dict, args, data_type = "train"):
    if data_type == "train":
        # Training set
#         if args.extrap:
        processed_dict = split_data_extrap(data_dict, dataset = args.dataset)
#         else:
#             processed_dict = split_data_interp(data_dict)

    else:
        # Test set
#         if args.extrap:
        processed_dict = split_data_extrap(data_dict, dataset = args.dataset)
#         else:
#             processed_dict = split_data_interp(data_dict)

    # add mask
    processed_dict = add_mask(processed_dict)

    # Subsample points or cut out the whole section of the timeline
#     if (args.sample_tp is not None) or (args.cut_tp is not None):
#         processed_dict = subsample_observed_data(processed_dict, 
#             n_tp_to_sample = args.sample_tp, 
#             n_points_to_cut = args.cut_tp)

    # if (args.sample_tp is not None):
    #     processed_dict = subsample_observed_data(processed_dict, 
    #         n_tp_to_sample = args.sample_tp)
    return processed_dict



def get_forecasting_res_by_time_steps(x, forecasted_x, x_mask):
    
    time_len = x.shape[1]
    
    rmse_list = torch.zeros(time_len)
    
    mae_list = torch.zeros(time_len)
    
    for t in range(time_len):
        rmse = torch.sqrt(torch.sum((x[:,t] - forecasted_x[:,t])**2*x_mask[:,t])/torch.sum(x_mask[:,t]))
        
        mae = (torch.sum(torch.abs(x[:,t] - forecasted_x[:,t])*x_mask[:,t])/torch.sum(x_mask[:,t]))
        
        rmse_list[t] = rmse
    
        mae_list[t] = mae
        
        
    print('forecasting error in time steps (rmse)')
    
    print(rmse_list)

    print('forecasting error in time steps (mae)')
    
    print(mae_list)
    
    return rmse_list, mae_list
