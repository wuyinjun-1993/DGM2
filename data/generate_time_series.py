'''
Created on Jun 17, 2020

'''
import os,sys
import numpy as np

import torch
import torch.nn as nn

from torch.distributions import uniform
from torch.utils.data import DataLoader



# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Models')


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/lib')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.abspath(__file__))


from lib.utils import *

from lib.load_imputed_data_GRUI import *



from scipy.stats import iqr

min_time_series_len = 10




data_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/" + data_dir

'''remove outlier with IQR method'''

def remove_outliers(dataset, masks):
    
#     masks = torch.ones_like(dataset)
    
    dataset_np = dataset.view(dataset.shape[0]*dataset.shape[1], dataset.shape[2])
    
    masks_np = masks.view(dataset.shape[0]*dataset.shape[1], dataset.shape[2])
    
    for i in range(dataset_np.shape[1]):
        
        masked_dataset = dataset_np[masks_np[:,i] != 0,i].numpy()
        
        iqr_score = iqr(masked_dataset)*1.5
        
        lower = np.quantile(masked_dataset, 0.25) - iqr_score
    
        higher = np.quantile(masked_dataset, 0.75) + iqr_score
        
        masks_np[(dataset_np[:,i] > higher), i] = 0
        
        masks_np[(dataset_np[:,i] < lower), i] = 0
        
#         masks_np[(dataset_np[:,i] > higher) + (dataset_np[:,i] < lower), i] = 0
        
        dataset_np[(dataset_np[:,i] > higher), i] = -1000
        
        dataset_np[(dataset_np[:,i] < lower), i] = -1000
        
#     iqr_score = iqr(dataset_np, axis = 0)*1.5
    
#     lower = np.quantile(dataset_np, 0.25, axis = 0) - iqr_score
#     
#     higher = np.quantile(dataset_np, 0.75, axis = 0) + iqr_score
    
    
    masks = masks_np.view(dataset.shape[0], dataset.shape[1], dataset.shape[2])
    
    new_dataset = dataset_np.view(dataset.shape[0], dataset.shape[1], dataset.shape[2])
    
    return new_dataset, masks
    
def remove_outliers2(dataset, masks):
    
#     masks = torch.ones_like(dataset)
    
    dataset_np = dataset.view(dataset.shape[0]*dataset.shape[1], dataset.shape[2])
    
    masks_np = masks.view(dataset.shape[0]*dataset.shape[1], dataset.shape[2])
    
    for i in range(dataset_np.shape[1]):
        
        non_masked_ids = (masks_np[:,i] != 0)
        
        masked_dataset = dataset_np[non_masked_ids,i]
        
        masked_mean = torch.mean(masked_dataset)
        
        masked_std = torch.sqrt(torch.mean((masked_dataset - masked_mean)**2))
        
        lower = masked_mean - 3*masked_std
        
        higher = masked_mean + 3*masked_std
        
        
#         print('bound at ::', i, lower, higher)
#         dataset_np[:,i]
#         
#         iqr_score = iqr(masked_dataset)*1.5
#         
#         lower = np.quantile(masked_dataset, 0.25) - iqr_score
#     
#         higher = np.quantile(masked_dataset, 0.75) + iqr_score
        
        masks_np[(dataset_np[:,i] > higher) + (dataset_np[:,i] < lower), i] = 0
#     iqr_score = iqr(dataset_np, axis = 0)*1.5
    
#     lower = np.quantile(dataset_np, 0.25, axis = 0) - iqr_score
#     
#     higher = np.quantile(dataset_np, 0.75, axis = 0) + iqr_score
    
    
    new_masks = masks_np.view(dataset.shape[0], dataset.shape[1], dataset.shape[2])
    
#     dataset[new_masks != masks]
    
    return dataset, new_masks    
    
    
    

def standardize_dataset(training_set, test_set, mask_train, mask_test):
    
    origin_training_set = training_set.clone()
    
    origin_test_set = test_set.clone()
    
    mean = torch.sum((training_set*mask_train).view(training_set.shape[0]*training_set.shape[1], training_set.shape[2]), dim = 0)/torch.sum(mask_train.view(training_set.shape[0]*training_set.shape[1], training_set.shape[2]), dim = 0)
    
    train_mean = mean.expand(training_set.shape[0], training_set.shape[1], mean.shape[0])
    
    std =  torch.sqrt(torch.sum(((training_set*mask_train).view(training_set.shape[0]*training_set.shape[1], training_set.shape[2]) - (train_mean*mask_train).view(training_set.shape[0]*training_set.shape[1], training_set.shape[2]))**2, dim = 0)/torch.sum(mask_train.view(training_set.shape[0]*training_set.shape[1], training_set.shape[2]), dim = 0))

    train_std = std.expand(training_set.shape[0], training_set.shape[1], std.shape[0])

    dims = (std != 0)

    training_set[:,:,dims] = (training_set[:,:, dims] - train_mean[:,:, dims])/train_std[:,:, dims]
    
    
    test_mean = mean.expand(test_set.shape[0], test_set.shape[1], mean.shape[0])
    
    test_std = std.expand(test_set.shape[0], test_set.shape[1], std.shape[0])
    
    test_set[:,:, dims] = (test_set[:,:, dims] - test_mean[:,:, dims])/test_std[:,:, dims]
    
    return training_set, test_set

def normalize_dataset(training_set, test_set, mask_train, mask_test):

    
    print('normalization start!!')
    
    x_max = torch.max((training_set*mask_train).view(training_set.shape[0]*training_set.shape[1], training_set.shape[2]), axis = 0)[0]
    
    x_min = torch.min((training_set*mask_train).view(training_set.shape[0]*training_set.shape[1], training_set.shape[2]), axis = 0)[0]
    
    range = x_max - x_min
    
    update_data = training_set.clone().view(training_set.shape[0]*training_set.shape[1], training_set.shape[2])
    
    update_test_data = test_set.clone().view(test_set.shape[0]*test_set.shape[1], test_set.shape[2])
    
    
#     print(average_value.shape)
#     
#     print(data)
#     
#     print(average_value)
#     
#     print(std_value)
    
    update_data[:,range!=0] = (update_data[:,range!=0] - x_min[range!=0])/range[range!=0]
    
    update_test_data[:,range!=0] = (update_test_data[:,range!=0] - x_min[range!=0])/range[range!=0]
    
#     data = data /std_value
    
    return update_data.view(training_set.shape[0], training_set.shape[1], training_set.shape[2]), update_test_data.view(test_set.shape[0], test_set.shape[1], test_set.shape[2])


def get_features_with_one_value(masks, masks2):
    
#     for i in range(masks.shape[2]):
    all_features = torch.sum(masks.view(masks.shape[0]*masks.shape[1], masks.shape[2]), 0)
    
    all_features2 = torch.sum(masks2.view(masks2.shape[0]*masks2.shape[1], masks2.shape[2]), 0)
    
    return (all_features > 0)*(all_features2 > 0)
    
    

def get_train_mean(data_obj, inference_len):
     
     
    train_sum = 0
     
    count = 0 
    
    for id, data_dict in enumerate(data_obj["train_dataloader"]):
         
        for k in range(data_dict["observed_lens"].shape[0]):
            
#             if dataset_name.startswith('mimic3'):
#                 len = data_dict["observed_lens"][k]
#             else:
#                 len = 48
             
            len = inference_len
            train_sum += torch.sum(data_dict["observed_data"][k,0:len]*data_dict['observed_mask'][k,0:len], dim = [0])
            
            count += torch.sum(data_dict['observed_mask'][k,0:len], dim = 0)
         
         
     
    train_mean = train_sum/count
     
    return train_mean


    
#     tr_train_set = train_set.view(train_set.shape[0]*train_set.shape[1], -1)
    
#     tr_train_set[]
    
#     tr_train_set[torch.sum(np.isnan(tr_train_set), 1) < train_set.shape[2]]
    
def check_delta_time_stamps(masks, time_stamps, exp_delta_time_stamps):
    
    all_ids = torch.tensor(list(range(time_stamps.shape[1]-1)))
    
    delta_time_stamps = torch.zeros_like(time_stamps)
    
    delta_time_stamps[:, all_ids + 1] = time_stamps[:,all_ids + 1] - time_stamps[:,all_ids]
    
    time_gap_tensors = torch.zeros(masks.shape[2], dtype = torch.float)

    res_delta_time_stamps = torch.zeros_like(masks, dtype = torch.float)

    for k in range(masks.shape[1]):
          
        res_delta_time_stamps[0,k] = time_gap_tensors + delta_time_stamps[0,k]
          
        time_gap_tensors= (1 - masks[0,k])*time_gap_tensors + (1 - masks[0,k])*delta_time_stamps[0,k]
        
    print('diff::', torch.norm(res_delta_time_stamps[0] - exp_delta_time_stamps[0]))
        
        
        
def check_remove_none(train_y, new_train_y):

    sample_id = 0
    
    count = 0
    
    for i in range(train_y[sample_id].shape[0]):
        num_nan = torch.sum(np.isnan(train_y[sample_id][i]))
        
        if not num_nan == train_y.shape[2]:
            for j in range(train_y[sample_id].shape[1]):
                if torch.isnan(train_y[sample_id,i,j]).item():
                    assert torch.isnan(new_train_y[sample_id, count, j])
                    continue
                
                else:
                    
                    if torch.isnan(new_train_y[sample_id, count, j]).item():
                        assert torch.isnan(train_y[sample_id,i,j])
                        continue
                    else:
                        
                        assert train_y[sample_id,i,j].item() == new_train_y[sample_id, count, j].item()
                
#                 print('here')
            count +=1
        else:
            continue   
    
#     shifted_ids = all_ids + 1
#     all_ids = all_ids.expand(time_stamps.shape[0], time_stamps.shape[1], all_ids.shape[0])

models_to_remove_none_time_stamps = [GRUD_method, 'DHMM_cluster_tlstm']

def parse_datasets(args, device):
    
    
    
    def basic_collate_fn(batch, time_steps, args = args, device = device, data_type = "train"):
        
        converted_batches = zip(*batch)
        
        id = 0
        
        for c_batch in converted_batches:
            if id == 0:
                batched_data = torch.stack(list(c_batch))
            
            if id == 1:
                batched_mask = torch.stack(list(c_batch))
            
            if id == 2:
                batched_origin_data = torch.stack(list(c_batch))
                
            if id == 3:
                batched_origin_masks = torch.stack(list(c_batch))
            
            if id == 4:
                batched_new_random_masks = torch.stack(list(c_batch))  
                
            if id == 5:
                batched_tensor_len = torch.tensor(list(c_batch))
            
            
            if id == 6:
                batched_time_stamps = torch.stack(list(c_batch))
            
            if id == 7:
                batched_delta_time_stamps = torch.stack(list(c_batch))
                
            
                
            if id == 8:
                batched_ids = torch.tensor(list(c_batch))
                
            id += 1
                
            
        
#         batch = torch.stack(batch)
        data_dict = {
            "data": batched_data, 
            "lens": batched_tensor_len,
            'origin_data': batched_origin_data,
            "origin_mask": batched_origin_masks,
            "time_stamps": batched_time_stamps,
            "delta_time_stamps": batched_delta_time_stamps,
            "new_random_mask": batched_new_random_masks,
            "time_steps": time_steps,
            "ids": batched_ids,
            "mask": batched_mask
            }
 
        data_dict = split_and_subsample_batch(data_dict, args, data_type = data_type)
        return data_dict


    dataset_name = args.dataset

    n_total_tp = args.timepoints + args.extrap
#     max_t_extrap = args.max_t / args.timepoints * n_total_tp
    
    max_t_extrap = 5 / args.timepoints * n_total_tp

    
    
    distribution = uniform.Uniform(torch.Tensor([0.0]),torch.Tensor([max_t_extrap]))
    time_steps_extrap =  distribution.sample(torch.Size([n_total_tp-1]))[:,0]
    time_steps_extrap = torch.cat((torch.Tensor([0.0]), time_steps_extrap))
    time_steps_extrap = torch.sort(time_steps_extrap)[0]


#     time_steps_extrap = torch.tensor(list(range(n_total_tp)))

    dataset_obj = None
    
   
    ##################################################################
    # Sample a periodic function
#     if dataset_name == "periodic":
#         dataset_obj = Periodic_1d(
#             init_freq = None, init_amplitude = 1.,
#             final_amplitude = 1., final_freq = None, 
#             z0 = 1.)
# 
# 
#     ##################################################################
# 
#         if dataset_obj is None:
#             raise Exception("Unknown dataset: {}".format(dataset_name))
#     
#     
#         if args.new:
#             dataset = dataset_obj.sample_traj(time_steps_extrap, n_samples = args.n, noise_weight = args.noise_weight)
#             
#             args.n = dataset.shape[0]
#             
#             if not os.path.exists(data_folder):
#                 os.makedirs(data_folder)
#             
#             if not os.path.exists(data_folder + "/" + synthetic_sub_data_dir):
#                 os.makedirs(data_folder + "/" + synthetic_sub_data_dir)
#             
#             time_steps_extrap = torch.tensor(list(range(n_total_tp)))
#             
#             torch.save(dataset, data_folder + "/" + synthetic_sub_data_dir + 'synthetic_data_tensor')
#             
#             torch.save(time_steps_extrap, data_folder + "/" + synthetic_sub_data_dir + 'time_steps')
#             
#             
#         else:
#             dataset = torch.load(data_folder + "/" + synthetic_sub_data_dir + 'synthetic_data_tensor')
#             
#             time_steps_extrap = torch.load(data_folder + "/" + synthetic_sub_data_dir + 'time_steps')
#             
#             args.n = dataset.shape[0]
# 
#     # Process small datasets
#     
# #         time_steps_extrap = torch.tensor(list(range(n_total_tp)))
#         
#         time_steps_extrap = time_steps_extrap.to(device)
#         
#         
#         train_y, test_y = split_train_test(dataset, train_fraq = 0.8)
#         
# #         masks = add_random_missing_values(dataset, args.missing_ratio)
#         
#         masks_train, random_train_masks = add_random_missing_values(train_y, torch.ones_like(train_y), args.missing_ratio)
#         
#         masks_test, random_test_masks = add_random_missing_values(test_y, torch.ones_like(test_y), args.missing_ratio)
#                 
# #         dataset = dataset.to(device)
#         
#         origin_train_mask = torch.ones_like(train_y)
#         
#         origin_test_mask = torch.ones_like(test_y)
#         
#         train_lens = torch.ones(train_y.shape[0], dtype = torch.long)*train_y.shape[1]
#             
#         test_lens = torch.ones(test_y.shape[0], dtype = torch.long)*test_y.shape[1]
#         
#         wrapped_train_y = MyDataset(train_y, masks_train, origin_train_mask, masks_train, train_lens)
#     
#         wrapped_test_y = MyDataset(test_y, masks_test, origin_test_mask, masks_test, test_lens)
# 
#     if dataset_name == 'pamap':
#         dataset = torch.load('.gitignore/' + pamap_folder + '/pamap_tensor').type(torch.FloatTensor)
#         time_steps_extrap = torch.tensor(list(range(dataset.shape[1])))
#         dataset = normalize_dataset(dataset)
#         args.n = dataset.shape[0]
#         
#     if dataset_name == 'shl':
#         
#         if args.new:
#         
#             dataset = torch.load(data_folder + shl_tensor_folder + '/shl_tensor').type(torch.FloatTensor)
#             time_steps_extrap = torch.tensor(list(range(dataset.shape[1])))
#             args.n = dataset.shape[0]
#     
#             train_y, test_y = split_train_test(dataset, train_fraq = 0.8)
#             
#             
#             train_time_stamps = torch.tensor(list(range(train_y.shape[1])))
#             
#             train_time_stamps = train_time_stamps.expand(train_y.shape[0], train_y.shape[1])
#             
#             
#             train_lens = torch.ones(train_y.shape[0], dtype = torch.long)*train_y.shape[1]
#             
#             test_lens = torch.ones(test_y.shape[0], dtype = torch.long)*test_y.shape[1]
#     
#             test_time_stamps = torch.tensor(list(range(test_y.shape[1])))
#     
#             test_time_stamps = test_time_stamps.expand(test_y.shape[0], test_y.shape[1])
#             
#             
# #             train_delta_time_stamps = get_delta_time_stamps(train_time_stamps)
# #             
# #             test_delta_time_stamps = get_delta_time_stamps(test_time_stamps)
# 
#             masks_train = torch.ones_like(train_y)
#     
# #             masks_train = remove_outliers2(train_y, masks_train)
#     
#             masks_test = torch.ones_like(test_y)
#             
#             train_delta_time_stamps = train_time_stamps.clone()
#             
#             test_delta_time_stamps = test_time_stamps.clone()
#             
#             if args.model == 'DHMM_cluster_tlstm':
#                 train_delta_time_stamps =  get_delta_time_stamps_all_dims(train_time_stamps)
#                 
#                 test_delta_time_stamps = get_delta_time_stamps_all_dims(test_time_stamps)
#             
#             if args.model == GRUD_method:
#                 train_delta_time_stamps = get_delta_time_stamps(masks_train, train_time_stamps)
#                 
#                 test_delta_time_stamps = get_delta_time_stamps(masks_test, test_time_stamps)
#     
# 
#     
# #             masks_test = remove_outliers2(test_y, masks_test)
#     
#             origin_train_masks = masks_train.clone()
#             
#             origin_test_masks = masks_test.clone()
#     
#             train_y, test_y = standardize_dataset(train_y, test_y, masks_train, masks_test)
#             masks_train, random_train_masks = add_random_missing_values(train_y, masks_train, args.missing_ratio)
#             
#             masks_test, random_test_masks = add_random_missing_values(test_y, masks_test, args.missing_ratio)
#             
#             
# #             train_y, test_y = normalize_dataset(train_y, test_y, masks_train, masks_test)
#             
#             
#     
#             
# #             dataset = normalize_dataset(dataset)
#     
#     
#     #         train_y = train_y.to(device)
#     #         
#     #         test_y = test_y.to(device)
#             
#             if args.model == 'Linear_regression':
#                 
#                 train_y_copy = train_y.clone()
#                 
#                 test_y_copy = test_y.clone()
#                 
#                 train_y_copy[masks_train == 0] = 0
#                 test_y_copy[masks_test == 0] = 0
#                 
#                 print(torch.norm((train_y_copy - train_y)*masks_train))
#                 
#                 train_y = train_y_copy
#                 
#                 test_y = test_y_copy
#             
#             wrapped_train_y = MyDataset(train_y, masks_train, origin_train_masks, random_train_masks, train_lens, train_time_stamps, train_delta_time_stamps)
#         
#             wrapped_test_y = MyDataset(test_y, masks_test, origin_test_masks, random_test_masks, test_lens, test_time_stamps, test_delta_time_stamps)
#             
#             
#             if not os.path.exists(data_folder):
#                 os.makedirs(data_folder)
#                 
#             if not os.path.exists(data_folder + shl_tensor_folder):
#                 os.makedirs(data_folder + shl_tensor_folder)
#             
#             torch.save(wrapped_train_y, data_folder + shl_tensor_folder + 'dataset_train_y')
#             
#             torch.save(wrapped_test_y, data_folder + shl_tensor_folder + 'dataset_test_y')
#             
#             torch.save(time_steps_extrap, data_folder + shl_tensor_folder + 'time_steps')
#         
#         else:
#             dataset = torch.load(data_folder + shl_tensor_folder + '/shl_tensor').type(torch.FloatTensor)
#             
#             wrapped_train_y = torch.load(data_folder + shl_tensor_folder + 'dataset_train_y')
#             
#             wrapped_test_y = torch.load(data_folder + shl_tensor_folder + 'dataset_test_y') 
# 
#             time_steps_extrap = torch.load(data_folder + shl_tensor_folder + 'time_steps')
#     
#             args.n = wrapped_train_y.data.shape[0] + wrapped_test_y.data.shape[0]
    
    
    if dataset_name.startswith(climate_data_name):
        
        if args.new:
        
            train_dataset = torch.load(os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/training_samples').type(torch.FloatTensor)
            train_y = train_dataset
            
            train_time_stamps = torch.tensor(list(range(train_y.shape[1])))
            
            train_time_stamps = train_time_stamps.expand(train_y.shape[0], train_y.shape[1])
            
            
            train_lens = torch.ones(train_y.shape[0], dtype = torch.long)*train_y.shape[1]
            
            
            
            masks_train = torch.load(os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/training_masks')
            
            test_dataset = torch.load(os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/test_samples').type(torch.FloatTensor)
            
            test_y = test_dataset
            
            test_lens = torch.ones(test_y.shape[0], dtype = torch.long)*test_y.shape[1]
            
            test_time_stamps = torch.tensor(list(range(test_y.shape[1])))
    
            test_time_stamps = test_time_stamps.expand(test_y.shape[0], test_y.shape[1])
            
            masks_test = torch.load(os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/test_masks')
#             test_lens = torch.load(os.path.join(data_folder, mimic3_data_dir) + '/mimic3_test_tensor_len').type(torch.LongTensor)
            
            if args.model in models_to_remove_none_time_stamps:
            
                new_train_y, masks_train, train_time_stamps, train_lens = remove_none_observations(train_y, masks_train, train_time_stamps, train_lens)
                
                new_test_y, masks_test, test_time_stamps, test_lens = remove_none_observations(test_y, masks_test, test_time_stamps, test_lens)
            
            
                check_remove_none(train_y, new_train_y)
            
                train_y = new_train_y
                
                test_y = new_test_y
            
            if args.model == 'DHMM_cluster_tlstm':
                train_delta_time_stamps =  get_delta_time_stamps_all_dims(train_time_stamps)
                
                test_delta_time_stamps = get_delta_time_stamps_all_dims(test_time_stamps)
            
            if args.model == GRUD_method:
                train_delta_time_stamps = get_delta_time_stamps(masks_train, train_time_stamps)
                
                test_delta_time_stamps = get_delta_time_stamps(masks_test, test_time_stamps)
            
#             check_delta_time_stamps(masks_train, train_time_stamps, train_delta_time_stamps)
            
            print(torch.norm(torch.sum(masks_train, 2) - torch.sum(1-np.isnan(train_y), 2)))
            
            
            print(torch.norm(torch.sum(masks_train, [1,2]) - torch.sum(1-np.isnan(train_y), [1,2])))
            
#             for k in range(masks_train.shape[0]):
#                 for p in range(masks_train.shape[1]):
#                     if not torch.sum(masks_train[k,p]) == torch.sum(1-np.isnan(train_y[k,p])):
#                         print('here')
#             
#                 if not torch.sum(masks_train[k]) == torch.sum(1-np.isnan(train_y[k])):
#                     print('here')
                    
                                
            assert torch.sum(masks_train, dtype=torch.double) == torch.sum(1-np.isnan(train_y), dtype=torch.double)
#             time_steps_extrap = dataset[:,:,0]
            
#             time_steps_extrap = torch.tensor(list(range(dataset.shape[1])))
    
    
            all_features_not_all_missing_values = get_features_with_one_value(masks_train, masks_test)
            
            train_y = train_y[:,:,all_features_not_all_missing_values]
            
            masks_train = masks_train[:,:,all_features_not_all_missing_values]
            
            
            
            
            
            
            train_y = train_y[train_lens >= min_time_series_len]
            
            train_time_stamps = train_time_stamps[train_lens >= min_time_series_len]
            
            if args.model == 'DHMM_cluster_tlstm' or args.model == GRUD_method:
                train_delta_time_stamps = train_delta_time_stamps[train_lens >= min_time_series_len]
            else:
                train_delta_time_stamps = train_time_stamps.clone()
            
            masks_train = masks_train[train_lens >= min_time_series_len]
            
            train_lens = train_lens[train_lens >= min_time_series_len]
            
            
            
            test_y = test_y[:,:,all_features_not_all_missing_values]
            
            
            
            masks_test = masks_test[:,:,all_features_not_all_missing_values]

            
            
            
            test_y = test_y[test_lens > 1]
            
            if args.model == 'DHMM_cluster_tlstm' or args.model == GRUD_method:
                test_delta_time_stamps = test_delta_time_stamps[test_lens >= 1]
            else:
                test_delta_time_stamps = test_time_stamps.clone()
            
            
            test_time_stamps = test_time_stamps[test_lens >= 1]
            
            masks_test = masks_test[test_lens > 1]
            
            test_lens = test_lens[test_lens > 1]




            train_y[train_y != train_y] = -1000
            
            test_y[test_y != test_y] = -1000
            
            args.n = train_y.shape[0] + test_y.shape[0]
            
            
            
#             train_y, test_y = split_train_test(dataset, train_fraq = 0.8)
    
#             masks_train = torch.ones_like(train_y)
    
            train_y, masks_train = remove_outliers2(train_y, masks_train)
    
#             masks_test = torch.ones_like(test_y)
    
            test_y, masks_test = remove_outliers2(test_y, masks_test)
    
            origin_train_masks = masks_train.clone()
            
            origin_test_masks = masks_test.clone()
            random_train_masks = torch.ones_like(origin_train_masks)
        
            random_test_masks = torch.ones_like(origin_test_masks)
    
            train_y, test_y = standardize_dataset(train_y, test_y, masks_train, masks_test)
    
            masks_train, random_train_masks = add_random_missing_values(train_y, masks_train, args.missing_ratio, climate_data_train_len)
             
            masks_test, random_test_masks = add_random_missing_values(test_y, masks_test, args.missing_ratio, climate_data_train_len)
            
            
#             train_y, test_y = normalize_dataset(train_y, test_y, masks_train, masks_test)
            
            
    
            
    #         dataset = normalize_dataset(dataset)
    
    
    #         train_y = train_y.to(device)
    #         
    #         test_y = test_y.to(device)
#             assert torch.sum(masks_train) == torch.sum(1-np.isnan(train_y))

            
#             upper_id = 1060
#             
#             wrapped_train_y = MyDataset(train_y[0:upper_id], masks_train[0:upper_id], origin_train_masks[0:upper_id], random_train_masks[0:upper_id], train_lens[0:upper_id], train_time_stamps[0:upper_id], train_delta_time_stamps[0:upper_id])
            
            if args.model == 'Linear_regression':
                
                train_y_copy = train_y.clone()
                
                test_y_copy = test_y.clone()
                
                train_y_copy[masks_train == 0] = 0
                test_y_copy[masks_test == 0] = 0
                
                print(torch.norm((train_y_copy - train_y)*masks_train))
                
                train_y = train_y_copy
                
                test_y = test_y_copy
            
            if args.model == cluster_ODE_method or args.model == l_ODE_method:
                train_time_stamps = train_time_stamps.type(torch.float)/train_time_stamps.shape[1]
                test_time_stamps = test_time_stamps.type(torch.float)/train_time_stamps.shape[1]
            
            wrapped_train_y = MyDataset(train_y, masks_train, origin_train_masks, random_train_masks, train_lens, train_time_stamps, train_delta_time_stamps)
        
            wrapped_test_y = MyDataset(test_y, masks_test, origin_test_masks, random_test_masks, test_lens, test_time_stamps, test_delta_time_stamps)
            
            
            
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
                
            if not os.path.exists(data_folder + climate_data_dir):
                os.makedirs(data_folder + climate_data_dir)
            
            torch.save(wrapped_train_y, os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/dataset_train_y')
            
            torch.save(wrapped_test_y, os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/dataset_test_y')
            
            torch.save(time_steps_extrap, os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/time_steps')
        
        else:
#             dataset = torch.load(data_folder + shl_tensor_folder + '/shl_tensor').type(torch.FloatTensor)
            
            wrapped_train_y = torch.load(os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/dataset_train_y')
            
            wrapped_test_y = torch.load(os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/dataset_test_y') 

            time_steps_extrap = torch.load(os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/time_steps')

            args.n = wrapped_train_y.data.shape[0] + wrapped_test_y.data.shape[0]
    
    if dataset_name == 'physionet':
        
        if args.new:
        
            train_dataset = torch.load(os.path.join(data_folder, physionet_data_dir) + '/train_dataset_tensor').type(torch.FloatTensor)
            train_y = train_dataset
            
            train_time_stamps = torch.tensor(list(range(train_y.shape[1])))
            
            train_time_stamps = train_time_stamps.expand(train_y.shape[0], train_y.shape[1])
            
            
            train_lens = torch.ones(train_y.shape[0], dtype = torch.long)*train_y.shape[1]
            
            
            
            masks_train = torch.load(os.path.join(data_folder, physionet_data_dir) + '/train_mask_tensor')
            
            test_dataset = torch.load(os.path.join(data_folder, physionet_data_dir) + '/test_dataset_tensor').type(torch.FloatTensor)
            
            test_y = test_dataset
            
            test_lens = torch.ones(test_y.shape[0], dtype = torch.long)*test_y.shape[1]
            
            test_time_stamps = torch.tensor(list(range(test_y.shape[1])))
    
            test_time_stamps = test_time_stamps.expand(test_y.shape[0], test_y.shape[1])
            
            masks_test = torch.load(os.path.join(data_folder, physionet_data_dir) + '/test_mask_tensor')
#             test_lens = torch.load(os.path.join(data_folder, mimic3_data_dir) + '/mimic3_test_tensor_len').type(torch.LongTensor)
            
            if args.model in models_to_remove_none_time_stamps:
            
                new_train_y, masks_train, train_time_stamps, train_lens = remove_none_observations(train_y, masks_train, train_time_stamps, train_lens)
                
                new_test_y, masks_test, test_time_stamps, test_lens = remove_none_observations(test_y, masks_test, test_time_stamps, test_lens)
            
            
                check_remove_none(train_y, new_train_y)
            
                train_y = new_train_y
                
                test_y = new_test_y
            
            if args.model == 'DHMM_cluster_tlstm':
                train_delta_time_stamps =  get_delta_time_stamps_all_dims(train_time_stamps)
                
                test_delta_time_stamps = get_delta_time_stamps_all_dims(test_time_stamps)
            
            if args.model == GRUD_method:
                train_delta_time_stamps = get_delta_time_stamps(masks_train, train_time_stamps)
                
                test_delta_time_stamps = get_delta_time_stamps(masks_test, test_time_stamps)
            
#             check_delta_time_stamps(masks_train, train_time_stamps, train_delta_time_stamps)
            
            print(torch.norm(torch.sum(masks_train, 2) - torch.sum(1-np.isnan(train_y), 2)))
            
            
            print(torch.norm(torch.sum(masks_train, [1,2]) - torch.sum(1-np.isnan(train_y), [1,2])))
            
#             for k in range(masks_train.shape[0]):
#                 for p in range(masks_train.shape[1]):
#                     if not torch.sum(masks_train[k,p]) == torch.sum(1-np.isnan(train_y[k,p])):
#                         print('here')
#             
#                 if not torch.sum(masks_train[k]) == torch.sum(1-np.isnan(train_y[k])):
#                     print('here')
                    
                                
#             assert torch.sum(masks_train, dtype=torch.double) == torch.sum(1-np.isnan(train_y), dtype=torch.double)
#             time_steps_extrap = dataset[:,:,0]
            
#             time_steps_extrap = torch.tensor(list(range(dataset.shape[1])))
    
    
            all_features_not_all_missing_values = get_features_with_one_value(masks_train, masks_test)
            
            train_y = train_y[:,:,all_features_not_all_missing_values]
            
            masks_train = masks_train[:,:,all_features_not_all_missing_values]
            
            
            
            
            
            
            train_y = train_y[train_lens >= min_time_series_len]
            
            train_time_stamps = train_time_stamps[train_lens >= min_time_series_len]
            
            if args.model == 'DHMM_cluster_tlstm' or args.model == GRUD_method:
                train_delta_time_stamps = train_delta_time_stamps[train_lens >= min_time_series_len]
            else:
                train_delta_time_stamps = train_time_stamps.clone()
            
            masks_train = masks_train[train_lens >= min_time_series_len]
            
            train_lens = train_lens[train_lens >= min_time_series_len]
            
            
            
            test_y = test_y[:,:,all_features_not_all_missing_values]
            
            
            
            masks_test = masks_test[:,:,all_features_not_all_missing_values]

            
            
            
            test_y = test_y[test_lens > 1]
            
            if args.model == 'DHMM_cluster_tlstm' or args.model == GRUD_method:
                test_delta_time_stamps = test_delta_time_stamps[test_lens >= 1]
            else:
                test_delta_time_stamps = test_time_stamps.clone()
            
            
            test_time_stamps = test_time_stamps[test_lens >= 1]
            
            masks_test = masks_test[test_lens > 1]
            
            test_lens = test_lens[test_lens > 1]




            train_y[train_y != train_y] = -1000
            
            test_y[test_y != test_y] = -1000
            
            args.n = train_y.shape[0] + test_y.shape[0]
            
            
            
#             train_y, test_y = split_train_test(dataset, train_fraq = 0.8)
    
#             masks_train = torch.ones_like(train_y)
    
            train_y, masks_train = remove_outliers(train_y, masks_train)
    
#             masks_test = torch.ones_like(test_y)
    
            test_y, masks_test = remove_outliers(test_y, masks_test)
    
            origin_train_masks = masks_train.clone()
            
            origin_test_masks = masks_test.clone()
            random_train_masks = torch.ones_like(origin_train_masks)
        
            random_test_masks = torch.ones_like(origin_test_masks)
    
            train_y, test_y = standardize_dataset(train_y, test_y, masks_train, masks_test)
    
            masks_train, random_train_masks = add_random_missing_values(train_y, masks_train, args.missing_ratio, physionet_data_train_len)
             
            masks_test, random_test_masks = add_random_missing_values(test_y, masks_test, args.missing_ratio, physionet_data_train_len)
            
            
#             train_y, test_y = normalize_dataset(train_y, test_y, masks_train, masks_test)
            
            
    
            
    #         dataset = normalize_dataset(dataset)
    
    
    #         train_y = train_y.to(device)
    #         
    #         test_y = test_y.to(device)
#             assert torch.sum(masks_train) == torch.sum(1-np.isnan(train_y))

            
#             upper_id = 1060
#             
#             wrapped_train_y = MyDataset(train_y[0:upper_id], masks_train[0:upper_id], origin_train_masks[0:upper_id], random_train_masks[0:upper_id], train_lens[0:upper_id], train_time_stamps[0:upper_id], train_delta_time_stamps[0:upper_id])
            
            if args.model == 'Linear_regression':
                
                train_y_copy = train_y.clone()
                
                test_y_copy = test_y.clone()
                
                train_y_copy[masks_train == 0] = 0
                test_y_copy[masks_test == 0] = 0
                
                print(torch.norm((train_y_copy - train_y)*masks_train))
                
                train_y = train_y_copy
                
                test_y = test_y_copy
                
                
            if args.model == cluster_ODE_method or args.model == l_ODE_method:
                train_time_stamps = train_time_stamps.type(torch.float)/train_time_stamps.shape[1]
                test_time_stamps = test_time_stamps.type(torch.float)/train_time_stamps.shape[1]
            
            wrapped_train_y = MyDataset(train_y, masks_train, origin_train_masks, random_train_masks, train_lens, train_time_stamps, train_delta_time_stamps)
        
            wrapped_test_y = MyDataset(test_y, masks_test, origin_test_masks, random_test_masks, test_lens, test_time_stamps, test_delta_time_stamps)
            
            
            
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
                
            if not os.path.exists(data_folder + physionet_data_dir):
                os.makedirs(data_folder + physionet_data_dir)
            
            torch.save(wrapped_train_y, data_folder + physionet_data_dir + 'dataset_train_y')
            
            torch.save(wrapped_test_y, data_folder + physionet_data_dir + 'dataset_test_y')
            
            torch.save(time_steps_extrap, data_folder + physionet_data_dir + 'time_steps')
        
        else:
#             dataset = torch.load(data_folder + shl_tensor_folder + '/shl_tensor').type(torch.FloatTensor)
            
            wrapped_train_y = torch.load(data_folder + physionet_data_dir + 'dataset_train_y')
            
            wrapped_test_y = torch.load(data_folder + physionet_data_dir + 'dataset_test_y') 

            time_steps_extrap = torch.load(data_folder + physionet_data_dir + 'time_steps')

            args.n = wrapped_train_y.data.shape[0] + wrapped_test_y.data.shape[0]
#     if dataset_name == 'mimic3_17' or dataset_name == 'mimic3_96' or dataset_name == 'mimic3_110':
    if dataset_name.startswith('mimic3'):
        
        if args.new:
        
            train_dataset = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), dataset_name) + '/mimic3_train_tensor').type(torch.FloatTensor)
            train_y = train_dataset[:,:,1:]
            
#             train_time_stamps = train_dataset[:,:,0] 
            
            single_train_time_stamp = torch.tensor(list(range(72)))
            
            train_time_stamps = single_train_time_stamp.view(1,72)
            
            train_time_stamps = train_time_stamps.repeat(train_dataset.shape[0], 1)
            
            
            masks_train = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), dataset_name) + '/mimic3_train_masks')[:,:,1:]
#             train_lens_exp = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), dataset_name) + '/mimic3_train_tensor_len0').type(torch.LongTensor)
            train_lens = torch.tensor(72)
            
            train_lens = train_lens.repeat(train_dataset.shape[0])
            
            
            test_dataset = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), dataset_name) + '/mimic3_test_tensor').type(torch.FloatTensor)
            
            test_y = test_dataset[:,:,1:]
            
#             test_time_stamps = test_dataset[:,:,0]
            single_test_time_stamp = torch.tensor(list(range(72)))
            
            test_time_stamps = single_test_time_stamp.view(1,72)
            
            test_time_stamps = test_time_stamps.repeat(test_dataset.shape[0], 1)
            
            masks_test = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), dataset_name) + '/mimic3_test_masks')[:,:,1:]
#             test_lens = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), dataset_name) + '/mimic3_test_tensor_len0').type(torch.LongTensor)
            test_lens = torch.tensor(72)
            
            test_lens = test_lens.repeat(test_dataset.shape[0])
            
            if args.model in models_to_remove_none_time_stamps:
            
                new_train_y, masks_train, train_time_stamps, train_lens = remove_none_observations(train_y, masks_train, train_time_stamps, train_lens)
                
                new_test_y, masks_test, test_time_stamps, test_lens = remove_none_observations(test_y, masks_test, test_time_stamps, test_lens)
            
            
                check_remove_none(train_y, new_train_y)
            
                train_y = new_train_y
                
                test_y = new_test_y
            
            train_delta_time_stamps = train_time_stamps.clone()
            
            test_delta_time_stamps = test_time_stamps.clone()
            
            if args.model == 'DHMM_cluster_tlstm':
                train_delta_time_stamps =  get_delta_time_stamps_all_dims(train_time_stamps)
                
                test_delta_time_stamps = get_delta_time_stamps_all_dims(test_time_stamps)
            
            if args.model == GRUD_method:
                train_delta_time_stamps = get_delta_time_stamps(masks_train, train_time_stamps)
                
                test_delta_time_stamps = get_delta_time_stamps(masks_test, test_time_stamps)
            
#             check_delta_time_stamps(masks_train, train_time_stamps, train_delta_time_stamps)
            
            assert torch.sum(masks_train) == torch.sum(1-np.isnan(train_y))
#             time_steps_extrap = dataset[:,:,0]
            
#             time_steps_extrap = torch.tensor(list(range(dataset.shape[1])))
    
    
            all_features_not_all_missing_values = get_features_with_one_value(masks_train.clone(), masks_train.clone())
            
            train_y = train_y[:,:,all_features_not_all_missing_values]
            
            masks_train = masks_train[:,:,all_features_not_all_missing_values]
            
            
            
            
            
            
#             train_y = train_y[train_lens >= min_time_series_len]
#             
#             train_time_stamps = train_time_stamps[train_lens >= min_time_series_len]
#             
#             if args.model == 'DHMM_cluster_tlstm' or args.model == GRUD_method:
#                 train_delta_time_stamps = train_delta_time_stamps[train_lens >= min_time_series_len]
#             else:
#                 train_delta_time_stamps = train_time_stamps.clone()
#             
#             masks_train = masks_train[train_lens >= min_time_series_len]
#             
#             train_lens = train_lens[train_lens >= min_time_series_len]
            
            
            
            test_y = test_y[:,:,all_features_not_all_missing_values]
            
            
            
            masks_test = masks_test[:,:,all_features_not_all_missing_values]

            
            
            
            test_y = test_y[test_lens > 1]
            
#             if args.model == 'DHMM_cluster_tlstm' or args.model == GRUD_method:
#                 test_delta_time_stamps = test_delta_time_stamps[test_lens >= 1]
#             else:
#                 test_delta_time_stamps = test_time_stamps.clone()
            
            
            test_time_stamps = test_time_stamps[test_lens >= 1]
            
            masks_test = masks_test[test_lens > 1]
            
            test_lens = test_lens[test_lens > 1]




            train_y[train_y != train_y] = -1000
            
            test_y[test_y != test_y] = -1000
            
            args.n = train_y.shape[0] + test_y.shape[0]
            
            
            
#             train_y, test_y = split_train_test(dataset, train_fraq = 0.8)
    
#             masks_train = torch.ones_like(train_y)
    
            
#     
#             masks_test = torch.ones_like(test_y)
            train_y, masks_train = remove_outliers2(train_y, masks_train)
            test_y, masks_test = remove_outliers2(test_y, masks_test)
    
            origin_train_masks = masks_train.clone()
            
            origin_test_masks = masks_test.clone()
    
    
            random_train_masks = torch.ones_like(origin_train_masks)
            
            random_test_masks = torch.ones_like(origin_test_masks)
    
            train_y, test_y = standardize_dataset(train_y, test_y, masks_train, masks_test)
    
            masks_train, random_train_masks = add_random_missing_values(train_y, masks_train, args.missing_ratio, mimic3_data_train_len)
             
            masks_test, random_test_masks = add_random_missing_values(test_y, masks_test, args.missing_ratio, mimic3_data_train_len)
            
            
#             train_y, test_y = normalize_dataset(train_y, test_y, masks_train, masks_test)
            
            
    
            
    #         dataset = normalize_dataset(dataset)
    
    
    #         train_y = train_y.to(device)
    #         
    #         test_y = test_y.to(device)
#             assert torch.sum(masks_train) == torch.sum(1-np.isnan(train_y))

            
#             upper_id = 1060
#             
#             wrapped_train_y = MyDataset(train_y[0:upper_id], masks_train[0:upper_id], origin_train_masks[0:upper_id], random_train_masks[0:upper_id], train_lens[0:upper_id], train_time_stamps[0:upper_id], train_delta_time_stamps[0:upper_id])
            
            if args.model == 'Linear_regression':
                
                train_y_copy = train_y.clone()
                
                test_y_copy = test_y.clone()
                
                train_y_copy[masks_train == 0] = 0
                test_y_copy[masks_test == 0] = 0
                
                print(torch.norm((train_y_copy - train_y)*masks_train))
                
                train_y = train_y_copy
                
                test_y = test_y_copy
            
            
            if args.model == cluster_ODE_method or args.model == l_ODE_method:
                train_time_stamps = train_time_stamps.type(torch.float)/train_time_stamps.shape[1]
                test_time_stamps = test_time_stamps.type(torch.float)/train_time_stamps.shape[1]
            
            wrapped_train_y = MyDataset(train_y, masks_train, origin_train_masks, random_train_masks, train_lens, train_time_stamps, train_delta_time_stamps)
        
            wrapped_test_y = MyDataset(test_y, masks_test, origin_test_masks, random_test_masks, test_lens, test_time_stamps, test_delta_time_stamps)
            
            
            
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
                
            if not os.path.exists(data_folder + mimic3_data_dir):
                os.makedirs(data_folder + mimic3_data_dir)
            
            if not os.path.exists(os.path.join(data_folder + mimic3_data_dir, dataset_name)):
                os.makedirs(os.path.join(data_folder + mimic3_data_dir, dataset_name))
            
            torch.save(wrapped_train_y, os.path.join(data_folder + mimic3_data_dir, dataset_name) + '/dataset_train_y')
            
            torch.save(wrapped_test_y, os.path.join(data_folder + mimic3_data_dir, dataset_name) + '/dataset_test_y')
            
            torch.save(time_steps_extrap, os.path.join(data_folder + mimic3_data_dir, dataset_name) + '/time_steps')
        
        else:
#             dataset = torch.load(data_folder + shl_tensor_folder + '/shl_tensor').type(torch.FloatTensor)
            
            wrapped_train_y = torch.load(os.path.join(data_folder + mimic3_data_dir, dataset_name) + '/dataset_train_y')
            
            wrapped_test_y = torch.load(os.path.join(data_folder + mimic3_data_dir, dataset_name) + '/dataset_test_y') 

            time_steps_extrap = torch.load(os.path.join(data_folder + mimic3_data_dir, dataset_name) + '/time_steps')

            args.n = wrapped_train_y.data.shape[0] + wrapped_test_y.data.shape[0]
    
    


    
    
    is_missing = torch.sum(wrapped_train_y.mask) < (wrapped_train_y.mask.shape[0]*wrapped_train_y.mask.shape[1]*wrapped_train_y.mask.shape[2]) 
    
#     args.n = len(dataset)
#     input_dim = dataset.size(-1)
    input_dim = wrapped_train_y.data.shape[-1]

    batch_size = args.batch_size
    train_dataloader = DataLoader(wrapped_train_y, batch_size = batch_size, shuffle=True,
        collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "train"))
    test_dataloader = DataLoader(wrapped_test_y, batch_size = batch_size, shuffle=True,
        collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "test"))
    
    data_objects = {#"dataset_obj": dataset_obj, 
                "train_dataloader": train_dataloader, 
                "test_dataloader": test_dataloader,
                "input_dim": input_dim,
                "n_train_batches": len(train_dataloader),
                "n_test_batches": len(test_dataloader)}

    train_mean = get_train_mean(data_objects, dataset_name)

    return data_objects, time_steps_extrap, is_missing, train_mean


def basic_collate_fn(batch, args, data_type = "train"):
         
        converted_batches = zip(*batch)
         
        id = 0
         
        for c_batch in converted_batches:
            if id == 0:
                batched_data = torch.stack(list(c_batch))
             
            if id == 1:
                batched_mask = torch.stack(list(c_batch))
             
            if id == 2:
                batched_origin_data = torch.stack(list(c_batch))
                 
            if id == 3:
                batched_origin_masks = torch.stack(list(c_batch))
             
            if id == 4:
                batched_new_random_masks = torch.stack(list(c_batch))  
                 
            if id == 5:
                batched_tensor_len = torch.tensor(list(c_batch))
             
             
            if id == 6:
                batched_time_stamps = torch.stack(list(c_batch))
             
            if id == 7:
                batched_delta_time_stamps = torch.stack(list(c_batch))
                 
             
                 
            if id == 8:
                batched_ids = torch.tensor(list(c_batch))
                 
            id += 1
                 
             
         
#         batch = torch.stack(batch)
        data_dict = {
            "data": batched_data, 
            "lens": batched_tensor_len,
            'origin_data': batched_origin_data,
            "origin_mask": batched_origin_masks,
            "time_stamps": batched_time_stamps,
            "delta_time_stamps": batched_delta_time_stamps,
            "new_random_mask": batched_new_random_masks,
#             "time_steps": time_steps,
            "ids": batched_ids,
            "mask": batched_mask
            }
  
        data_dict = split_and_subsample_batch(data_dict, args, data_type = data_type)
        return data_dict


def partition_validation_set(train_y, masks_train, origin_train_masks, random_train_masks, train_lens, train_time_stamps, train_delta_time_stamps):
    
    validation_size = int(train_y.shape[0]/8)
    
    all_ids = torch.randperm(train_y.shape[0])

#     new_train_ids = all_ids[]
    
    validation_ids = all_ids[0:validation_size]
    
    new_training_ids = all_ids[validation_size:]
    
    
    valid_y= train_y[validation_ids]
    
    valid_mask_train = masks_train[validation_ids]
    
    valid_origin_train_masks = origin_train_masks[validation_ids]
    
    valid_random_train_masks = random_train_masks[validation_ids]
    
    valid_train_lens = train_lens[validation_ids]
    
    valid_train_time_stamps = train_time_stamps[validation_ids]
    
    valid_train_delta_time_stamps = train_delta_time_stamps[validation_ids]
    
    
    wrapped_valid_y = MyDataset(valid_y, valid_mask_train, valid_origin_train_masks, valid_random_train_masks, valid_train_lens, valid_train_time_stamps, valid_train_delta_time_stamps)
    
    
    print('validation data size::', valid_y.shape[0])
    
    
    new_train_y = train_y[new_training_ids]
    
    new_masks_train = masks_train[new_training_ids]
    
    new_origin_train_masks = origin_train_masks[new_training_ids]
    
    new_random_train_masks = random_train_masks[new_training_ids]
    
    new_train_lens = train_lens[new_training_ids]
    
    new_train_time_stamps = train_time_stamps[new_training_ids]
    
    new_train_delta_time_stamps = train_delta_time_stamps[new_training_ids]
    
    
    wrapped_train_y = MyDataset(new_train_y, new_masks_train, new_origin_train_masks, new_random_train_masks, new_train_lens, new_train_time_stamps, new_train_delta_time_stamps)
    
    
    print('training data size::', new_train_y.shape[0])
    
    return wrapped_train_y, wrapped_valid_y

def generate_new_time_series(args):
    
    
    
    


    dataset_name = args.dataset

#     n_total_tp = args.timepoints + args.extrap
# #     max_t_extrap = args.max_t / args.timepoints * n_total_tp
#     
#     max_t_extrap = 5 / args.timepoints * n_total_tp
# 
#     
#     
#     distribution = uniform.Uniform(torch.Tensor([0.0]),torch.Tensor([max_t_extrap]))
#     time_steps_extrap =  distribution.sample(torch.Size([n_total_tp-1]))[:,0]
#     time_steps_extrap = torch.cat((torch.Tensor([0.0]), time_steps_extrap))
#     time_steps_extrap = torch.sort(time_steps_extrap)[0]
# 
# 
# #     time_steps_extrap = torch.tensor(list(range(n_total_tp)))
# 
#     dataset_obj = None
    
   
    ##################################################################
    # Sample a periodic function
#     if dataset_name == "periodic":
#         dataset_obj = Periodic_1d(
#             init_freq = None, init_amplitude = 1.,
#             final_amplitude = 1., final_freq = None, 
#             z0 = 1.)
# 
# 
#     ##################################################################
# 
#         if dataset_obj is None:
#             raise Exception("Unknown dataset: {}".format(dataset_name))
#     
#     
#         if args.new:
#             dataset = dataset_obj.sample_traj(time_steps_extrap, n_samples = args.n, noise_weight = args.noise_weight)
#             
#             args.n = dataset.shape[0]
#             
#             if not os.path.exists(data_folder):
#                 os.makedirs(data_folder)
#             
#             if not os.path.exists(data_folder + "/" + synthetic_sub_data_dir):
#                 os.makedirs(data_folder + "/" + synthetic_sub_data_dir)
#             
#             time_steps_extrap = torch.tensor(list(range(n_total_tp)))
#             
#             torch.save(dataset, data_folder + "/" + synthetic_sub_data_dir + 'synthetic_data_tensor')
#             
#             torch.save(time_steps_extrap, data_folder + "/" + synthetic_sub_data_dir + 'time_steps')
#             
#             
#         else:
#             dataset = torch.load(data_folder + "/" + synthetic_sub_data_dir + 'synthetic_data_tensor')
#             
#             time_steps_extrap = torch.load(data_folder + "/" + synthetic_sub_data_dir + 'time_steps')
#             
#             args.n = dataset.shape[0]
# 
#     # Process small datasets
#     
# #         time_steps_extrap = torch.tensor(list(range(n_total_tp)))
#         
#         time_steps_extrap = time_steps_extrap.to(device)
#         
#         
#         train_y, test_y = split_train_test(dataset, train_fraq = 0.8)
#         
# #         masks = add_random_missing_values(dataset, args.missing_ratio)
#         
#         masks_train, random_train_masks = add_random_missing_values(train_y, torch.ones_like(train_y), args.missing_ratio)
#         
#         masks_test, random_test_masks = add_random_missing_values(test_y, torch.ones_like(test_y), args.missing_ratio)
#                 
# #         dataset = dataset.to(device)
#         
#         origin_train_mask = torch.ones_like(train_y)
#         
#         origin_test_mask = torch.ones_like(test_y)
#         
#         train_lens = torch.ones(train_y.shape[0], dtype = torch.long)*train_y.shape[1]
#             
#         test_lens = torch.ones(test_y.shape[0], dtype = torch.long)*test_y.shape[1]
#         
#         wrapped_train_y = MyDataset(train_y, masks_train, origin_train_mask, masks_train, train_lens)
#     
#         wrapped_test_y = MyDataset(test_y, masks_test, origin_test_mask, masks_test, test_lens)
# 
#     if dataset_name == 'pamap':
#         dataset = torch.load('.gitignore/' + pamap_folder + '/pamap_tensor').type(torch.FloatTensor)
#         time_steps_extrap = torch.tensor(list(range(dataset.shape[1])))
#         dataset = normalize_dataset(dataset)
#         args.n = dataset.shape[0]
#         
#     if dataset_name == 'shl':
#         
#         if args.new:
#         
#             dataset = torch.load(data_folder + shl_tensor_folder + '/shl_tensor').type(torch.FloatTensor)
#             time_steps_extrap = torch.tensor(list(range(dataset.shape[1])))
#             args.n = dataset.shape[0]
#     
#             train_y, test_y = split_train_test(dataset, train_fraq = 0.8)
#             
#             
#             train_time_stamps = torch.tensor(list(range(train_y.shape[1])))
#             
#             train_time_stamps = train_time_stamps.expand(train_y.shape[0], train_y.shape[1])
#             
#             
#             train_lens = torch.ones(train_y.shape[0], dtype = torch.long)*train_y.shape[1]
#             
#             test_lens = torch.ones(test_y.shape[0], dtype = torch.long)*test_y.shape[1]
#     
#             test_time_stamps = torch.tensor(list(range(test_y.shape[1])))
#     
#             test_time_stamps = test_time_stamps.expand(test_y.shape[0], test_y.shape[1])
#             
#             
# #             train_delta_time_stamps = get_delta_time_stamps(train_time_stamps)
# #             
# #             test_delta_time_stamps = get_delta_time_stamps(test_time_stamps)
# 
#             masks_train = torch.ones_like(train_y)
#     
# #             masks_train = remove_outliers2(train_y, masks_train)
#     
#             masks_test = torch.ones_like(test_y)
#             
#             train_delta_time_stamps = train_time_stamps.clone()
#             
#             test_delta_time_stamps = test_time_stamps.clone()
#             
#             if args.model == 'DHMM_cluster_tlstm':
#                 train_delta_time_stamps =  get_delta_time_stamps_all_dims(train_time_stamps)
#                 
#                 test_delta_time_stamps = get_delta_time_stamps_all_dims(test_time_stamps)
#             
#             if args.model == GRUD_method:
#                 train_delta_time_stamps = get_delta_time_stamps(masks_train, train_time_stamps)
#                 
#                 test_delta_time_stamps = get_delta_time_stamps(masks_test, test_time_stamps)
#     
# 
#     
# #             masks_test = remove_outliers2(test_y, masks_test)
#     
#             origin_train_masks = masks_train.clone()
#             
#             origin_test_masks = masks_test.clone()
#     
#     
#             masks_train, random_train_masks = add_random_missing_values(train_y, masks_train, args.missing_ratio)
#             
#             masks_test, random_test_masks = add_random_missing_values(test_y, masks_test, args.missing_ratio)
#             
#             
# #             train_y, test_y = normalize_dataset(train_y, test_y, masks_train, masks_test)
#             
#             train_y, test_y = standardize_dataset(train_y, test_y, masks_train, masks_test)
#     
#             
# #             dataset = normalize_dataset(dataset)
#     
#     
#     #         train_y = train_y.to(device)
#     #         
#     #         test_y = test_y.to(device)
#             
#             if args.model == 'Linear_regression':
#                 
#                 train_y_copy = train_y.clone()
#                 
#                 test_y_copy = test_y.clone()
#                 
#                 train_y_copy[masks_train == 0] = 0
#                 test_y_copy[masks_test == 0] = 0
#                 
#                 print(torch.norm((train_y_copy - train_y)*masks_train))
#                 
#                 train_y = train_y_copy
#                 
#                 test_y = test_y_copy
#             
#             wrapped_train_y = MyDataset(train_y, masks_train, origin_train_masks, random_train_masks, train_lens, train_time_stamps, train_delta_time_stamps)
#         
#             wrapped_test_y = MyDataset(test_y, masks_test, origin_test_masks, random_test_masks, test_lens, test_time_stamps, test_delta_time_stamps)
#             
#             
#             if not os.path.exists(data_folder):
#                 os.makedirs(data_folder)
#                 
#             if not os.path.exists(data_folder + shl_tensor_folder):
#                 os.makedirs(data_folder + shl_tensor_folder)
#             
#             torch.save(wrapped_train_y, data_folder + shl_tensor_folder + 'dataset_train_y')
#             
#             torch.save(wrapped_test_y, data_folder + shl_tensor_folder + 'dataset_test_y')
#             
#             torch.save(time_steps_extrap, data_folder + shl_tensor_folder + 'time_steps')
#         
#         else:
#             dataset = torch.load(data_folder + shl_tensor_folder + '/shl_tensor').type(torch.FloatTensor)
#             
#             wrapped_train_y = torch.load(data_folder + shl_tensor_folder + 'dataset_train_y')
#             
#             wrapped_test_y = torch.load(data_folder + shl_tensor_folder + 'dataset_test_y') 
# 
#             time_steps_extrap = torch.load(data_folder + shl_tensor_folder + 'time_steps')
#     
#             args.n = wrapped_train_y.data.shape[0] + wrapped_test_y.data.shape[0]
    
    
    if dataset_name.startswith(climate_data_name):
        
        print('generate climate time series')
        
#         if args.new:
        
        train_dataset = torch.load(os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/training_samples').type(torch.FloatTensor)
        train_y = train_dataset
        
        origin_train_y = train_dataset.clone()
        
        train_time_stamps = torch.tensor(list(range(train_y.shape[1])))
        
        train_time_stamps = train_time_stamps.expand(train_y.shape[0], train_y.shape[1])
        
        
        train_lens = torch.ones(train_y.shape[0], dtype = torch.long)*train_y.shape[1]
        
        
        
        masks_train = torch.load(os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/training_masks')
        
        test_dataset = torch.load(os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/test_samples').type(torch.FloatTensor)
        
        test_y = test_dataset
        
        origin_test_y = test_dataset.clone()
        
        test_lens = torch.ones(test_y.shape[0], dtype = torch.long)*test_y.shape[1]
        
        test_time_stamps = torch.tensor(list(range(test_y.shape[1])))

        test_time_stamps = test_time_stamps.expand(test_y.shape[0], test_y.shape[1])
        
        masks_test = torch.load(os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/test_masks')
        
        
        train_delta_time_stamps = train_time_stamps.clone()
        
        test_delta_time_stamps = test_time_stamps.clone()
#             test_lens = torch.load(os.path.join(data_folder, mimic3_data_dir) + '/mimic3_test_tensor_len').type(torch.LongTensor)
        
#         if args.model in models_to_remove_none_time_stamps:
#         
#             new_train_y, masks_train, train_time_stamps, train_lens = remove_none_observations(train_y, masks_train, train_time_stamps, train_lens)
#             
#             new_test_y, masks_test, test_time_stamps, test_lens = remove_none_observations(test_y, masks_test, test_time_stamps, test_lens)
#         
#         
#             check_remove_none(train_y, new_train_y)
#         
#             train_y = new_train_y
#             
#             test_y = new_test_y
#         
#         if args.model == 'DHMM_cluster_tlstm':
#             train_delta_time_stamps =  get_delta_time_stamps_all_dims(train_time_stamps)
#             
#             test_delta_time_stamps = get_delta_time_stamps_all_dims(test_time_stamps)
#         
#         if args.model == GRUD_method:
#             train_delta_time_stamps = get_delta_time_stamps(masks_train, train_time_stamps)
#             
#             test_delta_time_stamps = get_delta_time_stamps(masks_test, test_time_stamps)
        
#             check_delta_time_stamps(masks_train, train_time_stamps, train_delta_time_stamps)
        
        print(torch.norm(torch.sum(masks_train, 2) - torch.sum(1-np.isnan(train_y), 2)))
        
        
        print(torch.norm(torch.sum(masks_train, [1,2]) - torch.sum(1-np.isnan(train_y), [1,2])))
        
#             for k in range(masks_train.shape[0]):
#                 for p in range(masks_train.shape[1]):
#                     if not torch.sum(masks_train[k,p]) == torch.sum(1-np.isnan(train_y[k,p])):
#                         print('here')
#             
#                 if not torch.sum(masks_train[k]) == torch.sum(1-np.isnan(train_y[k])):
#                     print('here')
                
                            
        assert torch.sum(masks_train, dtype=torch.double) == torch.sum(1-np.isnan(train_y), dtype=torch.double)
#             time_steps_extrap = dataset[:,:,0]
        
#             time_steps_extrap = torch.tensor(list(range(dataset.shape[1])))


        all_features_not_all_missing_values = get_features_with_one_value(masks_train, masks_test)
        
        train_y = train_y[:,:,all_features_not_all_missing_values]
        
        masks_train = masks_train[:,:,all_features_not_all_missing_values]
        
        
        
        
        
        
        train_y = train_y[train_lens >= min_time_series_len]
        
        train_time_stamps = train_time_stamps[train_lens >= min_time_series_len]
        
#         if args.model == 'DHMM_cluster_tlstm' or args.model == GRUD_method:
#             train_delta_time_stamps = train_delta_time_stamps[train_lens >= min_time_series_len]
#         else:
#             train_delta_time_stamps = train_time_stamps.clone()
        
        masks_train = masks_train[train_lens >= min_time_series_len]
        
        train_lens = train_lens[train_lens >= min_time_series_len]
        
        
        
        test_y = test_y[:,:,all_features_not_all_missing_values]
        
        
        
        masks_test = masks_test[:,:,all_features_not_all_missing_values]

        
        
        
        test_y = test_y[test_lens > 1]
        
#         if args.model == 'DHMM_cluster_tlstm' or args.model == GRUD_method:
#             test_delta_time_stamps = test_delta_time_stamps[test_lens >= 1]
#         else:
#             test_delta_time_stamps = test_time_stamps.clone()
        
        
        test_time_stamps = test_time_stamps[test_lens >= 1]
        
        masks_test = masks_test[test_lens > 1]
        
        test_lens = test_lens[test_lens > 1]




        train_y[train_y != train_y] = -1000
        
        test_y[test_y != test_y] = -1000
        
        args.n = train_y.shape[0] + test_y.shape[0]
        
        
        
#             train_y, test_y = split_train_test(dataset, train_fraq = 0.8)

#             masks_train = torch.ones_like(train_y)

        train_y, masks_train = remove_outliers2(train_y, masks_train)

#             masks_test = torch.ones_like(test_y)

        test_y, masks_test = remove_outliers2(test_y, masks_test)

        origin_train_masks = masks_train.clone()
        
        origin_test_masks = masks_test.clone()
        random_train_masks = torch.ones_like(origin_train_masks)
    
        random_test_masks = torch.ones_like(origin_test_masks)

        train_y, test_y = standardize_dataset(train_y, test_y, masks_train, masks_test)

        masks_train, random_train_masks = add_random_missing_values(train_y, masks_train, args.missing_ratio, climate_data_train_len)
         
        masks_test, random_test_masks = add_random_missing_values(test_y, masks_test, args.missing_ratio, climate_data_train_len)
        
        
#         curr_new_mask = origin_train_masks*(1-random_train_masks)
#             
#         assert (torch.nonzero(~(curr_new_mask ==  (1-random_train_masks)))).shape[0] == 0
        
#             train_y, test_y = normalize_dataset(train_y, test_y, masks_train, masks_test)
        
        

        
#         dataset = normalize_dataset(dataset)


#         train_y = train_y.to(device)
#         
#         test_y = test_y.to(device)
#             assert torch.sum(masks_train) == torch.sum(1-np.isnan(train_y))

        
#             upper_id = 1060
#             
#             wrapped_train_y = MyDataset(train_y[0:upper_id], masks_train[0:upper_id], origin_train_masks[0:upper_id], random_train_masks[0:upper_id], train_lens[0:upper_id], train_time_stamps[0:upper_id], train_delta_time_stamps[0:upper_id])
        
#         if args.model == 'Linear_regression':
#             
#             train_y_copy = train_y.clone()
#             
#             test_y_copy = test_y.clone()
#             
#             train_y_copy[masks_train == 0] = 0
#             test_y_copy[masks_test == 0] = 0
#             
#             print(torch.norm((train_y_copy - train_y)*masks_train))
#             
#             train_y = train_y_copy
#             
#             test_y = test_y_copy
#         
#         if args.model == cluster_ODE_method or args.model == l_ODE_method:
#             train_time_stamps = train_time_stamps.type(torch.float)/train_time_stamps.shape[1]
#             test_time_stamps = test_time_stamps.type(torch.float)/train_time_stamps.shape[1]
        
        
        wrapped_train_y, wrapped_valid_y = partition_validation_set(train_y, masks_train, origin_train_masks, random_train_masks, train_lens, train_time_stamps, train_delta_time_stamps)
        
#         wrapped_train_y = MyDataset(train_y, masks_train, origin_train_masks, random_train_masks, train_lens, train_time_stamps, train_delta_time_stamps)
    
        wrapped_test_y = MyDataset(test_y, masks_test, origin_test_masks, random_test_masks, test_lens, test_time_stamps, test_delta_time_stamps)
        
        
        
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            
        if not os.path.exists(data_folder + climate_data_dir):
            os.makedirs(data_folder + climate_data_dir)
        
        torch.save(wrapped_train_y, os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/dataset_train_y')
        
        torch.save(wrapped_valid_y, os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/dataset_valid_y')
        
        torch.save(wrapped_test_y, os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/dataset_test_y')
    
    
    
    if dataset_name.startswith(kddcup_data_name):
         
        print('generate climate time series')
         
#         if args.new:
         
        train_dataset = torch.load(os.path.join(data_folder, beijing_data_dir) + '/training_tensor').type(torch.FloatTensor)
        train_y = train_dataset
         
        origin_train_y = train_dataset.clone()
         
        train_time_stamps = torch.tensor(list(range(train_y.shape[1])))
         
        train_time_stamps = train_time_stamps.expand(train_y.shape[0], train_y.shape[1])
         
         
        train_lens = torch.ones(train_y.shape[0], dtype = torch.long)*train_y.shape[1]
         
         
         
        masks_train = torch.load(os.path.join(data_folder, beijing_data_dir) + '/training_mask')
         
        test_dataset = torch.load(os.path.join(data_folder, beijing_data_dir) + '/test_tensor').type(torch.FloatTensor)
         
        test_y = test_dataset
         
        origin_test_y = test_dataset.clone()
         
        test_lens = torch.ones(test_y.shape[0], dtype = torch.long)*test_y.shape[1]
         
        test_time_stamps = torch.tensor(list(range(test_y.shape[1])))
 
        test_time_stamps = test_time_stamps.expand(test_y.shape[0], test_y.shape[1])
         
        masks_test = torch.load(os.path.join(data_folder, beijing_data_dir) + '/test_mask')
         
         
        train_delta_time_stamps = train_time_stamps.clone()
         
        test_delta_time_stamps = test_time_stamps.clone()
#             test_lens = torch.load(os.path.join(data_folder, mimic3_data_dir) + '/mimic3_test_tensor_len').type(torch.LongTensor)
         
#         if args.model in models_to_remove_none_time_stamps:
#         
#             new_train_y, masks_train, train_time_stamps, train_lens = remove_none_observations(train_y, masks_train, train_time_stamps, train_lens)
#             
#             new_test_y, masks_test, test_time_stamps, test_lens = remove_none_observations(test_y, masks_test, test_time_stamps, test_lens)
#         
#         
#             check_remove_none(train_y, new_train_y)
#         
#             train_y = new_train_y
#             
#             test_y = new_test_y
#         
#         if args.model == 'DHMM_cluster_tlstm':
#             train_delta_time_stamps =  get_delta_time_stamps_all_dims(train_time_stamps)
#             
#             test_delta_time_stamps = get_delta_time_stamps_all_dims(test_time_stamps)
#         
#         if args.model == GRUD_method:
#             train_delta_time_stamps = get_delta_time_stamps(masks_train, train_time_stamps)
#             
#             test_delta_time_stamps = get_delta_time_stamps(masks_test, test_time_stamps)
         
#             check_delta_time_stamps(masks_train, train_time_stamps, train_delta_time_stamps)
         
        print(torch.norm(torch.sum(masks_train, 2) - torch.sum(1-np.isnan(train_y), 2)))
         
         
        print(torch.norm(torch.sum(masks_train, [1,2]) - torch.sum(1-np.isnan(train_y), [1,2])))
         
#             for k in range(masks_train.shape[0]):
#                 for p in range(masks_train.shape[1]):
#                     if not torch.sum(masks_train[k,p]) == torch.sum(1-np.isnan(train_y[k,p])):
#                         print('here')
#             
#                 if not torch.sum(masks_train[k]) == torch.sum(1-np.isnan(train_y[k])):
#                     print('here')
                 
                             
        assert torch.sum(masks_train, dtype=torch.double) == torch.sum(1-np.isnan(train_y), dtype=torch.double)
#             time_steps_extrap = dataset[:,:,0]
         
#             time_steps_extrap = torch.tensor(list(range(dataset.shape[1])))
 
 
        all_features_not_all_missing_values = get_features_with_one_value(masks_train, masks_test)
         
        train_y = train_y[:,:,all_features_not_all_missing_values]
         
        masks_train = masks_train[:,:,all_features_not_all_missing_values]
         
         
         
         
         
         
        train_y = train_y[train_lens >= min_time_series_len]
         
        train_time_stamps = train_time_stamps[train_lens >= min_time_series_len]
         
#         if args.model == 'DHMM_cluster_tlstm' or args.model == GRUD_method:
#             train_delta_time_stamps = train_delta_time_stamps[train_lens >= min_time_series_len]
#         else:
#             train_delta_time_stamps = train_time_stamps.clone()
         
        masks_train = masks_train[train_lens >= min_time_series_len]
         
        train_lens = train_lens[train_lens >= min_time_series_len]
         
         
         
        test_y = test_y[:,:,all_features_not_all_missing_values]
         
         
         
        masks_test = masks_test[:,:,all_features_not_all_missing_values]
 
         
         
         
        test_y = test_y[test_lens > 1]
         
#         if args.model == 'DHMM_cluster_tlstm' or args.model == GRUD_method:
#             test_delta_time_stamps = test_delta_time_stamps[test_lens >= 1]
#         else:
#             test_delta_time_stamps = test_time_stamps.clone()
         
         
        test_time_stamps = test_time_stamps[test_lens >= 1]
         
        masks_test = masks_test[test_lens > 1]
         
        test_lens = test_lens[test_lens > 1]
 
 
 
 
        train_y[train_y != train_y] = -1000
         
        test_y[test_y != test_y] = -1000
         
        args.n = train_y.shape[0] + test_y.shape[0]
         
         
         
#             train_y, test_y = split_train_test(dataset, train_fraq = 0.8)
 
#             masks_train = torch.ones_like(train_y)
 
        train_y, masks_train = remove_outliers2(train_y, masks_train)
 
#             masks_test = torch.ones_like(test_y)
 
        test_y, masks_test = remove_outliers2(test_y, masks_test)
 
        origin_train_masks = masks_train.clone()
         
        origin_test_masks = masks_test.clone()
        random_train_masks = torch.ones_like(origin_train_masks)
     
        random_test_masks = torch.ones_like(origin_test_masks)
 
        train_y, test_y = standardize_dataset(train_y, test_y, masks_train, masks_test)
 
        print(train_y.shape)
 
        masks_train, random_train_masks = add_random_missing_values(train_y, masks_train, args.missing_ratio, beijing_data_train_len)
          
        masks_test, random_test_masks = add_random_missing_values(test_y, masks_test, args.missing_ratio, beijing_data_train_len)
         
         
#             train_y, test_y = normalize_dataset(train_y, test_y, masks_train, masks_test)
         
         
 
         
#         dataset = normalize_dataset(dataset)
 
 
#         train_y = train_y.to(device)
#         
#         test_y = test_y.to(device)
#             assert torch.sum(masks_train) == torch.sum(1-np.isnan(train_y))
 
         
#             upper_id = 1060
#             
#             wrapped_train_y = MyDataset(train_y[0:upper_id], masks_train[0:upper_id], origin_train_masks[0:upper_id], random_train_masks[0:upper_id], train_lens[0:upper_id], train_time_stamps[0:upper_id], train_delta_time_stamps[0:upper_id])
         
#         if args.model == 'Linear_regression':
#             
#             train_y_copy = train_y.clone()
#             
#             test_y_copy = test_y.clone()
#             
#             train_y_copy[masks_train == 0] = 0
#             test_y_copy[masks_test == 0] = 0
#             
#             print(torch.norm((train_y_copy - train_y)*masks_train))
#             
#             train_y = train_y_copy
#             
#             test_y = test_y_copy
#         
#         if args.model == cluster_ODE_method or args.model == l_ODE_method:
#             train_time_stamps = train_time_stamps.type(torch.float)/train_time_stamps.shape[1]
#             test_time_stamps = test_time_stamps.type(torch.float)/train_time_stamps.shape[1]
         
         
        wrapped_train_y, wrapped_valid_y = partition_validation_set(train_y, masks_train, origin_train_masks, random_train_masks, train_lens, train_time_stamps, train_delta_time_stamps)
         
#         wrapped_train_y = MyDataset(train_y, masks_train, origin_train_masks, random_train_masks, train_lens, train_time_stamps, train_delta_time_stamps)
     
        wrapped_test_y = MyDataset(test_y, masks_test, origin_test_masks, random_test_masks, test_lens, test_time_stamps, test_delta_time_stamps)
         
         
         
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
             
        if not os.path.exists(data_folder + climate_data_dir):
            os.makedirs(data_folder + climate_data_dir)
         
        torch.save(wrapped_train_y, os.path.join(data_folder, beijing_data_dir) + '/dataset_train_y')
         
        torch.save(wrapped_test_y, os.path.join(data_folder, beijing_data_dir) + '/dataset_test_y')
         
        torch.save(wrapped_valid_y, os.path.join(data_folder, beijing_data_dir) + '/dataset_valid_y')
#         torch.save(time_steps_extrap, os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/time_steps')
         
#         else:
# #             dataset = torch.load(data_folder + shl_tensor_folder + '/shl_tensor').type(torch.FloatTensor)
#             
#             wrapped_train_y = torch.load(os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/dataset_train_y')
#             
#             wrapped_test_y = torch.load(os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/dataset_test_y') 
# 
#             time_steps_extrap = torch.load(os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name) + '/time_steps')
# 
#             args.n = wrapped_train_y.data.shape[0] + wrapped_test_y.data.shape[0]
#     
#     if dataset_name == 'physionet':
#         
# #         if args.new:
#         
#             train_dataset = torch.load(os.path.join(data_folder, physionet_data_dir) + '/train_dataset_tensor').type(torch.FloatTensor)
#             train_y = train_dataset
#             
# #             train_time_stamps = torch.tensor(list(range(train_y.shape[1])))
# #             
# #             train_time_stamps = train_time_stamps.expand(train_y.shape[0], train_y.shape[1])
#             
#             
#             train_time_stamps = torch.load(os.path.join(data_folder, physionet_data_dir) + '/train_time_stamps')
#             
#             train_lens = torch.ones(train_y.shape[0], dtype = torch.long)*train_y.shape[1]
#             
#             
#             
#             masks_train = torch.load(os.path.join(data_folder, physionet_data_dir) + '/train_mask_tensor')
#             
#             test_dataset = torch.load(os.path.join(data_folder, physionet_data_dir) + '/test_dataset_tensor').type(torch.FloatTensor)
#             
#             test_y = test_dataset
#             
#             test_lens = torch.ones(test_y.shape[0], dtype = torch.long)*test_y.shape[1]
#             
# #             test_time_stamps = torch.tensor(list(range(test_y.shape[1])))
# #     
# #             test_time_stamps = test_time_stamps.expand(test_y.shape[0], test_y.shape[1])
#             test_time_stamps = torch.load(os.path.join(data_folder, physionet_data_dir) + '/test_time_stamps')
#             
#             masks_test = torch.load(os.path.join(data_folder, physionet_data_dir) + '/test_mask_tensor')
#             
#             train_delta_time_stamps = train_time_stamps.clone()
#         
#             test_delta_time_stamps = test_time_stamps.clone()
# #             test_lens = torch.load(os.path.join(data_folder, mimic3_data_dir) + '/mimic3_test_tensor_len').type(torch.LongTensor)
#             
# #             if args.model in models_to_remove_none_time_stamps:
# #             
# #                 new_train_y, masks_train, train_time_stamps, train_lens = remove_none_observations(train_y, masks_train, train_time_stamps, train_lens)
# #                 
# #                 new_test_y, masks_test, test_time_stamps, test_lens = remove_none_observations(test_y, masks_test, test_time_stamps, test_lens)
# #             
# #             
# #                 check_remove_none(train_y, new_train_y)
# #             
# #                 train_y = new_train_y
# #                 
# #                 test_y = new_test_y
# #             
# #             if args.model == 'DHMM_cluster_tlstm':
# #                 train_delta_time_stamps =  get_delta_time_stamps_all_dims(train_time_stamps)
# #                 
# #                 test_delta_time_stamps = get_delta_time_stamps_all_dims(test_time_stamps)
# #             
# #             if args.model == GRUD_method:
# #                 train_delta_time_stamps = get_delta_time_stamps(masks_train, train_time_stamps)
# #                 
# #                 test_delta_time_stamps = get_delta_time_stamps(masks_test, test_time_stamps)
#             
# #             check_delta_time_stamps(masks_train, train_time_stamps, train_delta_time_stamps)
#             
#             print(torch.norm(torch.sum(masks_train, 2) - torch.sum(1-np.isnan(train_y), 2)))
#             
#             
#             print(torch.norm(torch.sum(masks_train, [1,2]) - torch.sum(1-np.isnan(train_y), [1,2])))
#             
# #             for k in range(masks_train.shape[0]):
# #                 for p in range(masks_train.shape[1]):
# #                     if not torch.sum(masks_train[k,p]) == torch.sum(1-np.isnan(train_y[k,p])):
# #                         print('here')
# #             
# #                 if not torch.sum(masks_train[k]) == torch.sum(1-np.isnan(train_y[k])):
# #                     print('here')
#                     
#                                 
# #             assert torch.sum(masks_train, dtype=torch.double) == torch.sum(1-np.isnan(train_y), dtype=torch.double)
# #             time_steps_extrap = dataset[:,:,0]
#             
# #             time_steps_extrap = torch.tensor(list(range(dataset.shape[1])))
#     
#     
#             all_features_not_all_missing_values = get_features_with_one_value(masks_train, masks_test)
#             
#             train_y = train_y[:,:,all_features_not_all_missing_values]
#             
#             masks_train = masks_train[:,:,all_features_not_all_missing_values]
#             
#             
#             
#             
#             
#             
#             train_y = train_y[train_lens >= min_time_series_len]
#             
#             train_time_stamps = train_time_stamps[train_lens >= min_time_series_len]
#             
# #             if args.model == 'DHMM_cluster_tlstm' or args.model == GRUD_method:
# #                 train_delta_time_stamps = train_delta_time_stamps[train_lens >= min_time_series_len]
# #             else:
# #                 train_delta_time_stamps = train_time_stamps.clone()
#             
#             masks_train = masks_train[train_lens >= min_time_series_len]
#             
#             train_lens = train_lens[train_lens >= min_time_series_len]
#             
#             
#             
#             test_y = test_y[:,:,all_features_not_all_missing_values]
#             
#             
#             
#             masks_test = masks_test[:,:,all_features_not_all_missing_values]
# 
#             
#             
#             
#             test_y = test_y[test_lens > 1]
#             
# #             if args.model == 'DHMM_cluster_tlstm' or args.model == GRUD_method:
# #                 test_delta_time_stamps = test_delta_time_stamps[test_lens >= 1]
# #             else:
# #                 test_delta_time_stamps = test_time_stamps.clone()
#             
#             
#             test_time_stamps = test_time_stamps[test_lens >= 1]
#             
#             masks_test = masks_test[test_lens > 1]
#             
#             test_lens = test_lens[test_lens > 1]
# 
# 
# 
# 
#             train_y[train_y != train_y] = -1000
#             
#             test_y[test_y != test_y] = -1000
#             
#             args.n = train_y.shape[0] + test_y.shape[0]
#             
#             
#             
# #             train_y, test_y = split_train_test(dataset, train_fraq = 0.8)
#     
# #             masks_train = torch.ones_like(train_y)
#     
#             train_y, masks_train = remove_outliers(train_y, masks_train)
#     
# #             masks_test = torch.ones_like(test_y)
#     
#             test_y, masks_test = remove_outliers(test_y, masks_test)
#     
#             origin_train_masks = masks_train.clone()
#             
#             origin_test_masks = masks_test.clone()
#             random_train_masks = torch.ones_like(origin_train_masks)
#         
#             random_test_masks = torch.ones_like(origin_test_masks)
#     
#             train_y, test_y = standardize_dataset(train_y, test_y, masks_train, masks_test)
#     
#             masks_train, random_train_masks = add_random_missing_values(train_y, masks_train, args.missing_ratio, physionet_data_train_len)
#              
#             masks_test, random_test_masks = add_random_missing_values(test_y, masks_test, args.missing_ratio, physionet_data_train_len)
#             
#             
# #             train_y, test_y = normalize_dataset(train_y, test_y, masks_train, masks_test)
#             
#             
#     
#             
#     #         dataset = normalize_dataset(dataset)
#     
#     
#     #         train_y = train_y.to(device)
#     #         
#     #         test_y = test_y.to(device)
# #             assert torch.sum(masks_train) == torch.sum(1-np.isnan(train_y))
# 
#             
# #             upper_id = 1060
# #             
# #             wrapped_train_y = MyDataset(train_y[0:upper_id], masks_train[0:upper_id], origin_train_masks[0:upper_id], random_train_masks[0:upper_id], train_lens[0:upper_id], train_time_stamps[0:upper_id], train_delta_time_stamps[0:upper_id])
#             
# #             if args.model == 'Linear_regression':
# #                 
# #                 train_y_copy = train_y.clone()
# #                 
# #                 test_y_copy = test_y.clone()
# #                 
# #                 train_y_copy[masks_train == 0] = 0
# #                 test_y_copy[masks_test == 0] = 0
# #                 
# #                 print(torch.norm((train_y_copy - train_y)*masks_train))
# #                 
# #                 train_y = train_y_copy
# #                 
# #                 test_y = test_y_copy
# #                 
# #                 
# #             if args.model == cluster_ODE_method or args.model == l_ODE_method:
# #                 train_time_stamps = train_time_stamps.type(torch.float)/train_time_stamps.shape[1]
# #                 test_time_stamps = test_time_stamps.type(torch.float)/train_time_stamps.shape[1]
#             
#             wrapped_train_y, wrapped_valid_y = partition_validation_set(train_y, masks_train, origin_train_masks, random_train_masks, train_lens, train_time_stamps, train_delta_time_stamps)
#             
# #             wrapped_train_y = MyDataset(train_y, masks_train, origin_train_masks, random_train_masks, train_lens, train_time_stamps, train_delta_time_stamps)
#         
#             wrapped_test_y = MyDataset(test_y, masks_test, origin_test_masks, random_test_masks, test_lens, test_time_stamps, test_delta_time_stamps)
#             
#             
#             
#             if not os.path.exists(data_folder):
#                 os.makedirs(data_folder)
#                 
#             if not os.path.exists(data_folder + physionet_data_dir):
#                 os.makedirs(data_folder + physionet_data_dir)
#             
#             torch.save(wrapped_train_y, data_folder + physionet_data_dir + 'dataset_train_y')
#             
#             torch.save(wrapped_valid_y, data_folder + physionet_data_dir + 'dataset_valid_y')
#             
#             torch.save(wrapped_test_y, data_folder + physionet_data_dir + 'dataset_test_y')
            
#             torch.save(time_steps_extrap, data_folder + physionet_data_dir + 'time_steps')
        
#         else:
# #             dataset = torch.load(data_folder + shl_tensor_folder + '/shl_tensor').type(torch.FloatTensor)
#             
#             wrapped_train_y = torch.load(data_folder + physionet_data_dir + 'dataset_train_y')
#             
#             wrapped_test_y = torch.load(data_folder + physionet_data_dir + 'dataset_test_y') 
# 
#             time_steps_extrap = torch.load(data_folder + physionet_data_dir + 'time_steps')
# 
#             args.n = wrapped_train_y.data.shape[0] + wrapped_test_y.data.shape[0]
#     if dataset_name == 'mimic3_17' or dataset_name == 'mimic3_96' or dataset_name == 'mimic3_110':
    if dataset_name.startswith(mimic_data_name):
         
#         if args.new:
         
        train_dataset = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), dataset_name) + '/mimic3_train_tensor').type(torch.FloatTensor)
        masks_train = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), dataset_name) + '/mimic3_train_masks')[:,:,1:]
 
        print(train_dataset.shape)
         
        train_y = train_dataset[:,:,1:]
         
        if dataset_name == 'mimic3_17_5':
             
            time_gap_in_hour = 1.0/12
             
            time_stamp_count = int(6/time_gap_in_hour)
             
            single_train_time_stamp = torch.tensor(list(range(time_stamp_count)))*time_gap_in_hour
             
            train_time_stamps = single_train_time_stamp.view(1,time_stamp_count)
             
            train_time_stamps = train_time_stamps.repeat(train_dataset.shape[0], 1)
             
            train_lens = torch.tensor(time_stamp_count)
         
            train_lens = train_lens.repeat(train_dataset.shape[0])
             
        else:
            
         
#             train_time_stamps = train_dataset[:,:,0] 
         
            single_train_time_stamp = torch.tensor(list(range(mimic3_data_len)))
             
            train_time_stamps = single_train_time_stamp.view(1,mimic3_data_len)
             
            train_time_stamps = train_time_stamps.repeat(train_dataset.shape[0], 1)
             
            train_lens = torch.tensor(mimic3_data_len)
         
            train_lens = train_lens.repeat(train_dataset.shape[0])
             
            train_y = train_y[:,0:mimic3_data_len]
             
            masks_train = masks_train[:, 0:mimic3_data_len]
         
#             train_lens_exp = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), dataset_name) + '/mimic3_train_tensor_len0').type(torch.LongTensor)
        print('non missing ratio::', torch.mean(masks_train))
         
         
        test_dataset = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), dataset_name) + '/mimic3_test_tensor').type(torch.FloatTensor)
        masks_test = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), dataset_name) + '/mimic3_test_masks')[:,:,1:]
 
         
        test_y = test_dataset[:,:,1:]
         
        if dataset_name == 'mimic3_17_5':
             
            time_gap_in_hour = 1.0/12
             
            time_stamp_count =int(6/time_gap_in_hour)
             
            single_test_time_stamp = torch.tensor(list(range(time_stamp_count)))*time_gap_in_hour
             
            test_time_stamps = single_test_time_stamp.view(1,time_stamp_count)
             
            test_time_stamps = test_time_stamps.repeat(test_dataset.shape[0], 1)
             
            test_lens = torch.tensor(time_stamp_count)
         
            test_lens = test_lens.repeat(test_dataset.shape[0])
             
        else:
             
         
#             test_time_stamps = test_dataset[:,:,0]
            single_test_time_stamp = torch.tensor(list(range(mimic3_data_len)))
             
            test_time_stamps = single_test_time_stamp.view(1,mimic3_data_len)
             
            test_time_stamps = test_time_stamps.repeat(test_dataset.shape[0], 1)
             
            test_lens = torch.tensor(mimic3_data_len)
         
            test_lens = test_lens.repeat(test_dataset.shape[0])
             
            test_y = test_y[:,0:mimic3_data_len]
             
            masks_test = masks_test[:, 0:mimic3_data_len]
         
#             test_lens = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), dataset_name) + '/mimic3_test_tensor_len0').type(torch.LongTensor)
         
         
#             if args.model in models_to_remove_none_time_stamps:
#             
#                 new_train_y, masks_train, train_time_stamps, train_lens = remove_none_observations(train_y, masks_train, train_time_stamps, train_lens)
#                 
#                 new_test_y, masks_test, test_time_stamps, test_lens = remove_none_observations(test_y, masks_test, test_time_stamps, test_lens)
#             
#             
#                 check_remove_none(train_y, new_train_y)
#             
#                 train_y = new_train_y
#                 
#                 test_y = new_test_y
         
        train_delta_time_stamps = train_time_stamps.clone()
         
        test_delta_time_stamps = test_time_stamps.clone()
         
#             if args.model == 'DHMM_cluster_tlstm':
#                 train_delta_time_stamps =  get_delta_time_stamps_all_dims(train_time_stamps)
#                 
#                 test_delta_time_stamps = get_delta_time_stamps_all_dims(test_time_stamps)
#             
#             if args.model == GRUD_method:
#                 train_delta_time_stamps = get_delta_time_stamps(masks_train, train_time_stamps)
#                 
#                 test_delta_time_stamps = get_delta_time_stamps(masks_test, test_time_stamps)
         
#             check_delta_time_stamps(masks_train, train_time_stamps, train_delta_time_stamps)
         
        assert torch.sum(masks_train) == torch.sum(1-np.isnan(train_y))
#             time_steps_extrap = dataset[:,:,0]
         
#             time_steps_extrap = torch.tensor(list(range(dataset.shape[1])))
 
 
        all_features_not_all_missing_values = get_features_with_one_value(masks_train.clone(), masks_train.clone())
         
        train_y = train_y[:,:,all_features_not_all_missing_values]
         
        masks_train = masks_train[:,:,all_features_not_all_missing_values]
         
         
         
         
         
         
#             train_y = train_y[train_lens >= min_time_series_len]
#             
#             train_time_stamps = train_time_stamps[train_lens >= min_time_series_len]
#             
#             if args.model == 'DHMM_cluster_tlstm' or args.model == GRUD_method:
#                 train_delta_time_stamps = train_delta_time_stamps[train_lens >= min_time_series_len]
#             else:
#                 train_delta_time_stamps = train_time_stamps.clone()
#             
#             masks_train = masks_train[train_lens >= min_time_series_len]
#             
#             train_lens = train_lens[train_lens >= min_time_series_len]
         
         
         
        test_y = test_y[:,:,all_features_not_all_missing_values]
         
         
         
        masks_test = masks_test[:,:,all_features_not_all_missing_values]
 
         
         
         
        test_y = test_y[test_lens > 1]
         
#             if args.model == 'DHMM_cluster_tlstm' or args.model == GRUD_method:
#                 test_delta_time_stamps = test_delta_time_stamps[test_lens >= 1]
#             else:
#                 test_delta_time_stamps = test_time_stamps.clone()
         
         
        test_time_stamps = test_time_stamps[test_lens >= 1]
         
        masks_test = masks_test[test_lens > 1]
         
        test_lens = test_lens[test_lens > 1]
 
 
 
 
        train_y[train_y != train_y] = -1000
         
        test_y[test_y != test_y] = -1000
         
        args.n = train_y.shape[0] + test_y.shape[0]
         
         
         
#             train_y, test_y = split_train_test(dataset, train_fraq = 0.8)
 
#             masks_train = torch.ones_like(train_y)
 
         
#     
#             masks_test = torch.ones_like(test_y)
        train_y, masks_train = remove_outliers2(train_y, masks_train)
        test_y, masks_test = remove_outliers2(test_y, masks_test)
 
        origin_train_masks = masks_train.clone()
         
        origin_test_masks = masks_test.clone()
 
 
        random_train_masks = torch.ones_like(origin_train_masks)
         
        random_test_masks = torch.ones_like(origin_test_masks)
 
        train_y, test_y = standardize_dataset(train_y, test_y, masks_train, masks_test)
 
        masks_train, random_train_masks = add_random_missing_values(train_y, masks_train, args.missing_ratio, mimic3_data_train_len)
          
        masks_test, random_test_masks = add_random_missing_values(test_y, masks_test, args.missing_ratio, mimic3_data_train_len)
         
         
#             train_y, test_y = normalize_dataset(train_y, test_y, masks_train, masks_test)
         
         
 
         
#         dataset = normalize_dataset(dataset)
 
 
#         train_y = train_y.to(device)
#         
#         test_y = test_y.to(device)
#             assert torch.sum(masks_train) == torch.sum(1-np.isnan(train_y))
 
         
#             upper_id = 1060
#             
#             wrapped_train_y = MyDataset(train_y[0:upper_id], masks_train[0:upper_id], origin_train_masks[0:upper_id], random_train_masks[0:upper_id], train_lens[0:upper_id], train_time_stamps[0:upper_id], train_delta_time_stamps[0:upper_id])
         
#             if args.model == 'Linear_regression':
#                 
#                 train_y_copy = train_y.clone()
#                 
#                 test_y_copy = test_y.clone()
#                 
#                 train_y_copy[masks_train == 0] = 0
#                 test_y_copy[masks_test == 0] = 0
#                 
#                 print(torch.norm((train_y_copy - train_y)*masks_train))
#                 
#                 train_y = train_y_copy
#                 
#                 test_y = test_y_copy
#             
#             
#             if args.model == cluster_ODE_method or args.model == l_ODE_method:
#                 train_time_stamps = train_time_stamps.type(torch.float)/train_time_stamps.shape[1]
#                 test_time_stamps = test_time_stamps.type(torch.float)/train_time_stamps.shape[1]
         
        wrapped_train_y, wrapped_valid_y = partition_validation_set(train_y, masks_train, origin_train_masks, random_train_masks, train_lens, train_time_stamps, train_delta_time_stamps)
         
#         wrapped_train_y = MyDataset(train_y, masks_train, origin_train_masks, random_train_masks, train_lens, train_time_stamps, train_delta_time_stamps)
     
        wrapped_test_y = MyDataset(test_y, masks_test, origin_test_masks, random_test_masks, test_lens, test_time_stamps, test_delta_time_stamps)
         
         
         
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
             
        if not os.path.exists(data_folder + mimic3_data_dir):
            os.makedirs(data_folder + mimic3_data_dir)
         
        if not os.path.exists(os.path.join(data_folder + mimic3_data_dir, dataset_name)):
            os.makedirs(os.path.join(data_folder + mimic3_data_dir, dataset_name))
         
        torch.save(wrapped_train_y, os.path.join(data_folder + mimic3_data_dir, dataset_name) + '/dataset_train_y')
         
        torch.save(wrapped_test_y, os.path.join(data_folder + mimic3_data_dir, dataset_name) + '/dataset_test_y')
         
        torch.save(wrapped_valid_y, os.path.join(data_folder + mimic3_data_dir, dataset_name) + '/dataset_valid_y')
         
#         torch.save(time_steps_extrap, os.path.join(data_folder + mimic3_data_dir, dataset_name) + '/time_steps')
         
#         else:
# #             dataset = torch.load(data_folder + shl_tensor_folder + '/shl_tensor').type(torch.FloatTensor)
#             
#             wrapped_train_y = torch.load(os.path.join(data_folder + mimic3_data_dir, dataset_name) + '/dataset_train_y')
#             
#             wrapped_test_y = torch.load(os.path.join(data_folder + mimic3_data_dir, dataset_name) + '/dataset_test_y') 
# 
#             time_steps_extrap = torch.load(os.path.join(data_folder + mimic3_data_dir, dataset_name) + '/time_steps')
# 
#             args.n = wrapped_train_y.data.shape[0] + wrapped_test_y.data.shape[0]
    
    


    
    
#     is_missing = torch.sum(wrapped_train_y.mask) < (wrapped_train_y.mask.shape[0]*wrapped_train_y.mask.shape[1]*wrapped_train_y.mask.shape[2]) 
#     
# #     args.n = len(dataset)
# #     input_dim = dataset.size(-1)
#     input_dim = wrapped_train_y.data.shape[-1]
# 
#     batch_size = args.batch_size
#     train_dataloader = DataLoader(wrapped_train_y, batch_size = batch_size, shuffle=True,
#         collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "train"))
#     test_dataloader = DataLoader(wrapped_test_y, batch_size = batch_size, shuffle=True,
#         collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "test"))
#     
#     data_objects = {#"dataset_obj": dataset_obj, 
#                 "train_dataloader": train_dataloader, 
#                 "test_dataloader": test_dataloader,
#                 "input_dim": input_dim,
#                 "n_train_batches": len(train_dataloader),
#                 "n_test_batches": len(test_dataloader)}
# 
#     train_mean = get_train_mean(data_objects, dataset_name)
# 
#     return data_objects, time_steps_extrap, is_missing, train_mean

# def load_time_series(args):
#     if not args.model == 'GRUI':
#         return load_time_series_non_grui(args)
#     else:
#         return load_time_series_grui(args)

def load_time_series_dataset(dataset_name):
#     if dataset_name.startswith('mimic3'):
# #         data_dir = mimic3_data_dir
#         data_dir = os.path.join(os.path.join(data_folder, mimic3_data_dir), dataset_name)
#         
#         training_data_file_name = 'dataset_train_y'
#         
#         
#         test_data_file_name = 'dataset_test_y'
#         
#         inference_len = mimic3_data_train_len
#         
# #         time_steps_file_name = 'time_steps'
#         
#     if dataset_name.startswith('physionet'):
# #         data_dir = climate_data_dir
#         
#         data_dir = os.path.join(data_folder, physionet_data_dir)
#         
#         training_data_file_name = 'dataset_train_y'
#         
#         
#         test_data_file_name = 'dataset_test_y'
#         
#         inference_len = physionet_data_train_len
        
#         time_steps_file_name = 'time_steps'
        
    if dataset_name.startswith(climate_data_name):
        data_dir = os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name)
        
        training_data_file_name = 'dataset_train_y'
        
        
        test_data_file_name = 'dataset_test_y'
        
        inference_len = climate_data_train_len
        
#     if dataset_name.startswith('beijing'):
#         data_dir = os.path.join(data_folder, beijing_data_dir)
#         
#         training_data_file_name = 'dataset_train_y'
#         
#         
#         test_data_file_name = 'dataset_test_y'
#         
# #         time_steps_file_name = 'time_steps'
#         
#         inference_len = beijing_data_train_len
    wrapped_train_y = torch.load(data_dir + '/' + training_data_file_name)
            
    wrapped_test_y = torch.load(data_dir + '/' + test_data_file_name) 
    
    return wrapped_train_y, wrapped_test_y, inference_len

# def load_time_series_grui(args):
#     
# #         dataset_name = 'mimic3_17'
#     
#     train_datapath = GRUI_train_dir + "/" + args.dataset
#     
#     test_datapath =  GRUI_test_dir + "/" + args.dataset
#     
#     print(train_datapath) 
#     
#     trainset_dataloader = ReadImputedData(train_datapath)
#     
#     trainset_dataloader.load()
#     
#     x_train = torch.tensor(trainset_dataloader.x)
#     
#     mask_train = torch.tensor(trainset_dataloader.m)
#     
#     delta_t_train = torch.tensor(trainset_dataloader.delta)
#     
#     testset_dataloader = ReadImputedData(test_datapath)
#     
#     testset_dataloader.load()
#     
#     x_test = torch.tensor(testset_dataloader.x)
#     
#     mask_test = torch.tensor(testset_dataloader.m)
#     
#     delta_t_test = torch.tensor(testset_dataloader.delta)
#     
#     wrapped_train_y, wrapped_test_y, inference_len = load_time_series_dataset(args.dataset)
#     
#     wrapped_train_y.data = x_train
#     
#     wrapped_test_y.data = x_test
#     
#     wrapped_train_y.mask = mask_train
#     
#     wrapped_test_y.mask = mask_test
#     
#     wrapped_train_y.delta_time_stamps = delta_t_train
#     
#     wrapped_test_y.delta_time_stamps = delta_t_test 
# 
# 
#     print('training data size::', wrapped_train_y.data.shape)
#     
#     print('test data size::', wrapped_test_y.data.shape)
# 
# 
#     print('inference missing ratio::', 1 - torch.mean(torch.cat([wrapped_train_y.mask[:, 0:inference_len], wrapped_test_y.mask[:, 0:inference_len]], 0)))
#     
#     print('forecasting missing ratio::', 1 - torch.mean(torch.cat([wrapped_train_y.mask[:, inference_len:], wrapped_test_y.mask[:, inference_len:]], 0)))
# 
#     
#     
#     train_dataloader = DataLoader(wrapped_train_y, batch_size = args.batch_size, shuffle=True,
#         collate_fn= lambda batch: basic_collate_fn(batch, args, data_type = "train"))
#     test_dataloader = DataLoader(wrapped_test_y, batch_size = args.batch_size, shuffle=True,
#         collate_fn= lambda batch: basic_collate_fn(batch, args, data_type = "test"))
#     
#     data_objects = {#"dataset_obj": dataset_obj, 
#                 "train_dataloader": train_dataloader, 
#                 "test_dataloader": test_dataloader,
#                 "input_dim": x_train.shape[2],
#                 "n_train_batches": len(train_dataloader),
#                 "n_test_batches": len(test_dataloader)}
# 
#     train_mean = get_train_mean(data_objects, inference_len)
# 
#     is_missing = torch.sum(wrapped_train_y.mask) < (wrapped_train_y.mask.shape[0]*wrapped_train_y.mask.shape[1]*wrapped_train_y.mask.shape[2])
# 
#     return data_objects, is_missing, train_mean
# #     print(x_train.shape)

def load_time_series(args):
    
    dataset_name = args.dataset
    
    data_dir = None
    
    training_data_file_name = None
    
    test_data_file_name = None
    
#     time_steps_file_name = None
    
    inference_len = 0
    
    if dataset_name.startswith(mimic_data_name):
#         data_dir = mimic3_data_dir
        data_dir = os.path.join(os.path.join(data_folder, mimic3_data_dir), dataset_name)
         
        training_data_file_name = 'dataset_train_y'
         
         
        test_data_file_name = 'dataset_test_y'
         
        valid_data_file_name = 'dataset_valid_y'
         
        inference_len = mimic3_data_train_len
         
#         time_steps_file_name = 'time_steps'
#         
#     if dataset_name.startswith('physionet'):
# #         data_dir = climate_data_dir
#         
#         data_dir = os.path.join(data_folder, physionet_data_dir)
#         
#         training_data_file_name = 'dataset_train_y'
#         
#         
#         test_data_file_name = 'dataset_test_y'
#         
#         valid_data_file_name = 'dataset_valid_y'
#         
#         inference_len = physionet_data_train_len
        
#         time_steps_file_name = 'time_steps'
        
    if dataset_name.startswith(climate_data_name):
        data_dir = os.path.join(os.path.join(data_folder, climate_data_dir), dataset_name)
        
        training_data_file_name = 'dataset_train_y'
        
        
        test_data_file_name = 'dataset_test_y'
        
        valid_data_file_name = 'dataset_valid_y'
        
        inference_len = climate_data_train_len
        
    if dataset_name.startswith(kddcup_data_name):
        data_dir = os.path.join(data_folder, beijing_data_dir)
         
        training_data_file_name = 'dataset_train_y'
         
        valid_data_file_name = 'dataset_valid_y'
         
        test_data_file_name = 'dataset_test_y'
        
#         time_steps_file_name = 'time_steps'
        
        inference_len = beijing_data_train_len
    wrapped_train_y = torch.load(data_dir + '/' + training_data_file_name)
    
    wrapped_valid_y = torch.load(data_dir + '/' + valid_data_file_name)
            
    wrapped_test_y = torch.load(data_dir + '/' + test_data_file_name) 

#     time_steps_extrap = torch.load(data_dir + '/' + time_steps_file_name)
    
    is_missing = torch.sum(wrapped_train_y.mask) < (wrapped_train_y.mask.shape[0]*wrapped_train_y.mask.shape[1]*wrapped_train_y.mask.shape[2]) 
    
    
#     if not dataset_name == 'mimic3_17_5':
    if args.model == cluster_ODE_method or args.model == l_ODE_method:
        new_train_time_stamps = wrapped_train_y.time_stamps.type(torch.float)/wrapped_train_y.time_stamps.shape[1]
        new_test_time_stamps = wrapped_test_y.time_stamps.type(torch.float)/wrapped_test_y.time_stamps.shape[1]
        new_valid_time_stamps = wrapped_valid_y.time_stamps.type(torch.float)/wrapped_valid_y.time_stamps.shape[1]
        
        
        
        wrapped_train_y.time_stamps = new_train_time_stamps
        wrapped_valid_y.time_stamps = new_valid_time_stamps
        wrapped_test_y.time_stamps = new_test_time_stamps
        
    
#     if args.model == 'Linear_regression':
#                 
#         train_y_copy = wrapped_train_y.data.clone()
#         
#         test_y_copy = wrapped_test_y.data.clone()
#         
#         train_y_copy[wrapped_train_y.mask == 0] = 0
#         test_y_copy[wrapped_test_y.mask == 0] = 0
#         
#         print(torch.norm((train_y_copy - wrapped_train_y.data)*wrapped_train_y.mask))
#         
#         wrapped_train_y.data = train_y_copy
#         
#         wrapped_test_y.data = test_y_copy
#         
#         
#         
#         train_y_copy = wrapped_train_y.origin_data.clone()
#         
#         test_y_copy = wrapped_test_y.origin_data.clone()
#         
#         train_y_copy[wrapped_train_y.origin_mask == 0] = 0
#         test_y_copy[wrapped_test_y.origin_mask == 0] = 0
#         
#         print(torch.norm((train_y_copy - wrapped_train_y.origin_data)*wrapped_train_y.origin_mask))
#         
#         wrapped_train_y.origin_data = train_y_copy
#         
#         wrapped_test_y.origin_data = test_y_copy
#     
#     
#     if args.model in models_to_remove_none_time_stamps:
#             
#         new_train_y, masks_train, train_time_stamps, train_lens = remove_none_observations(wrapped_train_y.data, wrapped_train_y.mask, wrapped_train_y.time_stamps, wrapped_train_y.lens)
#         
#         new_test_y, masks_test, test_time_stamps, test_lens = remove_none_observations(wrapped_test_y.data, wrapped_test_y.mask, wrapped_test_y.time_stamps, wrapped_test_y.lens)
#     
#     
#         check_remove_none(wrapped_train_y.data, new_train_y)
#     
#         wrapped_train_y.data = new_train_y
#         
#         wrapped_test_y.data = new_test_y
#         
#         wrapped_train_y.mask = masks_train
#         
#         wrapped_test_y.mask = masks_test
#         
#         wrapped_train_y.time_stamps = train_time_stamps
#         
#         wrapped_test_y.time_stamps = test_time_stamps
#     
#         wrapped_train_y.lens = train_lens
#         
#         wrapped_test_y.lens = test_lens    
    
    
#     if args.model == 'DHMM_cluster_tlstm':
#         train_delta_time_stamps =  get_delta_time_stamps_all_dims(wrapped_train_y.time_stamps)
#         
#         test_delta_time_stamps = get_delta_time_stamps_all_dims(wrapped_test_y.time_stamps)
#         
#         wrapped_train_y.delta_time_stamps = train_delta_time_stamps
#         
#         wrapped_test_y.delta_time_stamps = test_delta_time_stamps
            
#     if args.model == GRUD_method:
#         train_delta_time_stamps = get_delta_time_stamps(wrapped_train_y.mask, wrapped_train_y.time_stamps)
#         
#         test_delta_time_stamps = get_delta_time_stamps(wrapped_test_y.mask, wrapped_test_y.time_stamps)
#     
#     
#         wrapped_train_y.delta_time_stamps = train_delta_time_stamps
#         
#         wrapped_test_y.delta_time_stamps = test_delta_time_stamps
    
#             check_delta_time_stamps(masks_train, train_time_stamps, train_delta_time_stamps)
    
    
#     args.n = len(dataset)
#     input_dim = dataset.size(-1)
    input_dim = wrapped_train_y.data.shape[-1]

    args.n = wrapped_train_y.data.shape[0] + wrapped_test_y.data.shape[0]


    print('training data size::', wrapped_train_y.data.shape)
    
    print('validation data size::', wrapped_valid_y.data.shape)
    
    print('test data size::', wrapped_test_y.data.shape)


    print('inference missing ratio::', 1 - torch.mean(torch.cat([wrapped_train_y.mask[:, 0:inference_len], wrapped_valid_y.mask[:, 0:inference_len], wrapped_test_y.mask[:, 0:inference_len]], 0)))
    
    print('forecasting missing ratio::', 1 - torch.mean(torch.cat([wrapped_train_y.mask[:, inference_len:], wrapped_valid_y.mask[:, inference_len:], wrapped_test_y.mask[:, inference_len:]], 0)))

    batch_size = args.batch_size
    train_dataloader = DataLoader(wrapped_train_y, batch_size = batch_size, shuffle=True,
        collate_fn= lambda batch: basic_collate_fn(batch, args, data_type = "train"))
    valid_dataloader = DataLoader(wrapped_valid_y, batch_size = batch_size, shuffle=True,
        collate_fn= lambda batch: basic_collate_fn(batch, args, data_type = "train"))
    test_dataloader = DataLoader(wrapped_test_y, batch_size = batch_size, shuffle=True,
        collate_fn= lambda batch: basic_collate_fn(batch, args, data_type = "test"))
    
    data_objects = {#"dataset_obj": dataset_obj, 
                "train_dataloader": train_dataloader, 
                "valid_dataloader": valid_dataloader,
                "test_dataloader": test_dataloader,
                "input_dim": input_dim,
                "n_train_batches": len(train_dataloader),
                "n_test_batches": len(test_dataloader)}

    train_mean = get_train_mean(data_objects, inference_len)

    return data_objects, is_missing, train_mean
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
#     parser.add_argument('--data-path', type=str, default='./data/polyphonic/')
    parser.add_argument('--dataset', type=str, default='JSBChorales', help='name of dataset. SWDA or DailyDial')
    parser.add_argument('-ms', '--missing_ratio', type=float, default=0.00)
    
    args = parser.parse_args()
    generate_new_time_series(args)
    print('generate time series done!!!')
    
#     masks = torch.tensor([[[1,0],[1,1],[0,1],[1,0],[0,0],[1,0],[1,1]], [[1,0],[1,1],[0,1],[1,0],[0,0],[1,0],[1,1]]])
#     
#     time_stamps = torch.tensor([[0, 0.1, 0.6, 1.6, 2.2, 2.5, 3.1], [0, 0.1, 0.6, 1.6, 2.2, 2.5, 3.1]])
#     
#     print(masks.shape, time_stamps.shape)
#     
#     
#     
#     time_gap_tensors = get_delta_time_stamps(masks, time_stamps)
#     
#     print(time_gap_tensors)
    


