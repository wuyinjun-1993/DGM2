'''
Created on Jun 26, 2020

'''
import pandas as pd
import os, sys
import numpy as np
import argparse


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/lib')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from lib.utils import *


# positions = ['Bag', 'Hand', 'Hips', 'Torso']

positions = ['Bag']

motion_str = 'Motion'

shl_tensor_folder = 'shl/'

import torch


def read_single_obj_even_spaced_time_series(dir, selected_sample_count, length, overlap=5):
    
    selected_sample_tensors = []
    
    with pd.option_context('display.precision', 13):
        for i in range(len(positions)):
            file_name = dir + '/' + positions[i] + '_' + motion_str + '.txt'
        
            dr = pd.read_csv(file_name, delim_whitespace=True, header=None)
            
            samples_without_na = dr.dropna()
            
            
            selected_sample_tensor = torch.tensor(samples_without_na.reset_index().values)
            
            candidate_ids = torch.tensor(list(range(int(selected_sample_tensor.shape[0]/(length - overlap) - 1))), dtype = torch.int)*(length - overlap)
        
            print('candidate_ids::', candidate_ids.shape)
            
            selected_samples_ids = candidate_ids[torch.randperm(candidate_ids.shape[0])[0:selected_sample_count]]
            
            for j in range(selected_samples_ids.shape[0]):
                sub_selected_sample_tensor = selected_sample_tensor[int(selected_samples_ids[j].item()):int(selected_samples_ids[j].item()) + length]
            
                selected_sample_tensors.append(sub_selected_sample_tensor[:,2:])
        
        
#         print(selected_sample_tensor.shape)
#         
#         selected_sample_tensors.append(selected_sample_tensor)
    

    all_selected_sample_tensors = torch.stack(selected_sample_tensors, 0)
    
    print(all_selected_sample_tensors.shape)

    torch.save(all_selected_sample_tensors, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/" + data_dir + shl_tensor_folder + '/shl_tensor')
            
            


def read_single_obj_even_spaced_no_missing_values(dir, selected_sample_count):
    
    
    
    
    all_index_ids = []
    
    all_time_stamps = []
    
    df_list = []
    
    res_df = None
    
    with pd.option_context('display.precision', 13):
        for i in range(len(positions)):
            file_name = dir + '/' + positions[i] + '_' + motion_str + '.txt'
        
            dr = pd.read_csv(file_name, delim_whitespace=True, header=None)
            
            samples_without_na = dr.dropna()
            
    #         index_ids = list(samples_without_na.index)[0:selected_sample_count]
            
            samples_without_na.iloc[:,0] = samples_without_na.iloc[:,0].astype(int)
            
#             time_stamps = samples_without_na.iloc[:,0]
            
    #         index_ids = list(samples_without_na.index)
            
            if res_df is not None:
    #             time_stamps.join(all_time_stamps[-1], how = 'inner')
#                 res_df = samples_without_na.join(res_df, how = 'inner', on = '0')
                res_df = samples_without_na.merge(res_df, how = 'inner', on = 0, suffixes=('1','2'))
            
            else:
                res_df = samples_without_na
                
            
            
#             all_time_stamps.append(time_stamps)
#             
#             df_list.append(dr)
    
    
    index_ids = list(res_df.index)[0:selected_sample_count]
        
    selected_samples_without_na = res_df.loc[index_ids,:]
    
    print(selected_samples_without_na.iloc[:,0])
        
    selected_samples_array = torch.tensor(selected_samples_without_na.reset_index().values)
    
    
    return selected_samples_array
    
    
    
    
#     for i in range(len(index_ids)):
#         if index_ids[i] < 100000 and i > 0:
#             gap = index_ids[i] - index_ids[i-1]
#             print(gap)
    
    
    
    
#     print(dr)
    
    
    
    
    
    
if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser('process_shl')
    
    parser.add_argument('--dir', default = '/home/wuyinjun/pg_data/shl/SHLDataset_preview_v1/User1', help = 'size of gradients and parameters cached in GPU in deltagrad') 
    
    args = parser.parse_args()
    
    shl_data_dir = args.dir
    
    sub_dirs = [dI for dI in os.listdir(shl_data_dir) if os.path.isdir(os.path.join(shl_data_dir,dI))]

    
    read_single_obj_even_spaced_time_series(shl_data_dir + '/' + sub_dirs[0], 10000, 100)
    
    
#     selected_samples_arrays = []
#     
#     for i in range(len(sub_dirs)):
#         selected_samples_array = read_single_obj_even_spaced_no_missing_values(shl_data_dir + '/' + sub_dirs[i], 10000)
#         
#         print(selected_samples_array.shape)
#         
#         selected_samples_arrays.append(selected_samples_array)
#     
#     selected_samples_tensor = torch.stack(selected_samples_arrays)
#     
#     
#     tensor_store_dir = data_dir + '/shl'
#     
#     if not os.path.exists(tensor_store_dir):
#         os.makedirs(tensor_store_dir)
#     
#     
#     torch.save(selected_samples_tensor, tensor_store_dir + '/shl_tensor')
    
    
    
    