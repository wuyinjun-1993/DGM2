'''
Created on Jun 25, 2020

'''

import pandas as pd
import os, sys
import numpy as np

sys.path.append(os.path.abspath(__file__))
import torch


from utils import *

pamap_folder = 'PAMAP2/'

# subject_ids = ['101', '102', '103', '104', '105', '106', '107', '108']
subject_ids = ['101']

def read_single_obj_with_missing_values(file_name, selected_sample_count, length = 100):
    selected_sample_tensors = []
    
    for id in subject_ids:
    
    
        dr = pd.read_csv(file_name + id + '.dat', delim_whitespace=True, header=None)
        
        all_samples_with_na = dr.iloc[:,3:]
        
        time_stamps = dr.iloc[:,0].to_frame()
        
        
        all_samples_with_na = time_stamps.merge(all_samples_with_na, left_index = True, right_index = True)
        
        
        selected_sample_tensor = torch.tensor(all_samples_with_na.reset_index().values)
        
        
        candidate_ids = torch.tensor(list(range(int(selected_sample_tensor.shape[0]/(length/2) - 1))), dtype = torch.int)*(length/2)
        
        
        
        print('candidate_ids::', candidate_ids.shape)
        
        selected_samples_ids = candidate_ids[torch.randperm(candidate_ids.shape[0])[0:selected_sample_count]]
        
        for j in range(selected_samples_ids.shape[0]):
            sub_selected_sample_tensor = selected_sample_tensor[int(selected_samples_ids[j].item()):int(selected_samples_ids[j].item()) + length]
        
            selected_sample_tensors.append(sub_selected_sample_tensor[:,2:])
    
    
    all_selected_sample_tensors = torch.stack(selected_sample_tensors, 0)
    
        
    print(all_selected_sample_tensors.shape)

    torch.save(all_selected_sample_tensors, data_dir + pamap_folder + '/pamap_tensor')


def read_single_obj_even_spaced_no_missing_values(file_name, selected_sample_count, length = 10):
    
    
    selected_sample_tensors = []
    
    for id in subject_ids:
    
    
        dr = pd.read_csv(file_name + id + '.dat', delim_whitespace=True, header=None)
        
        samples_without_na = dr.iloc[:,3:]
        
        time_stamps = dr.iloc[:,0].to_frame()
        
        
        samples_without_na_before = time_stamps.merge(samples_without_na, left_index = True, right_index = True)
        
        samples_without_na = samples_without_na_before.dropna()
        
#         index_ids = list(samples_without_na.index)
#     
#         selected_samples_without_na = samples_without_na.loc[index_ids,:]
        
        selected_sample_tensor = torch.tensor(samples_without_na.reset_index().values)
        
        
        
        
        
        candidate_ids = torch.tensor(list(range(int(selected_sample_tensor.shape[0]/(length/2) - 1))), dtype = torch.int)*(length/2)
        
        print('candidate_ids::', candidate_ids.shape)
        
        selected_samples_ids = candidate_ids[torch.randperm(candidate_ids.shape[0])[0:selected_sample_count]]
        
        for j in range(selected_samples_ids.shape[0]):
            sub_selected_sample_tensor = selected_sample_tensor[int(selected_samples_ids[j].item()):int(selected_samples_ids[j].item()) + length]
        
            selected_sample_tensors.append(sub_selected_sample_tensor[:,3:])
        
        
#         print(selected_sample_tensor.shape)
#         
#         selected_sample_tensors.append(selected_sample_tensor)
    

    all_selected_sample_tensors = torch.stack(selected_sample_tensors, 0)
    
    print(all_selected_sample_tensors.shape)

    torch.save(all_selected_sample_tensors, data_dir + pamap_folder + '/pamap_tensor')
    
    
    
    
#     for i in range(len(index_ids)):
#         if index_ids[i] < 100000 and i > 0:
#             gap = index_ids[i] - index_ids[i-1]
#             print(gap)
    
    
    
    
#     print(dr)
    
    
    
    
    
    
if __name__ == '__main__':
    read_single_obj_even_spaced_no_missing_values(data_dir + pamap_folder + 'subject', 5000, 10)
    
    