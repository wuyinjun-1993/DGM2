'''
Created on Aug 10, 2020

'''
import pandas as pd 
import datetime

import numpy as np

dir = '../.gitignore/beijing_air/'

import torch


time_len = 36

train_len = 24

def get_masks(data):
    
    mask = ~torch.isnan(data)
    
    mask = mask.type(torch.float)
    
    return mask

def partition_tensors(data):
    ids = torch.tensor(range(int(data.shape[0]/time_len)))*time_len
    
    all_data = []
    for id in ids:
        curr_data = data[id: id+time_len]
        if curr_data.shape[0] < time_len:
            continue
        
        all_data.append(curr_data)
        
    all_data_tensor = torch.stack(all_data)
    
    print(all_data_tensor.shape)
    
    return all_data_tensor
    

with pd.option_context('display.precision', 13):
    file_name = dir + '/beijing_17_18_aq.csv'
        
    df = pd.read_csv(file_name)
    
    print(df.columns)
    
    df = df[['stationId', 'utc_time', 'PM2.5']]
    
    df['utc_time'] = pd.to_datetime(df['utc_time'], errors='coerce')
    
    station_IDs = list(df.stationId.unique())
    
    print('station ID count::', len(station_IDs))
    
    date_after = datetime.datetime(2017, 1, 1)
    
    print('date after::', date_after)
    
    date_before = datetime.datetime(2018, 1, 1)
    
    print('date before::', date_before)
    
    time_stamps = []
    
    for i in range(1, 13):
        for j in range(1, 32):
            for k in range(0, 24):
#                 for p in range(0, 60):
#                     for r in range(0, 60):
                try:
                    curr_date = datetime.datetime(2017, i, j, k, 0, 0)
                    
                    time_stamps.append(curr_date)
                    
#                             count = len(curr_df[curr_df['utc_time'] == curr_date])
#                         
#                     print(curr_date)
                except:
                    continue
    
    print('time stamp lenth::', len(time_stamps))
    
    time_stamp_dfs = pd.DataFrame(time_stamps, columns =['utc_time'])
    
    time_stamp_dfs.utc_time = pd.to_datetime(time_stamp_dfs.utc_time)
    
#     print(time_stamp_dfs)
#     
#     print(df['utc_time'])
    
    all_values = []
    
    for sid in station_IDs:
        curr_df = df[df['stationId'] == sid]
        
#         
        
#         print(len(curr_df))
        
        curr_df = curr_df[curr_df['utc_time'] >= date_after]
         
        curr_df = curr_df[curr_df['utc_time'] < date_before]
        
        curr_df = curr_df.drop_duplicates()
        
#         print(len(curr_df))
        
#         curr_time_stamp_list = list(curr_df['utc_time'])
        
#         for k in range(len(curr_time_stamp_list)):
#             print(curr_time_stamp_list[k], curr_time_stamp_list[k] in set(time_stamp_dfs['utc_time']))
#             
#             assert curr_time_stamp_list[k] in set(time_stamp_dfs['utc_time'])
        
        
#         remaining_time_stamps = set(time_stamp_dfs['utc_time']).difference(set(curr_df['utc_time']))
#         
#         null_values = np.zeros(len(remaining_time_stamps))
#         
#         null_values[:] = np.nan
#         
#         
#         
#         data = {'utc_time': list(remaining_time_stamps), 'PM2.5': null_values}
#         
#         other_dfs = pd.DataFrame(data)
        
        
        curr_df = curr_df.merge(time_stamp_dfs, on  = 'utc_time', how = 'outer')
        
        assert set(curr_df['utc_time']) == set(time_stamp_dfs['utc_time'])
        
        curr_df = curr_df.sort_values('utc_time')
        
        values = torch.tensor(list(curr_df['PM2.5']))
        
        all_values.append(values)
        
        print(len(curr_df))
        
        print(24*365)
    
    
    
    origin_value_tensor = torch.stack(all_values, 1)
    
    all_value_tensor = partition_tensors(origin_value_tensor)
    
    print(all_value_tensor.shape)
    
    tensor_len = all_value_tensor.shape[0]
    
    rand_ids = torch.randperm(tensor_len)
    
    train_ids = rand_ids[0:int(tensor_len*0.5)]
    
    test_ids = rand_ids[int(tensor_len*0.5):]
    
    
    training_tensor = all_value_tensor[train_ids]
    
    test_tensor = all_value_tensor[test_ids]
    
    training_mask = get_masks(training_tensor)
    
    test_mask = get_masks(test_tensor)
    
    torch.save(training_tensor, dir + 'training_tensor')
    
    torch.save(test_tensor, dir + 'test_tensor')

    torch.save(training_mask, dir + 'training_mask')
    
    torch.save(test_mask, dir + 'test_mask')
    
    assert torch.sum(torch.isnan(training_tensor)) == torch.sum(1-training_mask)
    
    print('nonmissing ratio::', torch.mean(training_mask))
    
    print(training_tensor.shape)
    
    print(test_tensor.shape)
    
    
    
    
       
    
    
