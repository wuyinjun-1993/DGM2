'''
Created on Jul 2, 2020

'''

import os

import pandas as pd

import numpy as np

import torch

import json
import random

mimic3_dir_train = '/home/wuyinjun/workspace/mimic3-benchmarks/data/'

mimic3_dir_test = '/home/wuyinjun/workspace/mimic3-benchmarks/data/'

mimic3_dir = '/home/wuyinjun/workspace/mimic3-benchmarks/data/'

time_series_file_name_prefix = 'episode'

time_series_file_name_suffix = 'timeseries.csv' 

time_series_file_name = 'episode1_timeseries.csv'


time_series_file_headers = ['Hours','Capillary refill rate','Diastolic blood pressure','Fraction inspired oxygen','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response','Glucose','Heart Rate','Height','Mean blood pressure','Oxygen saturation','Respiratory rate','Systolic blood pressure','Temperature','Weight','pH']

# time_series_file_headers = ['Hours','Diastolic blood pressure','Fraction inspired oxygen','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response','Glucose','Heart Rate','Height','Mean blood pressure','Oxygen saturation','Respiratory rate','Systolic blood pressure','Temperature','Weight','pH']

cast_columns = ['Glascow coma scale verbal response', 'Glascow coma scale total', 'Capillary refill rate', 'Glascow coma scale eye opening', 'Glascow coma scale motor response']


# max_time_stamp = 36


time_gap_in_hour = 5.0/60

time_stamp_lower_bound = 0

time_stamp_upper_bound = 72

time_stamp_training = 48


measurement_threshold = 100

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/lib')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.utils import *

data_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/" + data_dir

non_missing_ratio_thresholds = [0.8, 0.7,0.6, 0.5, 0.4,0.3, 0.2]

# periods_map = {
#     "all": (0, 0, 1, 0),
#     "first4days": (0, 0, 0, 4 * 24),
#     "first8days": (0, 0, 0, 8 * 24),
#     "last12hours": (1, -12, 1, 0),
#     "first25percent": (2, 25),
#     "first50percent": (2, 50)
# }
# 
# from scipy.stats import skew
# 
# all_functions = [min, max, np.mean, np.std, skew, len]
# 
# functions_map = {
#     "all": all_functions,
#     "len": [len],
#     "all_but_len": all_functions[:-1]
# }
# 
# def get_range(begin, end, period):
#     # first p %
#     if period[0] == 2:
#         return (begin, begin + (end - begin) * period[1] / 100.0)
#     # last p %
#     if period[0] == 3:
#         return (end - (end - begin) * period[1] / 100.0, end)
# 
#     if period[0] == 0:
#         L = begin + period[1]
#     else:
#         L = end + period[1]
# 
#     if period[2] == 0:
#         R = begin + period[3]
#     else:
#         R = end + period[3]
# 
#     return (L, R)
# 
# def calculate(channel_data, period, sub_period, functions):
#     if len(channel_data) == 0:
#         return np.full((len(functions, )), np.nan)
# 
#     L = channel_data[0][0]
#     R = channel_data[-1][0]
#     L, R = get_range(L, R, period)
#     L, R = get_range(L, R, sub_period)
# 
#     data = [x for (t, x) in channel_data
#             if L - 1e-6 < t < R + 1e-6]
# 
#     if len(data) == 0:
#         return np.full((len(functions, )), np.nan)
#     return np.array([fn(data) for fn in functions], dtype=np.float32)
# 
# def extract_features_single_episode(data_raw, period, functions):
#     global sub_periods
#     extracted_features = [np.concatenate([calculate(data_raw[i], period, sub_period, functions)
#                                           for sub_period in sub_periods],
#                                          axis=0)
#                           for i in range(len(data_raw))]
#     return np.concatenate(extracted_features, axis=0)
# 
# 
# def extract_features(data_raw, period, features):
#     period = periods_map[period]
#     functions = functions_map[features]
#     return np.array([extract_features_single_episode(x, period, functions)
#                      for x in data_raw])
# 
# 
# def convert_to_dict(data, header, channel_info):
#     """ convert data from readers output in to array of arrays format """
#     ret = [[] for i in range(data.shape[1] - 1)]
#     for i in range(1, data.shape[1]):
#         ret[i-1] = [(t, x) for (t, x) in zip(data[:, 0], data[:, i]) if x != ""]
#         channel = header[i]
#         if len(channel_info[channel]['possible_values']) != 0:
#             ret[i-1] = list(map(lambda x: (x[0], channel_info[channel]['values'][x[1]]), ret[i-1]))
#         ret[i-1] = list(map(lambda x: (float(x[0]), float(x[1])), ret[i-1]))
#     return ret
# 
# 
# def extract_features_from_rawdata(chunk, header, period, features):
#     with open(os.path.join(os.path.dirname(__file__), "resources/channel_info.json")) as channel_info_file:
#         channel_info = json.loads(channel_info_file.read())
#     data = [convert_to_dict(X, header, channel_info) for X in chunk]
#     return extract_features(data, period, features)
# 
# 
# def read_chunk(reader, chunk_size):
#     data = {}
#     for i in range(chunk_size):
#         ret = reader.read_next()
#         for k, v in ret.items():
#             if k not in data:
#                 data[k] = []
#             data[k].append(v)
#     data["header"] = data["header"][0]
#     return data
# 
# 
# def sort_and_shuffle(data, batch_size):
#     """ Sort data by the length and then make batches and shuffle them.
#         data is tuple (X1, X2, ..., Xn) all of them have the same length.
#         Usually data = (X, y).
#     """
#     assert len(data) >= 2
#     data = list(zip(*data))
# 
#     random.shuffle(data)
# 
#     old_size = len(data)
#     rem = old_size % batch_size
#     head = data[:old_size - rem]
#     tail = data[old_size - rem:]
#     data = []
# 
#     head.sort(key=(lambda x: x[0].shape[0]))
# 
#     mas = [head[i: i+batch_size] for i in range(0, len(head), batch_size)]
#     random.shuffle(mas)
# 
#     for x in mas:
#         data += x
#     data += tail
# 
#     data = list(zip(*data))
#     return data


def dataframe_from_csv(path, header=0, index_col=False):
    return pd.read_csv(path, header=header, index_col=index_col)

def pad_time_series(tensor):
    if tensor.shape[0] < time_stamp_upper_bound:
        tensor_copy = torch.zeros([time_stamp_upper_bound, tensor.shape[1]])
        tensor_copy[0:tensor.shape[0]] = tensor
        tensor_copy[tensor.shape[0]:time_stamp_upper_bound,:] = np.nan
        return tensor_copy
    return tensor


def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False

def remove_non_values_time_stamps(data, masks):
    time_stamp_masks = torch.sum(masks[:,:,1:], [0,-1])
    
    data = data[:,~(time_stamp_masks == 0)]
    
    masks = masks[:,~(time_stamp_masks == 0)]
    
    all_time_stamps = torch.tensor(range(int(time_stamp_lower_bound/time_gap_in_hour), int(time_stamp_upper_bound/time_gap_in_hour)))
    
    time_stamps = all_time_stamps[~(time_stamp_masks == 0)]
    
    print('non empty time stamp lenth::',len(time_stamps*time_gap_in_hour))
    
    return data, masks, time_stamps
    
    
    


def get_all_data_frames_tensors(mimic3_dir, fill_nan, select_17):
    
    
    all_tensors = []
    
    all_masks = []
    
    all_tensor_lens = []
    
    count = 0
    
    all_measurement_counts = []
    
    
    for sid_folder in os.listdir(mimic3_dir):
        
        
#         if count > 1394:
#             print('here')
#             break
        try:
            sid_int = int(sid_folder)
            
#             if not sid_int == 28166:
#                 continue
        
        except:
            continue
        
        for csv_file in os.listdir(os.path.join(mimic3_dir, sid_folder)):
            if csv_file.startswith(time_series_file_name_prefix) and csv_file.endswith(time_series_file_name_suffix):
                
#                 if count >= 9982:
#                     print('here')
                
                df = dataframe_from_csv(os.path.join(os.path.join(mimic3_dir, sid_folder), csv_file))
#                 df[cast_columns] = df[cast_columns].astype(str)

                
                if select_17:
                    df = df[time_series_file_headers]
                    df = replace_string_with_df(df)
                    
                else:
                    all_columns = list(df.columns)
                    
                    other_columns = list(set(all_columns).difference(set(time_series_file_headers)))
                    
                    final_other_columns = [time_series_file_headers[0]]
                    
                    final_other_columns.extend(other_columns)
                    
                    df = df[final_other_columns]
                    
                    
                
                
                len1 = len(df)
#                 print('length before filling::', len(df))
                df = df[df.applymap(isnumber)]
                len2 = len(df)
                
                assert (len1 == len2)
                
#                 print('length after filling::', len(df))
                
                df = df.astype(float)
                
                
                
                curr_tensor = process_dfs(df)
                
                print('id::', sid_folder, count)
                
                
                
                
                
#                 if int(sid_folder) == 21852:
#                     print('here')

                if torch.max(curr_tensor[:,0]).item() <= time_stamp_upper_bound:
                    continue
                
                if fill_nan:
                    clipped_tensors = fill_nan_value_to_missing_time_points(curr_tensor)
                else:
                    clipped_tensors = fill_nan_value_to_other_time_points(curr_tensor)
#                     clipped_tensors = curr_tensor
                
                if clipped_tensors.shape[0] <= 0:
                    continue
                
                if torch.max(clipped_tensors[:,0]).item() <= time_stamp_training:
                    continue
                
                
                curr_measurement_count = torch.sum(1-torch.isnan(clipped_tensors[clipped_tensors[:,0] < time_stamp_training,1:]).type(torch.LongTensor))
                
                curr_total_measurement_count = torch.sum(1-torch.isnan(clipped_tensors[:,1:]).type(torch.LongTensor))
                
                print('measurement count::', curr_measurement_count, curr_total_measurement_count)
                
                if curr_measurement_count.item() < 20:
                    print('here')
                
                
                if curr_measurement_count.item() < measurement_threshold:
                    continue
                
                
                all_measurement_counts.append(curr_measurement_count)
                
                
                
                assert clipped_tensors.shape[0] > 0
                
                all_tensor_lens.append(clipped_tensors.shape[0])
                
                clipped_tensors = pad_time_series(clipped_tensors)
                
                masks = get_all_masks(clipped_tensors)
                
                assert torch.sum(masks) == torch.sum(1-np.isnan(clipped_tensors))
                
                assert(torch.sum(1-np.isnan(clipped_tensors[0:all_tensor_lens[-1]])) == torch.sum(masks))
                
                
                all_masks.append(masks)
                
                print('count::', count)
                
                
                print(clipped_tensors.shape[0])
#                 print(sid_folder, csv_file)
#                 
#                 print(clipped_tensors[:,0], torch.max(curr_tensor[:,0]), clipped_tensors.shape[0])
                assert clipped_tensors.shape[0] == time_stamp_upper_bound/time_gap_in_hour
                
                all_tensors.append(clipped_tensors)
            
            
                count += 1   
                

    
    
    
    
    
    all_tensor_array = torch.stack(all_tensors)
    
    all_masks_array = torch.stack(all_masks)
    
    all_tensor_len_array = torch.tensor(all_tensor_lens)
    
    remove_non_values_time_stamps(all_tensor_array, all_masks_array)

    return all_tensor_array, all_masks_array, all_tensor_len_array

# def clean_str_to_value(df, variables):
#     with open(os.path.join(os.path.dirname(__file__), "../../mimic3models/resources/channel_info.json")) as channel_info_file:
#         channel_info = json.loads(channel_info_file.read())
#         
#     for var in variables:
#         
#         possible_values = channel_info[var]["possible_values"]
#         
#         if len(possible_values) > 0:
#             for val in possible_values:
#                 df.loc[(df['VARIABLE'] == var) & (df['VALUE'] == val), 'VALUE'] = str(channel_info[var]["values"][val])  
#     
#     
#     return df

def replace_string_with_df(df):
    
    with open(os.path.join(os.path.dirname(__file__), "channel_info.json")) as channel_info_file:
        channel_info = json.loads(channel_info_file.read())
    
    for k in range(len(time_series_file_headers)-1):
        
        if 'values' in channel_info[time_series_file_headers[k+1]]:
#             print('variables::', time_series_file_headers[k+1])
            
            replace_rule = channel_info[time_series_file_headers[k+1]]['values']
            
            try:
                df = df.replace({time_series_file_headers[k+1]: replace_rule})
            
            except:
                continue
            
            df[time_series_file_headers[k+1]] = df[time_series_file_headers[k+1]].astype(float)
    
    return df
    


def merge_two_rows_within_same_time_interval(row1, row2, count):
    
#     df_ = pd.DataFrame(index=index, columns=columns)
    
    res_now = torch.zeros(count)
    
    
    timestamp = row1[0]
    
    non_missing1 = 1 - np.isnan(row1)
    
    non_missing2 = 1 - np.isnan(row2)
    
    missing_ids = np.nonzero(np.isnan(row1)*np.isnan(row2)).view(-1)
    
    non_missing_ids = np.nonzero(non_missing1*non_missing2).view(-1)
    
    res_now[non_missing_ids] = row1[non_missing_ids] + row2[non_missing_ids]
    
    res_now[missing_ids] = np.nan
    
    non_missing_id1 = np.nonzero(non_missing1*np.isnan(row2)).view(-1)
    
    res_now[non_missing_id1] = row1[non_missing_id1]
    
    non_missing_id2 = np.nonzero(non_missing2*np.isnan(row1)).view(-1)
    
    res_now[non_missing_id2] = row1[non_missing_id2]
    
    res_now[0] = timestamp
    
    
    
#     for k in range(len(time_series_file_headers)-1):
#         if row1[k+1] is None and row2[k+1] is not None:
#             res_now[k+1] = row2[k+1]
#         else:
#             if row1[k+1] is not None and row2[k+1] is None:
#                 res_now[k+1] = row1[k+1]
#             else:
#                 if row1[k+1] is not None and row2[k+1] is not None:
#                     res_now[k+1] = row1[k+1] + row2[k+1] 
#                 else:
#                     res_now[k+1] = np.NaN
    return res_now
#         else:    


def extract_time_stamps(time_stamp_list):
    
    
    last_time_stamp = None
    
    
    count = 0
    
    unique_time_list = []
    
    unique_time_count_list = []
    
    
    for t in time_stamp_list:
        t_in_hour = int(t+0.5)
        
        if last_time_stamp is None:
            last_time_stamp = t_in_hour
            count = 1
        else:
            if last_time_stamp == t_in_hour:
                count += 1
            else:
                unique_time_list.append(last_time_stamp)
                unique_time_count_list.append(count)
                
                last_time_stamp = t_in_hour
                
                count = 1
                
    unique_time_list.append(last_time_stamp)
    unique_time_count_list.append(count)
        
    return unique_time_list,  unique_time_count_list 
        


def process_dfs(curr_df):
    
#     all_raw_data_tensor = []
#     
#     for i in range(len(all_dfs)):
#     curr_df = all_dfs[i]

#         curr_tensor = torch.tensor(curr_df.reset_index().values)

    

    time_stamp_list = list(curr_df[time_series_file_headers[0]])
    
    last_time_stamp = None
    
    
    count = 0
    
    id = 0
    
    np_time_list = np.array(time_stamp_list)/time_gap_in_hour
    
    if np.isnan(np_time_list).any():
        print('here')
        
    curr_df = curr_df.loc[~np.isnan(np_time_list)]
    
    np_time_list = np_time_list[~np.isnan(np_time_list)]
    
    unique_time_list, unique_time_count_list = extract_time_stamps(np_time_list)
    
    
    curr_tensor = torch.zeros([len(unique_time_list), len(curr_df.columns)])
    
    
    rid = 0
    
    for k in range(len(unique_time_list)):
        time_stamp = unique_time_list[k]
        
        time_stamp_count = unique_time_count_list[k]
        
        curr_row_tensor = None
        
        count_per_var = torch.zeros(len(curr_df.columns))
        
        
        for r in range(time_stamp_count):
            curr_row = curr_df.iloc[rid]
            
            if curr_row_tensor is None:
                curr_row_tensor = torch.tensor(list(curr_row.values), dtype = torch.float)
                curr_row_tensor[0] = time_stamp
                
                count_per_var += (1 - np.isnan(curr_row_tensor))
            else:
                
                this_row_tensor = torch.tensor(list(curr_row.values), dtype = torch.float)
                
                count_per_var += (1 - np.isnan(this_row_tensor))
                
                curr_row_tensor = merge_two_rows_within_same_time_interval(curr_row_tensor, this_row_tensor, len(list(curr_df.columns)))
        
            rid += 1
        
        curr_row_tensor[1:] = curr_row_tensor[1:]/time_stamp_count
        
        curr_tensor[k] = curr_row_tensor
    
    
    
    return curr_tensor
#         all_raw_data_tensor.append(curr_tensor)
        
#         for k in range(len(time_stamp_list)):
#             
#             curr_time_stamp = time_stamp_list[k]
#             
#             t_in_hour = int(curr_time_stamp)
#             
#             if last_time_stamp is None:
#                 last_time_stamp = t_in_hour
#                 
#             else:
#                 if last_time_stamp == t_in_hour:
#                     converted_tensor[id] += curr_tensor[k] 
#                     count += 1
#                 else:
#                     converted_tensor[id] =  curr_tensor[k]
#                     
#                     count = 0
#                     
#                     id += 1



def read_mimic_iii2(mimic3_dir, select_17):
    all_tensor_array, all_masks_array, all_tensor_len_array = get_all_data_frames_tensors(mimic3_dir, True, select_17)
    
    print('non missing ratio::', torch.sum(all_masks_array)/(all_masks_array.shape[0]*all_masks_array.shape[1]*all_masks_array.shape[2]))
    
#     dataset = all_tensor_array[:,:,1:]
#     
#     timestamps = all_tensor_array[:,:,0]
#     
#     new_dataset, new_data_masks, new_timestamps, new_x_lens =  remove_none_observations(dataset, all_masks_array[:,:,1:], timestamps, all_tensor_len_array)
#     
#     all_tensor_array2, all_masks_array2, all_tensor_len_array2 = get_all_data_frames_tensors(mimic3_dir, False)
#     
#     new_dataset2 = all_tensor_array2[:,:,1:]
#     
#     new_data_masks2 = all_masks_array2[:,:,1:]
#     
#     new_timestamps2 = all_tensor_array2[:,:,0]
#     
#     new_dataset[new_dataset != new_dataset] = -100
#     
#     new_dataset2[new_dataset2 != new_dataset2] = -100
#     
#     new_timestamps[new_timestamps != new_timestamps] = -100
#     
#     new_timestamps2[new_timestamps2 != new_timestamps2] = -100
#     
#     
#     print(torch.norm(new_dataset - new_dataset2))
#     
#     print(torch.norm(new_data_masks - new_data_masks2))
#     
#     print(torch.norm(new_timestamps - new_timestamps2))
#     
#     print(torch.norm(new_x_lens.type(torch.FloatTensor) - all_tensor_len_array2.type(torch.FloatTensor)))
    
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
                
    if not os.path.exists(os.path.join(data_folder, mimic3_data_dir)):
        os.makedirs(os.path.join(data_folder, mimic3_data_dir))
    
    
    assert torch.sum(all_masks_array) == torch.sum(1-np.isnan(all_tensor_array))
    
#     print('size::', all_tensor_array.shape)
    
    torch.save(all_tensor_array, os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + 'tensor')
    
#     torch.save(all_tensor_array[:,:,0], os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + train_str + '_time_stamps')
    
    torch.save(all_masks_array, os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + 'masks')
    
    torch.save(all_tensor_len_array, os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + 'tensor_len0')
    
    return all_tensor_array, all_masks_array, all_tensor_len_array




def read_mimic_iii(mimic3_dir, train_str):
    all_tensor_array, all_masks_array, all_tensor_len_array = get_all_data_frames_tensors(mimic3_dir, True)
    
    print('non missing ratio::', torch.sum(all_masks_array)/(all_masks_array.shape[0]*all_masks_array.shape[1]*all_masks_array.shape[2]))
    
#     dataset = all_tensor_array[:,:,1:]
#     
#     timestamps = all_tensor_array[:,:,0]
#     
#     new_dataset, new_data_masks, new_timestamps, new_x_lens =  remove_none_observations(dataset, all_masks_array[:,:,1:], timestamps, all_tensor_len_array)
#     
#     all_tensor_array2, all_masks_array2, all_tensor_len_array2 = get_all_data_frames_tensors(mimic3_dir, False)
#     
#     new_dataset2 = all_tensor_array2[:,:,1:]
#     
#     new_data_masks2 = all_masks_array2[:,:,1:]
#     
#     new_timestamps2 = all_tensor_array2[:,:,0]
#     
#     new_dataset[new_dataset != new_dataset] = -100
#     
#     new_dataset2[new_dataset2 != new_dataset2] = -100
#     
#     new_timestamps[new_timestamps != new_timestamps] = -100
#     
#     new_timestamps2[new_timestamps2 != new_timestamps2] = -100
#     
#     
#     print(torch.norm(new_dataset - new_dataset2))
#     
#     print(torch.norm(new_data_masks - new_data_masks2))
#     
#     print(torch.norm(new_timestamps - new_timestamps2))
#     
#     print(torch.norm(new_x_lens.type(torch.FloatTensor) - all_tensor_len_array2.type(torch.FloatTensor)))
    
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
                
    if not os.path.exists(os.path.join(data_folder, mimic3_data_dir)):
        os.makedirs(os.path.join(data_folder, mimic3_data_dir))
    
    
    assert torch.sum(all_masks_array) == torch.sum(1-np.isnan(all_tensor_array))
    
#     print('size::', all_tensor_array.shape)
    
    torch.save(all_tensor_array, os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + train_str + '_tensor')
    
#     torch.save(all_tensor_array[:,:,0], os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + train_str + '_time_stamps')
    
    torch.save(all_masks_array, os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + train_str + '_masks')
    
    torch.save(all_tensor_len_array, os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + train_str + '_tensor_len0')
    
    return all_tensor_array, all_masks_array, all_tensor_len_array


def get_all_masks(df_tensor):
    
    
    
    ids = np.nonzero(np.isnan(df_tensor.view(-1)))
    
#     ids = torch.nonzero(df_tensor.view(-1) == np.nan)
    
    masks = torch.ones_like(df_tensor.view(-1))
    
    masks[ids] = 0
    
    return masks.view(df_tensor.shape[0], df_tensor.shape[1])

    
    
def fill_nan_value_to_missing_time_points(df_tensor):
    timestamp_lists_lower = (df_tensor[:,0] >= int(time_stamp_lower_bound/time_gap_in_hour))
    
    timestamp_lists_upper = (df_tensor[:,0] < int(time_stamp_upper_bound/time_gap_in_hour))
    
    max_time_stamp = int(torch.max(df_tensor[:,0]).item())
    
    
    available_timestamps = df_tensor[timestamp_lists_lower*timestamp_lists_upper,0]
    
#     np.linspace()
    
    all_time_stamps = torch.tensor(range(int(time_stamp_lower_bound/time_gap_in_hour), int(min(time_stamp_upper_bound, max_time_stamp)/time_gap_in_hour)))
    
    other_timestamps = set((all_time_stamps).type(torch.long).numpy()).difference((set((available_timestamps).type(torch.long).numpy())))
    
    print(len(other_timestamps), len(available_timestamps), len(all_time_stamps))
    
    assert len(other_timestamps) + len(available_timestamps) == len(all_time_stamps)
    
    df_tensor_other_tps = torch.zeros([len(other_timestamps), df_tensor.shape[1]])
    
    df_tensor_other_tps[:] = np.nan
    
    df_tensor_other_tps[:,0] = torch.tensor(list(other_timestamps))
    
    
    df_tensor_curr_tps = df_tensor[timestamp_lists_lower*timestamp_lists_upper]
     
    df_all_tps = torch.cat([df_tensor_other_tps, df_tensor_curr_tps])
    
    sorted_ids = torch.argsort(df_all_tps[:,0])
    
    final_df_all_tps =  df_all_tps[sorted_ids]
    
    final_df_all_tps[:,0] = final_df_all_tps[:,0]*time_gap_in_hour
    
#     print(final_df_all_tps[:,0])
    
    print(torch.sum(~torch.isnan(df_all_tps)))
     
    return final_df_all_tps


def fill_nan_value_to_other_time_points(df_tensor):
    timestamp_lists_lower = (df_tensor[:,0] >= time_stamp_lower_bound)
    
    timestamp_lists_upper = (df_tensor[:,0] < time_stamp_upper_bound)
    
#     max_time_stamp = int(torch.max(df_tensor[:,0]).item())
#     
#     
#     available_timestamps = df_tensor[timestamp_lists_lower*timestamp_lists_upper,0]
#     
#     remaining_time_stamp_count = time_stamp_upper_bound - time_stamp_lower_bound - available_timestamps.shape[0]
#     
#     remaining_tensors = torch.zeros([remaining_time_stamp_count, df_tensor.shape[1]])
    
#     
#     all_time_stamps = torch.tensor(range(time_stamp_lower_bound, min(time_stamp_upper_bound, max_time_stamp)))
#     
#     other_timestamps = set(all_time_stamps.numpy()).difference(set(available_timestamps.numpy()))
#     
#     df_tensor_other_tps = torch.zeros([len(other_timestamps), len(time_series_file_headers)])
#     
#     df_tensor_other_tps[:] = np.nan
#     
#     df_tensor_other_tps[:,0] = torch.tensor(list(other_timestamps))
    
    
    df_tensor_curr_tps = df_tensor[timestamp_lists_lower*timestamp_lists_upper]
     
     
    
#     df_all_tps = torch.cat([df_tensor_curr_tps, remaining_tensors])
    
#     sorted_ids = torch.argsort(df_all_tps[:,0])
    
    final_df_all_tps =  df_tensor_curr_tps
     
     
    return final_df_all_tps


def further_processing(train_str, sub_dir):
    all_tensor_array = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir) + '/mimic3_' + train_str + '_tensor')
    
#     torch.save(all_tensor_array[:,:,0], os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + train_str + '_time_stamps')
    
    all_masks_array = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir) + '/mimic3_' + train_str + '_masks')
    
    all_tensor_len_array = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir) + '/mimic3_' + train_str + '_tensor_len0')
    
    
    all_tensor_array_list = []
    
    all_masks_array_list = []
    
    all_tensor_len_array_list = []
    
    for i in range(all_tensor_array.shape[0]):
        
        if i == 24018:
            print('here')
        
        all_time_stamps = all_tensor_array[i][:,0]
        
        ids = torch.sum(np.isnan(all_tensor_array[i,:,1:]), 1) < (all_tensor_array.shape[2]-1)
        non_nan_all_time_stamps = all_time_stamps[ids]
        
        T_max = torch.max(non_nan_all_time_stamps)
        
        if T_max.item() < time_stamp_training:
            continue
        
        all_tensor_array_list.append(all_tensor_array[i])
        all_masks_array_list.append(all_masks_array[i])
        all_tensor_len_array_list.append(all_tensor_len_array[i].item())
        
    all_tensor_array_copy = torch.stack(all_tensor_array_list, 0)
    
    all_masks_array_copy = torch.stack(all_masks_array_list, 0)
    
    all_tensor_len_array_copy = torch.tensor(all_tensor_len_array_list)
    
    print('size::', all_tensor_array_copy.shape)
    
    print('min len::', all_tensor_len_array_copy.min())
    
    torch.save(all_tensor_array_copy, os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir) + '/mimic3_' + train_str + '_tensor')
    
#     torch.save(all_tensor_array[:,:,0], os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + train_str + '_time_stamps')
    
    torch.save(all_masks_array_copy, os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir) + '/mimic3_' + train_str + '_masks')
    
    torch.save(all_tensor_len_array_copy, os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir) + '/mimic3_' + train_str + '_tensor_len0')
    


def get_missing_ratio_per_sample(data, mask):
    
    grouped_samples = {}
    
    for r in range(len(non_missing_ratio_thresholds)):
        grouped_samples[non_missing_ratio_thresholds[r]] = []
    
    for k in range(data.shape[0]):
        
        train_partition_time_stamp = 48
        
        curr_data = data[k:k+1, 0:train_partition_time_stamp]
            
        curr_mask = mask[k:k+1, 0:train_partition_time_stamp]
        
        non_missing_ratio = (torch.sum(curr_mask)*1.0/(curr_mask.shape[0]*curr_mask.shape[1]*curr_mask.shape[2])).item()
        
        
#         print(non_missing_ratio)
        
        for r in range(len(non_missing_ratio_thresholds)):
            if non_missing_ratio > non_missing_ratio_thresholds[r] and r == 0:
                grouped_samples[non_missing_ratio_thresholds[r]].append(k)
                break
            if non_missing_ratio > non_missing_ratio_thresholds[r] and non_missing_ratio < non_missing_ratio_thresholds[r-1]:
                grouped_samples[non_missing_ratio_thresholds[r]].append(k)
                break
        
        
        
#         curr_origin_mask = selected_origin_mask[k:k+1, 0:train_partition_time_stamp]
#         
#         curr_new_mask = selected_new_random_mask[k:k+1, 0:train_partition_time_stamp]
#         
#         curr_x_lens = torch.zeros_like(selected_lens[k:k+1])
#         
#         curr_x_lens[:] = train_partition_time_stamp
#         
# #         curr_x_lens = (selected_lens[k:k+1]*5/6).type(torch.LongTensor)
#         
#         curr_test_data = selected_data[k:k+1, train_partition_time_stamp:selected_lens[k]]
#         
#         curr_test_mask = selected_masks[k:k+1, train_partition_time_stamp:selected_lens[k]]
#         
#         curr_test_origin_mask = selected_origin_mask[k:k+1, train_partition_time_stamp:selected_lens[k]]
#         
#         curr_test_new_mask = selected_new_random_mask[k:k+1, train_partition_time_stamp:selected_lens[k]]
#         
#         curr_test_x_lens = (selected_lens[k:k+1] - curr_x_lens).type(torch.LongTensor)
    
    
    for k in range(len(non_missing_ratio_thresholds)):
        print('non_missing_ratio count::', non_missing_ratio_thresholds[k], len(grouped_samples[non_missing_ratio_thresholds[k]]))
    
    print(grouped_samples)
    
    return grouped_samples


def repartition_training_test_dataset(data_tensor, mask_tensor, len = 72, count = 20000):
    count_single_tensor = int(data_tensor.shape[1]/len)
    
    transformed_data_tensor = []
    
    transformed_mask_tensor = []
    
    for i in range(count_single_tensor):
        
        curr_data_tensor = data_tensor[:, i*len: (i+1)*len].clone()
        
        curr_mask_tensor = mask_tensor[:, i*len: (i+1)*len].clone()
        
        avg_mask = torch.mean(curr_mask_tensor, (1,2))
        
        selected_ids = (avg_mask > 0.1) 
        
        
        
        transformed_data_tensor.append(curr_data_tensor[selected_ids])
        
        print('mask average::', torch.mean(curr_mask_tensor[selected_ids]))
        
        transformed_mask_tensor.append(curr_mask_tensor[selected_ids])
    
    final_transformed_data_tensor = torch.cat(transformed_data_tensor, 0)
    
    print(final_transformed_data_tensor.shape[0])
    
    final_transformed_mask_tensor = torch.cat(transformed_mask_tensor, 0)
    
    print('non missing ratio::')
    
    return final_transformed_data_tensor, final_transformed_mask_tensor
    
    
    
    
    


def partition_training_test_dataset(sub_dir):
    
    all_tensor_array = torch.load(os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + 'tensor')
    
#     torch.save(all_tensor_array[:,:,0], os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + train_str + '_time_stamps')
    
    all_masks_array = torch.load(os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + 'masks')
    
    
    all_tensor_len_array = torch.load(os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + 'tensor_len0')
    
    if sub_dir == 'mimic3_17_5':
        partitioned_len = 72
        
        count = 20000
        
        final_all_tensor_array, final_all_masks_array = repartition_training_test_dataset(all_tensor_array, all_masks_array, len=partitioned_len, count=count)
        all_tensor_array, all_masks_array = final_all_tensor_array, final_all_masks_array
        
        print('data shape::', all_tensor_array.shape, all_masks_array.shape)
        
        all_tensor_len_array = torch.zeros((final_all_tensor_array.shape[0]))
        
        all_tensor_len_array[:] = partitioned_len
    
    
    count = int(all_tensor_array.shape[0]*0.8)
    
    random_ids = torch.randperm(all_tensor_array.shape[0])
    
    training_ids = random_ids[0:count]
    
    test_ids = random_ids[count:]
    
    training_tensor = all_tensor_array[training_ids]
    
    training_masks = all_masks_array[training_ids]
    
    training_len = all_tensor_len_array[training_ids]
    
    
    test_tensor = all_tensor_array[test_ids]
    
    test_masks = all_masks_array[test_ids]
    
    test_len = all_tensor_len_array[test_ids]
    
    
    print('training tensor shape::', training_tensor.shape)
    
    print('non missing ratio training::', torch.mean(training_masks))
    
    print('test tensor shape::', test_tensor.shape)
    
    print('non missing ratio test::', torch.mean(test_masks))
    
    if not os.path.exists(os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir)):
        os.makedirs(os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir))
    
    torch.save(training_tensor, os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir) + '/mimic3_train_tensor')
    
#     torch.save(all_tensor_array[:,:,0], os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + train_str + '_time_stamps')
    
    torch.save(training_masks, os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir) + '/mimic3_train_masks')
    
    torch.save(training_len, os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir) + '/mimic3_train_tensor_len0')
    
    torch.save(test_tensor, os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir) + '/mimic3_test_tensor')
    
#     torch.save(all_tensor_array[:,:,0], os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + train_str + '_time_stamps')
    
    torch.save(test_masks, os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir) + '/mimic3_test_masks')
    
    torch.save(test_len, os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir) + '/mimic3_test_tensor_len0')
    
#     print(all_tensor_array.shape)
    
#     get_missing_ratio_per_sample(all_tensor_array, all_masks_array)
    

def partition_training_test_dataset0(all_tensor_array, all_masks_array, all_tensor_len_array, sub_dir):
#     
#     all_tensor_array = torch.load(os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + 'tensor')
#     
# #     torch.save(all_tensor_array[:,:,0], os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + train_str + '_time_stamps')
#     
#     all_masks_array = torch.load(os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + 'masks')
#     
#     all_tensor_len_array = torch.load(os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + 'tensor_len0')
    
    count = int(all_tensor_array.shape[0]*0.8)
    
    random_ids = torch.randperm(all_tensor_array.shape[0])
    
    training_ids = random_ids[0:count]
    
    test_ids = random_ids[count:]
    
    training_tensor = all_tensor_array[training_ids]
    
    training_masks = all_masks_array[training_ids]
    
    training_len = all_tensor_len_array[training_ids]
    
    
    test_tensor = all_tensor_array[test_ids]
    
    test_masks = all_masks_array[test_ids]
    
    test_len = all_tensor_len_array[test_ids]
    
    
    print('training tensor shape::', training_tensor.shape)
    
    print('test tensor shape::', test_tensor.shape)
    
    print('training missing::', 1 - torch.mean(training_masks))
    
    print('test missing::', 1 - torch.mean(test_masks))
    
    if not os.path.exists(os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir)):
        os.makedirs(os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir))
    
    torch.save(training_tensor, os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir) + '/mimic3_train_tensor')
    
#     torch.save(all_tensor_array[:,:,0], os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + train_str + '_time_stamps')
    
    torch.save(training_masks, os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir) + '/mimic3_train_masks')
    
    torch.save(training_len, os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir) + '/mimic3_train_tensor_len0')
    
    torch.save(test_tensor, os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir) + '/mimic3_test_tensor')
    
#     torch.save(all_tensor_array[:,:,0], os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + train_str + '_time_stamps')
    
    torch.save(test_masks, os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir) + '/mimic3_test_masks')
    
    torch.save(test_len, os.path.join(os.path.join(data_folder, mimic3_data_dir), sub_dir) + '/mimic3_test_tensor_len0')






def parition_dataset_by_missing_ratio(data_folder, mimic3_data_dir, subfolder_src, subfolder_dst):
    
    
    all_tensor_array = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), subfolder_src) + '/mimic3_train_tensor')
    
    all_tensor_array_test = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), subfolder_src) + '/mimic3_test_tensor')
    
#     torch.save(all_tensor_array[:,:,0], os.path.join(data_folder, mimic3_data_dir) + '/mimic3_' + train_str + '_time_stamps')
    
    all_masks_array = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), subfolder_src) + '/mimic3_train_masks')
    
    all_masks_array_test = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), subfolder_src) + '/mimic3_test_masks')
    
    all_tensor_len_array = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), subfolder_src) + '/mimic3_train_tensor_len0')
    
    all_tensor_len_array_test = torch.load(os.path.join(os.path.join(data_folder, mimic3_data_dir), subfolder_src) + '/mimic3_test_tensor_len0')
    
    
    full_tensor_array = torch.cat([all_tensor_array, all_tensor_array_test],0)
    
    full_mask_array = torch.cat([all_masks_array, all_masks_array_test], 0)
    
    full_tensor_lens = torch.cat([all_tensor_len_array, all_tensor_len_array_test], 0)
    
    
    print('training tensor shape::', all_tensor_array.shape)
    
    print('test tensor shape::', all_tensor_array_test.shape)
    
    print('training missing::', 1 - torch.mean(all_masks_array))
    
    print('test missing::', 1 - torch.mean(all_masks_array_test))
    
    grouped_sample_ids = get_missing_ratio_per_sample(full_tensor_array, full_mask_array)
    
    
    selected_ids = []
    
    for k in range(2,5):
        selected_ids.extend(grouped_sample_ids[non_missing_ratio_thresholds[k]])
    
    print('selected id count::', len(selected_ids))
    
    grouped_samples = full_tensor_array[selected_ids]
    
    grouped_masks = full_mask_array[selected_ids]
    
    grouped_masks_lens = full_tensor_lens[selected_ids]
    
    if not os.path.exists(os.path.join(os.path.join(data_folder, mimic3_data_dir), subfolder_dst)):
        os.makedirs(os.path.join(os.path.join(data_folder, mimic3_data_dir), subfolder_dst))
    
    
    partition_training_test_dataset0(grouped_samples, grouped_masks, grouped_masks_lens, subfolder_dst)
    
    print('data shape::', grouped_samples.shape)
    
    print('masks::', torch.mean(grouped_masks))
    
#     torch.save(grouped_samples, os.path.join(os.path.join(data_folder, mimic3_data_dir), subfolder_dst) + '/mimic3_' + train_str + '_tensor')
#     
#     torch.save(grouped_masks, os.path.join(os.path.join(data_folder, mimic3_data_dir), subfolder_dst) + '/mimic3_' + train_str + '_masks')
#     
#     torch.save(grouped_masks_lens, os.path.join(os.path.join(data_folder, mimic3_data_dir), subfolder_dst) + '/mimic3_' + train_str + '_tensor_len0')
    
    
    
    


if __name__ == '__main__':
    
    
#     parition_dataset_by_missing_ratio(data_folder, mimic3_data_dir, 'mimic3_17', 'mimic3_17_group')
     
#     parition_dataset_by_missing_ratio(data_folder, mimic3_data_dir, 'mimic3_17', 'mimic3_17_group', 'test')
    
#     partition_training_test_dataset()
#     further_processing('train', 'mimic3_17')
#      
#     further_processing('test', 'mimic3_17')
    
#     all_tensor_array, all_masks_array, all_tensor_len_array = read_mimic_iii2(mimic3_dir, 'train')
    partition_training_test_dataset('mimic3_17_5')
    select_17 = True

#     all_tensor_array_train, all_masks_array_train, all_tensor_len_array_train = read_mimic_iii2(mimic3_dir_train, select_17)
      
#     all_tensor_array_test, all_masks_array_test, all_tensor_len_array_test = read_mimic_iii(mimic3_dir_test, 'test')
#     
#     
#     print('size::', all_tensor_array_train.shape)
#     
#     print('size::', all_tensor_array_test.shape)
    
    
    