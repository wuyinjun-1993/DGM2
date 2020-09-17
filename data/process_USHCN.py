'''
Created on Jul 17, 2020

'''

import torch

import numpy as np

from os import listdir
from os.path import isfile, join
u_year = 2000

l_year = 1900
import os

def check_month_continuous(climate_file):
    with open(climate_file) as fp:
        
        line = '1'

        last_station_ID = None

        last_month = 0
        
        last_year = 0

        while line:
            
            line = fp.readline()
            
            if len(line) <= 0:
                continue
            
            station_ID = line[0:6]
            
            year = int(line[6:10])
            
            
            if year < l_year or year > u_year:
                continue
            
            month = int(line[10:12])
            
            element = line[12:16]
            
#             print(line)
            
            if not element == 'TMAX':
                
                last_station_ID = station_ID
                
                
                
                continue 
            
            if last_station_ID is not None:
                if not last_station_ID == station_ID:
                    
                    last_month = month
                    
                    last_year = year
                    
                else:
                    
#                     print(last_month, last_year)
#                     
#                     print(month, year)
                    
                    if not (last_month + 1 == month or (last_year + 1 == year and last_month == 12 and month == 1)):
                        print(line)

                        print(station_ID)
                         
#                         print(month, year)


                    
                    last_month = month
                    
                    last_year = year
            
            else:
                last_month = month
                    
                last_year = year
                
                            
            last_station_ID = station_ID
            
#             line = fp.readline()
            
            

def process_one_climate_file(climate_file):
    

    days = 31

    with open(climate_file) as fp:
        
        line = '1'
        cnt = 1
        
        all_samples = {}
        
        
        last_station_ID = None
        
        
        
#         curr_values = []
#         
        curr_masks = []
        
        curr_time_series = []
        
        count = 0
        
        station_count = 0
        
        while line:
            
            
            
#             print('line ID::', count)
            
            count += 1
            
            line = fp.readline()
            
            if len(line) <= 0:
                continue
            
            station_ID = line[0:6]
            
            if station_ID not in all_samples:
                
                all_samples[station_ID] = []
            
            
            
#             if last_station_ID is not None:
#                 if not last_station_ID == station_ID:
#                     all_samples[last_station_ID] = {'values': curr_values, 'masks': curr_masks} 
#                 last_station_ID = station_ID
#                     
#             else:
#                 last_station_ID = station_ID
            
            
            year = int(line[6:10])
            
            month = int(line[10:12])
            
            element = line[12:16]
            
            
#             print(station_ID)
            
            
            
            start = 16
            
            if year < l_year or year > u_year:
                continue
            
            
            
            mask = []
                        
            all_values = []
            
            for k in range(days):
                value = int(line[start:start+5])
                
                mflag = line[start+5: start +6]
                
                fflag = line[start+6: start +7]
                
                sflag = line[start+7: start +8]
                
                if not fflag == ' ':
                    mask.append(0)
                    
                    all_values.append(np.nan)
                    
                    continue
                
                if value == -9999:
                    mask.append(0)
                    
                    all_values.append(np.nan)
                    
                    continue
                
                start += 8
                
                mask.append(1)
                all_values.append(value)
            
            
            curr_time_series.append(all_values)
                        
            curr_masks.append(mask)
        
            if len(curr_time_series) == 5:
            
                if last_station_ID is not None:
                    if not last_station_ID == station_ID:
                        
                        all_samples[station_ID].append({'values':torch.tensor(curr_time_series, dtype = torch.float), 'masks': torch.tensor(curr_masks, dtype = torch.long)})
                        
                        last_month = month
                        
                        last_year = year
                        
                        curr_time_series.clear()
                        
                        curr_masks.clear()
                        
                    else:
                        
    #                     print(last_month, last_year)
    #                     
    #                     print(month, year)
                        
                        if not (last_month + 1 == month or (last_year + 1 == year and last_month == 12 and month == 1)):
                            print(line)
    
                            print(station_ID)
                            
                            if len(all_samples[station_ID]) <= 0:
                                all_samples[station_ID].append({'values':torch.tensor(curr_time_series, dtype = torch.float), 'masks': torch.tensor(curr_masks, dtype = torch.long)})
                            else:
                                all_samples[station_ID].append({'values':torch.tensor(curr_time_series, dtype = torch.float), 'masks': torch.tensor(curr_masks, dtype = torch.long)})
                                
                            curr_time_series.clear()
                            
                            curr_masks.clear()
    
                        else:
                            
                            
                            if len(all_samples[station_ID]) <= 0:
                                all_samples[station_ID].append({'values':torch.tensor(curr_time_series, dtype = torch.float), 'masks': torch.tensor(curr_masks, dtype = torch.long)})
                            else:
#                                 print(all_samples[station_ID][-1]['values'], torch.tensor(curr_time_series, dtype = torch.float))
                                
                                all_samples[station_ID][-1]['values'] = torch.cat([all_samples[station_ID][-1]['values'], torch.tensor(curr_time_series, dtype = torch.float)], 1)
                                
                                all_samples[station_ID][-1]['masks'] = torch.cat([all_samples[station_ID][-1]['masks'], torch.tensor(curr_masks, dtype = torch.long)], 1)
                            
                            curr_time_series.clear()
                            
                            curr_masks.clear()
                        
                        last_month = month
                        
                        last_year = year
                
                else:
                    
                    all_samples[station_ID].append({'values':torch.tensor(curr_time_series, dtype = torch.float), 'masks': torch.tensor(curr_masks, dtype = torch.long)})
                    
                    last_month = month
                        
                    last_year = year
                    
                    curr_time_series.clear()
                            
                    curr_masks.clear()
                    
                    station_count += 1
                                
                last_station_ID = station_ID
            
            
        
        
        
    return all_samples
#         all_samples[last_station_ID] = {'values': curr_values, 'masks': curr_masks} 
            
#             all_values_tensor = torch.tensor(all_values)
              

def extract_values_masks(all_samples):
    
    
    sids = list(all_samples.keys())
    
#     for k in range(len(all_samples)):

    sample_values = []
    
    sample_masks = []

    for k in range(len(sids)):
        alL_sample_lists = all_samples[sids[k]]
        
        for j in range(len(alL_sample_lists)):
            sample_values.append(alL_sample_lists[j]['values'])
            
            sample_masks.append(alL_sample_lists[j]['masks'])
        
            
    return sample_values, sample_masks


def partition_by_lens(all_sample_values, all_sample_masks, sample_len):
    
    
    all_snippits = []
    
    all_masks = []
    
    for i in range(len(all_sample_values)):
        
        num_snippets = int(all_sample_values[i].shape[1]/sample_len)
        
        
        sample_value_transpose = torch.t(all_sample_values[i])
        
        sample_masks = torch.t(all_sample_masks[i])
        
        for k in range(num_snippets):
            
            
            curr_snippets = torch.zeros(sample_len, sample_value_transpose.shape[1])
            
            curr_masks = torch.zeros(sample_len, sample_value_transpose.shape[1])
            
            end_id = sample_value_transpose.shape[0]
            
            if end_id >= (k+1)*sample_len:
                curr_snippets[0:sample_len] = sample_value_transpose[k*sample_len:(k+1)*sample_len]
                curr_masks[0:sample_len] = sample_masks[k*sample_len:(k+1)*sample_len] 
                
            else:
                curr_snippets[0:end_id - k*sample_len] = sample_value_transpose[k*sample_len:end_id]
                curr_masks[0:end_id - k*sample_len] = sample_masks[k*sample_len:end_id]
            
            all_snippits.append(curr_snippets)
            
            all_masks.append(curr_masks)
    
    
    all_snippets_tensor = torch.stack(all_snippits, 0)
    
    all_mask_tensor = torch.stack(all_masks, 0)
    
    print('time series shape::', all_snippets_tensor.shape)
    
    return all_snippets_tensor, all_mask_tensor
    

def generate_time_series(all_sample_values, all_sample_masks, sample_count, sample_len):
    
    all_snippits, all_masks = partition_by_lens(all_sample_values, all_sample_masks, sample_len)
    
    
    
    
    
    selected_snippet_ids = torch.randperm(all_snippits.shape[0])[0:sample_count]
    
#     selected_snippet_ids = torch.tensor(list(range(sample_count)))
    
    print('selected sample count::', selected_snippet_ids.shape)
    
    selected_time_series = all_snippits[selected_snippet_ids]
    
    selected_masks = all_masks[selected_snippet_ids]
    
#     for i in range(len(selected_snippet_ids)):
#         all_snippits[selected_snippet_ids[i]]
    
    return selected_time_series, selected_masks
    
    
    
    
    
def partition_training_test_samples(selected_time_series, selected_masks, dir):
    
    training_sample_count = int(selected_time_series.shape[0]*0.8)
    
    selected_snippet_ids = torch.randperm(selected_time_series.shape[0])[0:training_sample_count]
    
    remaining_ids = set(range(selected_time_series.shape[0])).difference(set(selected_snippet_ids.tolist()))
    
    print(remaining_ids)
    
    remaining_ids_tensor = torch.tensor(list(remaining_ids), dtype = torch.long)
    
    training_samples = selected_time_series[selected_snippet_ids]
    
    training_masks = selected_masks[selected_snippet_ids]
    
    test_samples = selected_time_series[remaining_ids_tensor]
    
    test_masks = selected_masks[remaining_ids_tensor]
    
    
    torch.save(training_samples, dir + '/training_samples')
    
    torch.save(training_masks, dir + '/training_masks')
    
    torch.save(test_samples, dir + '/test_samples')
    
    torch.save(test_masks, dir + '/test_masks')
    
    
    



if __name__ == '__main__':
    path = '../.gitignore/climate/'
    
#     onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    
#     onlyfiles = ["state30_NY.txt", "state28_NJ.txt", "state36_PA.txt", "state06_CT.txt", "state19_MA.txt"]
    
#     onlyfiles = ["state30_NY.txt"]
    
    onlyfiles = ["state04_CA.txt"]
    
    all_sample_value_list = []
    
    all_sample_mask_list = []
    
    sample_mask_count = 0
    
    sample_count = 0
    
    
    f_count = 0
    
    for f in onlyfiles:
        print(f)
        
        curr_all_samples = process_one_climate_file(join(path,f))
        
        sample_value_list, sample_mask_list = extract_values_masks(curr_all_samples)
        
        all_sample_value_list.extend(sample_value_list)
        
        all_sample_mask_list.extend(sample_mask_list)
        
        f_count += 1
        
#         if f_count >= 2:
#             break
    
    
    selected_time_series, selected_masks = generate_time_series(all_sample_value_list, all_sample_mask_list, 5000, 100)
    
    print(selected_time_series.shape, selected_masks.shape)
    
    
    dir = '../.gitignore/climate/tensor/climate_CA'
    
#     dir = '../.gitignore/climate/tensor/climate_NY'
    
    if not os.path.exists(dir):
        os.makedirs(dir)
        
#     if not os.path.exists(os.path.join(dir, 'tensor')):
#         os.makedirs(os.path.join(dir, 'tensor'))
    
    torch.save(selected_time_series, dir + '/data')
    
    torch.save(selected_masks, dir + '/masks')
    
    partition_training_test_samples(selected_time_series, selected_masks, dir)
    
    print(torch.unique(selected_masks))
    
    
    
#         all_samples.append(curr_all_samples)
#         check_month_continuous(join(path,f))
    
            
            
            
    