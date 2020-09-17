
import argparse
import time
from datetime import datetime
import os,sys
from os.path import exists
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from data_loader import PolyphonicDataset
import models, configs
from helper import get_logger, gVar
from tensorboardX import SummaryWriter # install tensorboardX (pip install tensorboardX) before importing this package
from imputation import impute_with_mean
from torch.autograd import Variable


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/lib')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))




sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/imputation')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/lib')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/initialize')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(__file__))


# from initialize.initialize_kmeans import *




# print(os.path.dirname(os.path.abspath(__file__)) + '/lib')


from data.generate_time_series import *

from lib.utils import *

# cluster_models = ['DHMM_cluster', 'DHMM_cluster2', 'DHMM_cluster3', 'DHMM_cluster_tlstm']


def save_model(model, epoch):
    ckpt_path='./output/{}/{}/{}/models/model_epo{}.pkl'.format(args.model, args.expname, args.dataset, epoch)
    print("saving model to %s..." % ckpt_path)
    torch.save(model.state_dict(), ckpt_path)

def load_model(model, epoch):
    ckpt_path='./output/{}/{}/{}/models/model_epo{}.pkl'.format(args.model, args.expname, args.dataset, epoch)
    assert exists(ckpt_path), "epoch misspecified"
    print("loading model from %s..." % ckpt_path)
    model.load_state_dict(torch.load(ckpt_path))


# setup, training, and evaluation


def evaluate_imputation_errors(data_obj, model, is_GPU, device):
    with torch.no_grad():
        
        training_imputed_mse_loss = 0
        
        training_imputed_mse_loss2 = 0
        
        
        training_imputed_mae_loss = 0
        
        training_imputed_mae_loss2 = 0
        
        testing_imputed_mse_loss = 0
        
        testing_imputed_mse_loss2 = 0
        
        
        testing_imputed_mae_loss = 0
        
        testing_imputed_mae_loss2 = 0
        
        training_count = 0
        
        testing_count = 0
        
        model.evaluate = True
        for id, batch_dict in enumerate(data_obj["train_dataloader"]):            
            imputed_data,(imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2) = model.infer(batch_dict["observed_data"], batch_dict["origin_observed_data"], batch_dict['observed_mask'], batch_dict["observed_origin_mask"], batch_dict["observed_new_mask"], batch_dict["observed_lens"], batch_dict['data_to_predict'], batch_dict["origin_data_to_predict"], batch_dict['mask_predicted_data'], batch_dict['origin_mask_predicted_data'], batch_dict['new_mask_predicted_data'], batch_dict["lens_to_predict"], is_GPU, device)
            
            new_x_mask_count = torch.sum(1-batch_dict["observed_new_mask"])
            
            training_count += new_x_mask_count
            
            training_imputed_mse_loss += imputed_mse_loss**2*new_x_mask_count
            
            training_imputed_mse_loss2 += imputed_mse_loss2**2*new_x_mask_count
            
            training_imputed_mae_loss += imputed_loss*new_x_mask_count
        
            training_imputed_mae_loss2 += imputed_loss2*new_x_mask_count
            
            
        for id, batch_dict in enumerate(data_obj["test_dataloader"]):            
            imputed_data,(imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2) = model.infer(batch_dict["observed_data"], batch_dict["origin_observed_data"], batch_dict['observed_mask'], batch_dict["observed_origin_mask"], batch_dict["observed_new_mask"], batch_dict["observed_lens"], batch_dict['data_to_predict'], batch_dict["origin_data_to_predict"], batch_dict['mask_predicted_data'], batch_dict['origin_mask_predicted_data'], batch_dict['new_mask_predicted_data'], batch_dict["lens_to_predict"], is_GPU, device)
            
            new_x_mask_count = torch.sum(1-batch_dict["observed_new_mask"])
            
            testing_count += new_x_mask_count
            
            testing_imputed_mse_loss += imputed_mse_loss**2*new_x_mask_count
            
            testing_imputed_mse_loss2 += imputed_mse_loss2**2*new_x_mask_count
            
            testing_imputed_mae_loss += imputed_loss*new_x_mask_count
        
            testing_imputed_mae_loss2 += imputed_loss2*new_x_mask_count
        
        
        final_training_imputed_mse_loss = torch.sqrt(training_imputed_mse_loss/training_count)
        
        final_training_imputed_mse_loss2 = torch.sqrt(training_imputed_mse_loss2/training_count)
        
        final_training_imputed_mae_loss = training_imputed_mae_loss/training_count
        
        final_training_imputed_mae_loss2 = training_imputed_mae_loss2/training_count
        
        
        final_testing_imputed_mse_loss = torch.sqrt(testing_imputed_mse_loss/testing_count)
        
        final_testing_imputed_mse_loss2 = torch.sqrt(testing_imputed_mse_loss2/testing_count)
        
        final_testing_imputed_mae_loss = testing_imputed_mae_loss/testing_count
        
        final_testing_imputed_mae_loss2 = testing_imputed_mae_loss2/testing_count
        
        
        
        print('training imputation rmse loss::', final_training_imputed_mse_loss)
            
        print('training imputation rmse loss 2::', final_training_imputed_mse_loss2)
        
        print('training imputation mae loss::', final_training_imputed_mae_loss)
        
        print('training imputation mae loss 2::', final_training_imputed_mae_loss2)
        
        
        print('testing imputation rmse loss::', final_testing_imputed_mse_loss)
            
        print('testing imputation rmse loss 2::', final_testing_imputed_mse_loss2)
        
        print('testing imputation mae loss::', final_testing_imputed_mae_loss)
        
        print('testing imputation mae loss 2::', final_testing_imputed_mae_loss2)    
    

def test(data_obj, model, is_GPU, device):
    
    
    observed_test_data = []
    
    observed_test_mask = []
    
    observed_test_lens = []
    
    pred_test_data = []
    
    pred_test_mask = []
    
    pred_test_lens = []
    
    
    final_rmse_loss = 0
    
    final_rmse_loss2 = 0
    
    final_mae_losses = 0
    
    final_mae_losses2 = 0
    
    final_nll_loss = 0
    
    final_nll_loss2 = 0
    
    all_count1 = 0
    
    all_count2 = 0
    
    
    final_imputed_rmse_loss = 0
    
    final_imputed_mae_loss = 0


    final_imputed_rmse_loss2 = 0
    
    final_imputed_mae_loss2 = 0
   
    all_count3 = 0
    
    all_count4 = 0
    
    all_count5 = 0
    
    forecasting_rmse_list = 0
    
    forecasting_mae_list = 0
    
    forecasting_rmse_list2 = 0
    
    forecasting_mae_list2 = 0
    
    forecasting_count = 0
    
    with torch.no_grad():
    
        for id, data_dict in enumerate(data_obj["test_dataloader"]):    
        
        
            batch_dict = data_dict
                
    #         curr_seq_len = len(batch_dict['observed_tp'])
        
            rmse_loss, rmse_loss_count, mae_losses, mae_loss_count, nll_loss, nll_loss_count, list_res, imputed_res = model.test_samples(Variable(batch_dict["observed_data"]), Variable(batch_dict["origin_observed_data"]), Variable(batch_dict['observed_mask']), Variable(batch_dict["observed_origin_mask"]), Variable(batch_dict["observed_new_mask"]), Variable(batch_dict['observed_lens']), Variable(batch_dict['data_to_predict']), Variable(batch_dict["origin_data_to_predict"]), Variable(batch_dict['mask_predicted_data']), Variable(batch_dict['origin_mask_predicted_data']), Variable(batch_dict['new_mask_predicted_data']), Variable(batch_dict['lens_to_predict']), is_GPU, device, batch_dict["delta_time_stamps"], batch_dict["delta_time_stamps_to_predict"], batch_dict["time_stamps"], batch_dict["time_stamps_to_predict"])
            
#             imputed_res = None
            
            all_count1 += rmse_loss_count
    
            all_count2 += mae_loss_count
    
#             print(type(rmse_loss))
            if type(rmse_loss) is tuple and len(list(rmse_loss)) == 2:
                
                rmse_loss_list = list(rmse_loss)
                
                mae_loss_list = list(mae_losses)
                
                final_rmse_loss += (rmse_loss_list[0]**2)*rmse_loss_count
                
                final_mae_losses += (mae_loss_list[0])*mae_loss_count
                
                final_rmse_loss2 += (rmse_loss_list[1]**2)*rmse_loss_count
                
                final_mae_losses2 += (mae_loss_list[1])*mae_loss_count
            
            else:
                final_rmse_loss += (rmse_loss**2)*rmse_loss_count
                
                final_mae_losses += (mae_losses)*mae_loss_count
    
            if nll_loss_count is not None:
                
                if type(nll_loss) is tuple and len(list(nll_loss)) == 2:
                    nll_loss_list = list(nll_loss)
                    
                    final_nll_loss += (nll_loss_list[0])*nll_loss_count
                    
                    final_nll_loss2 += (nll_loss_list[1])*nll_loss_count
                    
                else:
                    final_nll_loss += (nll_loss)*nll_loss_count
                all_count3 += nll_loss_count
            else:
                final_nll_loss = None
                
                
            if imputed_res is not None:
                imputed_mae_res, imputed_mae_count, imputed_rmse_res, imputed_rmse_count = imputed_res
                
                if type(imputed_mae_res) is tuple:
                    
                    imputed_rmse_loss, imputed_rmse_loss2 = imputed_rmse_res
                    
                    imputed_mae_loss, imputed_mae_loss2 = imputed_mae_res
                    
                    final_imputed_rmse_loss += (imputed_rmse_loss ** 2)*imputed_rmse_count
                
                    final_imputed_mae_loss += (imputed_mae_loss)*imputed_mae_count
                    
                    final_imputed_rmse_loss2 += (imputed_rmse_loss2 ** 2)*imputed_rmse_count
                
                    final_imputed_mae_loss2 += (imputed_mae_loss2)*imputed_mae_count
                else:
                    
                    final_imputed_rmse_loss += (imputed_rmse_res ** 2)*imputed_rmse_count
                
                    final_imputed_mae_loss += (imputed_mae_res)*imputed_mae_count
                
                all_count4 += imputed_rmse_count
                
                all_count5 += imputed_mae_count
                
                
                
            if type(list_res[0]) is tuple:
                
                curr_forecasting_rmse_list = list(list_res[0])[0]
                
                curr_forecasting_mae_list = list(list_res[1])[0]
                
                curr_forecasting_rmse_list2 = list(list_res[0])[1]
                
                curr_forecasting_mae_list2 = list(list_res[1])[1]
                
                curr_forecasting_count = list_res[2]
                
                forecasting_rmse_list += (curr_forecasting_rmse_list**2)*curr_forecasting_count
                
                forecasting_mae_list += curr_forecasting_mae_list*curr_forecasting_count
                
                forecasting_rmse_list2 += (curr_forecasting_rmse_list2**2)*curr_forecasting_count
                
                forecasting_mae_list2 += curr_forecasting_mae_list2*curr_forecasting_count
                
                forecasting_count += curr_forecasting_count
                
            else:
                curr_forecasting_rmse_list = list_res[0]
                
                curr_forecasting_mae_list = list_res[1]
                
                curr_forecasting_count = list_res[2]
                
                forecasting_rmse_list += (curr_forecasting_rmse_list**2)*curr_forecasting_count
                
                forecasting_mae_list += curr_forecasting_mae_list*curr_forecasting_count
                
                forecasting_count += curr_forecasting_count
    final_rmse_loss = torch.sqrt(final_rmse_loss/all_count1)
    
    final_mae_losses = final_mae_losses/all_count2
    
    final_rmse_loss2 = torch.sqrt(final_rmse_loss2/all_count1)
    
    final_mae_losses2 = final_mae_losses2/all_count2
    
    print('test results::')
    
    print('test forecasting rmse loss::', final_rmse_loss)
        
    print('test forecasting mae loss::', final_mae_losses)

    print('test forecasting rmse loss 2::', final_rmse_loss2)
        
    print('test forecasting mae loss 2::', final_mae_losses2)
    
    if final_nll_loss is not None:
        final_nll_loss = final_nll_loss/all_count3
        
        final_nll_loss2 = final_nll_loss2/all_count3
        
        print('test forecasting neg likelihood::', final_nll_loss)
        
        print('test forecasting neg likelihood 2::', final_nll_loss2)  
    
    forecasting_rmse_list = torch.sqrt(forecasting_rmse_list/forecasting_count)
    
    forecasting_mae_list = forecasting_mae_list/forecasting_count


    forecasting_rmse_list2 = torch.sqrt(forecasting_rmse_list2/forecasting_count)
    
    forecasting_mae_list2 = forecasting_mae_list2/forecasting_count
    
    print('test forecasting rmse loss by time steps::')
    
    print(forecasting_rmse_list)
    
    print(forecasting_mae_list)
    
    print('test forecasting rmse loss 2 by time steps::')
    
    print(forecasting_rmse_list2)
    
    print(forecasting_mae_list2)
    
    
    if imputed_res is not None:
        final_imputed_rmse_loss = torch.sqrt(final_imputed_rmse_loss/all_count4)
        final_imputed_rmse_loss2 = torch.sqrt(final_imputed_rmse_loss2/all_count4)
        final_imputed_mae_loss = (final_imputed_mae_loss/all_count5)
        final_imputed_mae_loss2 = (final_imputed_mae_loss2/all_count5)
    
    print('test imputation rmse loss::', final_imputed_rmse_loss)
    
    print('test imputation mae loss::', final_imputed_mae_loss)
    
    print('test imputation rmse loss 2::', final_imputed_rmse_loss2)
    
    print('test imputation mae loss 2::', final_imputed_mae_loss2)
    
    if not os.path.exists(data_dir + output_dir):
        os.makedirs(data_dir + output_dir)
    torch.save(model, data_dir + output_dir + 'model')
    
    return (final_rmse_loss, final_mae_losses, final_rmse_loss2, final_mae_losses2, final_imputed_rmse_loss, final_imputed_mae_loss, final_imputed_rmse_loss2, final_imputed_mae_loss2)
  
#         observed_test_mask.append(batch_dict['observed_mask'])
#         
#         pred_test_mask.append(batch_dict['mask_predicted_data'])
# 
#         observed_test_data.append(batch_dict["observed_data"])
#         
#         pred_test_data.append(batch_dict['data_to_predict'])
#         
#         observed_test_lens.append(batch_dict['observed_lens'])
#         
#         pred_test_lens.append(batch_dict['lens_to_predict'])
        
    
#     return torch.cat(observed_test_data, 0), torch.cat(pred_test_data, 0), torch.cat(observed_test_mask, 0), torch.cat(pred_test_mask, 0), torch.cat(observed_test_lens, 0), torch.cat(pred_test_lens, 0)


def validate(data_obj, model, is_GPU, device):
    
    
    observed_test_data = []
    
    observed_test_mask = []
    
    observed_test_lens = []
    
    pred_test_data = []
    
    pred_test_mask = []
    
    pred_test_lens = []
    
    
    final_rmse_loss = 0
    
    final_rmse_loss2 = 0
    
    final_mae_losses = 0
    
    final_mae_losses2 = 0
    
    final_nll_loss = 0
    
    final_nll_loss2 = 0
    
    all_count1 = 0
    
    all_count2 = 0
    
    
    final_imputed_rmse_loss = 0
    
    final_imputed_mae_loss = 0


    final_imputed_rmse_loss2 = 0
    
    final_imputed_mae_loss2 = 0
   
    all_count3 = 0
    
    all_count4 = 0
    
    all_count5 = 0
    
    forecasting_rmse_list = 0
    
    forecasting_mae_list = 0
    
    forecasting_rmse_list2 = 0
    
    forecasting_mae_list2 = 0
    
    forecasting_count = 0
    
    with torch.no_grad():
    
        for id, data_dict in enumerate(data_obj["valid_dataloader"]):    
        
        
            batch_dict = data_dict
                
    #         curr_seq_len = len(batch_dict['observed_tp'])
        
            rmse_loss, rmse_loss_count, mae_losses, mae_loss_count, nll_loss, nll_loss_count, list_res, imputed_res = model.test_samples(Variable(batch_dict["observed_data"]), Variable(batch_dict["origin_observed_data"]), Variable(batch_dict['observed_mask']), Variable(batch_dict["observed_origin_mask"]), Variable(batch_dict["observed_new_mask"]), Variable(batch_dict['observed_lens']), Variable(batch_dict['data_to_predict']), Variable(batch_dict["origin_data_to_predict"]), Variable(batch_dict['mask_predicted_data']), Variable(batch_dict['origin_mask_predicted_data']), Variable(batch_dict['new_mask_predicted_data']), Variable(batch_dict['lens_to_predict']), is_GPU, device, batch_dict["delta_time_stamps"], batch_dict["delta_time_stamps_to_predict"], batch_dict["time_stamps"], batch_dict["time_stamps_to_predict"])
            
#             imputed_res = None
            
            all_count1 += rmse_loss_count
    
            all_count2 += mae_loss_count
    
#             print(type(rmse_loss))
            if type(rmse_loss) is tuple and len(list(rmse_loss)) == 2:
                
                rmse_loss_list = list(rmse_loss)
                
                mae_loss_list = list(mae_losses)
                
                final_rmse_loss += (rmse_loss_list[0]**2)*rmse_loss_count
                
                final_mae_losses += (mae_loss_list[0])*mae_loss_count
                
                final_rmse_loss2 += (rmse_loss_list[1]**2)*rmse_loss_count
                
                final_mae_losses2 += (mae_loss_list[1])*mae_loss_count
            
            else:
                final_rmse_loss += (rmse_loss**2)*rmse_loss_count
                
                final_mae_losses += (mae_losses)*mae_loss_count
    
            if nll_loss_count is not None:
                
                if type(nll_loss) is tuple and len(list(nll_loss)) == 2:
                    nll_loss_list = list(nll_loss)
                    
                    final_nll_loss += (nll_loss_list[0])*nll_loss_count
                    
                    final_nll_loss2 += (nll_loss_list[1])*nll_loss_count
                    
                else:
                    final_nll_loss += (nll_loss)*nll_loss_count
                all_count3 += nll_loss_count
            else:
                final_nll_loss = None
                
                
            if imputed_res is not None:
                imputed_mae_res, imputed_mae_count, imputed_rmse_res, imputed_rmse_count = imputed_res
                
                if type(imputed_mae_res) is tuple:
                    
                    imputed_rmse_loss, imputed_rmse_loss2 = imputed_rmse_res
                    
                    imputed_mae_loss, imputed_mae_loss2 = imputed_mae_res
                    
                    final_imputed_rmse_loss += (imputed_rmse_loss ** 2)*imputed_rmse_count
                
                    final_imputed_mae_loss += (imputed_mae_loss)*imputed_mae_count
                    
                    final_imputed_rmse_loss2 += (imputed_rmse_loss2 ** 2)*imputed_rmse_count
                
                    final_imputed_mae_loss2 += (imputed_mae_loss2)*imputed_mae_count
                else:
                    
                    final_imputed_rmse_loss += (imputed_rmse_res ** 2)*imputed_rmse_count
                
                    final_imputed_mae_loss += (imputed_mae_res)*imputed_mae_count
                
                all_count4 += imputed_rmse_count
                
                all_count5 += imputed_mae_count
                
                
                
            if type(list_res[0]) is tuple:
                
                curr_forecasting_rmse_list = list(list_res[0])[0]
                
                curr_forecasting_mae_list = list(list_res[1])[0]
                
                curr_forecasting_rmse_list2 = list(list_res[0])[1]
                
                curr_forecasting_mae_list2 = list(list_res[1])[1]
                
                curr_forecasting_count = list_res[2]
                
                forecasting_rmse_list += (curr_forecasting_rmse_list**2)*curr_forecasting_count
                
                forecasting_mae_list += curr_forecasting_mae_list*curr_forecasting_count
                
                forecasting_rmse_list2 += (curr_forecasting_rmse_list2**2)*curr_forecasting_count
                
                forecasting_mae_list2 += curr_forecasting_mae_list2*curr_forecasting_count
                
                forecasting_count += curr_forecasting_count
                
            else:
                curr_forecasting_rmse_list = list_res[0]
                
                curr_forecasting_mae_list = list_res[1]
                
                curr_forecasting_count = list_res[2]
                
                forecasting_rmse_list += (curr_forecasting_rmse_list**2)*curr_forecasting_count
                
                forecasting_mae_list += curr_forecasting_mae_list*curr_forecasting_count
                
                forecasting_count += curr_forecasting_count
    final_rmse_loss = torch.sqrt(final_rmse_loss/all_count1)
    
    final_mae_losses = final_mae_losses/all_count2
    
    final_rmse_loss2 = torch.sqrt(final_rmse_loss2/all_count1)
    
    final_mae_losses2 = final_mae_losses2/all_count2
    
    print('validation results::')
    
    print('validation forecasting rmse loss::', final_rmse_loss)
        
    print('validation forecasting mae loss::', final_mae_losses)

    print('validation forecasting rmse loss 2::', final_rmse_loss2)
        
    print('validation forecasting mae loss 2::', final_mae_losses2)
    
    if final_nll_loss is not None:
        final_nll_loss = final_nll_loss/all_count3
        
        final_nll_loss2 = final_nll_loss2/all_count3
        
        print('validation forecasting neg likelihood::', final_nll_loss)
        
        print('validation forecasting neg likelihood 2::', final_nll_loss2)  
    
    forecasting_rmse_list = torch.sqrt(forecasting_rmse_list/forecasting_count)
    
    forecasting_mae_list = forecasting_mae_list/forecasting_count


    forecasting_rmse_list2 = torch.sqrt(forecasting_rmse_list2/forecasting_count)
    
    forecasting_mae_list2 = forecasting_mae_list2/forecasting_count
    
    print('validation forecasting rmse loss by time steps::')
    
    print(forecasting_rmse_list)
    
    print(forecasting_mae_list)
    
    print('validation forecasting rmse loss 2 by time steps::')
    
    print(forecasting_rmse_list2)
    
    print(forecasting_mae_list2)
    
    
    if imputed_res is not None:
        final_imputed_rmse_loss = torch.sqrt(final_imputed_rmse_loss/all_count4)
        final_imputed_rmse_loss2 = torch.sqrt(final_imputed_rmse_loss2/all_count4)
        final_imputed_mae_loss = (final_imputed_mae_loss/all_count5)
        final_imputed_mae_loss2 = (final_imputed_mae_loss2/all_count5)
    
    print('validation imputation rmse loss::', final_imputed_rmse_loss)
    
    print('validation imputation mae loss::', final_imputed_mae_loss)
    
    print('validation imputation rmse loss 2::', final_imputed_rmse_loss2)
    
    print('validation imputation mae loss 2::', final_imputed_mae_loss2)
    
    if not os.path.exists(data_dir + output_dir):
        os.makedirs(data_dir + output_dir)
    torch.save(model, data_dir + output_dir + 'model')
    
    return final_rmse_loss
  
#         observed_test_mask.append(batch_dict['observed_mask'])
#         
#         pred_test_mask.append(batch_dict['mask_predicted_data'])
# 
#         observed_test_data.append(batch_dict["observed_data"])
#         
#         pred_test_data.append(batch_dict['data_to_predict'])
#         
#         observed_test_lens.append(batch_dict['observed_lens'])
#         
#         pred_test_lens.append(batch_dict['lens_to_predict'])
        
    
#     return torch.cat(observed_test_data, 0), torch.cat(pred_test_data, 0), torch.cat(observed_test_mask, 0), torch.cat(pred_test_mask, 0), torch.cat(observed_test_lens, 0), torch.cat(pred_test_lens, 0)

def print_test_res(all_valid_rmse_list, all_test_res, args):
    
    
    all_valid_rmse_array = np.array(all_valid_rmse_list)
    
#     for i in range(all_valid_rmse_array.shape[0]):
    
    
    
    
    
    
    selected_id = np.argmin(all_valid_rmse_array)
    
    test_res = all_test_res[selected_id]
    
    final_rmse_loss, final_mae_losses, final_rmse_loss2, final_mae_losses2, final_imputed_rmse_loss, final_imputed_mae_loss, final_imputed_rmse_loss2, final_imputed_mae_loss2 = test_res

    print('test results::')
    
    if args.model.startswith(cluster_ODE_method):
    
        rmse_loss = min(final_rmse_loss, final_rmse_loss2)
        
        mae_loss = min(final_mae_losses, final_mae_losses2)
        
        imputed_rmse_loss = min(final_imputed_rmse_loss, final_imputed_rmse_loss2)
        
        imputed_mae_loss = min(final_imputed_mae_loss, final_imputed_mae_loss2)
        
        print('test forecasting rmse loss::', rmse_loss)
            
        print('test forecasting mae loss::', mae_loss)
    
#         print('test forecasting rmse loss 2::', final_rmse_loss2)
#             
#         print('test forecasting mae loss 2::', final_mae_losses2)
        
        print('test imputation rmse loss::', imputed_rmse_loss)
        
        print('test imputation mae loss::', imputed_mae_loss)
        
    else:
        
        print('test forecasting rmse loss::', final_rmse_loss)
            
        print('test forecasting mae loss::', final_mae_losses)
    
#         print('test forecasting rmse loss 2::', final_rmse_loss2)
#             
#         print('test forecasting mae loss 2::', final_mae_losses2)
        
        print('test imputation rmse loss::', final_imputed_rmse_loss)
        
        print('test imputation mae loss::', final_imputed_mae_loss)
        
#         print('test imputation rmse loss 2::', final_imputed_rmse_loss2)
#         
#         print('test imputation mae loss 2::', final_imputed_mae_loss2)
    
#     if final_nll_loss is not None:
#         final_nll_loss = final_nll_loss/all_count3
#         
#         final_nll_loss2 = final_nll_loss2/all_count3
        
#     print('test forecasting neg likelihood::', final_nll_loss)
#     
#     print('test forecasting neg likelihood 2::', final_nll_loss2)  

def main(args):
    # setup logging
    
    
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
#     log = get_logger(args.log)
#     log(args)
    
    args.GRUD = False
    if args.model == GRUD_method:
        args.GRUD = True
    
#     timestamp = datetime.now().strftime('%Y%m%d%H%M')
#     tb_writer = SummaryWriter("./output/{}/{}/{}/logs/".format(args.model, args.expname, args.dataset)\
#                           +timestamp) if args.visual else None
    
    config=getattr(configs, 'config_'+args.model)()
    
        
#     train_set=PolyphonicDataset(args.data_path+'train.pkl')
#     valid_set=PolyphonicDataset(args.data_path+'valid.pkl')
#     test_set=PolyphonicDataset(args.data_path+'test.pkl')
#     data_obj, time_steps_extrap, is_missing, train_mean = parse_datasets(args, device)
    
    data_obj, is_missing, train_mean = load_time_series(args)

    config['cluster_num'] = args.cluster_num

    config['input_dim'] = data_obj['input_dim']
    
    config['phi_std'] = args.std
    
    config['epochs'] = args.epochs
    
    config['is_missing'] = is_missing    
#     if args.missing_ratio == 0:
#         if is_missing:
#             config['is_missing'] = True
#         else:
#             config['is_missing'] = False
#     else:
#         config['is_missing'] = True
    
    if args.use_gate:
        config['use_gate'] = True
    else:
        config['use_gate'] = False
    
    config['train_mean'] = train_mean
    
    config['gaussian'] = args.gaussian

    max_kl =args.max_kl
    
    is_GPU = args.GPU 
    
    if not is_GPU:
        device = torch.device("cpu")
    else:    
        GPU_ID = int(args.GPUID)
        device = torch.device("cuda:"+str(GPU_ID) if torch.cuda.is_available() else "cpu")
    config['device'] = device
    model = getattr(models, args.model)(config)

    model.init_params()
    model = model.to(device)
    
#     if args.model in cluster_models:
#         model.loss_on_missing = args.loss_missing
    
    
#     model = model.cuda()
#     if args.reload_from>=0:
#         load_model(model, args.reload_from)


#     model.gaussian = args.gaussian

#     if args.init:
#         
#         print('start initialize')
#         
#         impute_method = mean_impute(train_mean)
#         
#         save_dir = data_dir + '/' + output_dir
#         
#         init_kmeans = initial_kmeans(data_obj["train_dataloader"], impute_method, config['cluster_num'], save_dir)
# 
#         print('start kmeans')
#         
#         init_kmeans.start_selecting_samples()
# 
#         
#         if not args.model == 'DHMM_cluster2':
#         
#             centroids = init_kmeans.run_kmeans()
#             
#             model.init_phi_table(centroids, False)
#         
#         torch.save(init_kmeans.selected_sample_ids, data_dir + '/' + output_dir + 'selected_sample_ids')
#         
#         torch.save(init_kmeans.selected_time_ids, data_dir + '/' + output_dir + 'selected_time_ids')
# 
#         print('end initialize')
#     
#     if args.loadinit:
#         
#         
#         save_dir = data_dir + '/' + output_dir
# 
#         
#         
# #         init_kmeans = torch.load(data_dir + '/' + output_dir + 'init_kmeans_obj')
#         
#         
#         
#         selected_sample_ids = torch.load(data_dir + '/' + output_dir + 'selected_sample_ids')
#         
#         selected_time_ids = torch.load(data_dir + '/' + output_dir + 'selected_time_ids')
#         
#         impute_method = mean_impute(train_mean)
#         
#         init_kmeans = initial_kmeans(data_obj["train_dataloader"], impute_method, config['cluster_num'], save_dir)
#         
#         init_kmeans.selected_sample_ids = selected_sample_ids
#         
#         init_kmeans.selected_time_ids = selected_time_ids
#         
#         if not args.model == 'DHMM_cluster2':
#         
#             init_phi_table = torch.load(data_dir + '/' + output_dir + 'init_phi_table')
#             model.init_phi_table(init_phi_table, True)
        
    
    
#     sequence_len = len(time_steps_extrap)
#     
#     input_dim = data_obj["input_dim"]

#     dataset = dataset_obj.sample_traj(time_steps_extrap, n_samples = args.n, 
#     noise_weight = args.noise_weight)

    #################
    # TRAINING LOOP #
    #################
    times = [time.time()]
    
    
    
    wait_until_kl_inc = args.wait_epoch
    
    wait_until_gumbel = 0
    
    wait_until_sparsemax = 0
    
    itr = 0
    
    
    decay_period = 5
    
    test_period = 1
    
    rec_anneal = 0
    
    
    all_valid_rmse_list = []
    
    all_test_res = []
    
    for epoch in range(config['epochs']):
    
#         for itr in range(1, config['batch_size'] * (args.niters + 1)):
#         for itr in range(0, config['batch_size'], time_steps_extrap.shape[0]):
    
        epoch_nll = 0.0 # accumulator for our estimate of the negative log likelihood (or rather -elbo) for this epoch
        i_batch=1   
        n_slices=0
        
        
        if epoch >= 25:
            print('here')
        

        if epoch  < wait_until_kl_inc:
            kl_anneal = 0.0
        
        else:
#             if (epoch - wait_until_kl_inc) % decay_period == 0:
        

            print('max kl coefficient::', max_kl)
            
            kl_anneal = min((20-20*0.9**(((epoch - wait_until_kl_inc)*1.0)), max_kl))
            
        print('epoch::', epoch, kl_anneal)
#         print('KL::', kl_anneal)
    
        loss_records={}    
        for id, data_dict in enumerate(data_obj["train_dataloader"]):    
        
#             print('ids::',data_dict['ids'])
        
#             batch_dict = get_next_batch(data_dict)
            batch_dict = data_dict
            
#             curr_seq_len = len(batch_dict['observed_tp'])
            
#             print(batch_dict.keys())
            
#             print('id::', id, batch_dict['observed_data'].shape)
    
            
    #         if config['anneal_epochs'] > 0 and epoch < config['anneal_epochs']: # compute the KL annealing factor            
    #                 min_af = config['min_anneal']
    #                 kl_anneal = min_af+(1.0-min_af)*(float(i_batch+epoch*n_iters+1)/float(config['anneal_epochs']*n_iters))
    #         else:            
#             kl_anneal = 0.01 # by default the KL annealing factor is unity
    
#             print("sample count::", ids.shape[0])
            print(id)
            if id >= 1:
                print('here')
    
            
            loss_AE = model.train_AE(batch_dict["observed_data"], batch_dict["origin_observed_data"], batch_dict['observed_mask'], batch_dict["observed_origin_mask"], batch_dict["observed_new_mask"], batch_dict["observed_lens"], kl_anneal, batch_dict['data_to_predict'], batch_dict["origin_data_to_predict"], batch_dict['mask_predicted_data'], batch_dict['origin_mask_predicted_data'], batch_dict['new_mask_predicted_data'], batch_dict["lens_to_predict"], is_GPU, device, batch_dict["delta_time_stamps"], batch_dict["delta_time_stamps_to_predict"], batch_dict["time_stamps"], batch_dict["time_stamps_to_predict"])
            
            
            epoch_nll += loss_AE['train_loss_AE']
            i_batch=i_batch+1
            
            itr += 1
        
        
        if epoch % test_period == 0:
            print("test loss::")           
            
            valid_rmse = validate(data_obj, model, is_GPU, device)
            
            all_valid_rmse_list.append(valid_rmse)
            
#             model.test_samples(batch_dict["observed_data"], batch_dict['data_to_predict'], batch_dict['tp_to_predict'], curr_seq_len, is_GPU, device)
#             final_rmse_loss, final_mae_losses, final_rmse_loss2, final_mae_losses2, final_imputed_rmse_loss, final_imputed_mae_loss, final_imputed_rmse_loss2, final_imputed_mae_loss2

            test_res = test(data_obj, model, is_GPU, device)
    
            
            all_test_res.append(test_res)
#         if args.model == 'DHMM_cluster4' or args.model == 'DHMM_cluster2':
#             updated_centroids = init_kmeans.update_cluster(model)
#             
#             model.init_phi_table(updated_centroids, False)
    

    
    
    print('final test loss::')
    
    
    
    print_test_res(all_valid_rmse_list, all_test_res, args)
    
#     test(data_obj, model, is_GPU, device)
    
#     if args.missing_ratio > 0:
#         print('final imputation errors::')
#         
#         evaluate_imputation_errors(data_obj, model, is_GPU, device)
    
    
    
    if not os.path.exists(data_dir + output_dir):
        os.makedirs(data_dir + output_dir)
    torch.save(model, data_dir + output_dir + 'model')
    
#     if config['is_missing'] and args.model in cluster_models:
#         
#         
#         T_max = torch.max(data_obj["train_dataloader"].dataset.lens)
#         
# #         print(data_obj["train_dataloader"].dataset.lens)
#         
#         imputed_train_data = model.impute(data_obj["train_dataloader"].dataset.data, data_obj["train_dataloader"].dataset.mask, T_max)
#         
#         torch.save(imputed_train_data, data_dir + output_dir + 'imputed_train_data')
#         
#         
#         T_max_test = torch.max(data_obj["test_dataloader"].dataset.lens)
#         
#         imputed_test_data = model.impute(data_obj["test_dataloader"].dataset.data, data_obj["test_dataloader"].dataset.mask, T_max_test)
#         
#         torch.save(imputed_test_data, data_dir + output_dir + 'imputed_test_data')
    
#     if args.model == 'DHMM_cluster2':
#         T_max = torch.max(data_obj["train_dataloader"].dataset.lens)
#         
# #         print(data_obj["train_dataloader"].dataset.lens)
#         
#         imputed_train_data = model.impute(data_obj["train_dataloader"].dataset.data, data_obj["train_dataloader"].dataset.mask, T_max)
#         
#         torch.save(imputed_train_data, data_dir + output_dir + 'imputed_train_data')
#         
#         
#         imputed_kernel_train_data = model.x_kernel_encoder(imputed_train_data)
#         
#         torch.save(imputed_kernel_train_data, data_dir + output_dir + 'imputed_kernel_train_data')
#         
#         T_max_test = torch.max(data_obj["test_dataloader"].dataset.lens)
#         
#         imputed_test_data = model.impute(data_obj["test_dataloader"].dataset.data, data_obj["test_dataloader"].dataset.mask, T_max_test)
#         
#         imputed_kernel_test_data = model.x_kernel_encoder(imputed_test_data)
#         
#         torch.save(imputed_kernel_test_data, data_dir + output_dir + 'imputed_kernel_test_data')
#         
#         
#         
#         torch.save(imputed_test_data, data_dir + output_dir + 'imputed_test_data')
    
    
    
    
    
#     model.test_samples(batch_dict["observed_data"], batch_dict['data_to_predict'], batch_dict['tp_to_predict'], curr_seq_len)
       
#         n_slices=n_slices+x_lens.sum().item()
#         loss_records.update(loss_AE)   
# #         loss_records.update({'epo_nll':epoch_nll/n_slices})
#         times.append(time.time())
#         epoch_time = times[-1] - times[-2]
#         log("[Epoch %04d]\t\t(dt = %.3f sec)"%(epoch, epoch_time))
#         log(loss_records)
        
    
    
#     for id, data_dict in enumerate(data_obj["train_dataloader"]):
#         
#             batch_dict = get_next_batch(data_dict)
#             
#             curr_seq_len = len(batch_dict['observed_tp'])
#             
#             kl_anneal = 5.0 # by default the KL annealing factor is unity
#             loss_AE = model.train_AE(batch_dict["observed_data"], kl_anneal, curr_seq_len)
#     
#             model.evaluate_forecasting_errors(batch_dict, batch_size, curr_seq_len, h_now, c_now)
#         if args.visual:
#             for k, v in loss_records.items():
#                 tb_writer.add_scalar(k, v, epoch)
        # do evaluation on test and validation data and report results
#         if (epoch+1) % args.test_freq == 0:
#             save_model(model, epoch)
#             test_loader=torch.utils.data.DataLoader(dataset=test_set, batch_size=config['batch_size'], shuffle=False, num_workers=1)
#             for x, x_rev, x_lens in test_loader: 
#                 x, x_rev, x_lens = gVar(x), gVar(x_rev), gVar(x_lens)
#                 test_nll = model.valid(x, x_rev, x_lens) / x_lens.sum()
#             log("[val/test epoch %08d]  %.8f" % (epoch, test_nll))
        
        
    
#     for epoch in range(config['epochs']):
#             
#             
#             
#         
#         train_loader=torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True, num_workers=1)
#         train_data_iter=iter(train_loader)
#         n_iters=train_data_iter.__len__()
#         
#         epoch_nll = 0.0 # accumulator for our estimate of the negative log likelihood (or rather -elbo) for this epoch
#         i_batch=1   
#         n_slices=0
#         loss_records={}
#         while True:            
#             try: x, x_rev, x_lens = train_data_iter.next()                  
#             except StopIteration: break # end of epoch                 
#             x, x_rev, x_lens = gVar(x), gVar(x_rev), gVar(x_lens)
#             
#             if config['anneal_epochs'] > 0 and epoch < config['anneal_epochs']: # compute the KL annealing factor            
#                 min_af = config['min_anneal']
#                 kl_anneal = min_af+(1.0-min_af)*(float(i_batch+epoch*n_iters+1)/float(config['anneal_epochs']*n_iters))
#             else:            
#                 kl_anneal = 1.0 # by default the KL annealing factor is unity
#             
#             loss_AE = model.train_AE(x, x_rev, x_lens, kl_anneal)
#             
#             epoch_nll += loss_AE['train_loss_AE']
#             i_batch=i_batch+1
#             n_slices=n_slices+x_lens.sum().item()
#             
#         loss_records.update(loss_AE)   
#         loss_records.update({'epo_nll':epoch_nll/n_slices})
#         times.append(time.time())
#         epoch_time = times[-1] - times[-2]
#         log("[Epoch %04d]\t\t(dt = %.3f sec)"%(epoch, epoch_time))
#         log(loss_records)
#         if args.visual:
#             for k, v in loss_records.items():
#                 tb_writer.add_scalar(k, v, epoch)
#         # do evaluation on test and validation data and report results
#         if (epoch+1) % args.test_freq == 0:
#             save_model(model, epoch)
#             test_loader=torch.utils.data.DataLoader(dataset=test_set, batch_size=config['batch_size'], shuffle=False, num_workers=1)
#             for x, x_rev, x_lens in test_loader: 
#                 x, x_rev, x_lens = gVar(x), gVar(x_rev), gVar(x_lens)
#                 test_nll = model.valid(x, x_rev, x_lens) / x_lens.sum()
#             log("[val/test epoch %08d]  %.8f" % (epoch, test_nll))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--model', type=str, default='DHMM_cluster', help='model name')
    parser.add_argument('--dataset', type=str, default='climate_NY', help='name of dataset')
#     parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")

    parser.add_argument('-std',  type=float, default=0.5, help="std of the initial phi table")
    parser.add_argument('-b', '--batch-size', type=int, default=40)
    parser.add_argument('--wait_epoch', type=int, default=0)
    parser.add_argument('-e', '--epochs', type=int, default=500)
    parser.add_argument('--use_gate', action='store_true', help = 'use gate in the model')

    parser.add_argument('--GPU', action='store_true', help="GPU flag")
    
    parser.add_argument('-G', '--GPUID', type = int, help="GPU ID")
    
    parser.add_argument('--cluster_num', type = int, default = 20,  help="number of clusters")
    
    parser.add_argument('--max_kl', type = float, default = 1.0, help="max kl coefficient")
    
    parser.add_argument('--gaussian', type = float, default = 0.000001, help="gaussian coefficient")
    
    
#     parser.add_argument('-l', '--log', type=str, default='dmm.log')
    args = parser.parse_args()
    
    os.makedirs('./output/{args.model}/{args.expname}/{args.dataset}/models', exist_ok=True)
    main(args)
