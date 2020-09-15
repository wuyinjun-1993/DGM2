'''
Created on Jun 29, 2020

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class IDW_impute_low_high_pass(nn.Module):
    def __init__(self, dimension, device):
        
        super(IDW_impute_low_high_pass, self).__init__()
        
        self.k = nn.Parameter(torch.rand(dimension, device = device))
        
        self.log_gaussian_weight = nn.Parameter(torch.rand(dimension, device = device))
        
        self.log_cross_dim_coeff = nn.Parameter(torch.rand([dimension,dimension] , device = device))
        
        self.device = device
        
        self.dim = dimension


#     def compute_high_pass(self, observed_data, observed_mask, T_max):
        
    def compute_lambda(self, observed_data, observed_mask, weight_time_scale):
        
        observed_ones = torch.ones_like(observed_data)
        
        transformed_observed_ones = torch.transpose(torch.transpose(observed_ones*observed_mask, 0, 2), 1, 2)
        
        influence_agg_weights = torch.transpose(torch.bmm(transformed_observed_ones, weight_time_scale), 0, 1)
        
        return influence_agg_weights


    def get_influence_each_dim(self, observed_data, observed_mask, T_max, gaussian_weight):
        T_list = torch.tensor(range(T_max), device =self.device, dtype = torch.float)
        
        '''T_max, T_max'''
        
        T_list_gap = ((T_list.view(-1, 1) - T_list.view(1,-1))/10)**2
        
        '''T_max*T_max, m'''
        
        weighted_T_list_gap = -torch.mm(T_list_gap.view(-1,1), gaussian_weight.view(1,-1))
        
        
        '''m, T_max*T_max'''
        weight_time_scale = torch.t(torch.exp(weighted_T_list_gap)).view(self.dim, T_max, T_max)
        
        '''lambda: b, m, T_max'''
        influence_agg_weights = self.compute_lambda(observed_data, observed_mask, weight_time_scale)
        
#         for k in range(self.dim):
#             weight_time_scale[k].fill_diagonal_(0)
        
#         weight_time_scale_cp = weight_time_scale.clone()
        
#         for k in range(self.dim):
#             weight_time_scale_cp[k].fill_diagonal_(0)        
        '''b, T_max, m -> m, T_max, b -> m, b, T_max'''
        transformed_observed_data = torch.transpose(torch.transpose(observed_data*observed_mask, 0, 2), 1, 2)
        
        influence_each_dimension = torch.transpose(torch.bmm(transformed_observed_data, weight_time_scale), 0, 1)
        
        return influence_each_dimension, influence_agg_weights

    def forward(self, observed_data, observed_mask, T_max):
        gaussian_weight = F.softplus(self.log_gaussian_weight)

        cross_dim_coeff = F.softplus(self.log_cross_dim_coeff)
        
        gw_weight = F.softplus(self.k) + 1 
        

        influence_each_dimension, influence_agg_weights = self.get_influence_each_dim(observed_data, observed_mask, T_max, gaussian_weight)
        
        influence_each_dimension2, influence_agg_weights2 = self.get_influence_each_dim(observed_data, observed_mask, T_max, gw_weight)
        
        
        
        
        '''sum of lambda:: b, T_max'''
        influence_agg_weights_cross_dims = torch.sum(influence_agg_weights, 1)
        
        influence_agg_weights_cross_dims = influence_agg_weights_cross_dims.view(observed_data.shape[0], 1, T_max)
        
        '''m, b T_max -> b,m,T_max'''
        
        
        
        
        
#         influence_each_dimension_cp = torch.transpose(torch.bmm(transformed_observed_data, weight_time_scale_cp), 0, 1)
        
        cross_dim_coeff_rep = torch.t(cross_dim_coeff).repeat(observed_data.shape[0], 1, 1)
        
        
        
        agg_influence = torch.transpose(torch.bmm(cross_dim_coeff_rep, influence_each_dimension), 1, 2)
        
        
        '''b,m,T_max'''
        low_pass_res = agg_influence/influence_agg_weights_cross_dims.view(influence_agg_weights_cross_dims.shape[0], influence_agg_weights_cross_dims.shape[2], 1)
        
        
        
        
#         print(agg_influence.shape, influence_agg_weights_cross_dims.shape, influence_each_dimension2.shape, influence_agg_weights2.shape, low_pass_res.shape)
        high_pass_res = influence_each_dimension2/influence_agg_weights2 - torch.transpose(low_pass_res, 2, 1)
        
#         agg_influence_cp = torch.transpose(torch.bmm(cross_dim_coeff_rep, influence_each_dimension_cp), 1, 2)
        
#         res = agg_influence*(1-observed_mask) + observed_data*observed_mask
#         
#         agg_cp = agg_influence - ((torch.diag(cross_dim_coeff).view(-1)).repeat(observed_data.shape[0], observed_data.shape[1], 1)*observed_data*observed_mask)
#         
# #         res_cp = agg_influence_cp*(1-observed_mask) + observed_data*observed_mask
# #         
#         print(torch.norm(agg_influence*(1-observed_mask) - agg_cp*(1-observed_mask)))              
        
        return influence_agg_weights, low_pass_res, high_pass_res, agg_influence, influence_agg_weights_cross_dims.view(influence_agg_weights_cross_dims.shape[0], influence_agg_weights_cross_dims.shape[2], 1)
        
        
#     def forward2(self, observed_data, observed_mask, T_max):
#         gaussian_weight = F.softplus(self.log_gaussian_weight)
#  
#         cross_dim_coeff = F.softplus(self.log_cross_dim_coeff)
#          
#         T_list = torch.tensor(range(T_max), device =self.device, dtype = torch.float)
#          
#         '''T_max, T_max'''
#          
#         T_list_gap = (T_list.view(-1, 1) - T_list.view(1,-1))**2
#          
#         '''T_max*T_max, m'''
#          
#         weighted_T_list_gap = -torch.mm(T_list_gap.view(-1,1), gaussian_weight.view(1,-1))
#          
#          
#         '''m, T_max*T_max'''
#         weight_time_scale = torch.t(torch.exp(weighted_T_list_gap)).view(self.dim, T_max, T_max)
#          
# #         for k in range(self.dim):
# #             weight_time_scale[k].fill_diagonal_(0)
#          
#         '''b, T_max, m -> m, T_max, b -> m, b, T_max'''
#         transformed_observed_data = torch.transpose(torch.transpose(observed_data*observed_mask, 0, 2), 1, 2)
#          
#          
#         '''m, b T_max -> b,m,T_max'''
#         influence_each_dimension = torch.transpose(torch.bmm(transformed_observed_data, weight_time_scale), 0, 1)
#          
#         cross_dim_coeff_rep = torch.t(cross_dim_coeff).repeat(observed_data.shape[0], 1, 1)
#          
#          
#          
#         agg_influence = torch.transpose(torch.bmm(cross_dim_coeff_rep, influence_each_dimension), 1, 2)
#          
#         return agg_influence#, agg_influence    
        

    def forward2(self, observed_data, observed_mask, T_max):
 
         
        gaussian_weight = F.softplus(self.log_gaussian_weight)
 
        cross_dim_coeff = F.softplus(self.log_cross_dim_coeff)
 
        missing_ids = torch.nonzero(torch.zeros_like(observed_mask) == 0)
         
        k = 0
         
        exp_dim_coeff = None
         
        exp_transformed_input_data = None
                 
        exp_dim_influence = None
                 
        exp_imputed_value = None
        
        results = torch.zeros_like(observed_data)
         
         
        agg_single_weight_sum = torch.zeros_like(observed_data)
        
        
        for ids in missing_ids:
            s_id = ids[0]
            time_id = ids[1]
            dim_id = ids[2]
         
            T_list = torch.tensor(range(T_max), dtype = torch.float, device =self.device)
 
            T_diff = torch.t((((T_list - time_id)/10)**2).expand(self.dim, T_list.shape[0]))
             
             
             
            dim_coeff = torch.exp(-gaussian_weight.view(1, self.dim)*T_diff)
             
            transformed_input_data = observed_data[s_id]*observed_mask[s_id]
             
            dim_influence = torch.sum(transformed_input_data*dim_coeff, 0)
             
            agg_single_weight_sum[s_id, time_id] = torch.sum(dim_coeff*observed_mask[s_id],0)
             
#             other_dim_ids = list(range(self.dim))
#             
#             other_dim_ids.remove(dim_id)
             
#             if k == 0:
#                 print(dim_coeff)
#                 print(dim_influence)
                 
             
             
            imputed_value = torch.sum(dim_influence*cross_dim_coeff[:, dim_id])
             
            if k == 0:
#                 print(ids)
#                 print(dim_coeff)
#                  
#                 print(transformed_input_data)
#                  
#                 print(dim_influence)
#                 print(imputed_value)
                 
                exp_dim_coeff = dim_coeff
         
                exp_transformed_input_data = transformed_input_data
                         
                exp_dim_influence = dim_influence
                         
                exp_imputed_value = dim_influence
             
             
            results[s_id, time_id, dim_id] = imputed_value
             
            k+=1
             
             
        return results, agg_single_weight_sum
     
    