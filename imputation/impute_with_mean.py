'''
Created on Jul 6, 2020

'''
import torch
import torch.nn as nn


class mean_impute(nn.Module):
    def __init__(self, mean):
        
        super(mean_impute, self).__init__()
        
        self.mean = mean
        
#         self.log_gaussian_weight = nn.Parameter(torch.rand(dimension, device = device))
#         
#         self.log_cross_dim_coeff = nn.Parameter(torch.rand([dimension, dimension], device = device))
#         
#         self.device = device
#         
#         self.dim = dimension
        
    def forward(self, input, masks):
        dim = input.shape
        
        if len(masks.shape) == 3:
            mean_expanded = self.mean.expand(input.shape[0], input.shape[1], self.mean.shape[0])
        else:
            if len(masks.shape) == 2: 
                mean_expanded = self.mean.expand(input.shape[0], self.mean.shape[0])
        
                
        
#         missing_value_ids = torch.nonzero(masks.view(-1) == 0)
#         
#         reshaped_input = input.view(-1)
#         
#         reshaped_input[missing_value_ids] = self.mean
        
        res = input*masks + mean_expanded*(1-masks)
        
        return res
        
        
        
#         nn.Module.forward(self, *input)
        
        
        
        