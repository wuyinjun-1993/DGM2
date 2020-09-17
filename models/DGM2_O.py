
import argparse
import time
from os.path import exists
import numpy as np


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from modules import GatedTransition, PostNet, Encoder , PostNet_cluster,\
    Encoder_cluster
from helper import reverse_sequence, sequence_mask
from torch.distributions import normal

from torch.distributions import kl_divergence, Independent
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.autograd import Variable

from ODE_modules import *



import time

import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/imputation')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/lib')
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/initialize')
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/Sparsemax')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(__file__))


from imputation.inverse_distance_weighting import *
from lib.utils import *
# from initialize.initialize_kmeans import *
# from Sparsemax.Sparsemax import Sparsemax
from lib.encoder_decoder_cluster import *


data_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/" + data_dir

# def sample_gumbel(shape, is_GPU, eps=1e-20):
#     U = torch.rand(shape)
#     if is_GPU:
#         U = U.cuda()
#     return -torch.log(-torch.log(U + eps) + eps)
# 
# 
# def gumbel_softmax_sample(logits, temperature, is_GPU):
#     y = logits + sample_gumbel(logits.size(), is_GPU)
#     return F.softmax(y / temperature, dim=-1)
# 
# 
# def gumbel_softmax(logits, temperature, latent_dim, categorical_dim, is_GPU=False, hard=False):
#     """
#     ST-gumple-softmax
#     input: [*, n_class]
#     return: flatten --> [*, n_class] an one-hot vector
#     """
#     y = gumbel_softmax_sample(logits, temperature, is_GPU)
#     
#     if not hard:
#         return y.view(-1, latent_dim * categorical_dim)
# 
#     shape = y.size()
#     _, ind = y.max(dim=-1)
#     y_hard = torch.zeros_like(y).view(-1, shape[-1])
#     y_hard.scatter_(1, ind.view(-1, 1), 1)
#     y_hard = y_hard.view(*shape)
#     # Set gradients w.r.t. y_hard gradients w.r.t. y
#     y_hard = (y_hard - y).detach() + y
#     return y_hard.view(-1, latent_dim * categorical_dim)
# 
# sparsemax = Sparsemax(dim=-1)


class DGM2_O(nn.Module):
    """
    The Deep Markov Model
    """
    def __init__(self, config ):
        super(DGM2_O, self).__init__()
        
        
        
        
        
        self.input_dim = config['input_dim']
        
        self.h_dim = config['h_dim']
        
        self.s_dim = config['s_dim']
        
        self.centroid_max = config['d']
#         self.input_dim = config['input_dim']
        
        self.device = config['device']
        
        self.dropout = config['dropout']
        self.e_dim = config['e_dim']


#         self.latent_x_std = config['latent_x_std']

#         sparsemax.device = self.device

        
        
        
        
        self.cluster_num = config['cluster_num']
        
#         self.phi_std = config['phi_std']
        
        
        self.h_0 = torch.zeros(self.h_dim, device = config['device'])
        
        self.s_0 = torch.zeros(self.s_dim, device = config['device'])
        
        self.x_std = config['x_std']
        
#         self.sample_times = config['sampling_times']
        
        self.use_gate = config['use_gate']
        
        self.evaluate = False
        
#         self.loss_on_missing = False
        
#         self.t_thres = config['t_thres']
        
#         self.temp = config['temp_init']
        
        self.transfer_prob = True
        
        self.gaussian_prior_coeff = config['gaussian']
        
        self.pre_impute = True
        
        self.shift = 5
        
        self.use_shift = True
        
#         self.loss_with_mask = True
        
#         self.emission_z_dim = config['emission_z_dim']
#         self.emission_input_dim = config['emission_input_dim']
        
#         self.trans_dim = config['trans_dim']
#         self.rnn_dim = config['rnn_dim']
        self.clip_norm = config['clip_norm']
        
        self.emitter_z = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
#             nn.Linear(self.h_dim, self.emission_z_dim),
#             nn.ReLU(),
#             nn.Linear(self.emission_z_dim, self.emission_z_dim),
#             nn.ReLU(),
# # #             nn.Linear(self.emission_dim, self.emission_dim),
# #             
#             nn.Linear(self.emission_z_dim, self.cluster_num)
#             nn.Softmax()
            nn.Linear(self.h_dim, self.cluster_num),
            nn.Dropout(p = self.dropout)
#             nn.Sigmoid()
#             nn.ReLU()
        )
        
        
        
#         self.emiter_x_mean = nn.Sequential(
#             nn.Linear(self.z_dim, self.e_dim),
#             nn.ReLU(),
#             nn.Dropout(p = self.dropout),
# #             nn.Linear(self.e_dim, self.e_dim),
# #             nn.Dropout(p = self.dropout),
# #             nn.ReLU(),
#             nn.Linear(self.e_dim, self.input_dim)
# #             nn.ReLU(),
# #             nn.Dropout(p = self.dropout)
#             )
#         
#         
#         self.emiter_x_var = nn.Sequential(
#             nn.Linear(self.z_dim, self.e_dim),
#             nn.ReLU(),
#             nn.Dropout(p = self.dropout),
# #             nn.Linear(self.e_dim, self.e_dim),
# #             nn.Dropout(p = self.dropout),
#             nn.Linear(self.e_dim, self.input_dim),
#             nn.ReLU(),
#             nn.Dropout(p = self.dropout)
#             )
        
#         nn.init.normal_(self.emiter_x_var.weight, 0, self.x_std)
        
#         self.emitter_x = nn.Sequential(
# #             nn.Linear(self.z_dim, self.emission_input_dim),
# #             nn.ReLU(),
# #             nn.Linear(self.emission_input_dim, self.input_dim)
#             nn.Linear(self.z_dim, self.input_dim)
#         )
        
#         self.trans = GatedTransition()

#         self.latent =  config['latent']
        
        self.block = 'LSTM'
        
#         self.lstm_latent = True

#         if not self.latent:
        self.z_dim = self.input_dim
#         else:
#             
#             if not self.lstm_latent:
#                 self.z_dim = config['z_dim']
#                 
#             else:
#                 self.z_dim = self.input_dim

#         self.use_sparsemax = False
#         
#         self.use_gumbel = False
        
        self.use_transition_gen = True
        
        if config['is_missing']:
            self.impute = IDW_impute(self.input_dim, self.device)
        self.is_missing = config['is_missing'] 
        
        
        z0_dim = self.s_dim

        self.use_mask = True

        n_rec_dims = self.s_dim
        
        ode_func_net = create_net(n_rec_dims, n_rec_dims, nonlinear = nn.Tanh, dropout = self.dropout, n_layers = 0)
    
        rec_ode_func = ODEFunc(
            input_dim = self.s_dim, 
            latent_dim = n_rec_dims,
            ode_func_net = ode_func_net,
            device = self.device).to(self.device)

        z0_diffeq_solver = DiffeqSolver(self.s_dim, rec_ode_func, "euler", self.s_dim, 
            odeint_rtol = 1e-3, odeint_atol = 1e-4, device = self.device)
        
        
#         z0_diffeq_solver = DiffeqSolver(self.s_dim, rec_ode_func, "euler", self.s_dim, device = self.device)
        
        self.gru_update = GRU_unit_cluster(self.s_dim, self.input_dim, n_units = self.h_dim, device = self.device, use_mask = self.use_mask, dropout = self.dropout)
        
#         if self.transfer_prob:

        self.method_2 = False
        
        if self.method_2:
            self.postnet = Encoder_z0_ODE_RNN_cluster2(n_rec_dims, self.input_dim, self.cluster_num, z0_diffeq_solver, 
            z0_dim = z0_dim, n_gru_units = self.h_dim, GRU_update=self.gru_update, device = self.device, use_mask = self.use_mask, dropout = self.dropout).to(self.device)
        else:self.postnet = Encoder_z0_ODE_RNN_cluster(n_rec_dims, self.input_dim, self.cluster_num, z0_diffeq_solver, 
            z0_dim = z0_dim, n_gru_units = self.h_dim, GRU_update=self.gru_update, device = self.device, use_mask = self.use_mask, dropout = self.dropout).to(self.device)
#         self.trans = Decoder(self.h_dim, self.input_dim).to(self.device)
        
        ode_func_net2 = create_net(self.s_dim, self.s_dim,  nonlinear = nn.Tanh, dropout = self.dropout, n_layers = 0, n_units = 20)
     
        self.gen_ode_func = ODEFunc(
                input_dim = self.s_dim, 
                latent_dim = self.s_dim, 
                ode_func_net = ode_func_net2,
                device = self.device).to(self.device)
#         
#         
        self.prob_to_states = nn.Sequential(nn.Linear(self.cluster_num, self.s_dim), nn.Dropout(p = self.dropout))
#         
#         self.diffeq_solver = DiffeqSolver(self.s_dim, self.gen_ode_func, 'dopri5', self.s_dim, device = self.device, odeint_rtol = 1e-3, odeint_atol = 1e-4)
        
        
#         ode_func_net2 = create_net(self.cluster_num, self.cluster_num,  nonlinear = nn.Tanh, dropout = self.dropout, n_layers = -1)
#     
#         self.gen_ode_func = ODEFunc(
#                 input_dim = self.cluster_num, 
#                 latent_dim = self.cluster_num, 
#                 ode_func_net = ode_func_net2,
#                 device = self.device).to(self.device)
        
        
#         self.prob_to_states = nn.Sequential(nn.Linear(self.cluster_num, self.s_dim), nn.Dropout(p = self.dropout))
        
        self.diffeq_solver = DiffeqSolver(self.cluster_num, self.gen_ode_func, 'dopri5', self.s_dim, device = self.device, odeint_rtol = 1e-3, odeint_atol = 1e-4)
        
#         self.diffeq_solver = DiffeqSolver(self.cluster_num, self.gen_ode_func, 'dopri5', self.s_dim, device = self.device)
        
        
            
        self.gen_emitter_z = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
#             nn.Linear(self.h_dim, self.emission_z_dim),
#             nn.ReLU(),
#             nn.Linear(self.emission_z_dim, self.emission_z_dim),
#             nn.ReLU(),
# # #             nn.Linear(self.emission_dim, self.emission_dim),
# #             
#             nn.Linear(self.emission_z_dim, self.cluster_num)
#             nn.Softmax()
            nn.Linear(self.s_dim, self.cluster_num),
#             nn.Dropout(p = self.dropout)
#             nn.Sigmoid()
#             nn.ReLU()
        )
        
        self.trans = Decoder_ODE_RNN_cluster(n_rec_dims, self.cluster_num, self.cluster_num, self.diffeq_solver, 
            z0_dim = z0_dim, n_gru_units = self.h_dim, device = self.device, dropout = self.dropout).to(self.device)
        
        if self.use_gate:
            self.gate_func = nn.Sequential(
#                 nn.Linear(self.h_dim, 1),
                nn.Linear(self.s_dim, 1),
                nn.Dropout(p = self.dropout)
                )
        

        
#         if self.transfer_prob:
#             self.postnet = PostNet_cluster(self.cluster_num, self.s_dim, self.cluster_num, self.dropout, self.use_sparsemax, self.use_gumbel, self.sample_times, self.t_thres)
#         else:
#             self.postnet = PostNet_cluster(self.z_dim, self.s_dim, self.cluster_num, self.dropout, self.use_sparsemax, self.use_gumbel, self.sample_times, self.t_thres)
#         self.rnn = Encoder(None, self.input_dim, self.rnn_dim, False, 1)
        
#         self.trans = torch.nn.LSTM(self.cluster_num, self.h_dim)
#         self.x_encoder = torch.nn.LSTM(self.input_dim, self.s_dim, bidirectional = True)
        
        
#         if self.transfer_prob:
#             if self.block == 'GRU':
#                 self.trans = torch.nn.GRU(self.cluster_num, self.h_dim, dropout = self.dropout, batch_first = True)
#             else:
#     #             self.trans = Encoder_cluster(self.input_dim, self.h_dim, self.dropout, bidir = False, block = self.block)
#                 self.trans = torch.nn.LSTM(self.cluster_num, self.h_dim, dropout = self.dropout, batch_first = True)
#         
#         else:
#             if self.block == 'GRU':
#                 self.trans = torch.nn.GRU(self.z_dim, self.h_dim, dropout = self.dropout, batch_first = True)
#             else:
#     #             self.trans = Encoder_cluster(self.input_dim, self.h_dim, self.dropout, bidir = False, block = self.block)
#                 self.trans = torch.nn.LSTM(self.z_dim, self.h_dim, dropout = self.dropout, batch_first = True)
#         
# #             self.x_encoder = torch.nn.GRU(self.input_dim, self.s_dim, bidirectional = True, dropout = self.dropout)
#         self.x_encoder = Encoder_cluster(self.z_dim, self.s_dim, self.dropout, self.device, bidir = True, block = self.block)
#         
#         if self.use_gate:
#             self.gate_func = nn.Sequential(
# #                 nn.Linear(self.h_dim, 1),
#                 nn.Linear((1+self.x_encoder.bidir)*self.s_dim, 1),
#                 nn.Dropout(p = self.dropout)
#                 )
        
        
        self.phi_table = torch.zeros([self.input_dim, self.cluster_num], dtype = torch.float, device = config['device']) 
        
        self.phi_table = torch.nn.Parameter(self.phi_table)
        
        
        
#         if not self.latent:
#             self.z_dim = self.input_dim
#         else:
#             
#         
# #             self.x_kernel_encoder = nn.Sequential(
# #                 nn.Linear(self.input_dim, self.z_dim),
# # #                 nn.Dropout(p = self.dropout),
# # #                 nn.ReLU(),
# # #                 nn.Linear(self.e_dim, self.z_dim),
# # #                 nn.Dropout(p = self.dropout),
# # #                 nn.ReLU(),
# # #                 nn.Linear(self.e_dim, self.z_dim)
# #     #             nn.ReLU(),
# #     #             nn.Dropout(p = self.dropout)
# #                 ) 
#             if not self.lstm_latent:
#                 
#                 self.z_dim = config['z_dim']
#                 
#                 self.x_kernel_encoder = nn.Sequential(
#                     nn.Linear(self.input_dim, self.e_dim),
#                     nn.Dropout(p = self.dropout),
#                     nn.ReLU(),
#                     nn.Linear(self.e_dim, self.z_dim),
#     #                 nn.Dropout(p = self.dropout),
#     #                 nn.ReLU(),
#     #                 nn.Linear(self.e_dim, self.z_dim)
#         #             nn.ReLU(),
#         #             nn.Dropout(p = self.dropout)
#                     ) 
#                 
#                 self.x_kernel_decoder = nn.Sequential(
#                     nn.Linear(self.z_dim, self.e_dim),
#                     nn.Dropout(p = self.dropout),
#                     nn.ReLU(),
#     #                 nn.Linear(self.e_dim, self.e_dim),
#     #                 nn.Dropout(p = self.dropout),
#     #                 nn.ReLU(),
#                     nn.Linear(self.e_dim, self.input_dim)
#         #             nn.ReLU(),
#         #             nn.Dropout(p = self.dropout)
#                     )
#             else:
#                 self.z_dim = self.input_dim
#                 
#                 self.x_kernel_decoder = Encoder_cluster((1 + self.x_encoder.bidir)*self.s_dim, self.input_dim, self.dropout, self.device, bidir = False, block = self.block)
#         
#         if self.latent and self.lstm_latent:
#             self.phi_table = torch.zeros([(1+self.x_encoder.bidir)*self.s_dim, self.cluster_num], dtype = torch.float, device = config['device'])
#         else:
#             self.phi_table = torch.zeros([self.z_dim, self.cluster_num], dtype = torch.float, device = config['device']) 
#         self.init_phi_table()
         
        
#         else:
#             
# #             self.trans = torch.nn.LSTM(self.input_dim, self.h_dim, dropout = self.dropout)
# 
#             self.trans = Encoder_cluster(self.input_dim, self.h_dim, self.dropout, bidir = True, block = self.block)
# #             self.x_encoder = torch.nn.LSTM(self.input_dim, self.s_dim, bidirectional = True, dropout = self.dropout)
#             
#             self.x_encoder = Encoder_cluster(self.input_dim, self.s_dim, self.dropout, bidir = True, block = self.block)
            
        
        #nn.RNN(input_size=self.input_dim, hidden_size=self.rnn_dim, nonlinearity='relu', \
        #batch_first=True, bidirectional=False, num_layers=1)                   

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
#         self.z_0 = nn.Parameter(torch.zeros(self.z_dim))
#         self.z_q_0 = torch.ones(self.cluster_num)/self.cluster_num

        if self.transfer_prob:
            self.z_q_0 = torch.zeros(self.cluster_num, device = config['device'])
        else:
            self.z_q_0 = torch.zeros(self.z_dim, device = config['device'])
#         self.h_0 = nn.Parameter(torch.zeros(1, 1, self.rnn_dim))    
        
        self.optimizer = Adam(self.parameters(), lr=config['lr'], betas= (config['beta1'], config['beta2']))
#         self.optimizer = torch.optim.SGD(self.parameters(), lr=config['lr'])
    
    
#     def init_z_0(self):
    
    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, 
        mask = None, n_traj_samples = 1, run_backwards = False, mode = None):

        if isinstance(self.postnet, Encoder_z0_ODE_RNN_cluster) or \
            isinstance(self.postnet, Encoder_z0_RNN):

            truth_w_mask = truth
#             if mask is not None:
            if self.use_mask:
                truth_w_mask = torch.cat((truth, mask), -1)
            infer_probs, latent_y_states = self.postnet(
                truth_w_mask, truth_time_steps, run_backwards = run_backwards)

#             means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
#             sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
#             first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)

        else:
            raise Exception("Unknown encoder type {}".format(type(self.encoder_z0).__name__))
        
        if run_backwards:
            new_infer_probs = torch.flip(infer_probs, [1])
            
            infer_probs = new_infer_probs
            
            
            new_lat_y_states = torch.flip(latent_y_states, [1])

            latent_y_states = new_lat_y_states
#         first_point_enc_aug = torch.zeros([n_traj_samples, truth.shape[0], self.s_dim], device = self.device)
#         first_point_std = first_point_std.abs()
#         assert(torch.sum(first_point_std < 0) == 0.)

#         if self.use_poisson_proc:
#             n_traj_samples, n_traj, n_dims = first_point_enc.size()
#             # append a vector of zeros to compute the integral of lambda
#             zeros = torch.zeros([n_traj_samples, n_traj,self.input_dim]).to(get_device(truth))
#             first_point_enc_aug = torch.cat((first_point_enc, zeros), -1)
#             means_z0_aug = torch.cat((means_z0, zeros), -1)
#         else:
#             first_point_enc_aug = first_point_enc
#             means_z0_aug = means_z0
#             
#         assert(not torch.isnan(time_steps_to_predict).any())
#         assert(not torch.isnan(first_point_enc).any())
#         assert(not torch.isnan(first_point_enc_aug).any())

        # Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
        
#         minimum_step = self.postnet.minimum_step
#         
#         
#         n_intermediate_tp = max(2, ((t_i-prev_t) / minimum_step).int())
# 
#         time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
#         ode_sol = self.z0_diffeq_solver(prev_y_state, time_points)
        
#         print('latent y shape::', latent_y_states.shape)
        
        sol_y = self.diffeq_solver(latent_y_states[:,0], time_steps_to_predict)

#         if self.use_poisson_proc:
#             sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)
# 
#             assert(torch.sum(int_lambda[:,:,0,:]) == 0.)
#             assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)
#         print(sol_y.shape)
        gen_y_probs = F.softmax(self.gen_emitter_z(sol_y), -1)
#         pred_x = self.decoder(sol_y)
#         print(infer_probs.shape, gen_y_probs.shape)
        return torch.transpose(infer_probs.squeeze(0), 0, 1), gen_y_probs.squeeze(0), latent_y_states.squeeze(0)
    
    
    def get_reconstruction1(self, time_steps_to_predict, truth, truth_time_steps, 
        mask = None, n_traj_samples = 1, run_backwards = False, mode = None):

        if isinstance(self.postnet, Encoder_z0_ODE_RNN_cluster2) or \
            isinstance(self.postnet, Encoder_z0_ODE_RNN_cluster):

            truth_w_mask = truth
#             if mask is not None:
            if self.use_mask:
                truth_w_mask = torch.cat((truth, mask), -1)
            infer_probs, latent_y_states = self.postnet(
                truth_w_mask, truth_time_steps, run_backwards = run_backwards)

#             means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
#             sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
#             first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)

        else:
            raise Exception("Unknown encoder type {}".format(type(self.encoder_z0).__name__))
        
        if run_backwards:
            new_infer_probs = torch.flip(infer_probs, [1])
             
            infer_probs = new_infer_probs
             
             
            new_lat_y_states = torch.flip(latent_y_states, [1])
 
            latent_y_states = new_lat_y_states
#         first_point_enc_aug = torch.zeros([n_traj_samples, truth.shape[0], self.s_dim], device = self.device)
#         first_point_std = first_point_std.abs()
#         assert(torch.sum(first_point_std < 0) == 0.)

#         if self.use_poisson_proc:
#             n_traj_samples, n_traj, n_dims = first_point_enc.size()
#             # append a vector of zeros to compute the integral of lambda
#             zeros = torch.zeros([n_traj_samples, n_traj,self.input_dim]).to(get_device(truth))
#             first_point_enc_aug = torch.cat((first_point_enc, zeros), -1)
#             means_z0_aug = torch.cat((means_z0, zeros), -1)
#         else:
#             first_point_enc_aug = first_point_enc
#             means_z0_aug = means_z0
#             
#         assert(not torch.isnan(time_steps_to_predict).any())
#         assert(not torch.isnan(first_point_enc).any())
#         assert(not torch.isnan(first_point_enc_aug).any())

        # Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
        
#         minimum_step = self.postnet.minimum_step
#         
#         
#         n_intermediate_tp = max(2, ((t_i-prev_t) / minimum_step).int())
# 
#         time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
#         ode_sol = self.z0_diffeq_solver(prev_y_state, time_points)
        
#         print('latent y shape::', latent_y_states.shape)
        
#         h_0 = self.prob_to_states(infer_probs[0,0])
#         
#         h_0 = h_0.unsqueeze(0)
        
        
        sol_y = self.diffeq_solver(latent_y_states[:,0], time_steps_to_predict)

#         if self.use_poisson_proc:
#             sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)
# 
#             assert(torch.sum(int_lambda[:,:,0,:]) == 0.)
#             assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)
#         print(sol_y.shape)
        gen_y_probs = F.softmax(self.gen_emitter_z(sol_y), -1)
#         pred_x = self.decoder(sol_y)
#         print(infer_probs.shape, gen_y_probs.shape)
        return torch.transpose(infer_probs.squeeze(0), 0, 1), gen_y_probs.squeeze(0), latent_y_states.squeeze(0)
    
    
    def get_reconstruction2(self, time_steps_to_predict, truth, truth_time_steps, 
        mask = None, n_traj_samples = 1, run_backwards = False, mode = None):

        if isinstance(self.postnet, Encoder_z0_ODE_RNN_cluster) or \
            isinstance(self.postnet, Encoder_z0_ODE_RNN_cluster2):

            truth_w_mask = truth
            if self.use_mask:
                truth_w_mask = torch.cat((truth, mask), -1)
#             if mask is not None:
#                 truth_w_mask = torch.cat((truth, mask), -1)
            if torch.isnan(truth_w_mask).any():
                print('here')
            if torch.isnan(truth_time_steps).any():
                print('here')

            infer_probs, latent_y_states,_ = self.postnet.run_odernn(
                truth_w_mask, truth_time_steps, run_backwards = run_backwards)

#             infer_probs2, latent_y_states2,_ = self.postnet.run_odernn(
#                 truth_w_mask, truth_time_steps, run_backwards = run_backwards, exp_y_states = latent_y_states)
# 
#             print(torch.norm(latent_y_states - latent_y_states2))


             
#             print(torch.norm(next_latent_y_states - latent_y_states[:,0]))
            
#             means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
#             sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
#             first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)

        else:
            raise Exception("Unknown encoder type {}".format(type(self.encoder_z0).__name__))
        
        
        '''return lenth: T_max - 1'''
        gen_y_probs, gen_latent_y_states = self.trans(torch.transpose(infer_probs.squeeze(0), 1, 0), truth_time_steps, run_backwards = run_backwards)
#         gen_y_probs, gen_latent_y_states = self.trans(torch.transpose(infer_probs.squeeze(0), 1, 0), time_steps_to_predict, run_backwards = run_backwards)
        extra_kl_div = 0
        
            
        squeezed_infer_probs = torch.transpose(infer_probs.squeeze(0), 1, 0)
        
        
        selected_time_count = truth_time_steps.shape[0] -  self.shift
        
#         selected_time_steps = torch.tensor(list(range(selected_time_count))) + 1
        
#         predict_selcted_time_steps = selected_time_steps + self.shift
#         shifted_count = int(truth_time_steps.shape[0]/self.shift)
        
#         selected_time_steps = self.shift*torch.tensor()
        
#         shifted_infer_y_probs = squeezed_infer_probs[:,selected_time_steps]
#         
#         shifted_infer_y_states = latent_y_states[0][:, selected_time_steps]
        
#         shifted_time_steps_to_predict = time_steps_to_predict[selected_time_steps]
#         squeezed_gen_states = gen_latent_y_states.squeeze(0)
        count = 0
        
        for k in range(selected_time_count - 1):
            time_steps = torch.tensor([truth_time_steps[k].item(), truth_time_steps[k + self.shift + 1].item()])
            
            if k == 0:
                last_gen_probs, last_gen_states = self.trans.run_odernn_single_step(infer_probs[:,k].clone(), time_steps, prev_y_state = torch.zeros_like(gen_latent_y_states[:,0]))
            else:
                last_gen_probs, last_gen_states = self.trans.run_odernn_single_step(infer_probs[:,k].clone(), time_steps, prev_y_state = gen_latent_y_states[:,k-1].clone())
        
#             print(torch.norm(last_gen_probs - gen_y_probs[0,k]))
        
            extra_kl_div += self.kl_div(squeezed_infer_probs[:, k+self.shift+1].clone(), last_gen_probs.squeeze(0))
            
            count += 1
#         shifted_gen_y_probs, shifted_gen_latent_y_states = self.trans(shifted_infer_y_probs, shifted_time_steps_to_predict, run_backwards = run_backwards)
#         
#         squeezed_gen_probs = shifted_gen_y_probs.squeeze(0) 
#         
#         count = 0
#         
#         for k in range(selected_time_steps.shape[0] - 1):
# #                 if k*shifted_count + 1 < squeezed_infer_probs.shape[1]:
#             extra_kl_div += self.kl_div(shifted_infer_y_probs[:, k+1], squeezed_gen_probs[k])
#             count += 1
        
        extra_kl_div = torch.sum(extra_kl_div)/(count*truth.shape[0])
#             shifted_infer_y_states = latent_y_states.squeeze(0)[selected_time_steps]
            
            
            
            
#         first_point_enc_aug = torch.zeros([n_traj_samples, truth.shape[0], self.s_dim], device = self.device)
#         first_point_std = first_point_std.abs()
#         assert(torch.sum(first_point_std < 0) == 0.)

#         if self.use_poisson_proc:
#             n_traj_samples, n_traj, n_dims = first_point_enc.size()
#             # append a vector of zeros to compute the integral of lambda
#             zeros = torch.zeros([n_traj_samples, n_traj,self.input_dim]).to(get_device(truth))
#             first_point_enc_aug = torch.cat((first_point_enc, zeros), -1)
#             means_z0_aug = torch.cat((means_z0, zeros), -1)
#         else:
#             first_point_enc_aug = first_point_enc
#             means_z0_aug = means_z0
#             
#         assert(not torch.isnan(time_steps_to_predict).any())
#         assert(not torch.isnan(first_point_enc).any())
#         assert(not torch.isnan(first_point_enc_aug).any())

        # Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
#         sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)

#         if self.use_poisson_proc:
#             sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)
# 
#             assert(torch.sum(int_lambda[:,:,0,:]) == 0.)
#             assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)
#         print(sol_y.shape)
#         gen_y_probs = F.softmax(self.gen_emitter_z(sol_y), -1)
#         pred_x = self.decoder(sol_y)
        print(infer_probs.shape, gen_y_probs.shape)
        return torch.transpose(infer_probs.squeeze(0), 0, 1), torch.transpose(gen_y_probs.squeeze(0), 0, 1), latent_y_states.squeeze(0), gen_latent_y_states.squeeze(0), extra_kl_div
    
    
    def init_phi_table(self, init_values, is_tensor):
        
        
        
        
        if not is_tensor:
            self.phi_table.data.copy_(torch.t(torch.from_numpy(init_values)))
        else:
            self.phi_table.data.copy_(init_values)
         
        torch.save(self.phi_table, data_folder + '/' + output_dir + 'init_phi_table')
#         dataset = self.
#          
#          
#         init_kmeans = initial_kmeans(dataset, mask, impute, cluster_num)
#          
#          
#          
#         m = normal.Normal(torch.zeros(self.z_dim, device = self.device), 2*torch.ones(self.z_dim, device = self.device)*self.phi_std)
#         for i in range(self.phi_table.shape[1]):
#             x = m.sample()
#              
#             self.phi_table[:,i] = x 
#          
#         return x
    
    
    def generate_z(self, z_category, prior_cluster_probs, t, h_now=None):
        
        if self.use_gate:
            curr_gaussian_coeff = (self.gaussian_prior_coeff + F.sigmoid(self.gate_func(h_now)))/2
            
#             curr_gaussian_coeff = curr_gaussian_coeff.unsqueeze(0)
        else:
            curr_gaussian_coeff = self.gaussian_prior_coeff
        
#         z_catergory = self.emitter_z(h_t)
        
#         z_representation = F.gumbel_softmax(z_category.view(z_category.shape[1], z_category.shape[2]), tau=0.1)

        z_category = (1 - curr_gaussian_coeff)*z_category + curr_gaussian_coeff*prior_cluster_probs
        
#         print(z_category.shape)
        
        z_representation = z_category#.view(z_category.shape[1], z_category.shape[2])
        
#         if self.use_gumbel:
#         
#             samples = torch.multinomial(z_representation, 1)
#             
#             phi_z = torch.t(self.phi_table)[samples.view(-1)]
#             
#         else:
        phi_z = torch.t(torch.mm(self.phi_table, torch.t(z_representation)))
        
#         if t > self.t_thres:
# 
#             if not self.use_gumbel:
#                 z_representation = z_category.view(z_category.shape[1], z_category.shape[2])
#             else:
#                  
#     #             device = z_category.device
# #                 print(t, 'generate here')
#                 averaged_z_rep = 0
#                  
#                 log_prob = Variable(torch.log(z_category.view(z_category.shape[1], z_category.shape[2])))
#                  
#                 for k in range(self.sample_times):
#                     curr_z_rep = F.gumbel_softmax(log_prob, tau=0.5)
# #                     curr_z_rep = sparsemax(log_prob)
#                      
#                     averaged_z_rep += curr_z_rep
#                      
#                     del curr_z_rep
#                      
#     #             averaged_z_rep = averaged_z_rep.to(device)
#                              
#                 z_representation = averaged_z_rep/self.sample_times
#                 
#         else:
#             z_representation = z_category.view(z_category.shape[1], z_category.shape[2])
        
#         phi_z = torch.t(torch.mm(self.phi_table, torch.t(z_representation)))
        
        return phi_z, z_representation
    
    
    def compute_cluster_obj(self, all_distances, all_probabs, T_max, x_lens, x_dim):
        '''all_probabs: 1*cluster_num'''
        
        
        all_probabs_copy = all_probabs.view(1, 1, all_probabs.shape[0])
        
        all_probabs_copy = all_probabs_copy.repeat(all_distances.shape[0], T_max, 1)
        
        '''batch_size*T_max*cluster_num'''
        
        
        cluster_obj = torch.sum(all_distances**2*all_probabs_copy/((self.x_std**2)*(torch.sum(x_lens)*all_probabs.shape[0]*x_dim))) + 2*np.log(self.x_std) + 2*np.log(np.sqrt(2*np.pi))
        
        
#         cluster_obj = torch.log(torch.sum(-torch.exp(-all_distances**2/self.phi_std)*all_probabs_copy))  + 2*torch.log(self.x_std) + 2*np.log(np.sqrt(2*np.pi))
        
        return torch.sum(all_distances**2*all_probabs_copy/((self.x_std**2)*(torch.sum(x_lens)*all_probabs.shape[0]*x_dim))), cluster_obj
    
    
    def compute_cluster_obj_full(self, all_distances, all_probabs, T_max, x_masks, x_lens):
        
        '''all_distances:: T_max, self.cluster_num, batch_size, input_dim'''
        
        '''all_probabs: 1*cluster_num'''
        
        
        all_probabs_copy = all_probabs.view(1, all_probabs.shape[0], 1, 1)
        
        all_probabs_copy = all_probabs_copy.repeat(T_max, 1, all_distances.shape[2], all_distances.shape[3])
        
        
        all_probabs_copy2 = all_probabs.view(1, all_probabs.shape[0], 1)
        
        all_probabs_copy2 = all_probabs_copy2.repeat(T_max, 1, all_distances.shape[2])
#         all_masks = x_masks.view(x_masks.shape[0], 1, x_masks.shape[1], x_masks.shape[2])
        
        all_masks = torch.transpose(x_masks, 0, 1)
        
        all_masks = all_masks.view(all_masks.shape[0],1, all_masks.shape[1], all_masks.shape[2])
        
        all_masks_copy = all_masks.repeat(1, all_probabs.shape[0], 1, 1)
        
        all_masks_copy[all_distances == 0] = 1
        
        '''batch_size*T_max*cluster_num'''
        masked_distance = torch.sum(all_distances*all_masks_copy, 3)/(torch.sum(all_masks_copy, 3)*(self.x_std**2))*all_probabs_copy2
        
#         masked_distance[masked_distance != masked_distance] = 0
        
        agg_masked_distance = torch.sum(masked_distance)/torch.sum(x_lens*all_distances.shape[1])
        
        cluster_obj = torch.sum((all_distances*all_masks_copy/(self.x_std**2)*all_probabs_copy/torch.sum(all_masks_copy))) + 2*np.log(self.x_std) + 2*np.log(np.sqrt(2*np.pi))
        
        
#         cluster_obj = torch.log(torch.sum(-torch.exp(-all_distances**2/self.phi_std)*all_probabs_copy))  + 2*torch.log(self.x_std) + 2*np.log(np.sqrt(2*np.pi))
        
        return agg_masked_distance, agg_masked_distance + 2*np.log(self.x_std) + 2*np.log(np.sqrt(2*np.pi))
    
    
    
    def compute_cluster_obj_full2(self, all_distances, all_probabs, T_max, x_masks, x_lens):
        
        '''all_distances:: T_max, self.cluster_num, batch_size, input_dim'''
        
        '''all_probabs: 1*cluster_num'''
        
        
        all_probabs_copy = all_probabs.view(1, all_probabs.shape[0], 1, 1)
        
        all_probabs_copy = all_probabs_copy.repeat(T_max, 1, all_distances.shape[2], all_distances.shape[3])
        
        
        all_probabs_copy2 = all_probabs.view(1, all_probabs.shape[0], 1)
        
        all_probabs_copy2 = all_probabs_copy2.repeat(T_max, 1, all_distances.shape[2])
#         all_masks = x_masks.view(x_masks.shape[0], 1, x_masks.shape[1], x_masks.shape[2])
        
        all_masks = torch.transpose(x_masks, 0, 1)
        
        all_masks = all_masks.view(all_masks.shape[0],1, all_masks.shape[1], all_masks.shape[2])
        
        all_masks_copy = all_masks.repeat(1, all_probabs.shape[0], 1, 1)
#         
#         all_masks_copy[all_distances == 0] = 1
#         
#         '''batch_size*T_max*cluster_num'''
#         masked_distance = torch.sum(all_distances*all_masks_copy, 3)/(torch.sum(all_masks_copy, 3)*(self.x_std**2))*all_probabs_copy2
#         
# #         masked_distance[masked_distance != masked_distance] = 0
#         
#         agg_masked_distance = torch.sum(masked_distance)/torch.sum(x_lens*all_distances.shape[1])
        
#         cluster_obj = torch.sum((all_distances*all_masks_copy/(self.x_std**2)*all_probabs_copy/torch.sum(all_masks_copy))) + 2*np.log(self.x_std) + 2*np.log(np.sqrt(2*np.pi))
        
        cluster_obj = 0.5*(torch.sum((torch.sum(all_distances*all_masks_copy*all_probabs_copy, 1)/(self.x_std**2)/torch.sum(all_masks))) + 2*np.log(self.x_std) + 2*np.log(np.sqrt(2*np.pi)))
#         cluster_obj = torch.log(torch.sum(-torch.exp(-all_distances**2/self.phi_std)*all_probabs_copy))  + 2*torch.log(self.x_std) + 2*np.log(np.sqrt(2*np.pi))
        
        return 0.5*torch.sum((all_distances*all_masks_copy/(self.x_std**2)*all_probabs_copy/torch.sum(all_masks_copy))), cluster_obj

    
    
    def compute_cluster_obj_full2_2(self, cluster_losses, curr_full_rec_loss, all_distances, all_probabs, T_max, x_masks, x_lens):
        
        '''all_distances:: T_max, self.cluster_num, batch_size, input_dim'''
        
        '''all_probabs: 1*cluster_num'''
        
        
        all_probabs_copy = all_probabs.view(1, all_probabs.shape[0], 1, 1)
        
        all_probabs_copy = all_probabs_copy.repeat(T_max, 1, all_distances.shape[2], all_distances.shape[3])
        
        
        all_probabs_copy2 = all_probabs.view(1, all_probabs.shape[0], 1)
        
        all_probabs_copy2 = all_probabs_copy2.repeat(T_max, 1, all_distances.shape[2])
#         all_masks = x_masks.view(x_masks.shape[0], 1, x_masks.shape[1], x_masks.shape[2])
        
        all_masks = torch.transpose(x_masks, 0, 1)
        
        all_masks = all_masks.view(all_masks.shape[0],1, all_masks.shape[1], all_masks.shape[2])
        
        all_masks_copy = all_masks.repeat(1, all_probabs.shape[0], 1, 1)
#         
#         all_masks_copy[all_distances == 0] = 1
#         
#         '''batch_size*T_max*cluster_num'''
#         masked_distance = torch.sum(all_distances*all_masks_copy, 3)/(torch.sum(all_masks_copy, 3)*(self.x_std**2))*all_probabs_copy2
#         
# #         masked_distance[masked_distance != masked_distance] = 0
#         
#         agg_masked_distance = torch.sum(masked_distance)/torch.sum(x_lens*all_distances.shape[1])
        
#         cluster_obj = torch.sum((all_distances*all_masks_copy/(self.x_std**2)*all_probabs_copy/torch.sum(all_masks_copy))) + 2*np.log(self.x_std) + 2*np.log(np.sqrt(2*np.pi))
        
        cluster_obj = 0.5*(torch.sum((torch.sum(all_distances*all_masks_copy*all_probabs_copy, 1)/(self.x_std**2)/torch.sum(all_masks))) + 2*np.log(self.x_std) + 2*np.log(np.sqrt(2*np.pi)))
#         cluster_obj = torch.log(torch.sum(-torch.exp(-all_distances**2/self.phi_std)*all_probabs_copy))  + 2*torch.log(self.x_std) + 2*np.log(np.sqrt(2*np.pi))
        inter_res = (torch.sum(all_distances*all_masks_copy*all_probabs_copy, 1)/(self.x_std**2)*0.5 + 2*np.log(self.x_std) + 2*np.log(np.sqrt(2*np.pi)))*all_masks.squeeze(1)
        
        
        return 0.5*torch.sum((all_distances*all_masks_copy/(self.x_std**2)*all_probabs_copy/torch.sum(all_masks_copy))), cluster_obj
    
    def compute_distance_per_cluster(self, x_t):
        
        '''cluster_num*batch_size*dim'''
        
        curr_x_t = x_t.repeat(self.cluster_num, 1, 1)
        
#         curr_x_mask = x_mask.repeat(self.cluster_num, 1, 1)
        
        phi_table_transpose = torch.t(self.phi_table)
        
        phi_table_transpose = phi_table_transpose.reshape(phi_table_transpose.shape[0], 1, phi_table_transpose.shape[1])
        
        phi_table_transpose = phi_table_transpose.repeat(1,x_t.shape[0], 1)
        
        '''cluster_num*batch_size*dim'''
#         all_distances = (curr_x_t*curr_x_mask - phi_table_transpose*curr_x_mask)**2
        
        all_distances = torch.norm(curr_x_t - phi_table_transpose, dim=2)
        return torch.t(all_distances)
#         '''batch_size * cluster_num'''
#         return torch.t(all_distances)    
    def compute_distance_per_cluster_all(self, x_t, x_mask):
        
        '''cluster_num*batch_size*dim'''
        
        curr_x_t = x_t.repeat(self.cluster_num, 1, 1)
        
        curr_x_mask = x_mask.repeat(self.cluster_num, 1, 1)
        
        phi_table_transpose = torch.t(self.phi_table)
        
        phi_table_transpose = phi_table_transpose.reshape(phi_table_transpose.shape[0], 1, phi_table_transpose.shape[1])
        
        phi_table_transpose = phi_table_transpose.repeat(1,x_t.shape[0], 1)
        
        '''cluster_num*batch_size*dim'''
        all_distances = (curr_x_t*curr_x_mask - phi_table_transpose*curr_x_mask)**2
        
#         all_distances = torch.norm(curr_x_t - phi_table_transpose, dim=2)
        return all_distances
#         '''batch_size * cluster_num'''
#         return torch.t(all_distances)
        
        
    
    def generate_x(self, phi_z, z_t):
#         mean = self.emitter_x(phi_z)
#         print(phi_z.shape)
#         
#         print(self.emiter_x_mean)
        
        mean = phi_z
        
#         mean = mean_var[:,:self.input_dim]
#         
#         logvar = mean_var[:,self.input_dim:2*self.input_dim]
#         
#         std = F.softplus(logvar)
        '''0.1 for periodic'''
#         logvar = self.emiter_x_var(phi_z)
#          
#         std = torch.exp(0.5*logvar)
#         std = torch.clamp(torch.exp(self.emiter_x_var(phi_z)), max = self.x_std)
#         if self.latent:
#             std = self.latent_x_std*torch.ones_like(mean, device = self.device)
#         else:
        std = self.x_std*torch.ones_like(mean, device = self.device)
        
        logvar = 2*torch.log(std)
        
        
        epsilon = torch.randn(mean.shape, device=phi_z.device) # sampling z by re-parameterization
        x = epsilon * std + mean
        
#         m = normal.Normal(mean, std)
#         x = m.sample()
        
        return mean, logvar, mean
        
           
#     def kl_div(self, mu1, logvar1, mu2=None, logvar2=None):
#         one = torch.ones(1, device=mu1.device)
#         if mu2 is None: mu2=torch.zeros(1, device=mu1.device)
#         if logvar2 is None: logvar2=torch.zeros(1, device=mu1.device)
#         return torch.sum(0.5*(logvar2-logvar1+(torch.exp(logvar1)+(mu1-mu2).pow(2))/torch.exp(logvar2)-one), 1)             
    
    def kl_div(self, cat_1, cat_2):
        epsilon = 1e-5*torch.ones_like(cat_1)
        kl_div = torch.sum((cat_1+epsilon)*torch.log((cat_1 + epsilon)/(cat_2+epsilon)), 1)
        
        return kl_div
    
    
    def entropy(self, cat):
        epsilon = 1e-5*torch.ones_like(cat)
        kl_div = -torch.sum(cat*torch.log(cat+epsilon), 1)
        
        return kl_div
    


#     def infer2(self, x, T_max, x_to_predict, tp_to_predict):
#         """
#         infer q(z_{1:T}|x_{1:T}) (i.e. the variational distribution)
#         """
#         batch_size, _, input_dim = x.size()
# #         T_max = x_lens.max()
#         h_0 = self.h_0.expand(1, batch_size, self.s_dim).contiguous()
#         
# #         c_0 = self.c_0.expand(1, batch_size, self.s_dim).contiguous()
#         
#         
#         h_prev = self.h_0.expand(1, batch_size, self.h_0.size(0))
#         
#         c_prev = self.h_0.expand(1, batch_size, self.h_0.size(0))
#         
#         
#         z_1_category = torch.ones(self.cluster_num, dtype = torch.float, device = self.device)/self.cluster_num
#         
#         z_t_category_gen = z_1_category.expand(1, batch_size, z_1_category.size(0))
#         
#         phi_z, z_representation = self.generate_z(z_t_category_gen)
#         
#         
#         rnn_out,_= self.x_encoder(x) # push the observed x's through the rnn;
# #         rnn_out = reverse_sequence(rnn_out, x_lens) # reverse the time-ordering in the hidden state and un-pack it
#         rec_losses = torch.zeros((batch_size, T_max), device=x.device) 
#         kl_states = torch.zeros((batch_size, T_max), device=x.device)  
#         
#         l2_norm = torch.zeros((batch_size, T_max), device=x.device)
#         
#         z_t_category_infer = self.z_q_0.expand(batch_size,1, self.z_q_0.size(0)) # set z_prev=z_q_0 to setup the recursive conditioning in q(z_t|...)
#         
#         z_prev = F.gumbel_softmax(z_t_category_infer, tau=0.1)
#         
#         
#         for t in range(T_max):
# #             z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)# p(z_t| z_{t-1})
# 
# 
#             
#             kl_div = torch.sum(self.kl_div(torch.transpose(z_t_category_infer, 0, 1), z_t_category_gen),0)
#             
#             if np.isnan(kl_div.detach().numpy()).any():
#                 print('distribution 1::', z_t_category_gen)
#                 
#                 print('distribution 2::', z_t_category_infer)
#             
#             
#             
#             z_t, z_t_category_infer, phi_z_infer = self.postnet(z_prev, rnn_out[:,t,:], self.phi_table) #q(z_t | z_{t-1}, x_{t:T})
#             
#             kl_states[:,t] = kl_div
#             
#             
#             z_prev = z_t   
#             
#             if self.block == 'GRU':
#                 output, h_now = self.trans(phi_z_infer.view(1, phi_z_infer.shape[0], phi_z_infer.shape[2]), h_prev)# p(z_t| z_{t-1})
#             else:
#                 output, (h_now, c_now) = self.trans(phi_z_infer.view(1, phi_z_infer.shape[0], phi_z_infer.shape[2]), (h_prev, c_prev))# p(z_t| z_{t-1})
#             
#             z_t_category_gen = F.softmax(self.emitter_z(h_now), dim = 2)
#             
#             h_prev = h_now
#             
#             c_prev = c_now
#             
# #             if t < T_max - 1:            
#             
#             phi_z, z_representation = self.generate_z(z_t_category_gen)
#              
#             mean, std, logit_x_t = self.generate_x(phi_z)
# 
# #                 rec_loss = torch.norm(x[:,t+1,:] - mean)**2/(2*std**2) + torch.log(2*np.pi*std**2)/2
#             
#             rec_loss = torch.bmm(((x[:,t,:]-mean)/(std**2)).view(mean.shape[0],1,mean.shape[1]), (x[:,t,:]-mean).view(mean.shape[0],mean.shape[1],1)).view(-1) + (torch.log(2*np.pi*std**2)/2).view(-1) 
#             
#             curr_l2_norm = (x[:,t,:]-mean)**2
#             
# #             kl = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
# #             kl_states[:,t] = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
# #             logit_x_t = self.emitter(z_t).contiguous() # p(x_t|z_t)         
# #             rec_loss = nn.BCEWithLogitsLoss(reduction='none')(logit_x_t.view(-1), x[:,t,:].contiguous().view(-1)).view(batch_size, -1)
#             rec_losses[:,t] = rec_loss             
# 
#             l2_norm[:, t] = curr_l2_norm.view(-1)
# #         x_mask = sequence_mask(x_lens)
# #         x_mask = x_mask.gt(0).view(-1)
# #         rec_loss = rec_losses.view(-1).masked_select(x_mask).mean()
# #         kl_loss = kl_states.view(-1).masked_select(x_mask).mean()
# 
#         rec_loss = rec_losses.view(-1).mean()
#         kl_loss = kl_states.view(-1).mean()
#         
#         final_l2_norm = l2_norm.view(-1).mean()
#         
#         self.evaluate_forecasting_errors(x_to_predict, batch_size, tp_to_predict.shape[0], h_now, c_now)
#         
#           
# 
#         print('loss::', rec_loss, kl_loss, final_l2_norm)
# 
#         return rec_loss, kl_loss, h_now, c_now


    
    def compute_reconstruction_loss(self, x, mean, std, batch_size, mask):
        gaussian = Independent(Normal(loc = mean*mask, scale = std), 1)
        log_prob = gaussian.log_prob(x*mask) 
        log_prob = log_prob / batch_size
        
        return log_prob*(-1) 
    
    
    def compute_reconstruction_loss2(self, x, mean, std, batch_size):
        gaussian = Independent(Normal(loc = mean, scale = std), 1)
        log_prob = gaussian.log_prob(x) 
#         log_prob = log_prob / batch_size
        
        return log_prob*(-1) 
    
#     def compute_gaussian_probs(self, x, mean, std):
#         
#         
# #         gaussian = Independent(Normal(loc = mean[0][0], scale = std[0][0]), 0)
# #         log_prob = gaussian.log_prob(x[0][0])
#         
#         prob = -0.5*((x - mean)/std)**2 - torch.log((std*np.sqrt(2*np.pi)))
#         
# #         print(torch.norm(prob[0][0] - log_prob))
#         
#         return -prob
    def compute_gaussian_probs0(self, x, mean, logvar, mask):
    
    
#         gaussian = Independent(Normal(loc = mean[0][0], scale = std[0][0]), 0)
#         log_prob = gaussian.log_prob(x[0][0])
    
        std = torch.exp(0.5 * logvar)
        
        prob = 0.5*(((x - mean)/std)**2 + logvar + 2*np.log(np.sqrt(2*np.pi)))# + torch.log((std*np.sqrt(2*np.pi)))
        
    #         print(torch.norm(prob[0][0] - log_prob))
        
        return prob*mask, (x - mean)**2*mask
    
    def compute_rec_loss(self, joint_probs, prob_sums, full_curr_rnn_input, x_t, x_t_mask, h_now = None, curr_z_t_category_infer = None):

        phi_table_extend = (torch.t(self.phi_table)).clone()
        
        phi_table_extend = phi_table_extend.view(1, self.phi_table.shape[1], self.phi_table.shape[0])
        
        phi_table_extend = phi_table_extend.repeat(self.cluster_num, 1, 1) 
        
        phi_z_infer_full = torch.bmm(full_curr_rnn_input, phi_table_extend)
        mean_full, logvar_full, logit_x_t_full = self.generate_x(phi_z_infer_full, None)
        
        
        
        
        
#         if self.loss_on_missing:
#             
# #             if self.latent and self.lstm_latent:
# #                 x_t_full = rnn_out.view(1, rnn_out.shape[0], rnn_out.shape[1])
# #                 
# #                 x_t_full = x_t_full.repeat(self.cluster_num, 1, 1)
# #                 
# #                 curr_x_t_masks = torch.ones_like(rnn_out)
# #             else:
#             x_t_full = x_t.view(1, x_t.shape[0], x_t.shape[1])
#             
#             x_t_full = x_t_full.repeat(self.cluster_num, 1, 1)
#             
#             curr_x_t_masks = torch.ones_like(x_t)
#             
#             curr_full_rec_loss, curr_distances = self.compute_gaussian_probs0(x_t_full, mean_full, logvar_full, curr_x_t_masks)
            
            
#             curr_full_rec_loss2 = (x_t_full*curr_x_t_masks - mean_full*curr_x_t_masks)**2
            
#             print(curr_full_rec_loss2  -)
#         else:
            
#             if not self.lstm_latent:
        x_t_full = x_t.view(1, x_t.shape[0], x_t.shape[1])
        
        x_t_full = x_t_full.repeat(self.cluster_num, 1, 1)
        
#             if not self.latent:
        x_t_mask_full = x_t_mask.view(1, x_t_mask.shape[0], x_t_mask.shape[1])
        
        x_t_mask_full = x_t_mask_full.repeat(self.cluster_num, 1, 1)
            
#             else:
#                 x_t_mask_full = torch.ones_like(x_t_full)
#             else:
#                 
# #                 if self.latent:
# #                     x_t_full = rnn_out.view(1, rnn_out.shape[0], rnn_out.shape[1])
# #                     
# #                     x_t_full = x_t_full.repeat(self.cluster_num, 1, 1)
# #                     
# #                     x_t_mask_full = torch.ones_like(x_t_full)
# #                 else:
#                 x_t_full = x_t.view(1, x_t.shape[0], x_t.shape[1])
#             
#                 x_t_full = x_t_full.repeat(self.cluster_num, 1, 1)
#                 
#                 x_t_mask_full = x_t_mask.view(1, x_t_mask.shape[0], x_t_mask.shape[1])
#                 
#                 x_t_mask_full = x_t_mask_full.repeat(self.cluster_num, 1, 1)
#                 x_t_mask_full = x_t_mask_full.repeat(self.cluster_num, 1, 1)
            
                        
        curr_full_rec_loss, curr_distances = self.compute_gaussian_probs0(x_t_full, mean_full, logvar_full, x_t_mask_full)
            
#             curr_full_rec_loss2 = (x_t_full*x_t_mask_full - mean_full*x_t_mask_full)**2
        
        
        
#         print(curr_full_rec_loss.shape, joint_probs.shape)
        
        
        
#         t1 = time.time()
        
        if self.use_gate:
            curr_gaussian_coeff = (self.gaussian_prior_coeff + F.sigmoid(self.gate_func(h_now)))/2
        else:
            curr_gaussian_coeff = self.gaussian_prior_coeff
        
        full_rec_loss1 = torch.sum(curr_full_rec_loss*(1-curr_gaussian_coeff)*torch.t(joint_probs).unsqueeze(-1), 0)
        
        full_rec_loss2 = torch.sum(curr_full_rec_loss*(curr_gaussian_coeff)*prob_sums.view(prob_sums.shape[0], 1, 1), 0)
        
        l2_norm_loss = full_rec_loss1/(1-curr_gaussian_coeff)
        
        cluster_loss = full_rec_loss2/curr_gaussian_coeff
        

#         t2 = time.time()
#         
#         if not self.use_gate:
#             for k in range(self.cluster_num):
#                 full_rec_loss1 += curr_full_rec_loss[k]*((1-self.gaussian_prior_coeff)*joint_probs[:,k].view(curr_full_rec_loss[k].shape[0],1))
#                 
#                 full_rec_loss2 += curr_full_rec_loss[k]*((self.gaussian_prior_coeff)*prob_sums[k])
#         
#         else:
#             
#             curr_gaussian_coeff = F.sigmoid(self.gate_func(h_now))
#             
#             for k in range(self.cluster_num):
#                 full_rec_loss1 += curr_full_rec_loss[k]*((1-curr_gaussian_coeff)*joint_probs[:,k].view(curr_full_rec_loss[k].shape[0],1))
#                 
#                 full_rec_loss2 += curr_full_rec_loss[k]*((curr_gaussian_coeff)*prob_sums[k])
#         
#         t3 = time.time()
        
#         print(torch.norm(full_rec_loss1_2 - full_rec_loss1))
#         
#         print(torch.norm(full_rec_loss2_2 - full_rec_loss2))
#         
#         print('time 1::', t2 - t1)
#         
#         print('time 2::', t3 - t2)
        
#         full_logit_x_t = torch.mm(joint_probs, torch.t(self.phi_table))
        full_logit_x_t = torch.mm((1-curr_gaussian_coeff)*curr_z_t_category_infer + curr_gaussian_coeff*prob_sums.view(1,-1), torch.t(self.phi_table))
        
#         full_rec_loss1_2 = torch.sum(curr_full_rec_loss*(1-curr_gaussian_coeff))
#         
#         full_rec_loss1_2.backward()
#         
#         sum_loss = torch.sum(l2_norm_loss)
#         
#         sum_loss.backward()
        
        return full_rec_loss1, full_rec_loss2, full_logit_x_t, curr_full_rec_loss, l2_norm_loss, cluster_loss
    '''x, origin_x, x_mask, T_max, x_to_predict, origin_x_to_pred, x_to_predict_mask, tp_to_predict, is_GPU, device'''        
    
    def update_joint_probability(self, joint_probs, curr_rnn_output, batch_size, t, h_prev, c_prev, z_t_category_infer, shrinked_x_lens, x_t, x_t_mask):
        
        '''batch_size*cluster'''
        
#         z_t_category_gen = F.softmax(self.emitter_z(h_prev), dim = 2)

#         if not self.use_sparsemax:
        z_t_category_gen = F.softmax(self.emitter_z(h_prev), dim = 2)
#         else:
#             
#             logit_z_t = self.emitter_z(h_prev)
#             
#             z_t_category_gen = sparsemax(logit_z_t.view(logit_z_t.shape[1], logit_z_t.shape[2]))
#             
#             z_t_category_gen = z_t_category_gen.view(1, z_t_category_gen.shape[0], z_t_category_gen.shape[1])

        
        full_kl = self.kl_div(z_t_category_infer, z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))
        
        
#         updated_joint_probs = torch.zeros_like(joint_probs)
        
#         full_kl = 0
        
        full_rec_loss = 0
        
        full_logit_x_t = 0
        
        
#         curr_full_rec_loss = torch.zeros([self.cluster_num, batch_size, self.input_dim], device = self.device)
        
#         logit_x
        full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
        
        for k in range(self.cluster_num):
            curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            curr_rnn_input[:,k] = 1
            
            full_curr_rnn_input[k] = curr_rnn_input
        
        t1 = time.time()
        
        phi_table_extend = (torch.t(self.phi_table)).clone()
        
        phi_table_extend = phi_table_extend.view(1, self.cluster_num, self.z_dim)
        
        phi_table_extend = phi_table_extend.repeat(self.cluster_num, 1, 1) 
            
        phi_z_infer_full = torch.bmm(full_curr_rnn_input, phi_table_extend)
        
        
        curr_rnn_output_full = curr_rnn_output.view(1, curr_rnn_output.shape[0], curr_rnn_output.shape[1])
        
        curr_rnn_output_full = curr_rnn_output_full.repeat(self.cluster_num, 1, 1)
        
#         print(full_curr_rnn_input.shape, curr_rnn_output_full.shape)
            
        z_t, z_t_category_infer_full, _, z_category_infer_sparse = self.postnet(full_curr_rnn_input, curr_rnn_output_full, self.phi_table, t, self.temp)
        
        mean_full, logvar_full, logit_x_t_full = self.generate_x(phi_z_infer_full, None)
        
        
        
        
#         updated_joint_probs2 = torch.bmm(torch.transpose(z_t_category_infer_full, 1, 0), (joint_probs.view(x_t.shape[0], self.cluster_num, 1)))
        
        updated_joint_probs = torch.sum(z_t_category_infer_full*torch.t(joint_probs).view(joint_probs.shape[1], joint_probs.shape[0], 1), 0)
        
#         if self.loss_on_missing:
#                     
#             x_t_full = x_t.view(1, x_t.shape[0], x_t.shape[1])
#             
#             x_t_full = x_t_full.repeat(self.cluster_num, 1, 1)
#             
#             curr_x_t_masks = torch.ones_like(x_t)
#             
#             curr_full_rec_loss = compute_gaussian_probs0(x_t_full, mean_full, logvar_full, curr_x_t_masks)
#         else:
            
        x_t_full = x_t.view(1, x_t.shape[0], x_t.shape[1])
        
        x_t_full = x_t_full.repeat(self.cluster_num, 1, 1)
        
        x_t_mask_full = x_t_mask.view(1, x_t.shape[0], x_t.shape[1])
        
        x_t_mask_full = x_t_mask_full.repeat(self.cluster_num, 1, 1)
        
        
        curr_full_rec_loss = compute_gaussian_probs0(x_t_full, mean_full, logvar_full, x_t_mask_full)
        
        
#         t2 = time.time()
#         
#         for k in range(self.cluster_num):
#             
#             
#             curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
#             
#             curr_rnn_input[:,k] = 1
#             
#             phi_z_infer = torch.mm(curr_rnn_input, torch.t(self.phi_table))
#             
#             z_t, z_t_category_infer, _ = self.postnet(curr_rnn_input, curr_rnn_output, self.phi_table, t, self.temp)
# 
# #             full_kl += curr_kl*joint_probs[:,k]
#             
#             mean, logvar, logit_x_t = self.generate_x(phi_z_infer, None)
#             
#             if self.loss_on_missing:
#                     
#                 curr_x_t_masks = torch.ones_like(x_t)
#                 
#                 rec_loss = compute_gaussian_probs0(x_t, mean, logvar, curr_x_t_masks)
#             else:
#                 rec_loss = compute_gaussian_probs0(x_t, mean, logvar, x_t_mask)
#             
#             curr_full_rec_loss[k] = rec_loss
#             
# #             print(logit_x_t.shape)
#             
# #             curr_prob = compute_gaussian_probs0(x_t, mean, logvar, x_t_mask)
#             
#             updated_joint_probs += z_t_category_infer*joint_probs[:,k].view(logit_x_t.shape[0],1)
#             
#         
#         t3 = time.time()
#         
#         print('time1::', t2 -t1)
#         
#         print('time2::', t3 -t2)
#         
#         '''batch_size*dim'''
#         print(torch.norm(updated_joint_probs - updated_joint_probs2))
#         
#         print(torch.norm(full_rec_loss - curr_full_rec_loss))
        
        if self.transfer_prob:
                
            z_t_transfer = updated_joint_probs
            
            if self.block == 'GRU':
#                 output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
                output, h_now = self.trans(z_t_transfer.view(z_t_transfer.shape[0], 1, z_t_transfer.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
                
            else:
                output, (h_now, c_now) = self.trans(z_t_transfer.view(z_t_transfer.shape[0], 1, z_t_transfer.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
                
                
                
            z_t, z_t_category_infer, _ , z_category_infer_sparse= self.postnet(z_t_transfer, curr_rnn_output, self.phi_table, t, self.temp)
#                 output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
        else:
            
            phi_z_infer = torch.mm(updated_joint_probs, torch.t(self.phi_table))
            
            if self.block == 'GRU':
                output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
#                     output, h_now = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
                
            else:
#                     output, (h_now, c_now) = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
                output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
            
            z_t, z_t_category_infer, _,z_category_infer_sparse = self.postnet(phi_z_infer, curr_rnn_output, self.phi_table, t, self.temp)
        
        for k in range(self.cluster_num):
            full_rec_loss += curr_full_rec_loss[k]*updated_joint_probs[:,k].view(curr_full_rec_loss[k].shape[0],1)
        full_logit_x_t = torch.mm(updated_joint_probs, torch.t(self.phi_table))
        
        
        
        
        if np.isnan(full_kl.cpu().detach().numpy()).any():
            print('distribution 1::', z_t_category_gen)
            
            print('distribution 2::', z_t_category_infer)
        
#         self.optimizer.zero_grad()
#                 
#         torch.sum(full_rec_loss).backward(retain_graph=True)
        
        return updated_joint_probs, full_kl, h_now, c_now, full_rec_loss, full_logit_x_t, z_t_category_infer
    
    
    
#     def update_joint_probability2(self, joint_probs, batch_size, t, h_prev, c_prev, prev_z_t_category_infer, shrinked_x_lens, x_t, x_t_mask):
#         
#         '''batch_size*cluster'''
# #         if not self.use_sparsemax:
#         z_t_category_gen = F.softmax(self.emitter_z(h_prev), dim = 2)
# #         else:
# #             
# #             logit_z_t = self.emitter_z(h_prev)
# #             
# #             z_t_category_gen = sparsemax(logit_z_t.view(logit_z_t.shape[1], logit_z_t.shape[2]))
# #             
# #             z_t_category_gen = z_t_category_gen.view(1, z_t_category_gen.shape[0], z_t_category_gen.shape[1])
#         
#         full_kl = self.kl_div(prev_z_t_category_infer, z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))
#         
#         
#         
# #         updated_joint_probs = torch.zeros_like(joint_probs)
#         
# #         full_kl = 0
#         
# #         full_logit_x_t = 0
#         
#         
# #         curr_full_rec_loss = torch.zeros([self.cluster_num, batch_size, self.input_dim], device = self.device)
#         
# #         logit_x
# 
#         if self.transfer_prob:
# 
#             full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
#             
#             for k in range(self.cluster_num):
#                 curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
#                 
#                 curr_rnn_input[:,k] = 1
#                 
#                 full_curr_rnn_input[k] = curr_rnn_input
#                 
#         else:
#             full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.z_dim), dtype = torch.float, device = self.device)
#             
#             for k in range(self.cluster_num):
#                 curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
#                 
#                 curr_rnn_input[:,k] = 1
#                 
#                 full_curr_rnn_input[k] = torch.mm(curr_rnn_input, torch.t(self.phi_table))
#         
#         t1 = time.time()
#         
#         
#         
#         
#         curr_rnn_output_full = curr_rnn_output.view(1, curr_rnn_output.shape[0], curr_rnn_output.shape[1])
#         
#         curr_rnn_output_full = curr_rnn_output_full.repeat(self.cluster_num, 1, 1)
#         
#         
# #         print(full_curr_rnn_input.shape, curr_rnn_output_full.shape, t)
#         
#         z_t, z_t_category_infer_full, _,z_category_infer_sparse = self.postnet(full_curr_rnn_input, curr_rnn_output_full, self.phi_table, t, self.temp)
#         
#         
#         
#         
#         
#         
# #         updated_joint_probs2 = torch.bmm(torch.transpose(z_t_category_infer_full, 1, 0), (joint_probs.view(x_t.shape[0], self.cluster_num, 1)))
#         
#         updated_joint_probs = torch.sum(z_t_category_infer_full*torch.t(joint_probs).view(joint_probs.shape[1], joint_probs.shape[0], 1), 0)
#         
#         
#         
#         
# #         t2 = time.time()
# #         
# #         for k in range(self.cluster_num):
# #             
# #             
# #             curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
# #             
# #             curr_rnn_input[:,k] = 1
# #             
# #             phi_z_infer = torch.mm(curr_rnn_input, torch.t(self.phi_table))
# #             
# #             z_t, z_t_category_infer, _ = self.postnet(curr_rnn_input, curr_rnn_output, self.phi_table, t, self.temp)
# # 
# # #             full_kl += curr_kl*joint_probs[:,k]
# #             
# #             mean, logvar, logit_x_t = self.generate_x(phi_z_infer, None)
# #             
# #             if self.loss_on_missing:
# #                     
# #                 curr_x_t_masks = torch.ones_like(x_t)
# #                 
# #                 rec_loss = compute_gaussian_probs0(x_t, mean, logvar, curr_x_t_masks)
# #             else:
# #                 rec_loss = compute_gaussian_probs0(x_t, mean, logvar, x_t_mask)
# #             
# #             curr_full_rec_loss[k] = rec_loss
# #             
# # #             print(logit_x_t.shape)
# #             
# # #             curr_prob = compute_gaussian_probs0(x_t, mean, logvar, x_t_mask)
# #             
# #             updated_joint_probs += z_t_category_infer*joint_probs[:,k].view(logit_x_t.shape[0],1)
# #             
# #         
# #         t3 = time.time()
# #         
# #         print('time1::', t2 -t1)
# #         
# #         print('time2::', t3 -t2)
# #         
# #         '''batch_size*dim'''
# #         print(torch.norm(updated_joint_probs - updated_joint_probs2))
# #         
# #         print(torch.norm(full_rec_loss - curr_full_rec_loss))
#         
#         if self.use_sparsemax:
#             z_t_category_trans =  z_category_infer_sparse
#         else:
#             z_t_category_trans =  z_t_category_infer_full
#         
#         z_t_transfer = updated_joint_probs
#         z_t_transfer_infer = prev_z_t_category_infer
#         
#         if self.use_sparsemax:
#             z_t_transfer = sparsemax(torch.log(z_t_transfer + 1e-5))
#             z_t_transfer_infer = sparsemax(torch.log(z_t_transfer_infer + 1e-5))
#             
#         if self.transfer_prob:
#                 
#             
#             
#             if self.block == 'GRU':
# #                 output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
#                 output, h_now = self.trans(z_t_transfer_infer.view(z_t_transfer_infer.shape[0], 1, z_t_transfer_infer.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
#                 
#             else:
#                 output, (h_now, c_now) = self.trans(z_t_transfer_infer.view(z_t_transfer_infer.shape[0], 1, z_t_transfer_infer.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#                 
#                 
#                 
#             z_t, z_t_category_infer, _,z_category_infer_sparse = self.postnet(z_t_transfer_infer, curr_rnn_output, self.phi_table, t, self.temp)
# #                 output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#         else:
#             
#             phi_z_infer = torch.mm(z_t_transfer_infer, torch.t(self.phi_table))
#             
#             if self.block == 'GRU':
#                 output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
# #                     output, h_now = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
#                 
#             else:
# #                     output, (h_now, c_now) = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#                 output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#             
#             
#             phi_z_infer2 = torch.mm(z_t_transfer_infer, torch.t(self.phi_table))
#             z_t, z_t_category_infer, _,z_category_infer_sparse = self.postnet(phi_z_infer2, curr_rnn_output, self.phi_table, t, self.temp)
#         
#         
#         
#         
#         
#         if np.isnan(full_kl.cpu().detach().numpy()).any():
#             print('distribution 1::', z_t_category_gen)
#             
#             print('distribution 2::', z_t_category_infer)
#         
# #         self.optimizer.zero_grad()
# #                 
# #         torch.sum(full_rec_loss).backward(retain_graph=True)
#         
#         return updated_joint_probs, full_kl, h_now, c_now, z_t_category_infer
#     
    
    
    def infer2(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device):
        """
        infer q(z_{1:T}|x_{1:T}) (i.e. the variational distribution)
        """
        
#         assert torch.sum(x_mask) == torch.sum(1-np.isnan(x))
        T_max = x_lens.max().item()
        if is_GPU:
            x = x.to(device)
            
            x_to_predict = x_to_predict.to(device)
            
            x_mask = x_mask.to(device)
            
            x_to_predict_mask = x_to_predict_mask.to(device)
            
            origin_x = origin_x.to(device)
            
            origin_x_to_pred = origin_x_to_pred.to(device)
            
            
            origin_x_mask = origin_x_mask.to(device)
            
            new_x_mask = new_x_mask.to(device)
            
            
            
            x_to_predict_origin_mask = x_to_predict_origin_mask.to(device)
            
            x_to_predict_new_mask = x_to_predict_new_mask.to(device) 
        
            x_lens = x_lens.to(device)
        
            x_to_predict_lens = x_to_predict_lens.to(device)
        
        if self.is_missing:
            
#             t1 = time.time()
#             
#             imputed_x = self.impute(x, x_mask, T_max)
#             
#             t2 = time.time()
            
            
            imputed_x, interpolated_x = self.impute.forward2(x, x_mask, T_max)

            
#             t3 = time.time()
#             
#             print(t3 - t2)
#             print(t2 - t1)
                    
            x = imputed_x
        
        batch_size, _, input_dim = x.size()
        
        h_0 = self.h_0.expand(1, batch_size, self.h_dim).contiguous()
        
        
        
        
#         c_0 = self.c_0.expand(1, batch_size, self.s_dim).contiguous()
        
        
        h_prev = self.h_0.expand(1, batch_size, self.h_0.size(0))
        
        c_prev = self.h_0.expand(1, batch_size, self.h_0.size(0))
        
        
        z_1_category = torch.ones(self.cluster_num, dtype = torch.float, device = x.device)/self.cluster_num
        
        z_t_category_gen = z_1_category.expand(1, batch_size, z_1_category.size(0))
        
#         phi_z, z_representation = self.generate_z(z_t_category_gen, 0)
        
        
#         if np.any(x_lens.numpy()<= 0):
#             print(torch.nonzero(x_lens <= 0))
#             print('here')
        
#         print(torch.nonzero(x_lens <= 0))
        
        rnn_out,(last_h_n, last_c_n)= self.x_encoder(x, x_lens) # push the observed x's through the rnn;
        
#         self.x_encoder.forward2(x, x_lens, rnn_out, last_h_n, last_c_n)
        
        
        '''to be done'''
#         rnn_out2,(last_h_n2, last_c_n2)= self.x_encoder.forward2(x, x_lens) # push the observed x's through the rnn;
        
#         print(torch.norm(rnn_out - rnn_out2), torch.norm(last_h_n - last_h_n2), torch.norm(last_c_n - last_c_n2))
        
#         rnn_out = reverse_sequence(rnn_out, x_lens) # reverse the time-ordering in the hidden state and un-pack it
        rec_losses = torch.zeros((batch_size, T_max-1, input_dim), device=x.device) 
        kl_states = torch.zeros((batch_size, T_max), device=x.device)
        
        rmse_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
        
        mae_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)   
        
        
        cluster_distances = torch.zeros([batch_size, T_max, self.cluster_num], device = x.device)
        cluster_distances2 = torch.zeros([T_max, self.cluster_num, batch_size, input_dim], device = x.device)
        
        prob_sums = 0
        
#         negnill = torch.zeros()
        
        entropy_losses = torch.zeros((batch_size, T_max), device = x.device)
        
        '''z_q_*''' 
        z_prev = self.z_q_0.expand(batch_size,self.z_q_0.size(0)) # set z_prev=z_q_0 to setup the recursive conditioning in q(z_t|...)
        
        curr_x_lens = x_lens.clone()
        
        shrinked_x_lens = x_lens.clone()
        
        x_t = 0
        
        x_t_mask = 0
        
        curr_rnn_out = rnn_out[curr_x_lens > 0,0,:]
        
        single_time_steps = torch.ones_like(curr_x_lens)
        
        last_h_now = torch.zeros_like(h_prev)
        
        last_c_now = torch.zeros_like(c_prev)
        
        imputed_x2 = torch.zeros_like(x)
        
        imputed_x2[:,0] = x[:,0] 
        
        
        joint_probs = torch.zeros([T_max, batch_size, self.cluster_num], dtype = torch.float, device = self.device)
        
        for t in range(T_max):
#             z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)# p(z_t| z_{t-1})
            
            '''phi_z_infer: phi_{z_t}'''
            if t == 0:
                z_t, z_t_category_infer, _, z_category_infer_sparse = self.postnet(z_prev, curr_rnn_out, self.phi_table, t, self.temp) #q(z_t | z_{t-1}, x_{t:T})
                
                joint_probs[t] = z_t_category_infer
                
                kl = self.kl_div(z_t_category_infer, z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))
                
                if np.isnan(kl.cpu().detach().numpy()).any():
                    print('distribution 1::', z_t_category_gen)
                    
                    print('distribution 2::', z_t_category_infer)
                



                if self.transfer_prob:
                
                    z_t_transfer = z_t_category_infer
                    
                    if self.block == 'GRU':
        #                 output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
                        output, h_now = self.trans(z_t_transfer.view(z_t_transfer.shape[0], 1, z_t_transfer.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
                        
                    else:
                        output, (h_now, c_now) = self.trans(z_t_transfer.view(z_t_transfer.shape[0], 1, z_t_transfer.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
                        
                else:
                    
                    phi_z_infer = torch.mm(z_t_category_infer, torch.t(self.phi_table))
                    
                    if self.block == 'GRU':
                        output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
        #                     output, h_now = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
                        
                    else:
        #                     output, (h_now, c_now) = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
                        output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
                    
                kl_states[curr_x_lens > 0,t] = kl
                
            else:
                
                
                '''joint_probs, curr_rnn_output, batch_size, t, h_prev, c_prev, shrinked_x_lens'''
                '''updated_joint_probs, full_kl, h_now, c_now, full_rec_loss, full_logit_x_t'''
                updated_joint_probs, kl, h_now, c_now, full_rec_loss, logit_x_t, z_t_category_infer = self.update_joint_probability(joint_probs[t-1, curr_x_lens > 0], curr_rnn_out, torch.sum(curr_x_lens > 0), t, h_prev, c_prev, z_t_category_infer, shrinked_x_lens, x_t, x_t_mask)
                
#                 self.optimizer.zero_grad()
#                  
#                 torch.sum(full_rec_loss).backward(retain_graph=True)
                
                joint_probs[t, curr_x_lens > 0] = updated_joint_probs
                
#                 self.optimizer.zero_grad()
#                  
#                 torch.sum(full_rec_loss).backward(retain_graph=True)
                
                
                
                
                kl_states[curr_x_lens > 0,t] = kl
            
                rec_losses[curr_x_lens > 0,t-1] = full_rec_loss
            
#                 print('time::', t)
                
            
                rmse_loss = (x_t*x_t_mask - logit_x_t*x_t_mask)**2
                
                mae_loss = torch.abs(x_t*x_t_mask - logit_x_t*x_t_mask)

#                 mae_losses = torch.abs(x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])

            
                rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
                
                mae_losses[curr_x_lens > 0,:,t-1] = mae_loss
            
            
            
#             phi_z_infer = torch.mm(z_t, torch.t(self.phi_table))
#             phi_z_infer = torch.mm(z_t_category_infer, torch.t(self.phi_table))
#             
#             phi_z_infer2 = torch.mm(z_t, torch.t(self.phi_table))
#             
#             
#             if self.use_gumbel:
#                 print(t, torch.norm(phi_z_infer - phi_z_infer2))
            
            prob_sums += torch.sum(joint_probs[t], 0)
            
#             curr_cluster_distance_res = 0
#             
#             if self.loss_on_missing:
#                  
#                 curr_x_mask = torch.ones_like(x_mask[curr_x_lens > 0,t,:])
#                  
#                 curr_cluster_distance_res = self.compute_distance_per_cluster_all(x[curr_x_lens > 0,t,:], curr_x_mask)
#                  
#                 cluster_distances2[t, :, curr_x_lens > 0] = curr_cluster_distance_res
#             else:
                 
            curr_cluster_distance_res = self.compute_distance_per_cluster_all(x[curr_x_lens > 0,t,:], x_mask[curr_x_lens > 0,t,:])
             
            cluster_distances2[t, :, curr_x_lens > 0] = curr_cluster_distance_res 
#             
#             
#             sumed_cluster_distnace_res = torch.sqrt(torch.sum(curr_cluster_distance_res, 2))
#             
#             cluster_distances[curr_x_lens > 0, t] = self.compute_distance_per_cluster(x[curr_x_lens > 0,t,:])
#             
# #             print(torch.norm(torch.t(sumed_cluster_distnace_res) - cluster_distances[curr_x_lens > 0, t]))
# #             if self.use_sparsemax:
#             
# #             else:
# #                 kl = self.kl_div(z_t_category_infer, z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))
#             
#             entropy_loss = self.entropy(z_t_category_infer)
#             
#             
#             
#             entropy_losses[curr_x_lens > 0, t] = entropy_loss
#             
#             
#                
#             
# #             print(t, curr_x_lens)
#             
#             if self.transfer_prob:
#                 
#                 z_t_transfer = z_t_category_infer
#                 
#                 if self.block == 'GRU':
#     #                 output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
#                     output, h_now = self.trans(z_t_transfer.view(z_t_transfer.shape[0], 1, z_t_transfer.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
#                     
#                 else:
#                     output, (h_now, c_now) = self.trans(z_t_transfer.view(z_t_transfer.shape[0], 1, z_t_transfer.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
# #                 output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#             else:
#                 if self.block == 'GRU':
#                     output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
# #                     output, h_now = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
#                     
#                 else:
# #                     output, (h_now, c_now) = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#                     output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#                 
#                 
# #             if not self.use_sparsemax:
# 
# 
#             
#             
#             
# #             curr_x_lens[curr_x_lens < 0] = 0
#             
#             
#             if t >= 1:            
#             
#                 phi_z, z_representation = self.generate_z(z_t_category_gen, t)
#                 
# #                 phi_z = torch.t(torch.mm(self.phi_table, torch.t(z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))))
# #                 
# #                 phi_z2 = torch.t(torch.mm(self.phi_table, torch.t(z_representation)))
# #                 
# #                 if self.use_gumbel:
# #                     print(torch.norm(phi_z - phi_z2))
#                 
#                 
#                 mean, logvar, logit_x_t = self.generate_x(phi_z_infer, z_t)
#                 
#                 
#                 
#                 
#     
#                 imputed_x2[curr_x_lens > 0,t] = logit_x_t
#                 
#                 
# #                 rec_loss = torch.norm(x[:,t+1,:] - mean)**2/(2*std**2) + torch.log(2*np.pi*std**2)/2
#                 
# #                 rec_loss = torch.bmm(((x[:,t,:]-mean)/(std**2)).view(mean.shape[0],1,mean.shape[1]), (x[:,t,:]-mean).view(mean.shape[0],mean.shape[1],1)).view(-1) + (torch.log((2*np.pi)**x[:,t,:].shape[-1]*torch.prod(std, dim= 1))/2).view(-1) 
#                 
#                 if self.loss_on_missing:
#                     
#                     curr_x_t_masks = torch.ones_like(x_t)
#                     
#                     rec_loss = compute_gaussian_probs0(x_t, mean, logvar, curr_x_t_masks)
#                 else:
#                     rec_loss = compute_gaussian_probs0(x_t, mean, logvar, x_t_mask)
#                 
# #                 rec_loss = self.compute_reconstruction_loss2(x_t, mean, std, batch_size)
#                 
#                 
#     #             kl = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
#     #             kl_states[:,t] = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
#     #             logit_x_t = self.emitter(z_t).contiguous() # p(x_t|z_t)         
#     #             rec_loss = nn.BCEWithLogitsLoss(reduction='none')(logit_x_t.view(-1), x[:,t,:].contiguous().view(-1)).view(batch_size, -1)
#                 rec_losses[curr_x_lens > 0,t-1] = rec_loss
#                 
# #                 rmse_loss = torch.sqrt(torch.sum((x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])**2, dim = 1))
#                 
#                 rmse_loss = (x_t*x_t_mask - logit_x_t*x_t_mask)**2
#                 
#                 mae_loss = torch.abs(x_t*x_t_mask - logit_x_t*x_t_mask)
# 
# #                 mae_losses = torch.abs(x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])
# 
#             
#                 rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
#                 
#                 mae_losses[curr_x_lens > 0,:,t-1] = mae_loss


            curr_x_lens -= 1

            shrinked_x_lens -= 1
#             else:
#                 z_t_category_gen = F.gumbel_softmax(self.emitter_z(h_now), tau=0.001, dim = 2)
            
#             if t >= 21:
#                 print('here')
            
            h_prev = h_now[:,shrinked_x_lens > 0,:]
            
            if (curr_x_lens == 0).any():
                last_h_now[:, curr_x_lens == 0, :] = h_now[:,shrinked_x_lens <= 0,:]
            
            if self.block == 'LSTM':
                c_prev = c_now[:,shrinked_x_lens > 0,:]
                
                if (curr_x_lens == 0).any():
                    last_c_now[:, curr_x_lens == 0, :] = c_now[:,shrinked_x_lens <= 0,:] 
#             phi_z_infer = torch.mm(joint_probs, torch.t(self.phi_table))
#             if self.transfer_prob:
#                 z_prev = z_t[shrinked_x_lens > 0,:]
#             else:
#                 z_prev = phi_z_infer[shrinked_x_lens > 0,:]

#             if not self.use_sparsemax:
#                 
# #                 print(t, torch.sum(shrinked_x_lens > 0))
#                 z_t_category_gen = F.softmax(self.emitter_z(h_now[:,shrinked_x_lens > 0,:]), dim = 2)
#             else:
# #                 print(t, torch.sum(shrinked_x_lens > 0))
#                 
#                 if torch.sum(shrinked_x_lens > 0) > 0:
#                 
#                     logit_z_t = self.emitter_z(h_now[:,shrinked_x_lens > 0,:])
#                     
#                     z_t_category_gen = sparsemax(logit_z_t.view(logit_z_t.shape[1], logit_z_t.shape[2]))
#                 
#                     z_t_category_gen = z_t_category_gen.view(1, z_t_category_gen.shape[0], z_t_category_gen.shape[1])
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
                
                
                curr_rnn_out = rnn_out[curr_x_lens > 0,t+1,:]
            
            
            shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]
            
            
#             curr_x_lens = curr_x_lens[curr_x_lens > 0]

#         x_mask = sequence_mask(x_lens)
#         x_mask = x_mask.gt(0).view(-1)
#         rec_loss = rec_losses.view(-1).masked_select(x_mask).mean()
#         kl_loss = kl_states.view(-1).masked_select(x_mask).mean()

        

#         rec_loss = torch.sum(rec_losses)/torch.sum(x_lens-1)
        
        full_cluster_objs, cluster_objs = self.compute_cluster_obj(cluster_distances, prob_sums/torch.sum(x_lens), T_max, x_lens, input_dim)
        
        
#         if self.loss_on_missing:
#             
#             x_mask_full = torch.ones_like(x_mask)
#             
#             full_cluster_objs2, cluster_objs2 = self.compute_cluster_obj_full2(cluster_distances2, prob_sums/torch.sum(x_lens), T_max, x_mask_full, x_lens)
#         else:
        full_cluster_objs2, cluster_objs2 = self.compute_cluster_obj_full2(cluster_distances2, prob_sums/torch.sum(x_lens), T_max, x_mask, x_lens)
        
#         print(torch.norm(full_cluster_objs - full_cluster_objs2))
#          
#         print(torch.norm(cluster_objs - cluster_objs2))
        
#         if not self.loss_on_missing:
        rec_loss = torch.sum(rec_losses)/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss = torch.sum(rec_losses)/torch.sum(torch.ones_like(rec_losses))
        
#         for k in range(rec_losses.shape[0]):
#             print(rec_losses[k].mean())
        
        first_kl_loss = kl_states[:, 0].view(-1).mean()
        
        kl_loss = torch.sum(kl_states[:, 1:])/torch.sum(x_lens-1)
        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask[:,1:,:]))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask[:,1:,:])
        
        
        final_entropy_loss = entropy_losses.view(-1).mean()
        print('loss::', rec_loss, kl_loss)
        
        print('rmse loss::', final_rmse_loss)
        
        print('mae loss::', final_mae_losses)
        
        print('cluster objective::', cluster_objs2)
        
        interpolated_loss = 0
        
        if self.is_missing:
            interpolated_loss = torch.norm(interpolated_x*x_mask - x*x_mask)
            
            print('interpolate loss::', interpolated_loss)
        
        if torch.sum(1-new_x_mask) > 0:
            
            imputed_mse_loss = torch.sqrt(torch.sum((((origin_x - x)**2)*(1-new_x_mask)))/torch.sum(1-new_x_mask))
            
            imputed_mse_loss2 = torch.sqrt(torch.sum((((origin_x - imputed_x2)**2)*(1-new_x_mask)))/torch.sum(1-new_x_mask)) 
            
            imputed_loss = torch.sum((torch.abs(origin_x - x)*(1-new_x_mask)))/torch.sum(1-new_x_mask)
            
            imputed_loss2 = torch.sum((torch.abs(origin_x - imputed_x2)*(1-new_x_mask)))/torch.sum(1-new_x_mask) 
            
            print('training imputation rmse loss::', imputed_mse_loss)
            
            print('training imputation rmse loss 2::', imputed_mse_loss2)
            
            print('training imputation mae loss::', imputed_loss)
            
            print('training imputation mae loss 2::', imputed_loss2)
            
        
        
        prior_cluster_probs = prob_sums/torch.sum(x_lens)
        
        if self.block == 'GRU':
            self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, None, T_max, prior_cluster_probs)
        else:
            self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, last_c_now, T_max, prior_cluster_probs)
        


        
        print()
        
        if not self.evaluate:
            return rec_loss, kl_loss, first_kl_loss, final_rmse_loss, interpolated_loss, cluster_objs2
        else:
            
            if torch.sum(1-new_x_mask) > 0:
                return imputed_x2*(1-x_mask) + x*x_mask, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
            else:
                return imputed_x2*(1-x_mask) + x*x_mask, None
    
    
    def test_samples1(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_delta_time_stamps, x_to_predict_delta_time_stamps, x_time_stamps, x_to_predict_time_stamps):
        """
        infer q(z_{1:T}|x_{1:T}) (i.e. the variational distribution)
        """
        
#         assert torch.sum(x_mask) == torch.sum(1-np.isnan(x))
#         x_lens[:]=1

        T_max = x_lens.max().item()
        if is_GPU:
            x = x.to(device)
            
            x_to_predict = x_to_predict.to(device)
            
            x_mask = x_mask.to(device)
            
            x_to_predict_mask = x_to_predict_mask.to(device)
            
            origin_x = origin_x.to(device)
            
            origin_x_to_pred = origin_x_to_pred.to(device)
            
            
            origin_x_mask = origin_x_mask.to(device)
            
            new_x_mask = new_x_mask.to(device)
            
            
            
            x_to_predict_origin_mask = x_to_predict_origin_mask.to(device)
            
            x_to_predict_new_mask = x_to_predict_new_mask.to(device) 
        
            x_lens = x_lens.to(device)
        
            x_to_predict_lens = x_to_predict_lens.to(device)
        
            x_time_stamps = x_time_stamps.to(device)
            
            x_to_predict_time_stamps = x_to_predict_time_stamps.to(device)
        
        
        
        if self.is_missing:
            
#             t1 = time.time()
#             
#             imputed_x = self.impute(x, x_mask, T_max)
#             
#             t2 = time.time()
            
            
#             imputed_x, interpolated_x = self.impute.forward2(x, x_mask, x_time_stamps[0]*100)
# 
#         
# #             t3 = time.time()
# #             
# #             print(t3 - t2)
# #             print(t2 - t1)
#                     
#             x = imputed_x
            if self.pre_impute:
                imputed_x, interpolated_x = self.impute.forward2(x, x_mask, x_time_stamps[0]*100)
    # 
    #         
    # #             t3 = time.time()
    # #             
    # #             print(t3 - t2)
    # #             print(t2 - t1)
    #                     
                x = imputed_x
    #             
#             if torch.isnan(x).any():
#                 print(torch.nonzero(torch.isnan(x)))
            else:      
                x = x_mask*x
                 
                interpolated_x = x
        
        batch_size, _, input_dim = x.size()
        
        h_0 = self.h_0.expand(1, batch_size, self.h_dim).contiguous()
        
        if self.method_2:
            infer_probs, gen_probs, latent_y_states = self.get_reconstruction1(torch.cat([x_time_stamps[0].type(torch.FloatTensor), x_to_predict_time_stamps[0].type(torch.FloatTensor)], 0), x,x_time_stamps[0].type(torch.FloatTensor) ,x_mask, n_traj_samples = 1, run_backwards = True)
        else:
            infer_probs, gen_probs, latent_y_states = self.get_reconstruction(torch.cat([x_time_stamps[0].type(torch.FloatTensor), x_to_predict_time_stamps[0].type(torch.FloatTensor)], 0), x,x_time_stamps[0].type(torch.FloatTensor) ,x_mask, n_traj_samples = 1, run_backwards = True)
        
        rec_losses = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
            
        cluster_losses = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
        
        rec_losses_no_coeff = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
        
        cluster_losses_no_coeff = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
        
        imputed_x2 = torch.zeros_like(x)
        
        imputed_x2[:,0] = x[:,0]
        
        
        
        kl_states = torch.zeros((batch_size, T_max), device=x.device)
        
        rmse_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
        
        mae_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)   
        
        joint_probs = torch.zeros([T_max, batch_size, self.cluster_num], dtype = torch.float, device = self.device)

        curr_x_lens = x_lens.clone()
        
        shrinked_x_lens = x_lens.clone()
        
        x_t = None
        
        x_t_mask = None
        
        prob_sums = 0
        
        gen_prior = torch.ones([batch_size, self.cluster_num], device = self.device, dtype = torch.float)/self.cluster_num
        
        for t in range(T_max):
            
            
            full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
                curr_rnn_input[:,k] = 1
                full_curr_rnn_input[k] = curr_rnn_input
#             z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)# p(z_t| z_{t-1})
            
            '''phi_z_infer: phi_{z_t}'''
            
#             print(infer_probs[:,t].shape, gen_probs[:,t].shape)
            if t > 0:
                kl = self.kl_div(infer_probs[:,t], gen_probs[:,t])
            else:
                kl = self.kl_div(infer_probs[:,t], gen_prior)
                
                
            kl_states[:,t] = kl
            
#             print(infer_probs[:,t].shape, joint_probs[t, curr_x_lens > 0].shape)
            
            if t == 0:
                
                
                
                joint_probs[t, curr_x_lens > 0] = infer_probs[:,t].clone()
            else:
#                 updated_joint_probs, kl, h_now, c_now, z_t_category_infer = self.update_joint_probability2(joint_probs[t-1, curr_x_lens > 0], curr_rnn_out, torch.sum(curr_x_lens > 0), t, h_prev, c_prev,z_t_category_infer, shrinked_x_lens, x_t, x_t_mask)
#                 print(x_time_stamps[0,t] - x_time_stamps[0,t-1])
                
                
                if self.method_2:
#                     joint_probs[t, curr_x_lens > 0] = infer_probs[:,t].clone()
                    if isinstance(self.postnet, Encoder_z0_ODE_RNN_cluster):
                        updated_joint_probs = self.postnet.update_joint_probs(x_t.shape[0], joint_probs[t-1, curr_x_lens > 0], t, latent_y_states, (x_time_stamps[0,t] - x_time_stamps[0,t-1]).type(torch.float).to(self.device), full_curr_rnn_input)
                        
                        joint_probs[t, curr_x_lens > 0] = updated_joint_probs
                    else:
                        joint_probs[t, curr_x_lens > 0] = gen_probs[:,t].clone()
                else:
                    updated_joint_probs = self.postnet.update_joint_probs(x_t.shape[0], joint_probs[t-1, curr_x_lens > 0], t, latent_y_states, (x_time_stamps[0,t] - x_time_stamps[0,t-1]).type(torch.float).to(self.device), full_curr_rnn_input)
                    
    #                 print(updated_joint_probs.shape, joint_probs[t, curr_x_lens > 0].shape)
                    
                    joint_probs[t, curr_x_lens > 0] = updated_joint_probs
            
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
                
#             prob_sums += torch.sum(infer_probs[:,t], 0)
            if self.method_2 and isinstance(self.postnet, Encoder_z0_ODE_RNN_cluster):
                prob_sums += torch.sum(gen_probs[:,t], 0)
            else:
                prob_sums += torch.sum(infer_probs[:,t], 0)
            shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]
        
        prior_cluster_probs = prob_sums/torch.sum(x_lens)
        
        for t in range(T_max):
            
            full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
                curr_rnn_input[:,k] = 1
                full_curr_rnn_input[k] = curr_rnn_input
            
            if t >= 1:
#                 print('time step::', t)
                
                full_rec_loss1, full_rec_loss2, full_logit_x_t, curr_full_rec_loss, l2_norm_loss, cluster_loss = self.compute_rec_loss(joint_probs[t], prior_cluster_probs, full_curr_rnn_input, x_t, x_t_mask)
            
                rec_losses[curr_x_lens > 0,t-1] = full_rec_loss1
                
                cluster_losses[curr_x_lens > 0,t-1] = full_rec_loss2
                
                rec_losses_no_coeff[curr_x_lens > 0,t-1] = l2_norm_loss
                
                cluster_losses_no_coeff[curr_x_lens > 0,t-1] = cluster_loss
                
 
                
#                 print('time::', t)
                
                imputed_x2[curr_x_lens > 0,t] = full_logit_x_t
                rmse_loss = (x_t*x_t_mask - full_logit_x_t*x_t_mask)**2
                
                mae_loss = torch.abs(x_t*x_t_mask - full_logit_x_t*x_t_mask)

#                 mae_losses = torch.abs(x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])

            
                rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
                
                mae_losses[curr_x_lens > 0,:,t-1] = mae_loss
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
            
#         if (not self.loss_on_missing) and (not self.latent):
        rec_loss1 = torch.sum(rec_losses)/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss1 = torch.sum(rec_losses)/torch.sum(torch.ones_like(rec_losses))
        
#         if (not self.loss_on_missing) and (not self.latent):
        rec_loss2 = torch.sum(cluster_losses)/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss2 = torch.sum(cluster_losses)/torch.sum(torch.ones_like(cluster_losses))
        
        
#         if (not self.loss_on_missing) and (not self.latent):
        final_rec_loss = torch.sum(rec_losses_no_coeff)/torch.sum(x_mask[:,1:,:])
#         else:
#             final_rec_loss = torch.sum(rec_losses_no_coeff)/torch.sum(torch.ones_like(rec_losses_no_coeff))
        
#         if (not self.loss_on_missing) and (not self.latent):
        final_cluster_loss = torch.sum(cluster_losses_no_coeff)/torch.sum(x_mask[:,1:,:])
#         else:
#             final_cluster_loss = torch.sum(cluster_losses_no_coeff)/torch.sum(torch.ones_like(cluster_losses_no_coeff))
        
        
        
#         for k in range(rec_losses.shape[0]):
#             print(rec_losses[k].mean())
        
        first_kl_loss = kl_states[:, 0].view(-1).mean()
        
        kl_loss = torch.sum(kl_states[:, 1:])/torch.sum(x_lens-1)
        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask[:,1:,:]))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask[:,1:,:])
        
#         final_rec_loss.backward()
#          
#         kl_loss.backward()
#          
#         final_cluster_loss.backward()
#         
#         cluster_objs.backward()
        
        
        print('loss::', final_rec_loss, kl_loss)
        
        print('loss with coefficient::', rec_loss1, kl_loss)
        
        print('rmse loss::', final_rmse_loss)
        
        print('mae loss::', final_mae_losses)
        
        print('cluster objective::', final_cluster_loss)
        
        print('cluster objective with coefficient::', rec_loss2)
        
        
        final_ae_loss = 0

        imputed_loss = 0
        
        if self.is_missing:
            imputed_loss = torch.norm(interpolated_x*x_mask - x*x_mask)
            
            print('interpolate loss::', imputed_loss)
        
        if torch.sum(1-new_x_mask[:,1:]) > 0:
            
            imputed_mse_loss = torch.sqrt(torch.sum((((origin_x[:,1:] - x[:,1:])**2)*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:]))
            
            imputed_mse_loss2 = torch.sqrt(torch.sum((((origin_x[:,1:] - imputed_x2[:,1:])**2)*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:])) 
            
            imputed_loss = torch.sum((torch.abs(origin_x[:,1:] - x[:,1:])*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:])
            
            imputed_loss2 = torch.sum((torch.abs(origin_x[:,1:] - imputed_x2[:,1:])*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:]) 
            
            print('training imputation rmse loss::', imputed_mse_loss)
            
            print('training imputation rmse loss 2::', imputed_mse_loss2)
            
            print('training imputation mae loss::', imputed_loss)
            
            print('training imputation mae loss 2::', imputed_loss2)
            
            
        
        
#         if self.block == 'GRU':
#             self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max:], x_to_predict_time_stamps)
#         else:
#             self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max:], x_to_predict_time_stamps)


        if not self.evaluate:
            final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count, list_res = self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max:], x_to_predict_time_stamps)
        else:
            imputed_x2 = self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max:], x_to_predict_time_stamps)
        
        print()
        
        if not os.path.exists(data_folder + '/' + output_dir):
            os.makedirs(data_folder + '/' + output_dir)
        
        
        torch.save(self.phi_table, data_folder + '/' + output_dir + 'cluster_centroids')
        
        if not self.evaluate:
            if not torch.sum(1-new_x_mask[:,1:]) > 0:
                return final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count, list_res, None
            else:
                return final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count, list_res, (imputed_loss2, torch.sum(1-new_x_mask[:,1:]), imputed_mse_loss2, torch.sum(1-new_x_mask[:,1:]))

#             if not torch.sum(1-new_x_mask[:,1:]) > 0:
#                 return final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count, None
#             else:
#                 return final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count, (imputed_loss2, torch.sum(1-new_x_mask[:,1:]), imputed_mse_loss2, torch.sum(1-new_x_mask[:,1:]))

#             return final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count
        else:
            
            if torch.sum(1-new_x_mask[:,1:]) > 0:
                return imputed_x2, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
            else:
                return imputed_x2, None

#         self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max:], x_to_predict_time_stamps)
#         print()
#         
#         if not self.evaluate:
#             return rec_loss1, kl_loss, first_kl_loss, final_rmse_loss, imputed_loss, rec_loss2, final_ae_loss
#         else:
#             
#             if torch.sum(1-new_x_mask) > 0:
#                 return imputed_x2*(1-x_mask) + x*x_mask, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
#             else:
#                 return imputed_x2*(1-x_mask) + x*x_mask, None

    
    def infer0_1(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_time_stamps, x_to_predict_time_stamps):
        """
        infer q(z_{1:T}|x_{1:T}) (i.e. the variational distribution)
        """
        
#         assert torch.sum(x_mask) == torch.sum(1-np.isnan(x))
#         x_lens[:]=1

        T_max = x_lens.max().item()
        if is_GPU:
            x = x.to(device)
            
            x_to_predict = x_to_predict.to(device)
            
            x_mask = x_mask.to(device)
            
            x_to_predict_mask = x_to_predict_mask.to(device)
            
            origin_x = origin_x.to(device)
            
            origin_x_to_pred = origin_x_to_pred.to(device)
            
            
            origin_x_mask = origin_x_mask.to(device)
            
            new_x_mask = new_x_mask.to(device)
            
            
            
            x_to_predict_origin_mask = x_to_predict_origin_mask.to(device)
            
            x_to_predict_new_mask = x_to_predict_new_mask.to(device) 
        
            x_lens = x_lens.to(device)
        
            x_to_predict_lens = x_to_predict_lens.to(device)
        
            x_time_stamps = x_time_stamps.to(device)
            
            x_to_predict_time_stamps = x_to_predict_time_stamps.to(device)
        
        
        
        if self.is_missing:
            
#             t1 = time.time()
#             
#             imputed_x = self.impute(x, x_mask, T_max)
#             
#             t2 = time.time()
            
            
#             imputed_x, interpolated_x = self.impute.forward2(x, x_mask, x_time_stamps[0]*100)
# 
#         
# #             t3 = time.time()
# #             
# #             print(t3 - t2)
# #             print(t2 - t1)
#                     
#             x = imputed_x

            if self.pre_impute:
                imputed_x, interpolated_x = self.impute.forward2(x, x_mask, x_time_stamps[0]*100)
    # 
    #         
    # #             t3 = time.time()
    # #             
    # #             print(t3 - t2)
    # #             print(t2 - t1)
    #                     
                x = imputed_x
    #             
#             if torch.isnan(x).any():
#                 print(torch.nonzero(torch.isnan(x)))
            else:      
                x = x_mask*x
                 
                interpolated_x = x
        
        batch_size, _, input_dim = x.size()
        
        h_0 = self.h_0.expand(1, batch_size, self.h_dim).contiguous()
        
        infer_probs, gen_probs, latent_y_states = self.get_reconstruction(torch.cat([x_time_stamps[0].type(torch.FloatTensor), x_to_predict_time_stamps[0].type(torch.FloatTensor)], 0), x,x_time_stamps[0].type(torch.FloatTensor) ,x_mask, n_traj_samples = 1, run_backwards = True)
        
        rec_losses = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
            
        cluster_losses = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
        
        rec_losses_no_coeff = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
        
        cluster_losses_no_coeff = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
        
        imputed_x2 = torch.zeros_like(x)
        
        imputed_x2[:,0] = x[:,0]
        
        
        
        kl_states = torch.zeros((batch_size, T_max), device=x.device)
        
        rmse_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
        
        mae_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)   
        
        joint_probs = torch.zeros([T_max, batch_size, self.cluster_num], dtype = torch.float, device = self.device)

        curr_x_lens = x_lens.clone()
        
        shrinked_x_lens = x_lens.clone()
        
        x_t = None
        
        x_t_mask = None
        
        prob_sums = 0
        
        gen_prior = torch.ones([batch_size, self.cluster_num], device = self.device, dtype = torch.float)/self.cluster_num
        
        for t in range(T_max):
            
            
            full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
                curr_rnn_input[:,k] = 1
                full_curr_rnn_input[k] = curr_rnn_input
#             z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)# p(z_t| z_{t-1})
            
            '''phi_z_infer: phi_{z_t}'''
            
#             print(infer_probs[:,t].shape, gen_probs[:,t].shape)
            if t > 0:
                kl = self.kl_div(infer_probs[:,t], gen_probs[:,t])
            else:
                kl = self.kl_div(infer_probs[:,t], gen_prior)
                
                
            kl_states[:,t] = kl
            
#             print(infer_probs[:,t].shape, joint_probs[t, curr_x_lens > 0].shape)
            
            if t == 0:
                
                
                
                joint_probs[t, curr_x_lens > 0] = infer_probs[:,t].clone()
            else:
#                 updated_joint_probs, kl, h_now, c_now, z_t_category_infer = self.update_joint_probability2(joint_probs[t-1, curr_x_lens > 0], curr_rnn_out, torch.sum(curr_x_lens > 0), t, h_prev, c_prev,z_t_category_infer, shrinked_x_lens, x_t, x_t_mask)
#                 print(x_time_stamps[0,t] - x_time_stamps[0,t-1])
                updated_joint_probs = self.postnet.update_joint_probs(x_t.shape[0], joint_probs[t-1, curr_x_lens > 0], t, latent_y_states, (x_time_stamps[0,t] - x_time_stamps[0,t-1]).type(torch.float).to(self.device), full_curr_rnn_input)
                
#                 print(updated_joint_probs.shape, joint_probs[t, curr_x_lens > 0].shape)
                
                joint_probs[t, curr_x_lens > 0] = updated_joint_probs
            
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
                
            prob_sums += torch.sum(infer_probs[:,t], 0)
            shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]
        
        prior_cluster_probs = prob_sums/torch.sum(x_lens)
        
        for t in range(T_max):
            
            full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
                curr_rnn_input[:,k] = 1
                full_curr_rnn_input[k] = curr_rnn_input
            
            if t >= 1:
#                 print('time step::', t)
                
                full_rec_loss1, full_rec_loss2, full_logit_x_t, curr_full_rec_loss, l2_norm_loss, cluster_loss = self.compute_rec_loss(joint_probs[t], prior_cluster_probs, full_curr_rnn_input, x_t, x_t_mask)
            
                rec_losses[curr_x_lens > 0,t-1] = full_rec_loss1
                
                cluster_losses[curr_x_lens > 0,t-1] = full_rec_loss2
                
                rec_losses_no_coeff[curr_x_lens > 0,t-1] = l2_norm_loss
                
                cluster_losses_no_coeff[curr_x_lens > 0,t-1] = cluster_loss
                
 
                
#                 print('time::', t)
                
                imputed_x2[curr_x_lens > 0,t] = full_logit_x_t
                rmse_loss = (x_t*x_t_mask - full_logit_x_t*x_t_mask)**2
                
                mae_loss = torch.abs(x_t*x_t_mask - full_logit_x_t*x_t_mask)

#                 mae_losses = torch.abs(x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])

            
                rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
                
                mae_losses[curr_x_lens > 0,:,t-1] = mae_loss
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
            
#         if (not self.loss_on_missing) and (not self.latent):
        rec_loss1 = torch.sum(rec_losses)/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss1 = torch.sum(rec_losses)/torch.sum(torch.ones_like(rec_losses))
        
#         if (not self.loss_on_missing) and (not self.latent):
        rec_loss2 = torch.sum(cluster_losses)/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss2 = torch.sum(cluster_losses)/torch.sum(torch.ones_like(cluster_losses))
        
        
#         if (not self.loss_on_missing) and (not self.latent):
        final_rec_loss = torch.sum(rec_losses_no_coeff)/torch.sum(x_mask[:,1:,:])
#         else:
#             final_rec_loss = torch.sum(rec_losses_no_coeff)/torch.sum(torch.ones_like(rec_losses_no_coeff))
        
#         if (not self.loss_on_missing) and (not self.latent):
        final_cluster_loss = torch.sum(cluster_losses_no_coeff)/torch.sum(x_mask[:,1:,:])
#         else:
#             final_cluster_loss = torch.sum(cluster_losses_no_coeff)/torch.sum(torch.ones_like(cluster_losses_no_coeff))
        
        
        
#         for k in range(rec_losses.shape[0]):
#             print(rec_losses[k].mean())
        
        first_kl_loss = kl_states[:, 0].view(-1).mean()
        
        kl_loss = torch.sum(kl_states[:, 1:])/torch.sum(x_lens-1)
        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask[:,1:,:]))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask[:,1:,:])
        
#         final_rec_loss.backward()
#          
#         kl_loss.backward()
#          
#         final_cluster_loss.backward()
#         
#         cluster_objs.backward()
        
        
        print('loss::', final_rec_loss, kl_loss)
        
        print('loss with coefficient::', rec_loss1, kl_loss)
        
        print('rmse loss::', final_rmse_loss)
        
        print('mae loss::', final_mae_losses)
        
        print('cluster objective::', final_cluster_loss)
        
        print('cluster objective with coefficient::', rec_loss2)
        
        
        final_ae_loss = 0

        interpolated_loss = 0
        
        if self.is_missing:
            interpolated_loss = torch.norm(interpolated_x*x_mask - x*x_mask)
            
            print('interpolate loss::', interpolated_loss)
        
        if torch.sum(1-new_x_mask[:,1:]) > 0:
            
            imputed_mse_loss = torch.sqrt(torch.sum((((origin_x[:,1:] - x[:,1:])**2)*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:]))
            
            imputed_mse_loss2 = torch.sqrt(torch.sum((((origin_x[:,1:] - imputed_x2[:,1:])**2)*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:])) 
            
            imputed_loss = torch.sum((torch.abs(origin_x[:,1:] - x[:,1:])*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:])
            
            imputed_loss2 = torch.sum((torch.abs(origin_x[:,1:] - imputed_x2[:,1:])*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:]) 
            
            print('training imputation rmse loss::', imputed_mse_loss)
            
            print('training imputation rmse loss 2::', imputed_mse_loss2)
            
            print('training imputation mae loss::', imputed_loss)
            
            print('training imputation mae loss 2::', imputed_loss2)
            
            
        
        
#         if self.block == 'GRU':
#             self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max:], x_to_predict_time_stamps)
#         else:
#             self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max:], x_to_predict_time_stamps)

        self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max:], x_to_predict_time_stamps)
        print()
        
        if not self.evaluate:
            return rec_loss1, kl_loss, first_kl_loss, final_rmse_loss, interpolated_loss, rec_loss2, final_ae_loss
        else:
            
            if torch.sum(1-new_x_mask[:,1:]) > 0:
                return imputed_x2*(1-x_mask) + x*x_mask, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
            else:
                return imputed_x2*(1-x_mask) + x*x_mask, None

    def infer(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_time_stamps, x_to_predict_time_stamps):
        """
        infer q(z_{1:T}|x_{1:T}) (i.e. the variational distribution)
        """
        
#         assert torch.sum(x_mask) == torch.sum(1-np.isnan(x))
#         x_lens[:]=1

        T_max = x_lens.max().item()
        if is_GPU:
            x = x.to(device)
            
            x_to_predict = x_to_predict.to(device)
            
            x_mask = x_mask.to(device)
            
            x_to_predict_mask = x_to_predict_mask.to(device)
            
            origin_x = origin_x.to(device)
            
            origin_x_to_pred = origin_x_to_pred.to(device)
            
            
            origin_x_mask = origin_x_mask.to(device)
            
            new_x_mask = new_x_mask.to(device)
            
            
            
            x_to_predict_origin_mask = x_to_predict_origin_mask.to(device)
            
            x_to_predict_new_mask = x_to_predict_new_mask.to(device) 
        
            x_lens = x_lens.to(device)
        
            x_to_predict_lens = x_to_predict_lens.to(device)
        
            x_time_stamps = x_time_stamps.to(device)
            
            x_to_predict_time_stamps = x_to_predict_time_stamps.to(device)
        
        
        
        if self.is_missing:
            
#             t1 = time.time()
#             
#             imputed_x = self.impute(x, x_mask, T_max)
#             
#             t2 = time.time()
            
            
#             imputed_x, interpolated_x = self.impute.forward2(x, x_mask, x_time_stamps[0]*100)
# 
#         
# #             t3 = time.time()
# #             
# #             print(t3 - t2)
# #             print(t2 - t1)
#                     
#             x = imputed_x

            if self.pre_impute:
                imputed_x, interpolated_x = self.impute.forward2(x, x_mask, x_time_stamps[0]*100)
    # 
    #         
    # #             t3 = time.time()
    # #             
    # #             print(t3 - t2)
    # #             print(t2 - t1)
    #                     
                x = imputed_x
    #             
#             if torch.isnan(x).any():
#                 print(torch.nonzero(torch.isnan(x)))
            else:      
                x = x_mask*x
                 
                interpolated_x = x
        
        batch_size, _, input_dim = x.size()
        
        h_0 = self.h_0.expand(1, batch_size, self.h_dim).contiguous()
        
        infer_probs, gen_probs, latent_y_states = self.get_reconstruction1(torch.cat([x_time_stamps[0].type(torch.FloatTensor), x_to_predict_time_stamps[0].type(torch.FloatTensor)], 0), x,x_time_stamps[0].type(torch.FloatTensor) ,x_mask, n_traj_samples = 1, run_backwards = True)
        
        rec_losses = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
            
        cluster_losses = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
        
        rec_losses_no_coeff = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
        
        cluster_losses_no_coeff = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
        
        imputed_x2 = torch.zeros_like(x)
        
        imputed_x2[:,0] = x[:,0]
        
        
        
        kl_states = torch.zeros((batch_size, T_max), device=x.device)
        
        rmse_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
        
        mae_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)   
        
        joint_probs = torch.zeros([T_max, batch_size, self.cluster_num], dtype = torch.float, device = self.device)

        curr_x_lens = x_lens.clone()
        
        shrinked_x_lens = x_lens.clone()
        
        x_t = None
        
        x_t_mask = None
        
        prob_sums = 0
        
        gen_prior = torch.ones([batch_size, self.cluster_num], device = self.device, dtype = torch.float)/self.cluster_num
        
        for t in range(T_max):
            
            
            full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
                curr_rnn_input[:,k] = 1
                full_curr_rnn_input[k] = curr_rnn_input
#             z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)# p(z_t| z_{t-1})
            
            '''phi_z_infer: phi_{z_t}'''
            
#             print(infer_probs[:,t].shape, gen_probs[:,t].shape)
            if t > 0:
                kl = self.kl_div(infer_probs[:,t], gen_probs[:,t])
            else:
                kl = self.kl_div(infer_probs[:,t], gen_prior)
                
                
            kl_states[:,t] = kl
            
#             print(infer_probs[:,t].shape, joint_probs[t, curr_x_lens > 0].shape)
            
            if t == 0:
                
                
                
                joint_probs[t, curr_x_lens > 0] = infer_probs[:,t].clone()
            else:
#                 updated_joint_probs, kl, h_now, c_now, z_t_category_infer = self.update_joint_probability2(joint_probs[t-1, curr_x_lens > 0], curr_rnn_out, torch.sum(curr_x_lens > 0), t, h_prev, c_prev,z_t_category_infer, shrinked_x_lens, x_t, x_t_mask)
#                 print(x_time_stamps[0,t] - x_time_stamps[0,t-1])
                if isinstance(self.postnet, Encoder_z0_ODE_RNN_cluster):
                    updated_joint_probs = self.postnet.update_joint_probs(x_t.shape[0], joint_probs[t-1, curr_x_lens > 0], t, latent_y_states, (x_time_stamps[0,t] - x_time_stamps[0,t-1]).type(torch.float).to(self.device), full_curr_rnn_input)
                    
                    joint_probs[t, curr_x_lens > 0] = updated_joint_probs
                else:
                    joint_probs[t, curr_x_lens > 0] = gen_probs[:,t].clone()
#                 print(updated_joint_probs.shape, joint_probs[t, curr_x_lens > 0].shape)
                
#                 joint_probs[t, curr_x_lens > 0] = infer_probs[:,t].clone()
            
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
            if self.method_2 and isinstance(self.postnet, Encoder_z0_ODE_RNN_cluster):
                prob_sums += torch.sum(gen_probs[:,t], 0)
            else:
                prob_sums += torch.sum(infer_probs[:,t], 0)
            shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]
        
        prior_cluster_probs = prob_sums/torch.sum(x_lens)
        
        for t in range(T_max):
            
            full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
                curr_rnn_input[:,k] = 1
                full_curr_rnn_input[k] = curr_rnn_input
            
            if t >= 1:
#                 print('time step::', t)
                
                full_rec_loss1, full_rec_loss2, full_logit_x_t, curr_full_rec_loss, l2_norm_loss, cluster_loss = self.compute_rec_loss(joint_probs[t], prior_cluster_probs, full_curr_rnn_input, x_t, x_t_mask)
            
                rec_losses[curr_x_lens > 0,t-1] = full_rec_loss1
                
                cluster_losses[curr_x_lens > 0,t-1] = full_rec_loss2
                
                rec_losses_no_coeff[curr_x_lens > 0,t-1] = l2_norm_loss
                
                cluster_losses_no_coeff[curr_x_lens > 0,t-1] = cluster_loss
                
 
                
#                 print('time::', t)
                
                imputed_x2[curr_x_lens > 0,t] = full_logit_x_t
                rmse_loss = (x_t*x_t_mask - full_logit_x_t*x_t_mask)**2
                
                mae_loss = torch.abs(x_t*x_t_mask - full_logit_x_t*x_t_mask)

#                 mae_losses = torch.abs(x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])

            
                rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
                
                mae_losses[curr_x_lens > 0,:,t-1] = mae_loss
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
            
#         if (not self.loss_on_missing) and (not self.latent):
        rec_loss1 = torch.sum(rec_losses)/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss1 = torch.sum(rec_losses)/torch.sum(torch.ones_like(rec_losses))
        
#         if (not self.loss_on_missing) and (not self.latent):
        rec_loss2 = torch.sum(cluster_losses)/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss2 = torch.sum(cluster_losses)/torch.sum(torch.ones_like(cluster_losses))
        
        
#         if (not self.loss_on_missing) and (not self.latent):
        final_rec_loss = torch.sum(rec_losses_no_coeff)/torch.sum(x_mask[:,1:,:])
#         else:
#             final_rec_loss = torch.sum(rec_losses_no_coeff)/torch.sum(torch.ones_like(rec_losses_no_coeff))
        
#         if (not self.loss_on_missing) and (not self.latent):
        final_cluster_loss = torch.sum(cluster_losses_no_coeff)/torch.sum(x_mask[:,1:,:])
#         else:
#             final_cluster_loss = torch.sum(cluster_losses_no_coeff)/torch.sum(torch.ones_like(cluster_losses_no_coeff))
        
        
        
#         for k in range(rec_losses.shape[0]):
#             print(rec_losses[k].mean())
        
        first_kl_loss = kl_states[:, 0].view(-1).mean()
        
        kl_loss = torch.sum(kl_states[:, 1:])/torch.sum(x_lens-1)
        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask[:,1:,:]))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask[:,1:,:])
        
#         final_rec_loss.backward()
#          
#         kl_loss.backward()
#          
#         final_cluster_loss.backward()
#         
#         cluster_objs.backward()
        
        
        print('loss::', final_rec_loss, kl_loss)
        
        print('loss with coefficient::', rec_loss1, kl_loss)
        
        print('rmse loss::', final_rmse_loss)
        
        print('mae loss::', final_mae_losses)
        
        print('cluster objective::', final_cluster_loss)
        
        print('cluster objective with coefficient::', rec_loss2)
        
        
        final_ae_loss = 0

        interpolated_loss = 0
        
        if self.is_missing:
            interpolated_loss = torch.norm(interpolated_x*x_mask - x*x_mask)
            
            print('interpolate loss::', interpolated_loss)
        
        if torch.sum(1-new_x_mask[:,1:]) > 0:
            
            imputed_mse_loss = torch.sqrt(torch.sum((((origin_x[:,1:] - x[:,1:])**2)*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:]))
            
            imputed_mse_loss2 = torch.sqrt(torch.sum((((origin_x[:,1:] - imputed_x2[:,1:])**2)*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:])) 
            
            imputed_loss = torch.sum((torch.abs(origin_x[:,1:] - x[:,1:])*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:])
            
            imputed_loss2 = torch.sum((torch.abs(origin_x[:,1:] - imputed_x2[:,1:])*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:]) 
            
            print('training imputation rmse loss::', imputed_mse_loss)
            
            print('training imputation rmse loss 2::', imputed_mse_loss2)
            
            print('training imputation mae loss::', imputed_loss)
            
            print('training imputation mae loss 2::', imputed_loss2)
            
            
        
        
#         if self.block == 'GRU':
#             self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max:], x_to_predict_time_stamps)
#         else:
#             self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max:], x_to_predict_time_stamps)

        self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max:], x_to_predict_time_stamps)
        print()
        
        if not self.evaluate:
            return rec_loss1, kl_loss, first_kl_loss, final_rmse_loss, interpolated_loss, rec_loss2, final_ae_loss
        else:
            
            if torch.sum(1-new_x_mask[:,1:]) > 0:
                return imputed_x2*(1-x_mask) + x*x_mask, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
            else:
                return imputed_x2*(1-x_mask) + x*x_mask, None


    def infer0(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_time_stamps, x_to_predict_time_stamps):
        """
        infer q(z_{1:T}|x_{1:T}) (i.e. the variational distribution)
        """
        
#         assert torch.sum(x_mask) == torch.sum(1-np.isnan(x))
#         x_lens[:]=1

        T_max = x_lens.max().item()
        if is_GPU:
            x = x.to(device)
            
            x_to_predict = x_to_predict.to(device)
            
            x_mask = x_mask.to(device)
            
            x_to_predict_mask = x_to_predict_mask.to(device)
            
            origin_x = origin_x.to(device)
            
            origin_x_to_pred = origin_x_to_pred.to(device)
            
            
            origin_x_mask = origin_x_mask.to(device)
            
            new_x_mask = new_x_mask.to(device)
            
            
            
            x_to_predict_origin_mask = x_to_predict_origin_mask.to(device)
            
            x_to_predict_new_mask = x_to_predict_new_mask.to(device) 
        
            x_lens = x_lens.to(device)
        
            x_to_predict_lens = x_to_predict_lens.to(device)
        
        
            x_time_stamps = x_time_stamps.to(device)
            
            x_to_predict_time_stamps = x_to_predict_time_stamps.to(device)
        
        
#         x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, x_time_stamps, x_to_predict_time_stamps = remove_empty_time_steps(x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, x_time_stamps, x_to_predict_time_stamps)

        
        if self.is_missing:
            
#             t1 = time.time()
#             
#             imputed_x = self.impute(x, x_mask, T_max)
#             
#             t2 = time.time()
            
            if self.pre_impute:
                imputed_x, interpolated_x = self.impute.forward2(x, x_mask, x_time_stamps[0]*100)
    # 
    #         
    # #             t3 = time.time()
    # #             
    # #             print(t3 - t2)
    # #             print(t2 - t1)
    #                     
                x = imputed_x
    #             
#             if torch.isnan(x).any():
#                 print(torch.nonzero(torch.isnan(x)))
            else:      
                x = x_mask*x
                 
                interpolated_x = x
            
        
        batch_size, _, input_dim = x.size()
        
        h_0 = self.h_0.expand(1, batch_size, self.h_dim).contiguous()
        
        last_state = None
        
        if self.method_2:
            infer_probs, gen_probs, latent_y_states, gen_latent_y_states, extra_kl_div = self.get_reconstruction2(torch.cat([x_time_stamps[0].type(torch.FloatTensor), x_to_predict_time_stamps[0].type(torch.FloatTensor)], 0), x,x_time_stamps[0].type(torch.FloatTensor) ,x_mask, n_traj_samples = 1, run_backwards = True)
        else:
            infer_probs, gen_probs, latent_y_states, gen_latent_y_states, extra_kl_div = self.get_reconstruction2(torch.cat([x_time_stamps[0].type(torch.FloatTensor), x_to_predict_time_stamps[0].type(torch.FloatTensor)], 0), x,x_time_stamps[0].type(torch.FloatTensor) ,x_mask, n_traj_samples = 1, run_backwards = False)
            last_state = latent_y_states[-1]
            
            
        rec_losses = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
            
        cluster_losses = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
        
        rec_losses_no_coeff = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
        
        cluster_losses_no_coeff = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
        
        imputed_x2 = torch.zeros_like(x)
        
        imputed_x2[:,0] = x[:,0]
        
        
        
        kl_states = torch.zeros((batch_size, T_max), device=x.device)
        
        rmse_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
        
        mae_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)   
        
        joint_probs = torch.zeros([T_max, batch_size, self.cluster_num], dtype = torch.float, device = self.device)

        curr_x_lens = x_lens.clone()
        
        shrinked_x_lens = x_lens.clone()
        
        x_t = None
        
        x_t_mask = None
        
        prob_sums = 0
        
        gen_prior = torch.ones([batch_size, self.cluster_num], device = self.device, dtype = torch.float)/self.cluster_num
        
        for t in range(T_max):
            
            
            full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
                curr_rnn_input[:,k] = 1
                full_curr_rnn_input[k] = curr_rnn_input
#             z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)# p(z_t| z_{t-1})
            
            '''phi_z_infer: phi_{z_t}'''
            
#             print(infer_probs[:,t].shape, gen_probs[:,t].shape)
            if t > 0:
                kl = self.kl_div(infer_probs[:,t], gen_probs[:,t-1])
            else:
                kl = self.kl_div(infer_probs[:,t], gen_prior)
            
#             print('max kl divergence', t, torch.max(kl))
                
            kl_states[:,t] = kl
            
#             print(infer_probs[:,t].shape, joint_probs[t, curr_x_lens > 0].shape)
            
            if t == 0:
                
                
                
                joint_probs[t, curr_x_lens > 0] = infer_probs[:,t].clone()
            else:
#                 updated_joint_probs, kl, h_now, c_now, z_t_category_infer = self.update_joint_probability2(joint_probs[t-1, curr_x_lens > 0], curr_rnn_out, torch.sum(curr_x_lens > 0), t, h_prev, c_prev,z_t_category_infer, shrinked_x_lens, x_t, x_t_mask)
#                 print(x_time_stamps[0,t] - x_time_stamps[0,t-1])
                updated_joint_probs = self.postnet.update_joint_probs(x_t.shape[0], joint_probs[t-1, curr_x_lens > 0], t, latent_y_states, (x_time_stamps[0,t] - x_time_stamps[0,t-1]).type(torch.float).to(self.device), full_curr_rnn_input)
                
#                 print(updated_joint_probs.shape, joint_probs[t, curr_x_lens > 0].shape)
                
                joint_probs[t, curr_x_lens > 0] = updated_joint_probs
            
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
                
            prob_sums += torch.sum(infer_probs[:,t], 0)
            shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]
        
        prior_cluster_probs = prob_sums/torch.sum(x_lens)
        
        for t in range(T_max):
            
            full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
                curr_rnn_input[:,k] = 1
                full_curr_rnn_input[k] = curr_rnn_input
            
            if t >= 1:
#                 print('time step::', t)
                
                full_rec_loss1, full_rec_loss2, full_logit_x_t, curr_full_rec_loss, l2_norm_loss, cluster_loss = self.compute_rec_loss(joint_probs[t], prior_cluster_probs, full_curr_rnn_input, x_t, x_t_mask, latent_y_states[t-1], gen_probs[:,t-1])
            
                rec_losses[curr_x_lens > 0,t-1] = full_rec_loss1
                
                cluster_losses[curr_x_lens > 0,t-1] = full_rec_loss2
                
                rec_losses_no_coeff[curr_x_lens > 0,t-1] = l2_norm_loss
                
                cluster_losses_no_coeff[curr_x_lens > 0,t-1] = cluster_loss
                
 
                
#                 print('time::', t)
                
                imputed_x2[curr_x_lens > 0,t] = full_logit_x_t
                rmse_loss = (x_t*x_t_mask - full_logit_x_t*x_t_mask)**2
                
                mae_loss = torch.abs(x_t*x_t_mask - full_logit_x_t*x_t_mask)

#                 mae_losses = torch.abs(x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])

            
                rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
                
                mae_losses[curr_x_lens > 0,:,t-1] = mae_loss
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
            
#         if (not self.loss_on_missing) and (not self.latent):
        rec_loss1 = torch.sum(rec_losses)/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss1 = torch.sum(rec_losses)/torch.sum(torch.ones_like(rec_losses))
        
#         if (not self.loss_on_missing) and (not self.latent):
        rec_loss2 = torch.sum(cluster_losses)/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss2 = torch.sum(cluster_losses)/torch.sum(torch.ones_like(cluster_losses))
        
        
#         if (not self.loss_on_missing) and (not self.latent):
        final_rec_loss = torch.sum(rec_losses_no_coeff)/torch.sum(x_mask[:,1:,:])
#         else:
#             final_rec_loss = torch.sum(rec_losses_no_coeff)/torch.sum(torch.ones_like(rec_losses_no_coeff))
        
#         if (not self.loss_on_missing) and (not self.latent):
        final_cluster_loss = torch.sum(cluster_losses_no_coeff)/torch.sum(x_mask[:,1:,:])
#         else:
#             final_cluster_loss = torch.sum(cluster_losses_no_coeff)/torch.sum(torch.ones_like(cluster_losses_no_coeff))
        
        
        
#         for k in range(rec_losses.shape[0]):
#             print(rec_losses[k].mean())
        
        first_kl_loss = kl_states[:, 0].view(-1).mean()
        
        kl_loss = torch.sum(kl_states[:, 1:])/torch.sum(x_lens-1)
        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask[:,1:,:]))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask[:,1:,:])
        
#         final_rec_loss.backward()
#          
#         kl_loss.backward()
#          
#         final_cluster_loss.backward()
#         
#         cluster_objs.backward()
        
        
        print('loss::', final_rec_loss, kl_loss, extra_kl_div)
        
        print('loss with coefficient::', rec_loss1, kl_loss, extra_kl_div)
        
        print('rmse loss::', final_rmse_loss)
        
        print('mae loss::', final_mae_losses)
        
        print('cluster objective::', final_cluster_loss)
        
        print('cluster objective with coefficient::', rec_loss2)
        
        
        final_ae_loss = 0

        interpolated_loss = 0
        
        if self.is_missing:
            interpolated_loss = torch.norm(interpolated_x*x_mask - x*x_mask)
            
            print('interpolate loss::', interpolated_loss)
        
        if torch.sum(1-new_x_mask[:,1:]) > 0:
            
            imputed_mse_loss = torch.sqrt(torch.sum((((origin_x[:,1:] - x[:,1:])**2)*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:]))
            
            imputed_mse_loss2 = torch.sqrt(torch.sum((((origin_x[:,1:] - imputed_x2[:,1:])**2)*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:])) 
            
            imputed_loss = torch.sum((torch.abs(origin_x[:,1:] - x[:,1:])*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:])
            
            imputed_loss2 = torch.sum((torch.abs(origin_x[:,1:] - imputed_x2[:,1:])*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:]) 
            
            print('training imputation rmse loss::', imputed_mse_loss)
            
            print('training imputation rmse loss 2::', imputed_mse_loss2)
            
            print('training imputation mae loss::', imputed_loss)
            
            print('training imputation mae loss 2::', imputed_loss2)
            
            
        
        
#         if self.block == 'GRU':
#             self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max:], x_to_predict_time_stamps)
#         else:
#         self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max-1:], x_to_predict_time_stamps)

        forecasting_imputed_data = self.evaluate_forecasting_errors1(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, infer_probs[:, -1].unsqueeze(0), gen_latent_y_states[-1].unsqueeze(0) , x_time_stamps[0,-1], x_to_predict_time_stamps, last_state.unsqueeze(0))

        print()
        
        if not self.evaluate:
            
            if self.use_shift:
                return rec_loss1, kl_loss + extra_kl_div, first_kl_loss, final_rmse_loss, interpolated_loss, rec_loss2, final_ae_loss
            else:
                return rec_loss1, kl_loss, first_kl_loss, final_rmse_loss, interpolated_loss, rec_loss2, final_ae_loss
        else:
            
            if torch.sum(1-new_x_mask[:,1:]) > 0:
                return imputed_x2*(1-x_mask) + x*x_mask, forecasting_imputed_data*(1-x_to_predict_mask) + x_to_predict*x_to_predict_mask, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
            else:
                return imputed_x2*(1-x_mask) + x*x_mask, forecasting_imputed_data*(1-x_to_predict_mask) + x_to_predict*x_to_predict_mask, None

    def test_samples(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_delta_time_stamps, x_to_predict_delta_time_stamps, x_time_stamps, x_to_predict_time_stamps):
        """
        infer q(z_{1:T}|x_{1:T}) (i.e. the variational distribution)
        """
        
#         assert torch.sum(x_mask) == torch.sum(1-np.isnan(x))
#         x_lens[:]=1
        T_max = x_lens.max().item()
        if is_GPU:
            x = x.to(device)
            
            x_to_predict = x_to_predict.to(device)
            
            x_mask = x_mask.to(device)
            
            x_to_predict_mask = x_to_predict_mask.to(device)
            
            origin_x = origin_x.to(device)
            
            origin_x_to_pred = origin_x_to_pred.to(device)
            
            
            origin_x_mask = origin_x_mask.to(device)
            
            new_x_mask = new_x_mask.to(device)
            
            
            
            x_to_predict_origin_mask = x_to_predict_origin_mask.to(device)
            
            x_to_predict_new_mask = x_to_predict_new_mask.to(device) 
        
            x_lens = x_lens.to(device)
        
            x_to_predict_lens = x_to_predict_lens.to(device)
        
            x_time_stamps = x_time_stamps.to(device)
            
            x_to_predict_time_stamps = x_to_predict_time_stamps.to(device)
        
#         x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, x_time_stamps, x_to_predict_time_stamps = remove_empty_time_steps(x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, x_time_stamps, x_to_predict_time_stamps)

        
        if self.is_missing:
            
#             t1 = time.time()
#             
#             imputed_x = self.impute(x, x_mask, T_max)
#             
#             t2 = time.time()
            
            
#             imputed_x, interpolated_x = self.impute.forward2(x, x_mask, x_time_stamps[0]*100)
# 
#         
# #             t3 = time.time()
# #             
# #             print(t3 - t2)
# #             print(t2 - t1)
#                     
#             x = imputed_x

            if self.pre_impute:
                imputed_x, interpolated_x = self.impute.forward2(x, x_mask, x_time_stamps[0]*100)
    # 
    #         
    # #             t3 = time.time()
    # #             
    # #             print(t3 - t2)
    # #             print(t2 - t1)
    #                     
                x = imputed_x
    #             
#             if torch.isnan(x).any():
#                 print(torch.nonzero(torch.isnan(x)))
            else:      
                x = x_mask*x
                 
                interpolated_x = x
        
        batch_size, _, input_dim = x.size()
        
        h_0 = self.h_0.expand(1, batch_size, self.h_dim).contiguous()
        
        last_states = None
        
        if self.method_2:
            infer_probs, gen_probs, latent_y_states, gen_latent_y_states, extra_kl_div = self.get_reconstruction2(torch.cat([x_time_stamps[0].type(torch.FloatTensor), x_to_predict_time_stamps[0].type(torch.FloatTensor)], 0), x,x_time_stamps[0].type(torch.FloatTensor) ,x_mask, n_traj_samples = 1, run_backwards = True)
        else:
            infer_probs, gen_probs, latent_y_states, gen_latent_y_states, extra_kl_div = self.get_reconstruction2(torch.cat([x_time_stamps[0].type(torch.FloatTensor), x_to_predict_time_stamps[0].type(torch.FloatTensor)], 0), x,x_time_stamps[0].type(torch.FloatTensor) ,x_mask, n_traj_samples = 1, run_backwards = False)
            
            last_states = latent_y_states[-1]
        
#         infer_probs, gen_probs, latent_y_states, gen_latent_y_states = self.get_reconstruction2(torch.cat([x_time_stamps[0].type(torch.FloatTensor), x_to_predict_time_stamps[0].type(torch.FloatTensor)], 0), x,x_time_stamps[0].type(torch.FloatTensor) ,x_mask, n_traj_samples = 1)
        
        rec_losses = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
            
        cluster_losses = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
        
        rec_losses_no_coeff = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
        
        cluster_losses_no_coeff = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
        
        imputed_x2 = torch.zeros_like(x)
        
        imputed_x2[:,0] = x[:,0]
        
        
        
        kl_states = torch.zeros((batch_size, T_max), device=x.device)
        
        rmse_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
        
        mae_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)   
        
        joint_probs = torch.zeros([T_max, batch_size, self.cluster_num], dtype = torch.float, device = self.device)

        curr_x_lens = x_lens.clone()
        
        shrinked_x_lens = x_lens.clone()
        
        x_t = None
        
        x_t_mask = None
        
        prob_sums = 0
        
        gen_prior = torch.ones([batch_size, self.cluster_num], device = self.device, dtype = torch.float)/self.cluster_num
        
        for t in range(T_max):
            
            
            full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
                curr_rnn_input[:,k] = 1
                full_curr_rnn_input[k] = curr_rnn_input
#             z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)# p(z_t| z_{t-1})
            
            '''phi_z_infer: phi_{z_t}'''
            
#             print(infer_probs[:,t].shape, gen_probs[:,t].shape)
            if t > 0:
                kl = self.kl_div(infer_probs[:,t], gen_probs[:,t-1])
            else:
                kl = self.kl_div(infer_probs[:,t], gen_prior)
                
                
            kl_states[:,t] = kl
            
#             print(infer_probs[:,t].shape, joint_probs[t, curr_x_lens > 0].shape)
            
            if t == 0:
                
                
                
                joint_probs[t, curr_x_lens > 0] = infer_probs[:,t].clone()
            else:
#                 updated_joint_probs, kl, h_now, c_now, z_t_category_infer = self.update_joint_probability2(joint_probs[t-1, curr_x_lens > 0], curr_rnn_out, torch.sum(curr_x_lens > 0), t, h_prev, c_prev,z_t_category_infer, shrinked_x_lens, x_t, x_t_mask)
#                 print(x_time_stamps[0,t] - x_time_stamps[0,t-1])
                updated_joint_probs = self.postnet.update_joint_probs(x_t.shape[0], joint_probs[t-1, curr_x_lens > 0], t, latent_y_states, (x_time_stamps[0,t] - x_time_stamps[0,t-1]).type(torch.float).to(self.device), full_curr_rnn_input)
                
#                 print(updated_joint_probs.shape, joint_probs[t, curr_x_lens > 0].shape)
                
                joint_probs[t, curr_x_lens > 0] = updated_joint_probs
            
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
                
            prob_sums += torch.sum(infer_probs[:,t], 0)
            shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]
        
        prior_cluster_probs = prob_sums/torch.sum(x_lens)
        
        for t in range(T_max):
            
            full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
                curr_rnn_input[:,k] = 1
                full_curr_rnn_input[k] = curr_rnn_input
            
            if t >= 1:
#                 print('time step::', t)
                
                full_rec_loss1, full_rec_loss2, full_logit_x_t, curr_full_rec_loss, l2_norm_loss, cluster_loss = self.compute_rec_loss(joint_probs[t], prior_cluster_probs, full_curr_rnn_input, x_t, x_t_mask, latent_y_states[t-1], gen_probs[:,t-1])
            
                rec_losses[curr_x_lens > 0,t-1] = full_rec_loss1
                
                cluster_losses[curr_x_lens > 0,t-1] = full_rec_loss2
                
                rec_losses_no_coeff[curr_x_lens > 0,t-1] = l2_norm_loss
                
                cluster_losses_no_coeff[curr_x_lens > 0,t-1] = cluster_loss
                
 
                
#                 print('time::', t)
                
                imputed_x2[curr_x_lens > 0,t] = full_logit_x_t
                rmse_loss = (x_t*x_t_mask - full_logit_x_t*x_t_mask)**2
                
                mae_loss = torch.abs(x_t*x_t_mask - full_logit_x_t*x_t_mask)

#                 mae_losses = torch.abs(x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])

            
                rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
                
                mae_losses[curr_x_lens > 0,:,t-1] = mae_loss
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
            
#         if (not self.loss_on_missing) and (not self.latent):
        rec_loss1 = torch.sum(rec_losses)/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss1 = torch.sum(rec_losses)/torch.sum(torch.ones_like(rec_losses))
        
#         if (not self.loss_on_missing) and (not self.latent):
        rec_loss2 = torch.sum(cluster_losses)/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss2 = torch.sum(cluster_losses)/torch.sum(torch.ones_like(cluster_losses))
        
        
#         if (not self.loss_on_missing) and (not self.latent):
        final_rec_loss = torch.sum(rec_losses_no_coeff)/torch.sum(x_mask[:,1:,:])
#         else:
#             final_rec_loss = torch.sum(rec_losses_no_coeff)/torch.sum(torch.ones_like(rec_losses_no_coeff))
        
#         if (not self.loss_on_missing) and (not self.latent):
        final_cluster_loss = torch.sum(cluster_losses_no_coeff)/torch.sum(x_mask[:,1:,:])
#         else:
#             final_cluster_loss = torch.sum(cluster_losses_no_coeff)/torch.sum(torch.ones_like(cluster_losses_no_coeff))
        
        
        
#         for k in range(rec_losses.shape[0]):
#             print(rec_losses[k].mean())
        
        first_kl_loss = kl_states[:, 0].view(-1).mean()
        
        kl_loss = torch.sum(kl_states[:, 1:])/torch.sum(x_lens-1)
        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask[:,1:,:]))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask[:,1:,:])
        
#         final_rec_loss.backward()
#          
#         kl_loss.backward()
#          
#         final_cluster_loss.backward()
#         
#         cluster_objs.backward()
        
        
        print('loss::', final_rec_loss, kl_loss, extra_kl_div)
        
        print('loss with coefficient::', rec_loss1, kl_loss, extra_kl_div)
        
        print('rmse loss::', final_rmse_loss)
        
        print('mae loss::', final_mae_losses)
        
        print('cluster objective::', final_cluster_loss)
        
        print('cluster objective with coefficient::', rec_loss2)
        
        
        final_ae_loss = 0

        interpolated_loss = 0
        
        if self.is_missing:
            interpolated_loss = torch.norm(interpolated_x*x_mask - x*x_mask)
            
            print('interpolate loss::', interpolated_loss)
        
        if torch.sum(1-new_x_mask[:,1:]) > 0:
            
            imputed_mse_loss = torch.sqrt(torch.sum((((origin_x[:,1:] - x[:,1:])**2)*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:]))
            
            imputed_mse_loss2 = torch.sqrt(torch.sum((((origin_x[:,1:] - imputed_x2[:,1:])**2)*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:])) 
            
            imputed_loss = torch.sum((torch.abs(origin_x[:,1:] - x[:,1:])*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:])
            
            imputed_loss2 = torch.sum((torch.abs(origin_x[:,1:] - imputed_x2[:,1:])*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:]) 
            
            print('training imputation rmse loss::', imputed_mse_loss)
            
            print('training imputation rmse loss 2::', imputed_mse_loss2)
            
            print('training imputation mae loss::', imputed_loss)
            
            print('training imputation mae loss 2::', imputed_loss2)
            
            
        
        
#         if self.block == 'GRU':
#             self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max:], x_to_predict_time_stamps)
#         else:
#             self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max:], x_to_predict_time_stamps)


        if not self.evaluate:
            final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count, list_res = self.evaluate_forecasting_errors1(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, infer_probs[:, -1].unsqueeze(0), gen_latent_y_states[-1].unsqueeze(0) , x_time_stamps[0,-1], x_to_predict_time_stamps, last_states.unsqueeze(0))
        else:
            imputed_x2 = self.evaluate_forecasting_errors1(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, infer_probs[:, -1].unsqueeze(0), gen_latent_y_states[-1].unsqueeze(0) , x_time_stamps[0,-1], x_to_predict_time_stamps, last_states.unsqueeze(0))
#         if not self.evaluate:
#             final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count, list_res = self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max-1:], x_to_predict_time_stamps)
#         else:
#             imputed_x2 = self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max-1:], x_to_predict_time_stamps)
        
        print()
        
        if not os.path.exists(data_folder + '/' + output_dir):
            os.makedirs(data_folder + '/' + output_dir)
        
        
        torch.save(self.phi_table, data_folder + '/' + output_dir + 'cluster_centroids')
        
        if not self.evaluate:
            
            if not torch.sum(1-new_x_mask[:,1:]) > 0:
                return final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count, list_res, None
            else:
                return final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count, list_res, ((imputed_loss, imputed_loss2), torch.sum(1-new_x_mask[:,1:]), (imputed_mse_loss, imputed_mse_loss2), torch.sum(1-new_x_mask[:,1:]))

#             return final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count
        else:
            
            if torch.sum(1-new_x_mask[:,1:]) > 0:
                return imputed_x2, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
            else:
                return imputed_x2, None

#     self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_delta_time_stamps, x_to_predict_delta_time_stamps, x_time_stamps, x_to_predict_time_stamps

    def test_samples2(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_delta_time_stamps, x_to_predict_delta_time_stamps, x_time_stamps, x_to_predict_time_stamps):
        """
        infer q(z_{1:T}|x_{1:T}) (i.e. the variational distribution)
        """
        
#         assert torch.sum(x_mask) == torch.sum(1-np.isnan(x))
#         x_lens[:]=1

        T_max = x_lens.max().item()
        if is_GPU:
            x = x.to(device)
            
            x_to_predict = x_to_predict.to(device)
            
            x_mask = x_mask.to(device)
            
            x_to_predict_mask = x_to_predict_mask.to(device)
            
            origin_x = origin_x.to(device)
            
            origin_x_to_pred = origin_x_to_pred.to(device)
            
            
            origin_x_mask = origin_x_mask.to(device)
            
            new_x_mask = new_x_mask.to(device)
            
            
            
            x_to_predict_origin_mask = x_to_predict_origin_mask.to(device)
            
            x_to_predict_new_mask = x_to_predict_new_mask.to(device) 
        
            x_lens = x_lens.to(device)
        
            x_to_predict_lens = x_to_predict_lens.to(device)
        
            x_time_stamps = x_time_stamps.to(device)
            
            x_to_predict_time_stamps = x_to_predict_time_stamps.to(device)
        
        
        if self.is_missing:
            
#             t1 = time.time()
#             
#             imputed_x = self.impute(x, x_mask, T_max)
#             
#             t2 = time.time()
            
            
#             imputed_x, interpolated_x = self.impute.forward2(x, x_mask, x_time_stamps[0]*100)
# 
#         
# #             t3 = time.time()
# #             
# #             print(t3 - t2)
# #             print(t2 - t1)
#                     
#             x = imputed_x

            if self.pre_impute:
                imputed_x, interpolated_x = self.impute.forward2(x, x_mask, x_time_stamps[0]*100)
    # 
    #         
    # #             t3 = time.time()
    # #             
    # #             print(t3 - t2)
    # #             print(t2 - t1)
    #                     
                x = imputed_x
    #             
#             if torch.isnan(x).any():
#                 print(torch.nonzero(torch.isnan(x)))
            else:      
                x = x_mask*x
                 
                interpolated_x = x
        
        batch_size, _, input_dim = x.size()
        
        h_0 = self.h_0.expand(1, batch_size, self.h_dim).contiguous()
        
        
        infer_probs, gen_probs, latent_y_states = self.get_reconstruction(torch.cat([x_time_stamps[0].type(torch.FloatTensor), x_to_predict_time_stamps[0].type(torch.FloatTensor)], 0), x,x_time_stamps[0].type(torch.FloatTensor) ,x_mask, n_traj_samples = 1, run_backwards = True)
#         infer_probs, gen_probs, latent_y_states = self.get_reconstruction(torch.cat([x_time_stamps[0].type(torch.FloatTensor), x_to_predict_time_stamps[0].type(torch.FloatTensor)], 0), x, x_time_stamps[0].type(torch.FloatTensor),x_mask, n_traj_samples = 1)
        
        rec_losses = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
            
        cluster_losses = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
        
        rec_losses_no_coeff = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
        
        cluster_losses_no_coeff = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
        
        imputed_x2 = torch.zeros_like(x)
        
        imputed_x2[:,0] = x[:,0]
        
        
        
        kl_states = torch.zeros((batch_size, T_max), device=x.device)
        
        rmse_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
        
        mae_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)   
        
        joint_probs = torch.zeros([T_max, batch_size, self.cluster_num], dtype = torch.float, device = self.device)

        curr_x_lens = x_lens.clone()
        
        shrinked_x_lens = x_lens.clone()
        
        x_t = None
        
        x_t_mask = None
        
        prob_sums = 0
        
        full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
        for k in range(self.cluster_num):
            curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            curr_rnn_input[:,k] = 1
            full_curr_rnn_input[k] = curr_rnn_input
        
        for t in range(T_max):
#             z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)# p(z_t| z_{t-1})
            
            '''phi_z_infer: phi_{z_t}'''
            
            
                
            kl = self.kl_div(infer_probs[:,t], gen_probs[:,t])
                
                
            kl_states[:,t] = kl
            
            if t == 0:
                joint_probs[t, curr_x_lens > 0] = infer_probs[:,t].clone()
            else:
#                 updated_joint_probs, kl, h_now, c_now, z_t_category_infer = self.update_joint_probability2(joint_probs[t-1, curr_x_lens > 0], curr_rnn_out, torch.sum(curr_x_lens > 0), t, h_prev, c_prev,z_t_category_infer, shrinked_x_lens, x_t, x_t_mask)
                
                updated_joint_probs = self.postnet.update_joint_probs(x_t, joint_probs[t-1,curr_x_lens > 0], t, latent_y_states, (x_time_stamps[0,t] - x_time_stamps[0,t-1]).type(torch.float).to(self.device), full_curr_rnn_input)
                
                
                joint_probs[t, curr_x_lens > 0] = updated_joint_probs
            
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
                
            prob_sums += torch.sum(infer_probs[:,t], 0)
            shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]
        
        prior_cluster_probs = prob_sums/torch.sum(x_lens)
        
        for t in range(T_max):
            
            if t >= 1:
                full_rec_loss1, full_rec_loss2, full_logit_x_t, curr_full_rec_loss, l2_norm_loss, cluster_loss = self.compute_rec_loss(joint_probs[t], prior_cluster_probs, full_curr_rnn_input, x_t, x_t_mask)
            
                rec_losses[curr_x_lens > 0,t-1] = full_rec_loss1
                
                cluster_losses[curr_x_lens > 0,t-1] = full_rec_loss2
                
                rec_losses_no_coeff[curr_x_lens > 0,t-1] = l2_norm_loss
                
                cluster_losses_no_coeff[curr_x_lens > 0,t-1] = cluster_loss
                
 
                
#                 print('time::', t)
                
                imputed_x2[curr_x_lens > 0,t] = full_logit_x_t
                rmse_loss = (x_t*x_t_mask - full_logit_x_t*x_t_mask)**2
                
                mae_loss = torch.abs(x_t*x_t_mask - full_logit_x_t*x_t_mask)

#                 mae_losses = torch.abs(x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])

            
                rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
                
                mae_losses[curr_x_lens > 0,:,t-1] = mae_loss
            
            
#         if (not self.loss_on_missing) and (not self.latent):
        rec_loss1 = torch.sum(rec_losses)/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss1 = torch.sum(rec_losses)/torch.sum(torch.ones_like(rec_losses))
        
#         if (not self.loss_on_missing) and (not self.latent):
        rec_loss2 = torch.sum(cluster_losses)/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss2 = torch.sum(cluster_losses)/torch.sum(torch.ones_like(cluster_losses))
        
        
#         if (not self.loss_on_missing) and (not self.latent):
        final_rec_loss = torch.sum(rec_losses_no_coeff)/torch.sum(x_mask[:,1:,:])
#         else:
#             final_rec_loss = torch.sum(rec_losses_no_coeff)/torch.sum(torch.ones_like(rec_losses_no_coeff))
        
#         if (not self.loss_on_missing) and (not self.latent):
        final_cluster_loss = torch.sum(cluster_losses_no_coeff)/torch.sum(x_mask[:,1:,:])
#         else:
#             final_cluster_loss = torch.sum(cluster_losses_no_coeff)/torch.sum(torch.ones_like(cluster_losses_no_coeff))
        
        
        
#         for k in range(rec_losses.shape[0]):
#             print(rec_losses[k].mean())
        
        first_kl_loss = kl_states[:, 0].view(-1).mean()
        
        kl_loss = torch.sum(kl_states[:, 1:])/torch.sum(x_lens-1)
        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask[:,1:,:]))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask[:,1:,:])
        
        
        print('loss::', final_rec_loss, kl_loss)
        
        print('loss with coefficient::', rec_loss1, kl_loss)
        
        print('rmse loss::', final_rmse_loss)
        
        print('mae loss::', final_mae_losses)
        
        print('cluster objective::', final_cluster_loss)
        
        print('cluster objective with coefficient::', rec_loss2)
        
        
        final_ae_loss = 0

        interpolated_loss = 0
        
        if self.is_missing:
            interpolated_loss = torch.norm(interpolated_x*x_mask - x*x_mask)
            
            print('interpolate loss::', interpolated_loss)
        
        if torch.sum(1-new_x_mask[:,1:]) > 0:
            
            imputed_mse_loss = torch.sqrt(torch.sum((((origin_x[:,1:] - x[:,1:])**2)*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:]))
            
            imputed_mse_loss2 = torch.sqrt(torch.sum((((origin_x[:,1:] - imputed_x2[:,1:])**2)*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:])) 
            
            imputed_loss = torch.sum((torch.abs(origin_x[:,1:] - x[:,1:])*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:])
            
            imputed_loss2 = torch.sum((torch.abs(origin_x[:,1:] - imputed_x2[:,1:])*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:]) 
            
            print('training imputation rmse loss::', imputed_mse_loss)
            
            print('training imputation rmse loss 2::', imputed_mse_loss2)
            
            print('training imputation mae loss::', imputed_loss)
            
            print('training imputation mae loss 2::', imputed_loss2)
            
            
#         prior_cluster_probs = prob_sums/torch.sum(x_lens)
        
#         if self.block == 'GRU':
#             self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max:], x_to_predict_time_stamps)
#         else:
#             self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max:], x_to_predict_time_stamps)
            
            
        if not self.evaluate:
            final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count, list_res = self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max:], x_to_predict_time_stamps)
        else:
            imputed_x2 = self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max:], x_to_predict_time_stamps)
        
        print()
        
        if not os.path.exists(data_folder + '/' + output_dir):
            os.makedirs(data_folder + '/' + output_dir)
        
        
        torch.save(self.phi_table, data_folder + '/' + output_dir + 'cluster_centroids')
        
        if not self.evaluate:
            
            if not torch.sum(1-new_x_mask[:,1:]) > 0:
                return final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count, list_res, None
            else:
                return final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count, list_res, (imputed_loss2, torch.sum(1-new_x_mask[:,1:]), imputed_mse_loss2, torch.sum(1-new_x_mask[:,1:]))

#             return final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count
        else:
            
            if torch.sum(1-new_x_mask) > 0:
                return imputed_x2, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
            else:
                return imputed_x2, None
    
#     def infer(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_time_stamps, x_to_predict_time_stamps):
#         """
#         infer q(z_{1:T}|x_{1:T}) (i.e. the variational distribution)
#         """
#         
# #         assert torch.sum(x_mask) == torch.sum(1-np.isnan(x))
# #         x_lens[:]=1
# 
#         T_max = x_lens.max().item()
#         if is_GPU:
#             x = x.to(device)
#             
#             x_to_predict = x_to_predict.to(device)
#             
#             x_mask = x_mask.to(device)
#             
#             x_to_predict_mask = x_to_predict_mask.to(device)
#             
#             origin_x = origin_x.to(device)
#             
#             origin_x_to_pred = origin_x_to_pred.to(device)
#             
#             
#             origin_x_mask = origin_x_mask.to(device)
#             
#             new_x_mask = new_x_mask.to(device)
#             
#             
#             
#             x_to_predict_origin_mask = x_to_predict_origin_mask.to(device)
#             
#             x_to_predict_new_mask = x_to_predict_new_mask.to(device) 
#         
#             x_lens = x_lens.to(device)
#         
#             x_to_predict_lens = x_to_predict_lens.to(device)
#         
#         
#         
#         
#         
#         if self.is_missing:
#             
# #             t1 = time.time()
# #             
# #             imputed_x = self.impute(x, x_mask, T_max)
# #             
# #             t2 = time.time()
#             
#             
#             imputed_x, interpolated_x = self.impute.forward2(x, x_mask, T_max)
# 
#         
# #             t3 = time.time()
# #             
# #             print(t3 - t2)
# #             print(t2 - t1)
#                     
#             x = imputed_x
#         
#         batch_size, _, input_dim = x.size()
#         
#         h_0 = self.h_0.expand(1, batch_size, self.h_dim).contiguous()
#         
#         infer_probs, gen_y_probs = self.get_reconstruction(x_time_stamps[0].type(torch.FloatTensor), x, x_time_stamps[0].type(torch.FloatTensor),x_mask, n_traj_samples = 1, run_backwards = True)
#         
#         
# #         c_0 = self.c_0.expand(1, batch_size, self.s_dim).contiguous()
#         
#         
#         h_prev = self.h_0.expand(1, batch_size, self.h_0.size(0))
#         
#         c_prev = self.h_0.expand(1, batch_size, self.h_0.size(0))
#         
#         
#         z_1_category = torch.ones(self.cluster_num, dtype = torch.float, device = x.device)/self.cluster_num
#         
#         z_t_category_gen = z_1_category.expand(1, batch_size, z_1_category.size(0))
#         
# #         phi_z, z_representation = self.generate_z(z_t_category_gen, 0)
#         
#         
# #         if np.any(x_lens.numpy()<= 0):
# #             print(torch.nonzero(x_lens <= 0))
# #             print('here')
#         
# #         print(torch.nonzero(x_lens <= 0))
#         
#         if not self.latent:
#             rnn_out,(last_h_n, last_c_n)= self.x_encoder(x, x_lens) # push the observed x's through the rnn;
#         else:
#             if not self.lstm_latent:
#                 rnn_out,(last_h_n, last_c_n)= self.x_encoder(self.x_kernel_encoder(x), x_lens) # push the observed x's through the rnn;
#             else:
#                 rnn_out,(last_h_n, last_c_n)= self.x_encoder(x, x_lens) # push the observed x's through the rnn;
#                 
#                 
#         
# #         self.x_encoder.forward2(x, x_lens, rnn_out, last_h_n, last_c_n)
#         
#         
#         '''to be done'''
# #         rnn_out2,(last_h_n2, last_c_n2)= self.x_encoder.forward2(x, x_lens) # push the observed x's through the rnn;
#         
# #         print(torch.norm(rnn_out - rnn_out2), torch.norm(last_h_n - last_h_n2), torch.norm(last_c_n - last_c_n2))
#         
# #         rnn_out = reverse_sequence(rnn_out, x_lens) # reverse the time-ordering in the hidden state and un-pack it
#         
#         if self.latent and self.lstm_latent:
#             rec_losses = torch.zeros((batch_size, T_max-1, (1+self.x_encoder.bidir)*self.s_dim), device=x.device)
#         
#             cluster_losses = torch.zeros((batch_size, T_max  -1, (1+self.x_encoder.bidir)*self.s_dim), device=x.device) 
#             
#             rec_losses_no_coeff = torch.zeros((batch_size, T_max-1, (1+self.x_encoder.bidir)*self.s_dim), device=x.device)
#             
#             cluster_losses_no_coeff = torch.zeros((batch_size, T_max  -1, (1+self.x_encoder.bidir)*self.s_dim), device=x.device)
#             
#         else:
#             
#             rec_losses = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
#             
#             cluster_losses = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
#             
#             rec_losses_no_coeff = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
#             
#             cluster_losses_no_coeff = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
#          
#         kl_states = torch.zeros((batch_size, T_max), device=x.device)
#         
#         rmse_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
#         
#         mae_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)   
#         
#         
#         cluster_distances = torch.zeros([batch_size, T_max, self.cluster_num], device = x.device)
#         cluster_distances2 = torch.zeros([T_max, self.cluster_num, batch_size, input_dim], device = x.device)
#         
#         prob_sums = 0
#         
# #         negnill = torch.zeros()
#         
#         entropy_losses = torch.zeros((batch_size, T_max), device = x.device)
#         
#         '''z_q_*''' 
#         z_prev = self.z_q_0.expand(batch_size,self.z_q_0.size(0)) # set z_prev=z_q_0 to setup the recursive conditioning in q(z_t|...)
#         
#         curr_x_lens = x_lens.clone()
#         
#         shrinked_x_lens = x_lens.clone()
#         
#         x_t = 0
#         
#         x_t_mask = 0
#         
#         curr_rnn_out = rnn_out[curr_x_lens > 0,0,:]
#         
#         single_time_steps = torch.ones_like(curr_x_lens)
#         
#         last_h_now = torch.zeros_like(h_prev)
#         
#         last_c_now = torch.zeros_like(c_prev)
#         
#         imputed_x2 = torch.zeros_like(x)
#         
#         imputed_x2[:,0] = x[:,0] 
#         
#         last_rnn_out = torch.zeros(batch_size, self.s_dim*(1+self.x_encoder.bidir), device = self.device)
#         
#         joint_probs = torch.zeros([T_max, batch_size, self.cluster_num], dtype = torch.float, device = self.device)
#         
#         h_now_list = []
#         
#         decoded_h_n_list = []
#         
#         decoded_c_n_list = []
#         
#         last_decoded_h_n = torch.zeros([batch_size, self.z_dim], device = self.device)
#         
#         last_decoded_c_n = torch.zeros([batch_size, self.z_dim], device = self.device)
#         
#         if self.latent:
#             if not self.lstm_latent:
#                 ae_loss = torch.zeros((batch_size, T_max-1, input_dim), device=x.device)
#             
#             if self.lstm_latent:
#                 ae_loss = torch.zeros((batch_size, T_max, input_dim), device=x.device)
#         
#         
# #             if self.lstm_latent:
# #                 full_rnn_decoded_out, (last_decoded_h_n2, last_decoded_c_n2) = self.x_kernel_decoder(rnn_out, x_lens)
#         
#         
#         for t in range(T_max):
# #             z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)# p(z_t| z_{t-1})
#             
#             '''phi_z_infer: phi_{z_t}'''
#             
#             
#             
#             if t == 0:
# #                 z_t, z_t_category_infer, _, z_category_infer_sparse = self.postnet(z_prev, curr_rnn_out, self.phi_table, t, self.temp) #q(z_t | z_{t-1}, x_{t:T})
# #                 
# #                 joint_probs[t] = z_t_category_infer
#                 
#                 kl = self.kl_div(z_t_category_infer, z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))
#                 
#                 if np.isnan(kl.cpu().detach().numpy()).any():
#                     print('distribution 1::', z_t_category_gen)
#                     
#                     print('distribution 2::', z_t_category_infer)
#                 
#                 
#                 if self.use_sparsemax:
#                     z_t_category_trans =  sparsemax(torch.log(z_t_category_infer+1e-5))
#                 else:
#                     z_t_category_trans =  z_t_category_infer
#                 
#                 if self.transfer_prob:
#                     
#                     if self.block == 'GRU':
#         #                 output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
#                         output, h_now = self.trans(z_t_category_trans.view(z_t_category_trans.shape[0], 1, z_t_category_trans.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
#                         
#                     else:
#                         output, (h_now, c_now) = self.trans(z_t_category_trans.view(z_t_category_trans.shape[0], 1, z_t_category_trans.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#                         
#     #                 output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#                 else:
#                     
#                     phi_z_infer = torch.mm(z_t_category_trans, torch.t(self.phi_table))
#                     
#                     if self.block == 'GRU':
#                         output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
#         #                     output, h_now = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
#                         
#                     else:
#         #                     output, (h_now, c_now) = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#                         output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#                     
#                     
# #                 kl_states[curr_x_lens > 0,t] = kl
#                 
#             else:
#                 
#                 
#                 '''joint_probs, curr_rnn_output, batch_size, t, h_prev, c_prev, shrinked_x_lens'''
#                 '''updated_joint_probs, full_kl, h_now, c_now, full_rec_loss, full_logit_x_t'''
#                 updated_joint_probs, kl, h_now, c_now, z_t_category_infer = self.update_joint_probability2(joint_probs[t-1, curr_x_lens > 0], curr_rnn_out, torch.sum(curr_x_lens > 0), t, h_prev, c_prev,z_t_category_infer, shrinked_x_lens, x_t, x_t_mask)
#                 
# #                 self.optimizer.zero_grad()
# #                  
# #                 torch.sum(full_rec_loss).backward(retain_graph=True)
#                 
#                 joint_probs[t, curr_x_lens > 0] = updated_joint_probs
#                 
# #                 self.optimizer.zero_grad()
# #                  
# #                 torch.sum(full_rec_loss).backward(retain_graph=True)
#                 
#                 
#                 
#                 
#                 kl_states[curr_x_lens > 0,t] = kl
#             
#             
#             
# #             phi_z_infer = torch.mm(z_t, torch.t(self.phi_table))
# #             phi_z_infer = torch.mm(z_t_category_infer, torch.t(self.phi_table))
# #             
# #             phi_z_infer2 = torch.mm(z_t, torch.t(self.phi_table))
# #             
# #             
# #             if self.use_gumbel:
# #                 print(t, torch.norm(phi_z_infer - phi_z_infer2))
#             
#             prob_sums += torch.sum(z_t_category_infer, 0)
#             
#             
#             
# #             curr_cluster_distance_res = 0
# #             
# #             if self.loss_on_missing:
# #                  
# #                 curr_x_mask = torch.ones_like(x_mask[curr_x_lens > 0,t,:])
# #                  
# #                 curr_cluster_distance_res = self.compute_distance_per_cluster_all(x[curr_x_lens > 0,t,:], curr_x_mask)
# #                  
# #                 cluster_distances2[t, :, curr_x_lens > 0] = curr_cluster_distance_res
# #             else:
# #                  
# #                 curr_cluster_distance_res = self.compute_distance_per_cluster_all(x[curr_x_lens > 0,t,:], x_mask[curr_x_lens > 0,t,:])
# #                  
# #                 cluster_distances2[t, :, curr_x_lens > 0] = curr_cluster_distance_res 
# #             
# #             
# #             sumed_cluster_distnace_res = torch.sqrt(torch.sum(curr_cluster_distance_res, 2))
# #             
# #             cluster_distances[curr_x_lens > 0, t] = self.compute_distance_per_cluster(x[curr_x_lens > 0,t,:])
# #             
# # #             print(torch.norm(torch.t(sumed_cluster_distnace_res) - cluster_distances[curr_x_lens > 0, t]))
# # #             if self.use_sparsemax:
# #             
# # #             else:
# # #                 kl = self.kl_div(z_t_category_infer, z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))
# #             
# #             entropy_loss = self.entropy(z_t_category_infer)
# #             
# #             
# #             
# #             entropy_losses[curr_x_lens > 0, t] = entropy_loss
# #             
# #             
# #                
# #             
# # #             print(t, curr_x_lens)
# #             
# #             if self.transfer_prob:
# #                 
# #                 z_t_transfer = z_t_category_infer
# #                 
# #                 if self.block == 'GRU':
# #     #                 output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
# #                     output, h_now = self.trans(z_t_transfer.view(z_t_transfer.shape[0], 1, z_t_transfer.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
# #                     
# #                 else:
# #                     output, (h_now, c_now) = self.trans(z_t_transfer.view(z_t_transfer.shape[0], 1, z_t_transfer.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
# # #                 output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
# #             else:
# #                 if self.block == 'GRU':
# #                     output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
# # #                     output, h_now = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
# #                     
# #                 else:
# # #                     output, (h_now, c_now) = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
# #                     output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
# #                 
# #                 
# # #             if not self.use_sparsemax:
# # 
# # 
# #             
# #             
# #             
# # #             curr_x_lens[curr_x_lens < 0] = 0
# #             
# #             
# #             if t >= 1:            
# #             
# #                 phi_z, z_representation = self.generate_z(z_t_category_gen, t)
# #                 
# # #                 phi_z = torch.t(torch.mm(self.phi_table, torch.t(z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))))
# # #                 
# # #                 phi_z2 = torch.t(torch.mm(self.phi_table, torch.t(z_representation)))
# # #                 
# # #                 if self.use_gumbel:
# # #                     print(torch.norm(phi_z - phi_z2))
# #                 
# #                 
# #                 mean, logvar, logit_x_t = self.generate_x(phi_z_infer, z_t)
# #                 
# #                 
# #                 
# #                 
# #     
# #                 imputed_x2[curr_x_lens > 0,t] = logit_x_t
# #                 
# #                 
# # #                 rec_loss = torch.norm(x[:,t+1,:] - mean)**2/(2*std**2) + torch.log(2*np.pi*std**2)/2
# #                 
# # #                 rec_loss = torch.bmm(((x[:,t,:]-mean)/(std**2)).view(mean.shape[0],1,mean.shape[1]), (x[:,t,:]-mean).view(mean.shape[0],mean.shape[1],1)).view(-1) + (torch.log((2*np.pi)**x[:,t,:].shape[-1]*torch.prod(std, dim= 1))/2).view(-1) 
# #                 
# #                 if self.loss_on_missing:
# #                     
# #                     curr_x_t_masks = torch.ones_like(x_t)
# #                     
# #                     rec_loss = compute_gaussian_probs0(x_t, mean, logvar, curr_x_t_masks)
# #                 else:
# #                     rec_loss = compute_gaussian_probs0(x_t, mean, logvar, x_t_mask)
# #                 
# # #                 rec_loss = self.compute_reconstruction_loss2(x_t, mean, std, batch_size)
# #                 
# #                 
# #     #             kl = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
# #     #             kl_states[:,t] = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
# #     #             logit_x_t = self.emitter(z_t).contiguous() # p(x_t|z_t)         
# #     #             rec_loss = nn.BCEWithLogitsLoss(reduction='none')(logit_x_t.view(-1), x[:,t,:].contiguous().view(-1)).view(batch_size, -1)
# #                 rec_losses[curr_x_lens > 0,t-1] = rec_loss
# #                 
# # #                 rmse_loss = torch.sqrt(torch.sum((x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])**2, dim = 1))
# #                 
# #                 rmse_loss = (x_t*x_t_mask - logit_x_t*x_t_mask)**2
# #                 
# #                 mae_loss = torch.abs(x_t*x_t_mask - logit_x_t*x_t_mask)
# # 
# # #                 mae_losses = torch.abs(x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])
# # 
# #             
# #                 rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
# #                 
# #                 mae_losses[curr_x_lens > 0,:,t-1] = mae_loss
# 
# 
#             curr_x_lens -= 1
# 
#             shrinked_x_lens -= 1
# #             else:
# #                 z_t_category_gen = F.gumbel_softmax(self.emitter_z(h_now), tau=0.001, dim = 2)
#             
# #             if t >= 21:
# #                 print('here')
#             h_now_list.append(h_now)
#             
#             h_prev = h_now[:,shrinked_x_lens > 0,:]
#             
#             if (curr_x_lens == 0).any():
#                 last_h_now[:, curr_x_lens == 0, :] = h_now[:,shrinked_x_lens <= 0,:]
#                 last_rnn_out[curr_x_lens == 0] = rnn_out[curr_x_lens == 0,t,:]
#             if self.block == 'LSTM':
#                 c_prev = c_now[:,shrinked_x_lens > 0,:]
#                 
#                 if (curr_x_lens == 0).any():
#                     last_c_now[:, curr_x_lens == 0, :] = c_now[:,shrinked_x_lens <= 0,:] 
# #             phi_z_infer = torch.mm(joint_probs, torch.t(self.phi_table))
# #             if self.transfer_prob:
# #                 z_prev = z_t[shrinked_x_lens > 0,:]
# #             else:
# #                 z_prev = phi_z_infer[shrinked_x_lens > 0,:]
# 
# #             if not self.use_sparsemax:
# #                 
# # #                 print(t, torch.sum(shrinked_x_lens > 0))
# #                 z_t_category_gen = F.softmax(self.emitter_z(h_now[:,shrinked_x_lens > 0,:]), dim = 2)
# #             else:
# # #                 print(t, torch.sum(shrinked_x_lens > 0))
# #                 
# #                 if torch.sum(shrinked_x_lens > 0) > 0:
# #                 
# #                     logit_z_t = self.emitter_z(h_now[:,shrinked_x_lens > 0,:])
# #                     
# #                     z_t_category_gen = sparsemax(logit_z_t.view(logit_z_t.shape[1], logit_z_t.shape[2]))
# #                 
# #                     z_t_category_gen = z_t_category_gen.view(1, z_t_category_gen.shape[0], z_t_category_gen.shape[1])
#             
#             if t < T_max - 1:
#                 x_t = x[curr_x_lens > 0,t+1,:]
#                 
#                 x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
#                 
#                 
#                 curr_rnn_out = rnn_out[curr_x_lens > 0,t+1,:]
#                 
#             
#             
#             shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]
#         
#             
#         full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
#         
#         for k in range(self.cluster_num):
#             curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
#             
#             curr_rnn_input[:,k] = 1
#             
#             full_curr_rnn_input[k] = curr_rnn_input
#         
#         
#         curr_x_lens = x_lens.clone()
#         
#         
#         shrinked_x_lens = x_lens.clone()
#         
#         x_t = x[curr_x_lens > 0,0,:]
#                 
#         x_t_mask = x_mask[curr_x_lens > 0,0,:]
#         
#         
#         
#         
#         
#         decoded_h_n = None
#         
#         decoded_c_n = None
#         
#         decoded_h_n_gen = None
#         
#         decoded_c_n_gen = None
#         
#         for t in range(T_max):
#             
# #             print(t, torch.nonzero(curr_x_lens > 0).view(-1), joint_probs.shape)
# #             print(t, h_now_list[t])
#             
#             if t >= 1:
#                 
#                 if self.latent and not self.lstm_latent:
#                     input_x_t = self.x_kernel_encoder(x_t)
#                     if self.loss_on_missing:
#                         ae_loss[curr_x_lens > 0,t-1] = (self.x_kernel_decoder(input_x_t) - x_t)**2/(self.x_std**2)
#                     else:
#                         ae_loss[curr_x_lens > 0,t-1] = (self.x_kernel_decoder(input_x_t)*x_t_mask - x_t*x_t_mask)**2/(self.x_std**2)
#                 else:
#                     input_x_t = x_t
#                 
#                 
#                 
#                 
#                 full_rec_loss1, full_rec_loss2, full_logit_x_t, curr_full_rec_loss, l2_norm_loss, cluster_loss = self.compute_rec_loss(joint_probs[t, curr_x_lens > 0], prob_sums/torch.sum(x_lens), full_curr_rnn_input, rnn_out[curr_x_lens > 0,t,:], input_x_t, x_t_mask, curr_rnn_out)
#                 
#                 if self.latent:
#                     if not self.lstm_latent:
#                         full_logit_x_t = self.x_kernel_decoder(full_logit_x_t)
#                     else:
#                         full_logit_x_t, (decoded_h_n_gen, decoded_c_n_gen)  = self.x_kernel_decoder(full_logit_x_t.view(full_logit_x_t.shape[0], 1, full_logit_x_t.shape[1]), torch.ones(full_logit_x_t.shape[0], device = self.device), init_h = decoded_h_n_gen, init_c = decoded_c_n_gen)
#                 
#                         full_logit_x_t = full_logit_x_t.squeeze(1)
# #             print(t, full_rec_loss2.shape)
# 
#             
#             
#             
#                 
#                 rec_losses[curr_x_lens > 0,t-1] = full_rec_loss1
#                 
#                 cluster_losses[curr_x_lens > 0,t-1] = full_rec_loss2
#                 
#                 rec_losses_no_coeff[curr_x_lens > 0,t-1] = l2_norm_loss
#                 
#                 cluster_losses_no_coeff[curr_x_lens > 0,t-1] = cluster_loss
#                 
#  
#                 
# #                 print('time::', t)
#                 
#             
#                 rmse_loss = (x_t*x_t_mask - full_logit_x_t*x_t_mask)**2
#                 
#                 mae_loss = torch.abs(x_t*x_t_mask - full_logit_x_t*x_t_mask)
# 
# #                 mae_losses = torch.abs(x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])
# 
#             
#                 rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
#                 
#                 mae_losses[curr_x_lens > 0,:,t-1] = mae_loss
#             
#             
#             curr_rnn_out = rnn_out[curr_x_lens > 0,t,:]
#             
#             if self.latent and self.lstm_latent:
#                 
# #                 if t == 0:
# #                     rnn_decoded_out, (decoded_h_n, decoded_c_n) = self.x_kernel_decoder(curr_rnn_out.view(curr_rnn_out.shape[0], 1, curr_rnn_out.shape[1]), torch.ones(curr_rnn_out.shape[0], device = self.device))
# #                 else:
# #                     rnn_decoded_out, (decoded_h_n, decoded_c_n) = self.x_kernel_decoder(curr_rnn_out.view(curr_rnn_out.shape[0], 1, curr_rnn_out.shape[1]), torch.ones(curr_rnn_out.shape[0], device = self.device), init_h = decoded_h_n, init_c = decoded_c_n)
#                 
#                 if t == 0:
#                     rnn_decoded_out, (decoded_h_n, decoded_c_n) = self.x_kernel_decoder.rnn(curr_rnn_out.view(curr_rnn_out.shape[0], 1, curr_rnn_out.shape[1]))
#                 else:
#                     rnn_decoded_out, (decoded_h_n1, decoded_c_n1) = self.x_kernel_decoder.rnn(curr_rnn_out.view(curr_rnn_out.shape[0], 1, curr_rnn_out.shape[1]), (decoded_h_n, decoded_c_n))
# #                     rnn_decoded_out2, (decoded_h_n2, decoded_c_n2) = self.x_kernel_decoder(curr_rnn_out.view(curr_rnn_out.shape[0], 1, curr_rnn_out.shape[1]), torch.ones(curr_rnn_out.shape[0], device = self.device), init_h = decoded_h_n, init_c = decoded_c_n)
#                     
#                     
#                     decoded_h_n = decoded_h_n1
#                     decoded_c_n = decoded_c_n1
#                     
# #                     print(torch.norm(rnn_decoded_out - rnn_decoded_out2))
#                  
# #                 lstm_layer = nn.LSTM((1+self.x_encoder.bidir)*self.s_dim, self.input_dim, batch_first=True)    
#                 
# #                 full_rnn_decoded_out4, (last_decoded_h_n4, last_decoded_c_n4) = self.x_kernel_decoder.rnn(rnn_out[:,0:t+1])
# #                 
# #                 full_rnn_decoded_out3, (last_decoded_h_n3, last_decoded_c_n3) = self.x_kernel_decoder(rnn_out[:,0:t+1], (t+1)*torch.ones(curr_rnn_out.shape[0], device = self.device))
# #                 
# #                 for p in range(t+1):
# #                     if p == 0:
# #                         full_rnn_decoded_out5, (last_decoded_h_n5, last_decoded_c_n5) = self.x_kernel_decoder.rnn(rnn_out[:,p:p+1])
# #                     else:
# #                         full_rnn_decoded_out5, (last_decoded_h_n5, last_decoded_c_n5) = self.x_kernel_decoder.rnn(rnn_out[:,p:p+1], (last_decoded_h_n5, last_decoded_c_n5))
# #                 
# #                 print(torch.norm(rnn_decoded_out.squeeze(1) - full_rnn_decoded_out3[:,-1]))
#                 
#          
#                 decoded_h_n_list.append(decoded_h_n.clone())
#                  
#                 decoded_c_n_list.append(decoded_c_n.clone())
#                 
#                 curr_rnn_decoded_out = rnn_decoded_out.squeeze(1)
#                 
#                 if self.loss_on_missing:
#                     ae_loss[curr_x_lens > 0, t] = (curr_rnn_decoded_out - x_t)**2/(self.x_std**2)
#                 else:
#                     ae_loss[curr_x_lens > 0, t] = (curr_rnn_decoded_out*x_t_mask - x_t*x_t_mask)**2/(self.x_std**2)
#             
#             
#                 decoded_h_n_gen = decoded_h_n
#                 
#                 decoded_c_n_gen = decoded_c_n
#             
#             
#             curr_x_lens -= 1
#             shrinked_x_lens -=1
#             
#             
#             if t < T_max - 1:
#                 x_t = x[curr_x_lens > 0,t+1,:]
#                 
#                 x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
#                 
#             
#             if (curr_x_lens == 0).any():
#                 last_decoded_h_n[curr_x_lens == 0] = decoded_h_n[:, shrinked_x_lens == 0]
#                 last_decoded_c_n[curr_x_lens == 0] = decoded_c_n[:, shrinked_x_lens == 0] 
#                 
#             
#             if self.latent and self.lstm_latent:
#                 decoded_h_n = decoded_h_n[:, shrinked_x_lens > 0]
#                 decoded_c_n = decoded_c_n[:, shrinked_x_lens > 0]
#             
#             
#             
#             
#             shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]
#             
# #             curr_x_lens = curr_x_lens[curr_x_lens > 0]
# 
# #         x_mask = sequence_mask(x_lens)
# #         x_mask = x_mask.gt(0).view(-1)
# #         rec_loss = rec_losses.view(-1).masked_select(x_mask).mean()
# #         kl_loss = kl_states.view(-1).masked_select(x_mask).mean()
# 
#         
# 
# #         rec_loss = torch.sum(rec_losses)/torch.sum(x_lens-1)
#         
# #         full_cluster_objs, cluster_objs = self.compute_cluster_obj(cluster_distances, prob_sums/torch.sum(x_lens), T_max, x_lens, input_dim)
# #         
# #         
# #         if self.loss_on_missing:
# #              
# #             x_mask_full = torch.ones_like(x_mask)
# #              
# #             full_cluster_objs2, cluster_objs2 = self.compute_cluster_obj_full2(cluster_distances2, prob_sums/torch.sum(x_lens), T_max, x_mask_full, x_lens)
# #         else:
# #             full_cluster_objs2, cluster_objs2 = self.compute_cluster_obj_full2(cluster_distances2, prob_sums/torch.sum(x_lens), T_max, x_mask, x_lens)
#         
# #         print(torch.norm(full_cluster_objs - full_cluster_objs2))
# #          
# #         print(torch.norm(cluster_objs - cluster_objs2))
#         
# #         print(torch.norm(last_decoded_c_n - last_decoded_c_n2))
# #         
# #         print(torch.norm(last_decoded_h_n - last_decoded_h_n2))
#         
#         if (not self.loss_on_missing) and (not self.latent):
#             rec_loss1 = torch.sum(rec_losses)/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss1 = torch.sum(rec_losses)/torch.sum(torch.ones_like(rec_losses))
#         
#         if (not self.loss_on_missing) and (not self.latent):
#             rec_loss2 = torch.sum(cluster_losses)/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss2 = torch.sum(cluster_losses)/torch.sum(torch.ones_like(cluster_losses))
#         
#         
#         if (not self.loss_on_missing) and (not self.latent):
#             final_rec_loss = torch.sum(rec_losses_no_coeff)/torch.sum(x_mask[:,1:,:])
#         else:
#             final_rec_loss = torch.sum(rec_losses_no_coeff)/torch.sum(torch.ones_like(rec_losses_no_coeff))
#         
#         if (not self.loss_on_missing) and (not self.latent):
#             final_cluster_loss = torch.sum(cluster_losses_no_coeff)/torch.sum(x_mask[:,1:,:])
#         else:
#             final_cluster_loss = torch.sum(cluster_losses_no_coeff)/torch.sum(torch.ones_like(cluster_losses_no_coeff))
#         
#         
#         
# #         for k in range(rec_losses.shape[0]):
# #             print(rec_losses[k].mean())
#         
#         first_kl_loss = kl_states[:, 0].view(-1).mean()
#         
#         kl_loss = torch.sum(kl_states[:, 1:])/torch.sum(x_lens-1)
#         final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask[:,1:,:]))
#         
#         final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask[:,1:,:])
#         
#         
#         final_entropy_loss = entropy_losses.view(-1).mean()
#         print('loss::', final_rec_loss, kl_loss)
#         
#         print('loss with coefficient::', rec_loss1, kl_loss)
#         
#         print('rmse loss::', final_rmse_loss)
#         
#         print('mae loss::', final_mae_losses)
#         
#         print('cluster objective::', final_cluster_loss)
#         
#         print('cluster objective with coefficient::', rec_loss2)
#         
#         
#         final_ae_loss = 0
#         if self.latent:
#             if not self.lstm_latent:
#                 if not self.loss_on_missing:
#                     final_ae_loss = torch.sum(ae_loss)/torch.sum(x_mask[:,1:,:])
#                 else:
#                     final_ae_loss = torch.sum(ae_loss)/torch.sum(torch.ones_like(ae_loss))
#             else:
# #                 ae_loss = (x - rnn_decoded_out)**2
#                 
#                 if not self.loss_on_missing:
#                     final_ae_loss = torch.sum(ae_loss*x_mask)/torch.sum(x_mask)
#                 else:
#                     final_ae_loss = torch.sum(ae_loss)/torch.sum(torch.ones_like(ae_loss))
# 
#                 
#                 
#             print('autoencoder loss::', final_ae_loss)
#         
#         imputed_loss = 0
#         
#         if self.is_missing:
#             imputed_loss = torch.norm(interpolated_x*x_mask - x*x_mask)
#             
#             print('interpolate loss::', imputed_loss)
#         
#         if torch.sum(1-new_x_mask) > 0:
#             
#             imputed_mse_loss = torch.sqrt(torch.sum((((origin_x - x)**2)*(1-new_x_mask)))/torch.sum(1-new_x_mask))
#             
#             imputed_mse_loss2 = torch.sqrt(torch.sum((((origin_x - imputed_x2)**2)*(1-new_x_mask)))/torch.sum(1-new_x_mask)) 
#             
#             imputed_loss = torch.sum((torch.abs(origin_x - x)*(1-new_x_mask)))/torch.sum(1-new_x_mask)
#             
#             imputed_loss2 = torch.sum((torch.abs(origin_x - imputed_x2)*(1-new_x_mask)))/torch.sum(1-new_x_mask) 
#             
#             print('training imputation rmse loss::', imputed_mse_loss)
#             
#             print('training imputation rmse loss 2::', imputed_mse_loss2)
#             
#             print('training imputation mae loss::', imputed_loss)
#             
#             print('training imputation mae loss 2::', imputed_loss2)
#             
#         
#         
#         prior_cluster_probs = prob_sums/torch.sum(x_lens)
#         
#         if self.block == 'GRU':
#             self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, None, T_max, prior_cluster_probs, last_rnn_out, last_h_n, last_c_n, last_decoded_h_n.unsqueeze(0), last_decoded_c_n.unsqueeze(0))
#         else:
#             self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, last_c_now, T_max, prior_cluster_probs, last_rnn_out, last_h_n, last_c_n, last_decoded_h_n.unsqueeze(0), last_decoded_c_n.unsqueeze(0))
#         
# 
# 
#         
#         print()
#         
#         if not self.evaluate:
#             return rec_loss1, kl_loss, first_kl_loss, final_rmse_loss, imputed_loss, rec_loss2, final_ae_loss
#         else:
#             
#             if torch.sum(1-new_x_mask) > 0:
#                 return imputed_x2*(1-x_mask) + x*x_mask, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
#             else:
#                 return imputed_x2*(1-x_mask) + x*x_mask, None


    def test_samples_back(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_delta_time_stamps, x_to_predict_delta_time_stamps, x_time_stamps, x_to_predict_time_stamps):
        """
        infer q(z_{1:T}|x_{1:T}) (i.e. the variational distribution)
        """
        
#         assert torch.sum(x_mask) == torch.sum(1-np.isnan(x))
#         x_lens[:]=1

        T_max = x_lens.max().item()
        if is_GPU:
            x = x.to(device)
            
            x_to_predict = x_to_predict.to(device)
            
            x_mask = x_mask.to(device)
            
            x_to_predict_mask = x_to_predict_mask.to(device)
            
            origin_x = origin_x.to(device)
            
            origin_x_to_pred = origin_x_to_pred.to(device)
            
            
            origin_x_mask = origin_x_mask.to(device)
            
            new_x_mask = new_x_mask.to(device)
            
            
            
            x_to_predict_origin_mask = x_to_predict_origin_mask.to(device)
            
            x_to_predict_new_mask = x_to_predict_new_mask.to(device) 
        
            x_lens = x_lens.to(device)
        
            x_to_predict_lens = x_to_predict_lens.to(device)
        
        if self.is_missing:
            
#             t1 = time.time()
#             
#             imputed_x = self.impute(x, x_mask, T_max)
#             
#             t2 = time.time()
            
            
            imputed_x, interpolated_x = self.impute.forward2(x, x_mask, T_max)

            
#             t3 = time.time()
#             
#             print(t3 - t2)
#             print(t2 - t1)
                    
            x = imputed_x
        
        batch_size, _, input_dim = x.size()
        
        h_0 = self.h_0.expand(1, batch_size, self.h_dim).contiguous()
        
        
        
        
#         c_0 = self.c_0.expand(1, batch_size, self.s_dim).contiguous()
        
        
        h_prev = self.h_0.expand(1, batch_size, self.h_0.size(0))
        
        c_prev = self.h_0.expand(1, batch_size, self.h_0.size(0))
        
        
        z_1_category = torch.ones(self.cluster_num, dtype = torch.float, device = x.device)/self.cluster_num
        
        z_t_category_gen = z_1_category.expand(1, batch_size, z_1_category.size(0))
        
#         phi_z, z_representation = self.generate_z(z_t_category_gen, 0)
        
        
#         if np.any(x_lens.numpy()<= 0):
#             print(torch.nonzero(x_lens <= 0))
#             print('here')
        
#         print(torch.nonzero(x_lens <= 0))
        
#         if not self.latent:
        rnn_out,(last_h_n, last_c_n)= self.x_encoder(x, x_lens) # push the observed x's through the rnn;
#         else:
#             if not self.lstm_latent:
#                 rnn_out,(last_h_n, last_c_n)= self.x_encoder(self.x_kernel_encoder(x), x_lens) # push the observed x's through the rnn;
#             else:
#                 rnn_out,(last_h_n, last_c_n)= self.x_encoder(x, x_lens) # push the observed x's through the rnn;
                
                
        
#         self.x_encoder.forward2(x, x_lens, rnn_out, last_h_n, last_c_n)
        
        
        '''to be done'''
#         rnn_out2,(last_h_n2, last_c_n2)= self.x_encoder.forward2(x, x_lens) # push the observed x's through the rnn;
        
#         print(torch.norm(rnn_out - rnn_out2), torch.norm(last_h_n - last_h_n2), torch.norm(last_c_n - last_c_n2))
        
#         rnn_out = reverse_sequence(rnn_out, x_lens) # reverse the time-ordering in the hidden state and un-pack it
        
#         if self.latent and self.lstm_latent:
#             rec_losses = torch.zeros((batch_size, T_max-1, (1+self.x_encoder.bidir)*self.s_dim), device=x.device)
#         
#             cluster_losses = torch.zeros((batch_size, T_max  -1, (1+self.x_encoder.bidir)*self.s_dim), device=x.device) 
#             
#             rec_losses_no_coeff = torch.zeros((batch_size, T_max-1, (1+self.x_encoder.bidir)*self.s_dim), device=x.device)
#             
#             cluster_losses_no_coeff = torch.zeros((batch_size, T_max  -1, (1+self.x_encoder.bidir)*self.s_dim), device=x.device)
#             
#         else:
            
        rec_losses = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
        
        cluster_losses = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
        
        rec_losses_no_coeff = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
        
        cluster_losses_no_coeff = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
         
        kl_states = torch.zeros((batch_size, T_max), device=x.device)
        
        rmse_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
        
        mae_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)   
        
        
        cluster_distances = torch.zeros([batch_size, T_max, self.cluster_num], device = x.device)
        cluster_distances2 = torch.zeros([T_max, self.cluster_num, batch_size, input_dim], device = x.device)
        
        prob_sums = 0
        
#         negnill = torch.zeros()
        
        entropy_losses = torch.zeros((batch_size, T_max), device = x.device)
        
        '''z_q_*''' 
        z_prev = self.z_q_0.expand(batch_size,self.z_q_0.size(0)) # set z_prev=z_q_0 to setup the recursive conditioning in q(z_t|...)
        
        curr_x_lens = x_lens.clone()
        
        shrinked_x_lens = x_lens.clone()
        
        x_t = 0
        
        x_t_mask = 0
        
        curr_rnn_out = rnn_out[curr_x_lens > 0,0,:]
        
        single_time_steps = torch.ones_like(curr_x_lens)
        
        last_h_now = torch.zeros_like(h_prev)
        
        last_c_now = torch.zeros_like(c_prev)
        
        imputed_x2 = torch.zeros_like(x)
        
        imputed_x2[:,0] = x[:,0] 
        
        last_rnn_out = torch.zeros(batch_size, self.s_dim*(1+self.x_encoder.bidir), device = self.device)
        
        joint_probs = torch.zeros([T_max, batch_size, self.cluster_num], dtype = torch.float, device = self.device)
        
        h_now_list = []
        
        decoded_h_n_list = []
        
        decoded_c_n_list = []
        
        last_decoded_h_n = torch.zeros([batch_size, self.z_dim], device = self.device)
        
        last_decoded_c_n = torch.zeros([batch_size, self.z_dim], device = self.device)
        
#         if self.latent:
#             if not self.lstm_latent:
#                 ae_loss = torch.zeros((batch_size, T_max-1, input_dim), device=x.device)
#             
#             if self.lstm_latent:
#                 ae_loss = torch.zeros((batch_size, T_max, input_dim), device=x.device)
        
        
#             if self.lstm_latent:
#                 full_rnn_decoded_out, (last_decoded_h_n2, last_decoded_c_n2) = self.x_kernel_decoder(rnn_out, x_lens)
        
        
        for t in range(T_max):
#             z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)# p(z_t| z_{t-1})
            
            '''phi_z_infer: phi_{z_t}'''
            
            
            
            if t == 0:
                z_t, z_t_category_infer, _, z_category_infer_sparse = self.postnet(z_prev, curr_rnn_out, self.phi_table, t, self.temp) #q(z_t | z_{t-1}, x_{t:T})
                
                joint_probs[t] = z_t_category_infer
                
                kl = self.kl_div(z_t_category_infer, z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))
                
                if np.isnan(kl.cpu().detach().numpy()).any():
                    print('distribution 1::', z_t_category_gen)
                    
                    print('distribution 2::', z_t_category_infer)
                
                
#                 if self.use_sparsemax:
#                     z_t_category_trans =  sparsemax(torch.log(z_t_category_infer+1e-5))
#                 else:
                z_t_category_trans =  z_t_category_infer
                
                if self.transfer_prob:
                    
                    if self.block == 'GRU':
        #                 output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
                        output, h_now = self.trans(z_t_category_trans.view(z_t_category_trans.shape[0], 1, z_t_category_trans.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
                        
                    else:
                        output, (h_now, c_now) = self.trans(z_t_category_trans.view(z_t_category_trans.shape[0], 1, z_t_category_trans.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
                        
    #                 output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
                else:
                    
                    phi_z_infer = torch.mm(z_t_category_trans, torch.t(self.phi_table))
                    
                    if self.block == 'GRU':
                        output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
        #                     output, h_now = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
                        
                    else:
        #                     output, (h_now, c_now) = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
                        output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
                    
                    
#                 kl_states[curr_x_lens > 0,t] = kl
                
            else:
                
                
                '''joint_probs, curr_rnn_output, batch_size, t, h_prev, c_prev, shrinked_x_lens'''
                '''updated_joint_probs, full_kl, h_now, c_now, full_rec_loss, full_logit_x_t'''
                updated_joint_probs, kl, h_now, c_now, z_t_category_infer = self.update_joint_probability2(joint_probs[t-1, curr_x_lens > 0], curr_rnn_out, torch.sum(curr_x_lens > 0), t, h_prev, c_prev,z_t_category_infer, shrinked_x_lens, x_t, x_t_mask)
                
#                 self.optimizer.zero_grad()
#                  
#                 torch.sum(full_rec_loss).backward(retain_graph=True)
                
                joint_probs[t, curr_x_lens > 0] = updated_joint_probs
                
#                 self.optimizer.zero_grad()
#                  
#                 torch.sum(full_rec_loss).backward(retain_graph=True)
                
                
                
                
                kl_states[curr_x_lens > 0,t] = kl
            
            
            
#             phi_z_infer = torch.mm(z_t, torch.t(self.phi_table))
#             phi_z_infer = torch.mm(z_t_category_infer, torch.t(self.phi_table))
#             
#             phi_z_infer2 = torch.mm(z_t, torch.t(self.phi_table))
#             
#             
#             if self.use_gumbel:
#                 print(t, torch.norm(phi_z_infer - phi_z_infer2))
            
            prob_sums += torch.sum(z_t_category_infer, 0)
            
            
            
#             curr_cluster_distance_res = 0
#             
#             if self.loss_on_missing:
#                  
#                 curr_x_mask = torch.ones_like(x_mask[curr_x_lens > 0,t,:])
#                  
#                 curr_cluster_distance_res = self.compute_distance_per_cluster_all(x[curr_x_lens > 0,t,:], curr_x_mask)
#                  
#                 cluster_distances2[t, :, curr_x_lens > 0] = curr_cluster_distance_res
#             else:
#                  
#                 curr_cluster_distance_res = self.compute_distance_per_cluster_all(x[curr_x_lens > 0,t,:], x_mask[curr_x_lens > 0,t,:])
#                  
#                 cluster_distances2[t, :, curr_x_lens > 0] = curr_cluster_distance_res 
#             
#             
#             sumed_cluster_distnace_res = torch.sqrt(torch.sum(curr_cluster_distance_res, 2))
#             
#             cluster_distances[curr_x_lens > 0, t] = self.compute_distance_per_cluster(x[curr_x_lens > 0,t,:])
#             
# #             print(torch.norm(torch.t(sumed_cluster_distnace_res) - cluster_distances[curr_x_lens > 0, t]))
# #             if self.use_sparsemax:
#             
# #             else:
# #                 kl = self.kl_div(z_t_category_infer, z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))
#             
#             entropy_loss = self.entropy(z_t_category_infer)
#             
#             
#             
#             entropy_losses[curr_x_lens > 0, t] = entropy_loss
#             
#             
#                
#             
# #             print(t, curr_x_lens)
#             
#             if self.transfer_prob:
#                 
#                 z_t_transfer = z_t_category_infer
#                 
#                 if self.block == 'GRU':
#     #                 output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
#                     output, h_now = self.trans(z_t_transfer.view(z_t_transfer.shape[0], 1, z_t_transfer.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
#                     
#                 else:
#                     output, (h_now, c_now) = self.trans(z_t_transfer.view(z_t_transfer.shape[0], 1, z_t_transfer.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
# #                 output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#             else:
#                 if self.block == 'GRU':
#                     output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
# #                     output, h_now = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
#                     
#                 else:
# #                     output, (h_now, c_now) = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#                     output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#                 
#                 
# #             if not self.use_sparsemax:
# 
# 
#             
#             
#             
# #             curr_x_lens[curr_x_lens < 0] = 0
#             
#             
#             if t >= 1:            
#             
#                 phi_z, z_representation = self.generate_z(z_t_category_gen, t)
#                 
# #                 phi_z = torch.t(torch.mm(self.phi_table, torch.t(z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))))
# #                 
# #                 phi_z2 = torch.t(torch.mm(self.phi_table, torch.t(z_representation)))
# #                 
# #                 if self.use_gumbel:
# #                     print(torch.norm(phi_z - phi_z2))
#                 
#                 
#                 mean, logvar, logit_x_t = self.generate_x(phi_z_infer, z_t)
#                 
#                 
#                 
#                 
#     
#                 imputed_x2[curr_x_lens > 0,t] = logit_x_t
#                 
#                 
# #                 rec_loss = torch.norm(x[:,t+1,:] - mean)**2/(2*std**2) + torch.log(2*np.pi*std**2)/2
#                 
# #                 rec_loss = torch.bmm(((x[:,t,:]-mean)/(std**2)).view(mean.shape[0],1,mean.shape[1]), (x[:,t,:]-mean).view(mean.shape[0],mean.shape[1],1)).view(-1) + (torch.log((2*np.pi)**x[:,t,:].shape[-1]*torch.prod(std, dim= 1))/2).view(-1) 
#                 
#                 if self.loss_on_missing:
#                     
#                     curr_x_t_masks = torch.ones_like(x_t)
#                     
#                     rec_loss = compute_gaussian_probs0(x_t, mean, logvar, curr_x_t_masks)
#                 else:
#                     rec_loss = compute_gaussian_probs0(x_t, mean, logvar, x_t_mask)
#                 
# #                 rec_loss = self.compute_reconstruction_loss2(x_t, mean, std, batch_size)
#                 
#                 
#     #             kl = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
#     #             kl_states[:,t] = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
#     #             logit_x_t = self.emitter(z_t).contiguous() # p(x_t|z_t)         
#     #             rec_loss = nn.BCEWithLogitsLoss(reduction='none')(logit_x_t.view(-1), x[:,t,:].contiguous().view(-1)).view(batch_size, -1)
#                 rec_losses[curr_x_lens > 0,t-1] = rec_loss
#                 
# #                 rmse_loss = torch.sqrt(torch.sum((x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])**2, dim = 1))
#                 
#                 rmse_loss = (x_t*x_t_mask - logit_x_t*x_t_mask)**2
#                 
#                 mae_loss = torch.abs(x_t*x_t_mask - logit_x_t*x_t_mask)
# 
# #                 mae_losses = torch.abs(x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])
# 
#             
#                 rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
#                 
#                 mae_losses[curr_x_lens > 0,:,t-1] = mae_loss


            curr_x_lens -= 1

            shrinked_x_lens -= 1
#             else:
#                 z_t_category_gen = F.gumbel_softmax(self.emitter_z(h_now), tau=0.001, dim = 2)
            
#             if t >= 21:
#                 print('here')
            h_now_list.append(h_now)
            
            h_prev = h_now[:,shrinked_x_lens > 0,:]
            
            if (curr_x_lens == 0).any():
                last_h_now[:, curr_x_lens == 0, :] = h_now[:,shrinked_x_lens <= 0,:]
                last_rnn_out[curr_x_lens == 0] = rnn_out[curr_x_lens == 0,t,:]
            if self.block == 'LSTM':
                c_prev = c_now[:,shrinked_x_lens > 0,:]
                
                if (curr_x_lens == 0).any():
                    last_c_now[:, curr_x_lens == 0, :] = c_now[:,shrinked_x_lens <= 0,:] 
#             phi_z_infer = torch.mm(joint_probs, torch.t(self.phi_table))
#             if self.transfer_prob:
#                 z_prev = z_t[shrinked_x_lens > 0,:]
#             else:
#                 z_prev = phi_z_infer[shrinked_x_lens > 0,:]

#             if not self.use_sparsemax:
#                 
# #                 print(t, torch.sum(shrinked_x_lens > 0))
#                 z_t_category_gen = F.softmax(self.emitter_z(h_now[:,shrinked_x_lens > 0,:]), dim = 2)
#             else:
# #                 print(t, torch.sum(shrinked_x_lens > 0))
#                 
#                 if torch.sum(shrinked_x_lens > 0) > 0:
#                 
#                     logit_z_t = self.emitter_z(h_now[:,shrinked_x_lens > 0,:])
#                     
#                     z_t_category_gen = sparsemax(logit_z_t.view(logit_z_t.shape[1], logit_z_t.shape[2]))
#                 
#                     z_t_category_gen = z_t_category_gen.view(1, z_t_category_gen.shape[0], z_t_category_gen.shape[1])
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
                
                
                curr_rnn_out = rnn_out[curr_x_lens > 0,t+1,:]
                
            
            
            shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]
        
            
        full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
        
        for k in range(self.cluster_num):
            curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            curr_rnn_input[:,k] = 1
            
            full_curr_rnn_input[k] = curr_rnn_input
        
        
        curr_x_lens = x_lens.clone()
        
        
        shrinked_x_lens = x_lens.clone()
        
        x_t = x[curr_x_lens > 0,0,:]
                
        x_t_mask = x_mask[curr_x_lens > 0,0,:]
        
        
        
        
        
        decoded_h_n = None
        
        decoded_c_n = None
        
        decoded_h_n_gen = None
        
        decoded_c_n_gen = None
        
        for t in range(T_max):
            
#             print(t, torch.nonzero(curr_x_lens > 0).view(-1), joint_probs.shape)
#             print(t, h_now_list[t])
            
            if t >= 1:
                
#                 if self.latent and not self.lstm_latent:
#                     input_x_t = self.x_kernel_encoder(x_t)
#                     if self.loss_on_missing:
#                         ae_loss[curr_x_lens > 0,t-1] = (self.x_kernel_decoder(input_x_t) - x_t)**2/(self.x_std**2)
#                     else:
#                         ae_loss[curr_x_lens > 0,t-1] = (self.x_kernel_decoder(input_x_t)*x_t_mask - x_t*x_t_mask)**2/(self.x_std**2)
#                 else:
                input_x_t = x_t
            
                
                
                
                full_rec_loss1, full_rec_loss2, full_logit_x_t, curr_full_rec_loss, l2_norm_loss, cluster_loss = self.compute_rec_loss(joint_probs[t, curr_x_lens > 0], prob_sums/torch.sum(x_lens), full_curr_rnn_input, rnn_out[curr_x_lens > 0,t,:], input_x_t, x_t_mask, curr_rnn_out)
                
#                 if self.latent:
#                     if not self.lstm_latent:
#                         full_logit_x_t = self.x_kernel_decoder(full_logit_x_t)
#                     else:
#                         full_logit_x_t, (decoded_h_n_gen, decoded_c_n_gen)  = self.x_kernel_decoder(full_logit_x_t.view(full_logit_x_t.shape[0], 1, full_logit_x_t.shape[1]), torch.ones(full_logit_x_t.shape[0], device = self.device), init_h = decoded_h_n_gen, init_c = decoded_c_n_gen)
#                 
#                         full_logit_x_t = full_logit_x_t.squeeze(1)
#             print(t, full_rec_loss2.shape)

            
            
            
                
                rec_losses[curr_x_lens > 0,t-1] = full_rec_loss1
                
                cluster_losses[curr_x_lens > 0,t-1] = full_rec_loss2
                
                rec_losses_no_coeff[curr_x_lens > 0,t-1] = l2_norm_loss
                
                cluster_losses_no_coeff[curr_x_lens > 0,t-1] = cluster_loss
                
 
                
#                 print('time::', t)
                
            
                rmse_loss = (x_t*x_t_mask - full_logit_x_t*x_t_mask)**2
                
                mae_loss = torch.abs(x_t*x_t_mask - full_logit_x_t*x_t_mask)

#                 mae_losses = torch.abs(x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])

            
                rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
                
                mae_losses[curr_x_lens > 0,:,t-1] = mae_loss
            
            
            curr_rnn_out = rnn_out[curr_x_lens > 0,t,:]
            
#             if self.latent and self.lstm_latent:
#                 
# #                 if t == 0:
# #                     rnn_decoded_out, (decoded_h_n, decoded_c_n) = self.x_kernel_decoder(curr_rnn_out.view(curr_rnn_out.shape[0], 1, curr_rnn_out.shape[1]), torch.ones(curr_rnn_out.shape[0], device = self.device))
# #                 else:
# #                     rnn_decoded_out, (decoded_h_n, decoded_c_n) = self.x_kernel_decoder(curr_rnn_out.view(curr_rnn_out.shape[0], 1, curr_rnn_out.shape[1]), torch.ones(curr_rnn_out.shape[0], device = self.device), init_h = decoded_h_n, init_c = decoded_c_n)
#                 
#                 if t == 0:
#                     rnn_decoded_out, (decoded_h_n, decoded_c_n) = self.x_kernel_decoder.rnn(curr_rnn_out.view(curr_rnn_out.shape[0], 1, curr_rnn_out.shape[1]))
#                 else:
#                     rnn_decoded_out, (decoded_h_n1, decoded_c_n1) = self.x_kernel_decoder.rnn(curr_rnn_out.view(curr_rnn_out.shape[0], 1, curr_rnn_out.shape[1]), (decoded_h_n, decoded_c_n))
# #                     rnn_decoded_out2, (decoded_h_n2, decoded_c_n2) = self.x_kernel_decoder(curr_rnn_out.view(curr_rnn_out.shape[0], 1, curr_rnn_out.shape[1]), torch.ones(curr_rnn_out.shape[0], device = self.device), init_h = decoded_h_n, init_c = decoded_c_n)
#                     
#                     
#                     decoded_h_n = decoded_h_n1
#                     decoded_c_n = decoded_c_n1
#                     
# #                     print(torch.norm(rnn_decoded_out - rnn_decoded_out2))
#                  
# #                 lstm_layer = nn.LSTM((1+self.x_encoder.bidir)*self.s_dim, self.input_dim, batch_first=True)    
#                 
# #                 full_rnn_decoded_out4, (last_decoded_h_n4, last_decoded_c_n4) = self.x_kernel_decoder.rnn(rnn_out[:,0:t+1])
# #                 
# #                 full_rnn_decoded_out3, (last_decoded_h_n3, last_decoded_c_n3) = self.x_kernel_decoder(rnn_out[:,0:t+1], (t+1)*torch.ones(curr_rnn_out.shape[0], device = self.device))
# #                 
# #                 for p in range(t+1):
# #                     if p == 0:
# #                         full_rnn_decoded_out5, (last_decoded_h_n5, last_decoded_c_n5) = self.x_kernel_decoder.rnn(rnn_out[:,p:p+1])
# #                     else:
# #                         full_rnn_decoded_out5, (last_decoded_h_n5, last_decoded_c_n5) = self.x_kernel_decoder.rnn(rnn_out[:,p:p+1], (last_decoded_h_n5, last_decoded_c_n5))
# #                 
# #                 print(torch.norm(rnn_decoded_out.squeeze(1) - full_rnn_decoded_out3[:,-1]))
#                 
#          
#                 decoded_h_n_list.append(decoded_h_n.clone())
#                  
#                 decoded_c_n_list.append(decoded_c_n.clone())
#                 
#                 curr_rnn_decoded_out = rnn_decoded_out.squeeze(1)
#                 
#                 if self.loss_on_missing:
#                     ae_loss[curr_x_lens > 0, t] = (curr_rnn_decoded_out - x_t)**2
#                 else:
#                     ae_loss[curr_x_lens > 0, t] = (curr_rnn_decoded_out*x_t_mask - x_t*x_t_mask)**2
#             
#             
#                 decoded_h_n_gen = decoded_h_n
#                 
#                 decoded_c_n_gen = decoded_c_n
            
            
            curr_x_lens -= 1
            shrinked_x_lens -=1
            
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
                
            
            if (curr_x_lens == 0).any():
                last_decoded_h_n[curr_x_lens == 0] = decoded_h_n[:, shrinked_x_lens == 0]
                last_decoded_c_n[curr_x_lens == 0] = decoded_c_n[:, shrinked_x_lens == 0] 
                
            
#             if self.latent and self.lstm_latent:
#                 decoded_h_n = decoded_h_n[:, shrinked_x_lens > 0]
#                 decoded_c_n = decoded_c_n[:, shrinked_x_lens > 0]
            
            
            
            
            shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]
            
#             curr_x_lens = curr_x_lens[curr_x_lens > 0]

#         x_mask = sequence_mask(x_lens)
#         x_mask = x_mask.gt(0).view(-1)
#         rec_loss = rec_losses.view(-1).masked_select(x_mask).mean()
#         kl_loss = kl_states.view(-1).masked_select(x_mask).mean()

        

#         rec_loss = torch.sum(rec_losses)/torch.sum(x_lens-1)
        
#         full_cluster_objs, cluster_objs = self.compute_cluster_obj(cluster_distances, prob_sums/torch.sum(x_lens), T_max, x_lens, input_dim)
#         
#         
#         if self.loss_on_missing:
#              
#             x_mask_full = torch.ones_like(x_mask)
#              
#             full_cluster_objs2, cluster_objs2 = self.compute_cluster_obj_full2(cluster_distances2, prob_sums/torch.sum(x_lens), T_max, x_mask_full, x_lens)
#         else:
#             full_cluster_objs2, cluster_objs2 = self.compute_cluster_obj_full2(cluster_distances2, prob_sums/torch.sum(x_lens), T_max, x_mask, x_lens)
        
#         print(torch.norm(full_cluster_objs - full_cluster_objs2))
#          
#         print(torch.norm(cluster_objs - cluster_objs2))
        
#         print(torch.norm(last_decoded_c_n - last_decoded_c_n2))
#         
#         print(torch.norm(last_decoded_h_n - last_decoded_h_n2))
        
#         if (not self.loss_on_missing) and (not self.latent):
        rec_loss1 = torch.sum(rec_losses)/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss1 = torch.sum(rec_losses)/torch.sum(torch.ones_like(rec_losses))
        
#         if (not self.loss_on_missing) and (not self.latent):
        rec_loss2 = torch.sum(cluster_losses)/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss2 = torch.sum(cluster_losses)/torch.sum(torch.ones_like(cluster_losses))
        
        
#         if (not self.loss_on_missing) and (not self.latent):
        final_rec_loss = torch.sum(rec_losses_no_coeff)/torch.sum(x_mask[:,1:,:])
#         else:
#             final_rec_loss = torch.sum(rec_losses_no_coeff)/torch.sum(torch.ones_like(rec_losses_no_coeff))
        
#         if (not self.loss_on_missing) and (not self.latent):
        final_cluster_loss = torch.sum(cluster_losses_no_coeff)/torch.sum(x_mask[:,1:,:])
#         else:
#             final_cluster_loss = torch.sum(cluster_losses_no_coeff)/torch.sum(torch.ones_like(cluster_losses_no_coeff))
        
        
        
#         for k in range(rec_losses.shape[0]):
#             print(rec_losses[k].mean())
        
        first_kl_loss = kl_states[:, 0].view(-1).mean()
        
        kl_loss = torch.sum(kl_states[:, 1:])/torch.sum(x_lens-1)
        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask[:,1:,:]))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask[:,1:,:])
        
        
        final_entropy_loss = entropy_losses.view(-1).mean()
        print('loss::', final_rec_loss, kl_loss)
        
        print('loss with coefficient::', rec_loss1, kl_loss)
        
        print('rmse loss::', final_rmse_loss)
        
        print('mae loss::', final_mae_losses)
        
        print('cluster objective::', final_cluster_loss)
        
        print('cluster objective with coefficient::', rec_loss2)
        
        
        final_ae_loss = 0
#         if self.latent:
#             if not self.lstm_latent:
#                 if not self.loss_on_missing:
#                     final_ae_loss = torch.sum(ae_loss)/torch.sum(x_mask[:,1:,:])
#                 else:
#                     final_ae_loss = torch.sum(ae_loss)/torch.sum(torch.ones_like(ae_loss))
#             else:
# #                 ae_loss = (x - rnn_decoded_out)**2
#                 
#                 if not self.loss_on_missing:
#                     final_ae_loss = torch.sum(ae_loss*x_mask)/torch.sum(x_mask)
#                 else:
#                     final_ae_loss = torch.sum(ae_loss)/torch.sum(torch.ones_like(ae_loss))
# 
#                 
#                 
#             print('autoencoder loss::', final_ae_loss)
        
        imputed_loss = 0
        
        if self.is_missing:
            imputed_loss = torch.norm(interpolated_x*x_mask - x*x_mask)
            
            print('interpolate loss::', imputed_loss)
        
        if torch.sum(1-new_x_mask) > 0:
            
            imputed_mse_loss = torch.sqrt(torch.sum((((origin_x - x)**2)*(1-new_x_mask)))/torch.sum(1-new_x_mask))
            
            imputed_mse_loss2 = torch.sqrt(torch.sum((((origin_x - imputed_x2)**2)*(1-new_x_mask)))/torch.sum(1-new_x_mask)) 
            
            imputed_loss = torch.sum((torch.abs(origin_x - x)*(1-new_x_mask)))/torch.sum(1-new_x_mask)
            
            imputed_loss2 = torch.sum((torch.abs(origin_x - imputed_x2)*(1-new_x_mask)))/torch.sum(1-new_x_mask) 
            
            print('training imputation rmse loss::', imputed_mse_loss)
            
            print('training imputation rmse loss 2::', imputed_mse_loss2)
            
            print('training imputation mae loss::', imputed_loss)
            
            print('training imputation mae loss 2::', imputed_loss2)
            
        
        
        prior_cluster_probs = prob_sums/torch.sum(x_lens)
        
        if not self.evaluate:
            final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count = self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, last_c_now, T_max, prior_cluster_probs, last_rnn_out, last_h_n, last_c_n, last_decoded_h_n.unsqueeze(0), last_decoded_c_n.unsqueeze(0))
        else:
            imputed_x2 = self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, last_c_now, T_max, prior_cluster_probs, last_rnn_out,last_h_n, last_c_n, last_decoded_h_n.unsqueeze(0), last_decoded_c_n.unsqueeze(0))
        
        print()
        
        if not os.path.exists(data_folder + '/' + output_dir):
            os.makedirs(data_folder + '/' + output_dir)
        
        
        torch.save(self.phi_table, data_folder + '/' + output_dir + 'cluster_centroids')
        
#         torch.save(all_mean, data_folder + '/' + output_dir + 'all_mean')
#         
#         torch.save(all_log_var, data_folder + '/' + output_dir + 'all_log_var')
#         
#         torch.save(all_probs, data_folder + '/' + output_dir + 'all_probs')
        
        
        
#         del x, x_to_predict,origin_x, origin_x_to_predict, x_mask, x_to_predict_mask, x_origin_mask, x_new_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_lens, x_to_predict_lens
        
        
        
        
        
        
        
        
        
        if not self.evaluate:
            return final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count
        else:
            
            if torch.sum(1-new_x_mask) > 0:
                return imputed_x2, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
            else:
                return imputed_x2, None


    '''self, x, origin_x, x_mask, x_origin_mask, x_new_mask, x_lens, x_to_predict, origin_x_to_predict, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_delta_time_stamps, x_to_predict_delta_time_stamps, x_time_stamps, x_to_predict_time_stamps'''

    def test_samples0(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_delta_time_stamps, x_to_predict_delta_time_stamps, x_time_stamps, x_to_predict_time_stamps):
        """
        infer q(z_{1:T}|x_{1:T}) (i.e. the variational distribution)
        """
        
#         assert torch.sum(x_mask) == torch.sum(1-np.isnan(x))
        T_max = x_lens.max().item()
        if is_GPU:
            x = x.to(device)
            
            x_to_predict = x_to_predict.to(device)
            
            x_mask = x_mask.to(device)
            
            x_to_predict_mask = x_to_predict_mask.to(device)
            
            origin_x = origin_x.to(device)
            
            origin_x_to_pred = origin_x_to_pred.to(device)
            
            
            origin_x_mask = origin_x_mask.to(device)
            
            new_x_mask = new_x_mask.to(device)
            
            
            
            x_to_predict_origin_mask = x_to_predict_origin_mask.to(device)
            
            x_to_predict_new_mask = x_to_predict_new_mask.to(device) 
        
            x_lens = x_lens.to(device)
        
            x_to_predict_lens = x_to_predict_lens.to(device)
        
        if self.is_missing:
            
#             t1 = time.time()
#             
#             imputed_x = self.impute(x, x_mask, T_max)
#             
#             t2 = time.time()
            
            
#             imputed_x, interpolated_x = self.impute.forward2(x, x_mask, T_max)
# 
#             
# #             t3 = time.time()
# #             
# #             print(t3 - t2)
# #             print(t2 - t1)
#                     
#             x = imputed_x
            if self.pre_impute:
                imputed_x, interpolated_x = self.impute.forward2(x, x_mask, x_time_stamps[0]*100)
    # 
    #         
    # #             t3 = time.time()
    # #             
    # #             print(t3 - t2)
    # #             print(t2 - t1)
    #                     
                x = imputed_x
    #             
#             if torch.isnan(x).any():
#                 print(torch.nonzero(torch.isnan(x)))
            else:      
                x = x_mask*x
                 
                interpolated_x = x
        
        batch_size, _, input_dim = x.size()
        
        h_0 = self.h_0.expand(1, batch_size, self.h_dim).contiguous()
        
        
        
        
#         c_0 = self.c_0.expand(1, batch_size, self.s_dim).contiguous()
        
        
        h_prev = self.h_0.expand(1, batch_size, self.h_0.size(0))
        
        c_prev = self.h_0.expand(1, batch_size, self.h_0.size(0))
        
        
        z_1_category = torch.ones(self.cluster_num, dtype = torch.float, device = x.device)/self.cluster_num
        
        z_t_category_gen = z_1_category.expand(1, batch_size, z_1_category.size(0))
        
#         phi_z, z_representation = self.generate_z(z_t_category_gen, 0)
        
        
#         if np.any(x_lens.numpy()<= 0):
#             print(torch.nonzero(x_lens <= 0))
#             print('here')
        
#         print(torch.nonzero(x_lens <= 0))
        
#         if not self.latent:
        rnn_out,(last_h_n, last_c_n)= self.x_encoder(x, x_lens) # push the observed x's through the rnn;
#         else:
#             if not self.lstm_latent:
#                 rnn_out,(last_h_n, last_c_n)= self.x_encoder(self.x_kernel_encoder(x), x_lens) # push the observed x's through the rnn;
#             else:
#                 rnn_out,(last_h_n, last_c_n)= self.x_encoder(x, x_lens) # push the observed x's thr
        
#         self.x_encoder.forward2(x, x_lens, rnn_out, last_h_n, last_c_n)
        
        
        '''to be done'''
#         rnn_out2,(last_h_n2, last_c_n2)= self.x_encoder.forward2(x, x_lens) # push the observed x's through the rnn;
        
#         print(torch.norm(rnn_out - rnn_out2), torch.norm(last_h_n - last_h_n2), torch.norm(last_c_n - last_c_n2))
        
#         rnn_out = reverse_sequence(rnn_out, x_lens) # reverse the time-ordering in the hidden state and un-pack it
        rec_losses = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
        
        cluster_losses = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 

        rec_losses_with_no_coeff = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
        
        cluster_losses_with_no_coeff = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
         
        kl_states = torch.zeros((batch_size, T_max), device=x.device)
        
        rmse_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
        
        mae_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)   
        
        
        cluster_distances = torch.zeros([batch_size, T_max, self.cluster_num], device = x.device)
        cluster_distances2 = torch.zeros([T_max, self.cluster_num, batch_size, input_dim], device = x.device)
        
        prob_sums = 0
        
#         negnill = torch.zeros()
        
        entropy_losses = torch.zeros((batch_size, T_max), device = x.device)
        
        '''z_q_*''' 
        z_prev = self.z_q_0.expand(batch_size,self.z_q_0.size(0)) # set z_prev=z_q_0 to setup the recursive conditioning in q(z_t|...)
        
        curr_x_lens = x_lens.clone()
        
        shrinked_x_lens = x_lens.clone()
        
        x_t = 0
        
        x_t_mask = 0
        
        curr_rnn_out = rnn_out[curr_x_lens > 0,0,:]
        
        single_time_steps = torch.ones_like(curr_x_lens)
        
        last_h_now = torch.zeros_like(h_prev)
        
        last_c_now = torch.zeros_like(c_prev)
        
        imputed_x2 = torch.zeros_like(x)
        
        imputed_x2[:,0] = x[:,0] 
        
        last_rnn_out = torch.zeros(batch_size, self.s_dim*(1+self.x_encoder.bidir), device = self.device)
        
        joint_probs = torch.zeros([T_max, batch_size, self.cluster_num], dtype = torch.float, device = self.device)
        
        h_now_list = []
        
#         if self.latent:
#             if not self.lstm_latent:
#                 ae_loss = torch.zeros((batch_size, T_max-1, input_dim), device=x.device)
#             
#             if self.lstm_latent:
#                 ae_loss = torch.zeros((batch_size, T_max, input_dim), device=x.device)
#         
#         
#             if self.lstm_latent:
#                 rnn_decoded_out, (last_decoded_h_n, last_decoded_c_n) = self.x_kernel_decoder(rnn_out, x_lens)
        
        
        
        for t in range(T_max):
#             z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)# p(z_t| z_{t-1})
            
#             if self.latent and self.lstm_latent:
# #                 rnn_decoded_out, (decoded_h_n, decoded_c_n) = self.x_kernel_decoder(curr_rnn_out)
# #         
# #         
# #                 decoded_h_n_list.append(decoded_h_n)
# #                 
# #                 decoded_c_n_list.append(decoded_c_n)
#                 
#                 curr_rnn_decoded_out = rnn_decoded_out[curr_x_lens > 0, t]
#                 
#                 if self.loss_on_missing:
#                     ae_loss[curr_x_lens > 0, t] = (curr_rnn_decoded_out - x_t)**2
#                 else:
#                     ae_loss[curr_x_lens > 0, t] = (curr_rnn_decoded_out*x_t_mask - x_t*x_t_mask)**2
            
            '''phi_z_infer: phi_{z_t}'''
            if t == 0:
                z_t, z_t_category_infer, _ , z_category_infer_sparse= self.postnet(z_prev, curr_rnn_out, self.phi_table, t, self.temp) #q(z_t | z_{t-1}, x_{t:T})
                
                joint_probs[t] = z_t_category_infer
                
                kl = self.kl_div(z_t_category_infer, z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))
                
                if np.isnan(kl.cpu().detach().numpy()).any():
                    print('distribution 1::', z_t_category_gen)
                    
                    print('distribution 2::', z_t_category_infer)
                
                
                if self.transfer_prob:
                    
                    if self.block == 'GRU':
        #                 output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
                        output, h_now = self.trans(z_t_category_infer.view(z_t_category_infer.shape[0], 1, z_t_category_infer.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
                        
                    else:
                        output, (h_now, c_now) = self.trans(z_t_category_infer.view(z_t_category_infer.shape[0], 1, z_t_category_infer.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
                        
    #                 output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
                else:
                    
                    phi_z_infer = torch.mm(z_t_category_infer, torch.t(self.phi_table))
                    
                    if self.block == 'GRU':
                        output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
        #                     output, h_now = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
                        
                    else:
        #                     output, (h_now, c_now) = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
                        output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
                    
                    
#                 kl_states[curr_x_lens > 0,t] = kl
                
            else:
                
                
                '''joint_probs, curr_rnn_output, batch_size, t, h_prev, c_prev, shrinked_x_lens'''
                '''updated_joint_probs, full_kl, h_now, c_now, full_rec_loss, full_logit_x_t'''
                updated_joint_probs, kl, h_now, c_now, z_t_category_infer = self.update_joint_probability2(joint_probs[t-1, curr_x_lens > 0], curr_rnn_out, torch.sum(curr_x_lens > 0), t, h_prev, c_prev,z_t_category_infer, shrinked_x_lens, x_t, x_t_mask)
                
#                 self.optimizer.zero_grad()
#                  
#                 torch.sum(full_rec_loss).backward(retain_graph=True)
                
                joint_probs[t, curr_x_lens > 0] = updated_joint_probs
                
#                 self.optimizer.zero_grad()
#                  
#                 torch.sum(full_rec_loss).backward(retain_graph=True)
                
                
                
                
                kl_states[curr_x_lens > 0,t] = kl
            
            
            
#             phi_z_infer = torch.mm(z_t, torch.t(self.phi_table))
#             phi_z_infer = torch.mm(z_t_category_infer, torch.t(self.phi_table))
#             
#             phi_z_infer2 = torch.mm(z_t, torch.t(self.phi_table))
#             
#             
#             if self.use_gumbel:
#                 print(t, torch.norm(phi_z_infer - phi_z_infer2))
            
            prob_sums += torch.sum(z_t_category_infer, 0)
            
            
            
#             curr_cluster_distance_res = 0
#             
#             if self.loss_on_missing:
#                  
#                 curr_x_mask = torch.ones_like(x_mask[curr_x_lens > 0,t,:])
#                  
#                 curr_cluster_distance_res = self.compute_distance_per_cluster_all(x[curr_x_lens > 0,t,:], curr_x_mask)
#                  
#                 cluster_distances2[t, :, curr_x_lens > 0] = curr_cluster_distance_res
#             else:
#                  
#                 curr_cluster_distance_res = self.compute_distance_per_cluster_all(x[curr_x_lens > 0,t,:], x_mask[curr_x_lens > 0,t,:])
#                  
#                 cluster_distances2[t, :, curr_x_lens > 0] = curr_cluster_distance_res 
#             
#             
#             sumed_cluster_distnace_res = torch.sqrt(torch.sum(curr_cluster_distance_res, 2))
#             
#             cluster_distances[curr_x_lens > 0, t] = self.compute_distance_per_cluster(x[curr_x_lens > 0,t,:])
#             
# #             print(torch.norm(torch.t(sumed_cluster_distnace_res) - cluster_distances[curr_x_lens > 0, t]))
# #             if self.use_sparsemax:
#             
# #             else:
# #                 kl = self.kl_div(z_t_category_infer, z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))
#             
#             entropy_loss = self.entropy(z_t_category_infer)
#             
#             
#             
#             entropy_losses[curr_x_lens > 0, t] = entropy_loss
#             
#             
#                
#             
# #             print(t, curr_x_lens)
#             
#             if self.transfer_prob:
#                 
#                 z_t_transfer = z_t_category_infer
#                 
#                 if self.block == 'GRU':
#     #                 output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
#                     output, h_now = self.trans(z_t_transfer.view(z_t_transfer.shape[0], 1, z_t_transfer.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
#                     
#                 else:
#                     output, (h_now, c_now) = self.trans(z_t_transfer.view(z_t_transfer.shape[0], 1, z_t_transfer.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
# #                 output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#             else:
#                 if self.block == 'GRU':
#                     output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
# #                     output, h_now = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
#                     
#                 else:
# #                     output, (h_now, c_now) = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#                     output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#                 
#                 
# #             if not self.use_sparsemax:
# 
# 
#             
#             
#             
# #             curr_x_lens[curr_x_lens < 0] = 0
#             
#             
#             if t >= 1:            
#             
#                 phi_z, z_representation = self.generate_z(z_t_category_gen, t)
#                 
# #                 phi_z = torch.t(torch.mm(self.phi_table, torch.t(z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))))
# #                 
# #                 phi_z2 = torch.t(torch.mm(self.phi_table, torch.t(z_representation)))
# #                 
# #                 if self.use_gumbel:
# #                     print(torch.norm(phi_z - phi_z2))
#                 
#                 
#                 mean, logvar, logit_x_t = self.generate_x(phi_z_infer, z_t)
#                 
#                 
#                 
#                 
#     
#                 imputed_x2[curr_x_lens > 0,t] = logit_x_t
#                 
#                 
# #                 rec_loss = torch.norm(x[:,t+1,:] - mean)**2/(2*std**2) + torch.log(2*np.pi*std**2)/2
#                 
# #                 rec_loss = torch.bmm(((x[:,t,:]-mean)/(std**2)).view(mean.shape[0],1,mean.shape[1]), (x[:,t,:]-mean).view(mean.shape[0],mean.shape[1],1)).view(-1) + (torch.log((2*np.pi)**x[:,t,:].shape[-1]*torch.prod(std, dim= 1))/2).view(-1) 
#                 
#                 if self.loss_on_missing:
#                     
#                     curr_x_t_masks = torch.ones_like(x_t)
#                     
#                     rec_loss = compute_gaussian_probs0(x_t, mean, logvar, curr_x_t_masks)
#                 else:
#                     rec_loss = compute_gaussian_probs0(x_t, mean, logvar, x_t_mask)
#                 
# #                 rec_loss = self.compute_reconstruction_loss2(x_t, mean, std, batch_size)
#                 
#                 
#     #             kl = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
#     #             kl_states[:,t] = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
#     #             logit_x_t = self.emitter(z_t).contiguous() # p(x_t|z_t)         
#     #             rec_loss = nn.BCEWithLogitsLoss(reduction='none')(logit_x_t.view(-1), x[:,t,:].contiguous().view(-1)).view(batch_size, -1)
#                 rec_losses[curr_x_lens > 0,t-1] = rec_loss
#                 
# #                 rmse_loss = torch.sqrt(torch.sum((x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])**2, dim = 1))
#                 
#                 rmse_loss = (x_t*x_t_mask - logit_x_t*x_t_mask)**2
#                 
#                 mae_loss = torch.abs(x_t*x_t_mask - logit_x_t*x_t_mask)
# 
# #                 mae_losses = torch.abs(x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])
# 
#             
#                 rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
#                 
#                 mae_losses[curr_x_lens > 0,:,t-1] = mae_loss


            curr_x_lens -= 1

            shrinked_x_lens -= 1
#             else:
#                 z_t_category_gen = F.gumbel_softmax(self.emitter_z(h_now), tau=0.001, dim = 2)
            
#             if t >= 21:
#                 print('here')
            h_now_list.append(h_now)
            
            h_prev = h_now[:,shrinked_x_lens > 0,:]
            
            if (curr_x_lens == 0).any():
                last_h_now[:, curr_x_lens == 0, :] = h_now[:,shrinked_x_lens <= 0,:]
                last_rnn_out[curr_x_lens == 0] = rnn_out[curr_x_lens == 0,t,:]
            
            if self.block == 'LSTM':
                c_prev = c_now[:,shrinked_x_lens > 0,:]
                
                if (curr_x_lens == 0).any():
                    last_c_now[:, curr_x_lens == 0, :] = c_now[:,shrinked_x_lens <= 0,:] 
#             phi_z_infer = torch.mm(joint_probs, torch.t(self.phi_table))
#             if self.transfer_prob:
#                 z_prev = z_t[shrinked_x_lens > 0,:]
#             else:
#                 z_prev = phi_z_infer[shrinked_x_lens > 0,:]

#             if not self.use_sparsemax:
#                 
# #                 print(t, torch.sum(shrinked_x_lens > 0))
#                 z_t_category_gen = F.softmax(self.emitter_z(h_now[:,shrinked_x_lens > 0,:]), dim = 2)
#             else:
# #                 print(t, torch.sum(shrinked_x_lens > 0))
#                 
#                 if torch.sum(shrinked_x_lens > 0) > 0:
#                 
#                     logit_z_t = self.emitter_z(h_now[:,shrinked_x_lens > 0,:])
#                     
#                     z_t_category_gen = sparsemax(logit_z_t.view(logit_z_t.shape[1], logit_z_t.shape[2]))
#                 
#                     z_t_category_gen = z_t_category_gen.view(1, z_t_category_gen.shape[0], z_t_category_gen.shape[1])
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
                
                
                curr_rnn_out = rnn_out[curr_x_lens > 0,t+1,:]
            shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]
        
            
        full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
        
        for k in range(self.cluster_num):
            curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            curr_rnn_input[:,k] = 1
            
            full_curr_rnn_input[k] = curr_rnn_input
        
        
        curr_x_lens = x_lens.clone()
        
        
        x_t = x[curr_x_lens > 0,0,:]
                
        x_t_mask = x_mask[curr_x_lens > 0,0,:]
        
#         if self.latent:
#             ae_loss = torch.zeros((batch_size, T_max-1, input_dim), device=x.device)
        
        for t in range(T_max):
            
#             print(t, torch.nonzero(curr_x_lens > 0).view(-1), joint_probs.shape)
            


            if t >= 1:
                
#                 if self.latent:
#                     input_x_t = self.x_kernel_encoder(x_t)
#                     
#                     if self.loss_on_missing:
#                         ae_loss[curr_x_lens > 0,t-1] = (self.x_kernel_decoder(input_x_t) - x_t)**2/(self.x_std**2)
#                     else:
#                         ae_loss[curr_x_lens > 0,t-1] = (self.x_kernel_decoder(input_x_t)*x_t_mask - x_t*x_t_mask)**2/(self.x_std**2)
#                     
#                     
#                 else:
                input_x_t = x_t
                
                full_rec_loss1, full_rec_loss2, full_logit_x_t, curr_full_rec_loss, l2_norm_loss, cluster_loss = self.compute_rec_loss(joint_probs[t, curr_x_lens > 0], prob_sums/torch.sum(x_lens), full_curr_rnn_input, input_x_t, x_t_mask, curr_rnn_out)
                
#                 if self.latent:
#                     full_logit_x_t = self.x_kernel_decoder(full_logit_x_t)
                
#             print(t, full_rec_loss2.shape)

            
            
            
                
                rec_losses[curr_x_lens > 0,t-1] = full_rec_loss1
                
                cluster_losses[curr_x_lens > 0,t-1] = full_rec_loss2
                
                rec_losses_with_no_coeff[curr_x_lens > 0,t-1] = l2_norm_loss
                
                cluster_losses_with_no_coeff[curr_x_lens > 0,t-1] = cluster_loss
                
#                 print('time::', t)
                
            
                rmse_loss = (x_t*x_t_mask - full_logit_x_t*x_t_mask)**2
                
                mae_loss = torch.abs(x_t*x_t_mask - full_logit_x_t*x_t_mask)

#                 mae_losses = torch.abs(x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])

            
                rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
                
                mae_losses[curr_x_lens > 0,:,t-1] = mae_loss
            
            
            curr_x_lens -= 1
               
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
                
                
            curr_rnn_out = rnn_out[curr_x_lens > 0,t,:]
#             curr_x_lens = curr_x_lens[curr_x_lens > 0]

#         x_mask = sequence_mask(x_lens)
#         x_mask = x_mask.gt(0).view(-1)
#         rec_loss = rec_losses.view(-1).masked_select(x_mask).mean()
#         kl_loss = kl_states.view(-1).masked_select(x_mask).mean()

        

#         rec_loss = torch.sum(rec_losses)/torch.sum(x_lens-1)
        
#         full_cluster_objs, cluster_objs = self.compute_cluster_obj(cluster_distances, prob_sums/torch.sum(x_lens), T_max, x_lens, input_dim)
#         
#         
#         if self.loss_on_missing:
#              
#             x_mask_full = torch.ones_like(x_mask)
#              
#             full_cluster_objs2, cluster_objs2 = self.compute_cluster_obj_full2(cluster_distances2, prob_sums/torch.sum(x_lens), T_max, x_mask_full, x_lens)
#         else:
#             full_cluster_objs2, cluster_objs2 = self.compute_cluster_obj_full2(cluster_distances2, prob_sums/torch.sum(x_lens), T_max, x_mask, x_lens)
        
#         print(torch.norm(full_cluster_objs - full_cluster_objs2))
#          
#         print(torch.norm(cluster_objs - cluster_objs2))
        
#         if (not self.loss_on_missing) and (not self.latent):
        rec_loss1 = torch.sum(rec_losses)/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss1 = torch.sum(rec_losses)/torch.sum(torch.ones_like(rec_losses))
        
#         if (not self.loss_on_missing) and (not self.latent):
        rec_loss2 = torch.sum(cluster_losses*x_mask[:,1:,:])/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss2 = torch.sum(cluster_losses*torch.ones_like(cluster_losses))/torch.sum(torch.ones_like(cluster_losses))
        
#         if (not self.loss_on_missing) and (not self.latent):
        final_rec_loss = torch.sum(rec_losses_with_no_coeff)/torch.sum(x_mask[:,1:,:])
#         else:
#             final_rec_loss = torch.sum(rec_losses_with_no_coeff)/torch.sum(torch.ones_like(rec_losses_with_no_coeff))
        
#         if (not self.loss_on_missing) and (not self.latent):
        final_cluster_loss = torch.sum(cluster_losses_with_no_coeff*x_mask[:,1:,:])/torch.sum(x_mask[:,1:,:])
#         else:
#             final_cluster_loss = torch.sum(cluster_losses_with_no_coeff*torch.ones_like(cluster_losses_with_no_coeff))/torch.sum(torch.ones_like(cluster_losses_with_no_coeff))
        
#         for k in range(rec_losses.shape[0]):
#             print(rec_losses[k].mean())
        
        
        final_ae_loss = 0
#         if self.latent:
#             if not self.lstm_latent:
#                 if not self.loss_on_missing:
#                     final_ae_loss = torch.sum(ae_loss)/torch.sum(x_mask[:,1:,:])
#                 else:
#                     final_ae_loss = torch.sum(ae_loss)/torch.sum(torch.ones_like(ae_loss))
#             else:
# #                 ae_loss = (x - rnn_decoded_out)**2
#                 
#                 if not self.loss_on_missing:
#                     final_ae_loss = torch.sum(ae_loss*x_mask)/torch.sum(x_mask)
#                 else:
#                     final_ae_loss = torch.sum(ae_loss)/torch.sum(torch.ones_like(ae_loss))
#                 
#                 
#             print('autoencoder loss::', final_ae_loss)
        
        
        
        
        
        
        first_kl_loss = kl_states[:, 0].view(-1).mean()
        
        kl_loss = torch.sum(kl_states[:, 1:])/torch.sum(x_lens-1)
        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask[:,1:,:]))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask[:,1:,:])
        
        
        final_entropy_loss = entropy_losses.view(-1).mean()
        print('loss::', final_rec_loss, kl_loss)
        
        print('loss with coefficient::', rec_loss1, kl_loss)
        
        print('rmse loss::', final_rmse_loss)
        
        print('mae loss::', final_mae_losses)
        
        print('cluster objective::', rec_loss2)
        
        print('cluster objective with coefficient:', final_cluster_loss)
        
        interpolated_loss = 0
        
        if self.is_missing:
            interpolated_loss = torch.norm(interpolated_x*x_mask - x*x_mask)
            
            print('interpolate loss::', interpolated_loss)
        
        if torch.sum(1-new_x_mask[:,1:]) > 0:
            
            imputed_mse_loss = torch.sqrt(torch.sum((((origin_x[:,1:] - x[:,1:])**2)*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:]))
            
            imputed_mse_loss2 = torch.sqrt(torch.sum((((origin_x[:,1:] - imputed_x2[:,1:])**2)*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:])) 
            
            imputed_loss = torch.sum((torch.abs(origin_x[:,1:] - x[:,1:])*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:])
            
            imputed_loss2 = torch.sum((torch.abs(origin_x[:,1:] - imputed_x2[:,1:])*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:]) 
            
            print('training imputation rmse loss::', imputed_mse_loss)
            
            print('training imputation rmse loss 2::', imputed_mse_loss2)
            
            print('training imputation mae loss::', imputed_loss)
            
            print('training imputation mae loss 2::', imputed_loss2)
            
        
        
        prior_cluster_probs = prob_sums/torch.sum(x_lens)
        
#         if self.block == 'GRU':
#             self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, None, T_max, prior_cluster_probs)
#         else:
#             self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, last_c_now, T_max, prior_cluster_probs)
        
        if not self.evaluate:
            final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count = self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, last_c_now, T_max, prior_cluster_probs, last_rnn_out, last_h_n, last_c_n, last_decoded_h_n.unsqueeze(0), last_decoded_c_n.unsqueeze(0))
        else:
            imputed_x2 = self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, last_c_now, T_max, prior_cluster_probs, last_rnn_out,last_h_n, last_c_n, last_decoded_h_n.unsqueeze(0), last_decoded_c_n.unsqueeze(0))
        
        print()
        
        if not os.path.exists(data_folder + '/' + output_dir):
            os.makedirs(data_folder + '/' + output_dir)
        
        
        torch.save(self.phi_table, data_folder + '/' + output_dir + 'cluster_centroids')
        
#         torch.save(all_mean, data_folder + '/' + output_dir + 'all_mean')
#         
#         torch.save(all_log_var, data_folder + '/' + output_dir + 'all_log_var')
#         
#         torch.save(all_probs, data_folder + '/' + output_dir + 'all_probs')
        
        
        
#         del x, x_to_predict,origin_x, origin_x_to_predict, x_mask, x_to_predict_mask, x_origin_mask, x_new_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_lens, x_to_predict_lens
        
        
        
        
        
        
        
        
        
        if not self.evaluate:
#             return final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count
            if not torch.sum(1-new_x_mask[:,1:]) > 0:
                return final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count, None
            else:
                return final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count, (imputed_loss2, torch.sum(1-new_x_mask[:,1:]), imputed_mse_loss2, torch.sum(1-new_x_mask[:,1:]))

        else:
            
            if torch.sum(1-new_x_mask) > 0:
                return imputed_x2, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
            else:
                return imputed_x2, None

        
#         print()
#         
#         if not self.evaluate:
#             return rec_loss1, kl_loss, first_kl_loss, final_rmse_loss, imputed_loss, rec_loss2
#         else:
#             
#             if torch.sum(1-new_x_mask) > 0:
#                 return imputed_x2*(1-x_mask) + x*x_mask, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
#             else:
#                 return imputed_x2*(1-x_mask) + x*x_mask, None

    '''self, x, origin_x, x_mask, x_origin_mask, x_new_mask, x_lens, x_to_predict, origin_x_to_predict, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_delta_time_stamps, x_to_predict_delta_time_stamps, x_time_stamps, x_to_predict_time_stamps'''
#     def test_samples(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_delta_time_stamps, x_to_predict_delta_time_stamps, x_time_stamps, x_to_predict_time_stamps):
#         """
#         infer q(z_{1:T}|x_{1:T}) (i.e. the variational distribution)
#         """
#         
# #         assert torch.sum(x_mask) == torch.sum(1-np.isnan(x))
#         T_max = x_lens.max().item()
#         if is_GPU:
#             x = x.to(device)
#             
#             x_to_predict = x_to_predict.to(device)
#             
#             x_mask = x_mask.to(device)
#             
#             x_to_predict_mask = x_to_predict_mask.to(device)
#             
#             origin_x = origin_x.to(device)
#             
#             origin_x_to_pred = origin_x_to_pred.to(device)
#             
#             
#             origin_x_mask = origin_x_mask.to(device)
#             
#             new_x_mask = new_x_mask.to(device)
#             
#             
#             
#             x_to_predict_origin_mask = x_to_predict_origin_mask.to(device)
#             
#             x_to_predict_new_mask = x_to_predict_new_mask.to(device) 
#         
#             x_lens = x_lens.to(device)
#         
#             x_to_predict_lens = x_to_predict_lens.to(device)
#         
#         if self.is_missing:
#             
# #             t1 = time.time()
# #             
# #             imputed_x = self.impute(x, x_mask, T_max)
# #             
# #             t2 = time.time()
#             
#             
#             imputed_x, interpolated_x = self.impute.forward2(x, x_mask, T_max)
# 
#             
# #             t3 = time.time()
# #             
# #             print(t3 - t2)
# #             print(t2 - t1)
#                     
#             x = imputed_x
#         
#         batch_size, _, input_dim = x.size()
#         
#         h_0 = self.h_0.expand(1, batch_size, self.h_dim).contiguous()
#         
#         
#         
#         
# #         c_0 = self.c_0.expand(1, batch_size, self.s_dim).contiguous()
#         
#         
#         h_prev = self.h_0.expand(1, batch_size, self.h_0.size(0))
#         
#         c_prev = self.h_0.expand(1, batch_size, self.h_0.size(0))
#         
#         
#         z_1_category = torch.ones(self.cluster_num, dtype = torch.float, device = x.device)/self.cluster_num
#         
#         z_t_category_gen = z_1_category.expand(1, batch_size, z_1_category.size(0))
#         
# #         phi_z, z_representation = self.generate_z(z_t_category_gen, 0)
#         
#         
# #         if np.any(x_lens.numpy()<= 0):
# #             print(torch.nonzero(x_lens <= 0))
# #             print('here')
#         
# #         print(torch.nonzero(x_lens <= 0))
#         
#         rnn_out,(last_h_n, last_c_n)= self.x_encoder(x, x_lens) # push the observed x's through the rnn;
#         
# #         self.x_encoder.forward2(x, x_lens, rnn_out, last_h_n, last_c_n)
#         
#         
#         '''to be done'''
# #         rnn_out2,(last_h_n2, last_c_n2)= self.x_encoder.forward2(x, x_lens) # push the observed x's through the rnn;
#         
# #         print(torch.norm(rnn_out - rnn_out2), torch.norm(last_h_n - last_h_n2), torch.norm(last_c_n - last_c_n2))
#         
# #         rnn_out = reverse_sequence(rnn_out, x_lens) # reverse the time-ordering in the hidden state and un-pack it
#         rec_losses = torch.zeros((batch_size, T_max-1, input_dim), device=x.device) 
#         kl_states = torch.zeros((batch_size, T_max), device=x.device)
#         
#         rmse_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
#         
#         mae_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)   
#         
#         
#         cluster_distances = torch.zeros([batch_size, T_max, self.cluster_num], device = x.device)
#         cluster_distances2 = torch.zeros([T_max, self.cluster_num, batch_size, input_dim], device = x.device)
#         
#         prob_sums = 0
#         
# #         negnill = torch.zeros()
#         
#         entropy_losses = torch.zeros((batch_size, T_max), device = x.device)
#         
#         '''z_q_*''' 
#         z_prev = self.z_q_0.expand(batch_size,self.z_q_0.size(0)) # set z_prev=z_q_0 to setup the recursive conditioning in q(z_t|...)
#         
#         curr_x_lens = x_lens.clone()
#         
#         shrinked_x_lens = x_lens.clone()
#         
#         x_t = 0
#         
#         x_t_mask = 0
#         
#         curr_rnn_out = rnn_out[curr_x_lens > 0,0,:]
#         
#         single_time_steps = torch.ones_like(curr_x_lens)
#         
#         last_h_now = torch.zeros_like(h_prev)
#         
#         last_c_now = torch.zeros_like(c_prev)
#         
#         imputed_x2 = torch.zeros_like(x)
#         
#         imputed_x2[:,0] = x[:,0] 
#         
#         
#         joint_probs = torch.zeros([T_max, batch_size, self.cluster_num], dtype = torch.float, device = self.device)
#         
#         for t in range(T_max):
# #             z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)# p(z_t| z_{t-1})
#             
#             '''phi_z_infer: phi_{z_t}'''
#             if t == 0:
#                 z_t, z_t_category_infer, _ = self.postnet(z_prev, curr_rnn_out, self.phi_table, t, self.temp) #q(z_t | z_{t-1}, x_{t:T})
#                 
#                 joint_probs[t] = z_t_category_infer
#                 
#                 kl = self.kl_div(z_t_category_infer, z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))
#                 
#                 if np.isnan(kl.cpu().detach().numpy()).any():
#                     print('distribution 1::', z_t_category_gen)
#                     
#                     print('distribution 2::', z_t_category_infer)
#                 
#                 h_now = h_prev
#                 
#                 if not self.block == 'GRU':
#                     c_now = c_prev
#                 
#                 kl_states[curr_x_lens > 0,t] = kl
#                 
#             else:
#                 
#                 
#                 '''joint_probs, curr_rnn_output, batch_size, t, h_prev, c_prev, shrinked_x_lens'''
#                 '''updated_joint_probs, full_kl, h_now, c_now, full_rec_loss, full_logit_x_t'''
#                 updated_joint_probs, kl, h_now, c_now, full_rec_loss, logit_x_t = self.update_joint_probability(joint_probs[t-1, curr_x_lens > 0], curr_rnn_out, torch.sum(curr_x_lens > 0), t, h_prev, c_prev,shrinked_x_lens, x_t, x_t_mask)
#                 
# #                 self.optimizer.zero_grad()
# #                  
# #                 torch.sum(full_rec_loss).backward(retain_graph=True)
#                 
#                 joint_probs[t, curr_x_lens > 0] = updated_joint_probs
#                 
# #                 self.optimizer.zero_grad()
# #                  
# #                 torch.sum(full_rec_loss).backward(retain_graph=True)
#                 
#                 
#                 
#                 
#                 kl_states[curr_x_lens > 0,t] = kl
#             
#                 rec_losses[curr_x_lens > 0,t-1] = full_rec_loss
#             
# #                 print('time::', t)
#                 
#             
#                 rmse_loss = (x_t*x_t_mask - logit_x_t*x_t_mask)**2
#                 
#                 mae_loss = torch.abs(x_t*x_t_mask - logit_x_t*x_t_mask)
# 
# #                 mae_losses = torch.abs(x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])
# 
#             
#                 rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
#                 
#                 mae_losses[curr_x_lens > 0,:,t-1] = mae_loss
#             
#             
#             
# #             phi_z_infer = torch.mm(z_t, torch.t(self.phi_table))
# #             phi_z_infer = torch.mm(z_t_category_infer, torch.t(self.phi_table))
# #             
# #             phi_z_infer2 = torch.mm(z_t, torch.t(self.phi_table))
# #             
# #             
# #             if self.use_gumbel:
# #                 print(t, torch.norm(phi_z_infer - phi_z_infer2))
#             
#             prob_sums += torch.sum(joint_probs[t], 0)
#             
# #             curr_cluster_distance_res = 0
# #             
#             if self.loss_on_missing:
#                  
#                 curr_x_mask = torch.ones_like(x_mask[curr_x_lens > 0,t,:])
#                  
#                 curr_cluster_distance_res = self.compute_distance_per_cluster_all(x[curr_x_lens > 0,t,:], curr_x_mask)
#                  
#                 cluster_distances2[t, :, curr_x_lens > 0] = curr_cluster_distance_res
#             else:
#                  
#                 curr_cluster_distance_res = self.compute_distance_per_cluster_all(x[curr_x_lens > 0,t,:], x_mask[curr_x_lens > 0,t,:])
#                  
#                 cluster_distances2[t, :, curr_x_lens > 0] = curr_cluster_distance_res 
# #             
# #             
# #             sumed_cluster_distnace_res = torch.sqrt(torch.sum(curr_cluster_distance_res, 2))
# #             
# #             cluster_distances[curr_x_lens > 0, t] = self.compute_distance_per_cluster(x[curr_x_lens > 0,t,:])
# #             
# # #             print(torch.norm(torch.t(sumed_cluster_distnace_res) - cluster_distances[curr_x_lens > 0, t]))
# # #             if self.use_sparsemax:
# #             
# # #             else:
# # #                 kl = self.kl_div(z_t_category_infer, z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))
# #             
# #             entropy_loss = self.entropy(z_t_category_infer)
# #             
# #             
# #             
# #             entropy_losses[curr_x_lens > 0, t] = entropy_loss
# #             
# #             
# #                
# #             
# # #             print(t, curr_x_lens)
# #             
# #             if self.transfer_prob:
# #                 
# #                 z_t_transfer = z_t_category_infer
# #                 
# #                 if self.block == 'GRU':
# #     #                 output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
# #                     output, h_now = self.trans(z_t_transfer.view(z_t_transfer.shape[0], 1, z_t_transfer.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
# #                     
# #                 else:
# #                     output, (h_now, c_now) = self.trans(z_t_transfer.view(z_t_transfer.shape[0], 1, z_t_transfer.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
# # #                 output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
# #             else:
# #                 if self.block == 'GRU':
# #                     output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
# # #                     output, h_now = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
# #                     
# #                 else:
# # #                     output, (h_now, c_now) = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
# #                     output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
# #                 
# #                 
# # #             if not self.use_sparsemax:
# # 
# # 
# #             
# #             
# #             
# # #             curr_x_lens[curr_x_lens < 0] = 0
# #             
# #             
# #             if t >= 1:            
# #             
# #                 phi_z, z_representation = self.generate_z(z_t_category_gen, t)
# #                 
# # #                 phi_z = torch.t(torch.mm(self.phi_table, torch.t(z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))))
# # #                 
# # #                 phi_z2 = torch.t(torch.mm(self.phi_table, torch.t(z_representation)))
# # #                 
# # #                 if self.use_gumbel:
# # #                     print(torch.norm(phi_z - phi_z2))
# #                 
# #                 
# #                 mean, logvar, logit_x_t = self.generate_x(phi_z_infer, z_t)
# #                 
# #                 
# #                 
# #                 
# #     
# #                 imputed_x2[curr_x_lens > 0,t] = logit_x_t
# #                 
# #                 
# # #                 rec_loss = torch.norm(x[:,t+1,:] - mean)**2/(2*std**2) + torch.log(2*np.pi*std**2)/2
# #                 
# # #                 rec_loss = torch.bmm(((x[:,t,:]-mean)/(std**2)).view(mean.shape[0],1,mean.shape[1]), (x[:,t,:]-mean).view(mean.shape[0],mean.shape[1],1)).view(-1) + (torch.log((2*np.pi)**x[:,t,:].shape[-1]*torch.prod(std, dim= 1))/2).view(-1) 
# #                 
# #                 if self.loss_on_missing:
# #                     
# #                     curr_x_t_masks = torch.ones_like(x_t)
# #                     
# #                     rec_loss = compute_gaussian_probs0(x_t, mean, logvar, curr_x_t_masks)
# #                 else:
# #                     rec_loss = compute_gaussian_probs0(x_t, mean, logvar, x_t_mask)
# #                 
# # #                 rec_loss = self.compute_reconstruction_loss2(x_t, mean, std, batch_size)
# #                 
# #                 
# #     #             kl = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
# #     #             kl_states[:,t] = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
# #     #             logit_x_t = self.emitter(z_t).contiguous() # p(x_t|z_t)         
# #     #             rec_loss = nn.BCEWithLogitsLoss(reduction='none')(logit_x_t.view(-1), x[:,t,:].contiguous().view(-1)).view(batch_size, -1)
# #                 rec_losses[curr_x_lens > 0,t-1] = rec_loss
# #                 
# # #                 rmse_loss = torch.sqrt(torch.sum((x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])**2, dim = 1))
# #                 
# #                 rmse_loss = (x_t*x_t_mask - logit_x_t*x_t_mask)**2
# #                 
# #                 mae_loss = torch.abs(x_t*x_t_mask - logit_x_t*x_t_mask)
# # 
# # #                 mae_losses = torch.abs(x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])
# # 
# #             
# #                 rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
# #                 
# #                 mae_losses[curr_x_lens > 0,:,t-1] = mae_loss
# 
# 
#             curr_x_lens -= 1
# 
#             shrinked_x_lens -= 1
# #             else:
# #                 z_t_category_gen = F.gumbel_softmax(self.emitter_z(h_now), tau=0.001, dim = 2)
#             
# #             if t >= 21:
# #                 print('here')
#             
#             h_prev = h_now[:,shrinked_x_lens > 0,:]
#             
#             if (curr_x_lens == 0).any():
#                 last_h_now[:, curr_x_lens == 0, :] = h_now[:,shrinked_x_lens <= 0,:]
#             
#             if self.block == 'LSTM':
#                 c_prev = c_now[:,shrinked_x_lens > 0,:]
#                 
#                 if (curr_x_lens == 0).any():
#                     last_c_now[:, curr_x_lens == 0, :] = c_now[:,shrinked_x_lens <= 0,:] 
# #             phi_z_infer = torch.mm(joint_probs, torch.t(self.phi_table))
# #             if self.transfer_prob:
# #                 z_prev = z_t[shrinked_x_lens > 0,:]
# #             else:
# #                 z_prev = phi_z_infer[shrinked_x_lens > 0,:]
# 
# #             if not self.use_sparsemax:
# #                 
# # #                 print(t, torch.sum(shrinked_x_lens > 0))
# #                 z_t_category_gen = F.softmax(self.emitter_z(h_now[:,shrinked_x_lens > 0,:]), dim = 2)
# #             else:
# # #                 print(t, torch.sum(shrinked_x_lens > 0))
# #                 
# #                 if torch.sum(shrinked_x_lens > 0) > 0:
# #                 
# #                     logit_z_t = self.emitter_z(h_now[:,shrinked_x_lens > 0,:])
# #                     
# #                     z_t_category_gen = sparsemax(logit_z_t.view(logit_z_t.shape[1], logit_z_t.shape[2]))
# #                 
# #                     z_t_category_gen = z_t_category_gen.view(1, z_t_category_gen.shape[0], z_t_category_gen.shape[1])
#             
#             if t < T_max - 1:
#                 x_t = x[curr_x_lens > 0,t+1,:]
#                 
#                 x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
#                 
#                 
#                 curr_rnn_out = rnn_out[curr_x_lens > 0,t+1,:]
#             
#             
#             shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]
#             
#             
# #             curr_x_lens = curr_x_lens[curr_x_lens > 0]
# 
# #         x_mask = sequence_mask(x_lens)
# #         x_mask = x_mask.gt(0).view(-1)
# #         rec_loss = rec_losses.view(-1).masked_select(x_mask).mean()
# #         kl_loss = kl_states.view(-1).masked_select(x_mask).mean()
# 
#         
# 
# #         rec_loss = torch.sum(rec_losses)/torch.sum(x_lens-1)
#         
#         full_cluster_objs, cluster_objs = self.compute_cluster_obj(cluster_distances, prob_sums/torch.sum(x_lens), T_max, x_lens, input_dim)
#         
#         
#         if self.loss_on_missing:
#             
#             x_mask_full = torch.ones_like(x_mask)
#             
#             full_cluster_objs2, cluster_objs2 = self.compute_cluster_obj_full2(cluster_distances2, prob_sums/torch.sum(x_lens), T_max, x_mask_full, x_lens)
#         else:
#             full_cluster_objs2, cluster_objs2 = self.compute_cluster_obj_full2(cluster_distances2, prob_sums/torch.sum(x_lens), T_max, x_mask, x_lens)
#         
# #         print(torch.norm(full_cluster_objs - full_cluster_objs2))
# #          
# #         print(torch.norm(cluster_objs - cluster_objs2))
#         
#         if not self.loss_on_missing:
#             rec_loss = torch.sum(rec_losses)/torch.sum(x_mask[:,1:,:])
#         else:
#             rec_loss = torch.sum(rec_losses)/torch.sum(torch.ones_like(rec_losses))
#         
# #         for k in range(rec_losses.shape[0]):
# #             print(rec_losses[k].mean())
#         
#         first_kl_loss = kl_states[:, 0].view(-1).mean()
#         
#         kl_loss = torch.sum(kl_states[:, 1:])/torch.sum(x_lens-1)
#         final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask[:,1:,:]))
#         
#         final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask[:,1:,:])
#         
#         
#         final_entropy_loss = entropy_losses.view(-1).mean()
#         print('loss::', rec_loss, kl_loss)
#         
#         print('rmse loss::', final_rmse_loss)
#         
#         print('mae loss::', final_mae_losses)
#         
#         print('cluster objective::', cluster_objs2)
#         
#         imputed_loss = 0
#         
#         if self.is_missing:
#             imputed_loss = torch.norm(interpolated_x*x_mask - x*x_mask)
#             
#             print('interpolate loss::', imputed_loss)
#         
#         if torch.sum(1-new_x_mask) > 0:
#             
#             imputed_mse_loss = torch.sqrt(torch.sum((((origin_x - x)**2)*(1-new_x_mask)))/torch.sum(1-new_x_mask))
#             
#             imputed_mse_loss2 = torch.sqrt(torch.sum((((origin_x - imputed_x2)**2)*(1-new_x_mask)))/torch.sum(1-new_x_mask)) 
#             
#             imputed_loss = torch.sum((torch.abs(origin_x - x)*(1-new_x_mask)))/torch.sum(1-new_x_mask)
#             
#             imputed_loss2 = torch.sum((torch.abs(origin_x - imputed_x2)*(1-new_x_mask)))/torch.sum(1-new_x_mask) 
#             
#             print('training imputation rmse loss::', imputed_mse_loss)
#             
#             print('training imputation rmse loss 2::', imputed_mse_loss2)
#             
#             print('training imputation mae loss::', imputed_loss)
#             
#             print('training imputation mae loss 2::', imputed_loss2)
#             
#         
#         
#         prior_cluster_probs = prob_sums/torch.sum(x_lens)
#         
# #         if self.block == 'GRU':
# #             self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, None, T_max, prior_cluster_probs)
# #         else:
# #             self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, last_c_now, T_max, prior_cluster_probs)
#         if not self.evaluate:
#             final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count = self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, last_c_now, T_max, prior_cluster_probs)
#         else:
#             imputed_x2 = self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, last_c_now, T_max, prior_cluster_probs)
# 
# 
# 
#         
#         print()
#         
#         
#         
#         if not self.evaluate:
#             return final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count
#         else:
#             
#             if torch.sum(1-new_x_mask) > 0:
#                 return imputed_x2, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
#             else:
#                 return imputed_x2, None
#         
        
#         if not self.evaluate:
#             return rec_loss, kl_loss, first_kl_loss, final_rmse_loss, imputed_loss, cluster_objs2
#         else:
#             
#             if torch.sum(1-new_x_mask) > 0:
#                 return imputed_x2*(1-x_mask) + x*x_mask, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
#             else:
#                 return imputed_x2*(1-x_mask) + x*x_mask, None

    def infer_back(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device):
        """
        infer q(z_{1:T}|x_{1:T}) (i.e. the variational distribution)
        """
        
#         assert torch.sum(x_mask) == torch.sum(1-np.isnan(x))
        T_max = x_lens.max().item()
        if is_GPU:
            x = x.to(device)
            
            x_to_predict = x_to_predict.to(device)
            
            x_mask = x_mask.to(device)
            
            x_to_predict_mask = x_to_predict_mask.to(device)
            
            origin_x = origin_x.to(device)
            
            origin_x_to_pred = origin_x_to_pred.to(device)
            
            
            origin_x_mask = origin_x_mask.to(device)
            
            new_x_mask = new_x_mask.to(device)
            
            
            
            x_to_predict_origin_mask = x_to_predict_origin_mask.to(device)
            
            x_to_predict_new_mask = x_to_predict_new_mask.to(device) 
        
            x_lens = x_lens.to(device)
        
            x_to_predict_lens = x_to_predict_lens.to(device)
        
        if self.is_missing:
            
#             t1 = time.time()
#             
#             imputed_x = self.impute(x, x_mask, T_max)
#             
#             t2 = time.time()
            
            
            imputed_x, interpolated_x = self.impute.forward2(x, x_mask, T_max)

            
#             t3 = time.time()
#             
#             print(t3 - t2)
#             print(t2 - t1)
                    
            x = imputed_x
        
        batch_size, _, input_dim = x.size()
        
        h_0 = self.h_0.expand(1, batch_size, self.h_dim).contiguous()
        
        
        
        
#         c_0 = self.c_0.expand(1, batch_size, self.s_dim).contiguous()
        
        
        h_prev = self.h_0.expand(1, batch_size, self.h_0.size(0))
        
        c_prev = self.h_0.expand(1, batch_size, self.h_0.size(0))
        
        
        z_1_category = torch.ones(self.cluster_num, dtype = torch.float, device = x.device)/self.cluster_num
        
        z_t_category_gen = z_1_category.expand(1, batch_size, z_1_category.size(0))
        
        phi_z, z_representation = self.generate_z(z_t_category_gen, 0)
        
        
#         if np.any(x_lens.numpy()<= 0):
#             print(torch.nonzero(x_lens <= 0))
#             print('here')
        
#         print(torch.nonzero(x_lens <= 0))
        
        rnn_out,(last_h_n, last_c_n)= self.x_encoder(x, x_lens) # push the observed x's through the rnn;
        
#         self.x_encoder.forward2(x, x_lens, rnn_out, last_h_n, last_c_n)
        
        
        '''to be done'''
#         rnn_out2,(last_h_n2, last_c_n2)= self.x_encoder.forward2(x, x_lens) # push the observed x's through the rnn;
        
#         print(torch.norm(rnn_out - rnn_out2), torch.norm(last_h_n - last_h_n2), torch.norm(last_c_n - last_c_n2))
        
#         rnn_out = reverse_sequence(rnn_out, x_lens) # reverse the time-ordering in the hidden state and un-pack it
        rec_losses = torch.zeros((batch_size, T_max-1, input_dim), device=x.device) 
        kl_states = torch.zeros((batch_size, T_max), device=x.device)
        
        rmse_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
        
        mae_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)   
        
        
        cluster_distances = torch.zeros([batch_size, T_max, self.cluster_num], device = x.device)
        cluster_distances2 = torch.zeros([T_max, self.cluster_num, batch_size, input_dim], device = x.device)
        
        prob_sums = 0
        
#         negnill = torch.zeros()
        
        entropy_losses = torch.zeros((batch_size, T_max), device = x.device)
        
        '''z_q_*''' 
        z_prev = self.z_q_0.expand(batch_size,self.z_q_0.size(0)) # set z_prev=z_q_0 to setup the recursive conditioning in q(z_t|...)
        
        curr_x_lens = x_lens.clone()
        
        shrinked_x_lens = x_lens.clone()
        
        x_t = 0
        
        x_t_mask = 0
        
        curr_rnn_out = rnn_out[curr_x_lens > 0,0,:]
        
        single_time_steps = torch.ones_like(curr_x_lens)
        
        last_h_now = torch.zeros_like(h_prev)
        
        last_c_now = torch.zeros_like(c_prev)
        
        imputed_x2 = torch.zeros_like(x)
        
        imputed_x2[:,0] = x[:,0] 
        
        for t in range(T_max):
#             z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)# p(z_t| z_{t-1})
            
            '''phi_z_infer: phi_{z_t}'''
            z_t, z_t_category_infer, _, z_category_infer_sparse = self.postnet(z_prev, curr_rnn_out, self.phi_table, t, self.temp) #q(z_t | z_{t-1}, x_{t:T})
            
            phi_z_infer = torch.mm(z_t, torch.t(self.phi_table))
#             phi_z_infer = torch.mm(z_t_category_infer, torch.t(self.phi_table))
#             
#             phi_z_infer2 = torch.mm(z_t, torch.t(self.phi_table))
#             
#             
#             if self.use_gumbel:
#                 print(t, torch.norm(phi_z_infer - phi_z_infer2))
            
            prob_sums += torch.sum(z_t_category_infer, 0)
            
            curr_cluster_distance_res = 0
            
            if self.loss_on_missing:
                
                curr_x_mask = torch.ones_like(x_mask[curr_x_lens > 0,t,:])
                
                curr_cluster_distance_res = self.compute_distance_per_cluster_all(x[curr_x_lens > 0,t,:], curr_x_mask)
                
                cluster_distances2[t, :, curr_x_lens > 0] = curr_cluster_distance_res
            else:
                
                curr_cluster_distance_res = self.compute_distance_per_cluster_all(x[curr_x_lens > 0,t,:], x_mask[curr_x_lens > 0,t,:])
                
                cluster_distances2[t, :, curr_x_lens > 0] = curr_cluster_distance_res 
            
            
            sumed_cluster_distnace_res = torch.sqrt(torch.sum(curr_cluster_distance_res, 2))
            
            cluster_distances[curr_x_lens > 0, t] = self.compute_distance_per_cluster(x[curr_x_lens > 0,t,:])
            
#             print(torch.norm(torch.t(sumed_cluster_distnace_res) - cluster_distances[curr_x_lens > 0, t]))
#             if self.use_sparsemax:
            kl = self.kl_div(z_t_category_infer, z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))
#             else:
#                 kl = self.kl_div(z_t_category_infer, z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))
            
            entropy_loss = self.entropy(z_t_category_infer)
            
            if np.isnan(kl.cpu().detach().numpy()).any():
                print('distribution 1::', z_t_category_gen)
                
                print('distribution 2::', z_t_category_infer)
            
            
            kl_states[curr_x_lens > 0,t] = kl
            
            entropy_losses[curr_x_lens > 0, t] = entropy_loss
            
            
               
            
#             print(t, curr_x_lens)
            
            if self.transfer_prob:
                
                z_t_transfer = z_t_category_infer
                
                if self.block == 'GRU':
    #                 output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
                    output, h_now = self.trans(z_t_transfer.view(z_t_transfer.shape[0], 1, z_t_transfer.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
                    
                else:
                    output, (h_now, c_now) = self.trans(z_t_transfer.view(z_t_transfer.shape[0], 1, z_t_transfer.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#                 output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
            else:
                if self.block == 'GRU':
                    output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
#                     output, h_now = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
                    
                else:
#                     output, (h_now, c_now) = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
                    output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
                
                
#             if not self.use_sparsemax:


            
            
            
#             curr_x_lens[curr_x_lens < 0] = 0
            
            
            if t >= 1:            
            
#                 phi_z, z_representation = self.generate_z(z_t_category_gen, t)
                
#                 phi_z = torch.t(torch.mm(self.phi_table, torch.t(z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))))
#                 
#                 phi_z2 = torch.t(torch.mm(self.phi_table, torch.t(z_representation)))
#                 
#                 if self.use_gumbel:
#                     print(torch.norm(phi_z - phi_z2))
                
                
                mean, logvar, logit_x_t = self.generate_x(phi_z_infer, z_t)
                
                
                
                
    
                imputed_x2[curr_x_lens > 0,t] = logit_x_t
                
                
#                 rec_loss = torch.norm(x[:,t+1,:] - mean)**2/(2*std**2) + torch.log(2*np.pi*std**2)/2
                
#                 rec_loss = torch.bmm(((x[:,t,:]-mean)/(std**2)).view(mean.shape[0],1,mean.shape[1]), (x[:,t,:]-mean).view(mean.shape[0],mean.shape[1],1)).view(-1) + (torch.log((2*np.pi)**x[:,t,:].shape[-1]*torch.prod(std, dim= 1))/2).view(-1) 
                
                if self.loss_on_missing:
                    
                    curr_x_t_masks = torch.ones_like(x_t)
                    
                    rec_loss = compute_gaussian_probs0(x_t, mean, logvar, curr_x_t_masks)
                else:
                    rec_loss = compute_gaussian_probs0(x_t, mean, logvar, x_t_mask)
                
#                 rec_loss = self.compute_reconstruction_loss2(x_t, mean, std, batch_size)
                
                
    #             kl = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
    #             kl_states[:,t] = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
    #             logit_x_t = self.emitter(z_t).contiguous() # p(x_t|z_t)         
    #             rec_loss = nn.BCEWithLogitsLoss(reduction='none')(logit_x_t.view(-1), x[:,t,:].contiguous().view(-1)).view(batch_size, -1)
                rec_losses[curr_x_lens > 0,t-1] = rec_loss
                
#                 rmse_loss = torch.sqrt(torch.sum((x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])**2, dim = 1))
                
                rmse_loss = (x_t*x_t_mask - logit_x_t*x_t_mask)**2
                
                mae_loss = torch.abs(x_t*x_t_mask - logit_x_t*x_t_mask)

#                 mae_losses = torch.abs(x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])

            
                rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
                
                mae_losses[curr_x_lens > 0,:,t-1] = mae_loss


            curr_x_lens -= 1

            shrinked_x_lens -= 1
#             else:
#                 z_t_category_gen = F.gumbel_softmax(self.emitter_z(h_now), tau=0.001, dim = 2)
            
#             if t >= 21:
#                 print('here')
            
            h_prev = h_now[:,shrinked_x_lens > 0,:]
            
            if (curr_x_lens == 0).any():
                last_h_now[:, curr_x_lens == 0, :] = h_now[:,shrinked_x_lens <= 0,:]
            
            if self.block == 'LSTM':
                c_prev = c_now[:,shrinked_x_lens > 0,:]
                
                if (curr_x_lens == 0).any():
                    last_c_now[:, curr_x_lens == 0, :] = c_now[:,shrinked_x_lens <= 0,:] 
            
            if self.transfer_prob:
                z_prev = z_t[shrinked_x_lens > 0,:]
            else:
                z_prev = phi_z_infer[shrinked_x_lens > 0,:]

#             if not self.use_sparsemax:
                
#                 print(t, torch.sum(shrinked_x_lens > 0))
            z_t_category_gen = F.softmax(self.emitter_z(h_now[:,shrinked_x_lens > 0,:]), dim = 2)
#             else:
# #                 print(t, torch.sum(shrinked_x_lens > 0))
#                 
#                 if torch.sum(shrinked_x_lens > 0) > 0:
#                 
#                     logit_z_t = self.emitter_z(h_now[:,shrinked_x_lens > 0,:])
#                     
#                     z_t_category_gen = sparsemax(logit_z_t.view(logit_z_t.shape[1], logit_z_t.shape[2]))
#                 
#                     z_t_category_gen = z_t_category_gen.view(1, z_t_category_gen.shape[0], z_t_category_gen.shape[1])
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
                
                
                curr_rnn_out = rnn_out[curr_x_lens > 0,t+1,:]
            
            
            shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]
            
            
#             curr_x_lens = curr_x_lens[curr_x_lens > 0]

#         x_mask = sequence_mask(x_lens)
#         x_mask = x_mask.gt(0).view(-1)
#         rec_loss = rec_losses.view(-1).masked_select(x_mask).mean()
#         kl_loss = kl_states.view(-1).masked_select(x_mask).mean()

        

#         rec_loss = torch.sum(rec_losses)/torch.sum(x_lens-1)
        
        full_cluster_objs, cluster_objs = self.compute_cluster_obj(cluster_distances, prob_sums/torch.sum(x_lens), T_max, x_lens, input_dim)
        
        
        if self.loss_on_missing:
            
            x_mask_full = torch.ones_like(x_mask)
            
            full_cluster_objs2, cluster_objs2 = self.compute_cluster_obj_full2(cluster_distances2, prob_sums/torch.sum(x_lens), T_max, x_mask_full, x_lens)
        else:
            full_cluster_objs2, cluster_objs2 = self.compute_cluster_obj_full2(cluster_distances2, prob_sums/torch.sum(x_lens), T_max, x_mask, x_lens)
        
#         print(torch.norm(full_cluster_objs - full_cluster_objs2))
#          
#         print(torch.norm(cluster_objs - cluster_objs2))
        
        if not self.loss_on_missing:
            rec_loss = torch.sum(rec_losses)/torch.sum(x_mask[:,1:,:])
        else:
            rec_loss = torch.sum(rec_losses)/torch.sum(torch.ones_like(rec_losses))
        
#         for k in range(rec_losses.shape[0]):
#             print(rec_losses[k].mean())
        
        first_kl_loss = kl_states[:, 0].view(-1).mean()
        
        kl_loss = torch.sum(kl_states[:, 1:])/torch.sum(x_lens-1)
        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask[:,1:,:]))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask[:,1:,:])
        
        
        final_entropy_loss = entropy_losses.view(-1).mean()
        print('loss::', rec_loss, kl_loss)
        
        print('rmse loss::', final_rmse_loss)
        
        print('mae loss::', final_mae_losses)
        
        print('cluster objective::', cluster_objs2)
        
        imputed_loss = 0
        
        if self.is_missing:
            imputed_loss = torch.norm(interpolated_x*x_mask - x*x_mask)
            
            print('interpolate loss::', imputed_loss)
        
        if torch.sum(1-new_x_mask) > 0:
            
            imputed_mse_loss = torch.sqrt(torch.sum((((origin_x - x)**2)*(1-new_x_mask)))/torch.sum(1-new_x_mask))
            
            imputed_mse_loss2 = torch.sqrt(torch.sum((((origin_x - imputed_x2)**2)*(1-new_x_mask)))/torch.sum(1-new_x_mask)) 
            
            imputed_loss = torch.sum((torch.abs(origin_x - x)*(1-new_x_mask)))/torch.sum(1-new_x_mask)
            
            imputed_loss2 = torch.sum((torch.abs(origin_x - imputed_x2)*(1-new_x_mask)))/torch.sum(1-new_x_mask) 
            
            print('training imputation rmse loss::', imputed_mse_loss)
            
            print('training imputation rmse loss 2::', imputed_mse_loss2)
            
            print('training imputation mae loss::', imputed_loss)
            
            print('training imputation mae loss 2::', imputed_loss2)
            
            
        if self.block == 'GRU':
            self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, None, T_max)
        else:
            self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, last_c_now, T_max)
        


        
        print()
        
        if not self.evaluate:
            return rec_loss, kl_loss, first_kl_loss, final_rmse_loss, imputed_loss, cluster_objs2
        else:
            
            if torch.sum(1-new_x_mask) > 0:
                return imputed_x2*(1-x_mask) + x*x_mask, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
            else:
                return imputed_x2*(1-x_mask) + x*x_mask, None
    
    
    def evaluate_forecasting_errors0(self, x, origin_x_to_pred, x_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max_train, prior_cluster_probs, gen_probs, x_to_predict_time_stamps):


#         rec_losses = torch.zeros((batch_size, T_max), device=x.device)
        T_max = x_to_predict_lens.max().cpu().item()
        
#         if self.is_missing:
#             imputed_x, interpolated_x = self.impute.forward2(x[:,0:T_max,:], x_mask[:,0:T_max,:], T_max)
#             x = imputed_x
        
        rmse_losses = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
        
        mae_losses = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
        
#         if self.latent and self.lstm_latent:
#             neg_nll_losses = torch.zeros((x.shape[0], (1+self.x_encoder.bidir)*self.s_dim, T_max), device=x.device)
#         else:
        neg_nll_losses = torch.zeros((x.shape[0], self.z_dim, T_max), device=x.device)
        
        curr_x_to_predict_lens = x_to_predict_lens.clone()
        
#         one_time_step_arr = torch.ones_like(curr_x_to_predict_lens)
        
        shrinked_x_to_predict_lens = x_to_predict_lens.clone() 
        
        imputed_x2 = torch.zeros_like(x)
        
        
#         last_h_n2 = last_h_n.clone()
#         
#         last_c_n2 = last_c_n.clone()
        
#         first_point_enc_aug = torch.zeros([1, x.shape[0], self.s_dim], device = self.device)
# 
#         sol_y = self.diffeq_solver(first_point_enc_aug, x_to_predict_time_stamps.type(torch.float))
        
        
        
        for t in range(T_max):
            
#             print(t)
            
#             if not self.use_sparsemax:
            phi_z, z_representation = self.generate_z(gen_probs[:,t], prior_cluster_probs, T_max_train + t, None)
            
#             print(phi_z.shape)
            
            mean, std, logit_x_t = self.generate_x(phi_z, z_representation)
            
#         if not self.latent:
            imputed_x2[curr_x_to_predict_lens>0,t,:] = logit_x_t
#             else:
#                 if not self.lstm_latent:
#                     logit_x_t = self.x_kernel_decoder(logit_x_t)
#                     
#                     imputed_x2[curr_x_to_predict_lens>0,t,:] = logit_x_t
#                 else:
#                     
# #                     print(t, last_decoded_c_n.shape, logit_x_t.shape)
#                     
#                     logit_x_t, (last_decoded_h_n, last_decoded_c_n)  = self.x_kernel_decoder(logit_x_t.view(logit_x_t.shape[0], 1, logit_x_t.shape[1]), torch.ones(logit_x_t.shape[0], device = self.device), init_h = last_decoded_h_n, init_c = last_decoded_c_n)
#                     
#                     logit_x_t = logit_x_t.squeeze(1)
#                     
#                     imputed_x2[curr_x_to_predict_lens>0,t,:] = logit_x_t

            
#             print(last_h_n.shape, last_c_n.shape, logit_x_t.shape)
#             if self.use_gate:
#                 curr_rnn_out, (last_h_n, last_c_n) = self.x_encoder(logit_x_t.view(logit_x_t.shape[0], 1, logit_x_t.shape[1]), torch.ones(logit_x_t.shape[0], device = self.device), init_h = last_h_n, init_c = last_c_n)
            
#             if self.latent and self.lstm_latent:
                
                
#             rec_loss = (x[:,t,:] - logit_x_t)**2

#             print('print shape')
#             
#             print(logit_x_t.shape)
#             
#             print(x[:,t,:].shape)
#             
#             print(x_mask[:,t,:].shape)
            
#             rec_loss = torch.sqrt(torch.sum((x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])**2, dim = 1))
            
            rec_loss = (x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t*x_mask[curr_x_to_predict_lens>0,t,:])**2
            
            mae_loss = torch.abs(x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t*x_mask[curr_x_to_predict_lens>0,t,:])
            
            rmse_losses[curr_x_to_predict_lens>0,:,t] = rec_loss
            
            mae_losses[curr_x_to_predict_lens>0,:,t] = mae_loss
            
#             print(mean.shape, std.shape, torch.sum((curr_x_to_predict_lens>0)))
            
#             if not self.latent:
            neg_nll_losses[curr_x_to_predict_lens>0,:,t] = compute_gaussian_probs0(x[curr_x_to_predict_lens>0,t,:], mean, std, x_mask[curr_x_to_predict_lens>0,t,:])
#             else:
#                 if not self.lstm_latent:
#                     encoded_x_t = self.x_kernel_encoder(x[curr_x_to_predict_lens>0,t,:])
#                     
#                     neg_nll_losses[curr_x_to_predict_lens>0,:,t] = compute_gaussian_probs0(encoded_x_t, mean, std, torch.ones_like(encoded_x_t))
#                 else:
# #                     encoded_x_t = self.x_encoder(x[curr_x_to_predict_lens>0,t,:])
#                     
#                     x_t = x[curr_x_to_predict_lens>0,t,:]
#                 
#                     curr_rnn_out2, (new_last_h_n2, new_last_c_n2) = self.x_encoder(x_t.view(x_t.shape[0], 1, x_t.shape[1]), torch.ones(x_t.shape[0], device = self.device), init_h = last_h_n2, init_c = last_c_n2)
# 
# #                     curr_rnn_out3, (new_last_h_n3, new_last_c_n3) = self.x_encoder.rnn(x_t.view(x_t.shape[0], 1, x_t.shape[1]), (last_h_n2, last_c_n2))
# #                     
# #                     print(torch.norm(curr_rnn_out2 - curr_rnn_out3))
# #                     
# #                     print(torch.norm(new_last_h_n2 - new_last_h_n3))
#                     
#                     last_h_n2, last_c_n2 = new_last_h_n2, new_last_c_n2
#                     
#                     encoded_x_t = curr_rnn_out2.squeeze(1)
#                     
#                     neg_nll_losses[curr_x_to_predict_lens>0,:,t] = compute_gaussian_probs0(encoded_x_t, mean, std, torch.ones_like(encoded_x_t))
            
#             neg_nll_losses[curr_x_to_predict_lens>0,:,t] = self.compute_reconstruction_loss2(x[curr_x_to_predict_lens>0,t,:], mean[curr_x_to_predict_lens>0,:], std[curr_x_to_predict_lens>0,:], torch.sum((curr_x_to_predict_lens>0)))
            
            
#             print('time stamps::', t)
#             z_t_category_gen_trans = z_t_category_gen
#             if self.use_sparsemax:
#                 z_t_category_gen_trans = sparsemax(torch.log(z_t_category_gen+1e-5))
#             
#             if self.transfer_prob:
#                 if self.block == 'GRU':
#                     output, h_now = self.trans(z_t_category_gen_trans.view(z_t_category_gen_trans.shape[1], 1, z_t_category_gen_trans.shape[2]), h_now)# p(z_t| z_{t-1})
#     #                 output, h_now = self.trans(phi_z.view(phi_z.shape[0], 1, phi_z.shape[1]), h_now)# p(z_t| z_{t-1})
#                 else:
#                     output, (h_now, c_now) = self.trans(z_t_category_gen_trans.view(z_t_category_gen_trans.shape[1], 1, z_t_category_gen_trans.shape[2]), (h_now, c_now))# p(z_t| z_{t-1})
#             
#             else:
#                 
#                 phi_z_transfer = torch.t(torch.mm(self.phi_table, torch.t(z_t_category_gen_trans.squeeze(0))))
#                 
#                 if self.block == 'GRU':
#                     output, h_now = self.trans(phi_z_transfer.view(phi_z_transfer.shape[0], 1, phi_z_transfer.shape[1]), h_now)# p(z_t| z_{t-1})
# #                     output, h_now = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
#                     
#                 else:
# #                     output, (h_now, c_now) = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#                     output, (h_now, c_now) = self.trans(phi_z_transfer.view(phi_z_transfer.shape[0], 1, phi_z_transfer.shape[1]), (h_now, c_now))# p(z_t| z_{t-1})
# 
# #                 output, (h_now, c_now) = self.trans(phi_z.view(phi_z.shape[0], 1, phi_z.shape[1]), (h_now, c_now))# p(z_t| z_{t-1})
#             

#             
#             h_now = h_now[:,shrinked_x_to_predict_lens > 0]
# 
#             if self.latent and self.lstm_latent:
#                 last_decoded_c_n = last_decoded_c_n[:,shrinked_x_to_predict_lens > 0]
#                 last_decoded_h_n = last_decoded_h_n[:,shrinked_x_to_predict_lens > 0]
#             
#             
#             if self.use_gate:
#                 curr_rnn_out = curr_rnn_out[shrinked_x_to_predict_lens > 0,0]
#                 
#                 last_h_n = last_h_n[:,shrinked_x_to_predict_lens > 0]
#             
#                 last_c_n = last_c_n[:,shrinked_x_to_predict_lens > 0]
#                 
#                 last_h_n2 = last_h_n2[:,shrinked_x_to_predict_lens > 0]
#             
#                 last_c_n2 = last_c_n2[:,shrinked_x_to_predict_lens > 0]
#             
#             if self.block == 'LSTM':
#                 c_now = c_now[:,shrinked_x_to_predict_lens > 0]
#             
#             phi_z = phi_z[shrinked_x_to_predict_lens > 0]
            
            curr_x_to_predict_lens -= 1
             
            shrinked_x_to_predict_lens -= 1
            
            shrinked_x_to_predict_lens = shrinked_x_to_predict_lens[shrinked_x_to_predict_lens > 0]
            
#             if t < T_max - 1:            
            
            
             
            
            
#             print(z_representation.shape)
        
        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask)
        
        final_nll_loss = torch.sum(neg_nll_losses)/torch.sum(x_mask)
        
        if np.isnan(neg_nll_losses.cpu().detach().numpy()).any():
            print('here')
        
        
        print('forecasting rmse loss::', final_rmse_loss)
        
        print('forecasting mae loss::', final_mae_losses)
        
        print('forecasting neg likelihood::', final_nll_loss)
        
#         if self.is_missing:
#             print('forecasting interpolate loss:', torch.norm(interpolated_x*x_mask - x*x_mask))
        
        if torch.sum(1-x_to_predict_new_mask) > 0:
            imputed_loss = torch.sum((torch.abs(origin_x_to_pred - x)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask)
            imputed_loss2 = torch.sum((torch.abs(origin_x_to_pred - imputed_x2)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask)
            
            imputed_mse_loss = torch.sqrt(torch.sum((((origin_x_to_pred - x)**2)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask))
            
            imputed_mse_loss2 = torch.sqrt(torch.sum((((origin_x_to_pred - imputed_x2)**2)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask))
            
            print('forecasting imputation rmse loss::', imputed_mse_loss)
            print('forecasting imputation rmse loss 2::', imputed_mse_loss2)
            
            print('forecasting imputation mae loss::', imputed_loss)
            print('forecasting imputation mae loss 2::', imputed_loss2)
        
        all_masks = torch.sum(x_mask)
        
        rmse_list, mae_list = get_forecasting_res_by_time_steps(x, imputed_x2, x_mask)
        
        if not self.evaluate:
            return final_rmse_loss, torch.sum(x_mask), final_mae_losses, torch.sum(x_mask), final_nll_loss, torch.sum(x_mask), (rmse_list, mae_list, torch.sum(x_mask))
        else:
            return imputed_x2*(1-x_mask) + x*x_mask
        
    
    
    
#     def evaluate_forecasting_errors2(self, x, origin_x_to_pred, x_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max_train, prior_cluster_probs, gen_probs, x_to_predict_time_stamps):
# 
# 
# #         rec_losses = torch.zeros((batch_size, T_max), device=x.device)
#         T_max = x_to_predict_lens.max().cpu().item()
#         
# #         if self.is_missing:
# #             imputed_x, interpolated_x = self.impute.forward2(x[:,0:T_max,:], x_mask[:,0:T_max,:], T_max)
# #             x = imputed_x
#         
#         rmse_losses = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
#         
#         mae_losses = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
#         
#         if self.latent and self.lstm_latent:
#             neg_nll_losses = torch.zeros((x.shape[0], (1+self.x_encoder.bidir)*self.s_dim, T_max), device=x.device)
#         else:
#             neg_nll_losses = torch.zeros((x.shape[0], self.z_dim, T_max), device=x.device)
#         
#         curr_x_to_predict_lens = x_to_predict_lens.clone()
#         
# #         one_time_step_arr = torch.ones_like(curr_x_to_predict_lens)
#         
#         shrinked_x_to_predict_lens = x_to_predict_lens.clone() 
#         
#         imputed_x2 = torch.zeros_like(x)
#         
#         
# #         last_h_n2 = last_h_n.clone()
# #         
# #         last_c_n2 = last_c_n.clone()
#         
# #         first_point_enc_aug = torch.zeros([1, x.shape[0], self.s_dim], device = self.device)
# # 
# #         sol_y = self.diffeq_solver(first_point_enc_aug, x_to_predict_time_stamps.type(torch.float))
#         
#         
#         
#         for t in range(T_max):
#             
# #             print(t)
#             
# #             if not self.use_sparsemax:
#             phi_z, z_representation = self.generate_z(gen_probs[:,t], prior_cluster_probs, T_max_train + t, None)
#             
# #             print(phi_z.shape)
#             
#             mean, std, logit_x_t = self.generate_x(phi_z, z_representation)
#             
# #         if not self.latent:
#             imputed_x2[curr_x_to_predict_lens>0,t,:] = logit_x_t
# #             else:
# #                 if not self.lstm_latent:
# #                     logit_x_t = self.x_kernel_decoder(logit_x_t)
# #                     
# #                     imputed_x2[curr_x_to_predict_lens>0,t,:] = logit_x_t
# #                 else:
# #                     
# # #                     print(t, last_decoded_c_n.shape, logit_x_t.shape)
# #                     
# #                     logit_x_t, (last_decoded_h_n, last_decoded_c_n)  = self.x_kernel_decoder(logit_x_t.view(logit_x_t.shape[0], 1, logit_x_t.shape[1]), torch.ones(logit_x_t.shape[0], device = self.device), init_h = last_decoded_h_n, init_c = last_decoded_c_n)
# #                     
# #                     logit_x_t = logit_x_t.squeeze(1)
# #                     
# #                     imputed_x2[curr_x_to_predict_lens>0,t,:] = logit_x_t
# 
#             
# #             print(last_h_n.shape, last_c_n.shape, logit_x_t.shape)
# #             if self.use_gate:
# #                 curr_rnn_out, (last_h_n, last_c_n) = self.x_encoder(logit_x_t.view(logit_x_t.shape[0], 1, logit_x_t.shape[1]), torch.ones(logit_x_t.shape[0], device = self.device), init_h = last_h_n, init_c = last_c_n)
#             
# #             if self.latent and self.lstm_latent:
#                 
#                 
# #             rec_loss = (x[:,t,:] - logit_x_t)**2
# 
# #             print('print shape')
# #             
# #             print(logit_x_t.shape)
# #             
# #             print(x[:,t,:].shape)
# #             
# #             print(x_mask[:,t,:].shape)
#             
# #             rec_loss = torch.sqrt(torch.sum((x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])**2, dim = 1))
#             
#             rec_loss = (x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t*x_mask[curr_x_to_predict_lens>0,t,:])**2
#             
#             mae_loss = torch.abs(x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t*x_mask[curr_x_to_predict_lens>0,t,:])
#             
#             rmse_losses[curr_x_to_predict_lens>0,:,t] = rec_loss
#             
#             mae_losses[curr_x_to_predict_lens>0,:,t] = mae_loss
#             
# #             print(mean.shape, std.shape, torch.sum((curr_x_to_predict_lens>0)))
#             
# #             if not self.latent:
#             neg_nll_losses[curr_x_to_predict_lens>0,:,t] = compute_gaussian_probs0(x[curr_x_to_predict_lens>0,t,:], mean, std, x_mask[curr_x_to_predict_lens>0,t,:])
# #             else:
# #                 if not self.lstm_latent:
# #                     encoded_x_t = self.x_kernel_encoder(x[curr_x_to_predict_lens>0,t,:])
# #                     
# #                     neg_nll_losses[curr_x_to_predict_lens>0,:,t] = compute_gaussian_probs0(encoded_x_t, mean, std, torch.ones_like(encoded_x_t))
# #                 else:
# # #                     encoded_x_t = self.x_encoder(x[curr_x_to_predict_lens>0,t,:])
# #                     
# #                     x_t = x[curr_x_to_predict_lens>0,t,:]
# #                 
# #                     curr_rnn_out2, (new_last_h_n2, new_last_c_n2) = self.x_encoder(x_t.view(x_t.shape[0], 1, x_t.shape[1]), torch.ones(x_t.shape[0], device = self.device), init_h = last_h_n2, init_c = last_c_n2)
# # 
# # #                     curr_rnn_out3, (new_last_h_n3, new_last_c_n3) = self.x_encoder.rnn(x_t.view(x_t.shape[0], 1, x_t.shape[1]), (last_h_n2, last_c_n2))
# # #                     
# # #                     print(torch.norm(curr_rnn_out2 - curr_rnn_out3))
# # #                     
# # #                     print(torch.norm(new_last_h_n2 - new_last_h_n3))
# #                     
# #                     last_h_n2, last_c_n2 = new_last_h_n2, new_last_c_n2
# #                     
# #                     encoded_x_t = curr_rnn_out2.squeeze(1)
# #                     
# #                     neg_nll_losses[curr_x_to_predict_lens>0,:,t] = compute_gaussian_probs0(encoded_x_t, mean, std, torch.ones_like(encoded_x_t))
#             
# #             neg_nll_losses[curr_x_to_predict_lens>0,:,t] = self.compute_reconstruction_loss2(x[curr_x_to_predict_lens>0,t,:], mean[curr_x_to_predict_lens>0,:], std[curr_x_to_predict_lens>0,:], torch.sum((curr_x_to_predict_lens>0)))
#             
#             
# #             print('time stamps::', t)
# #             z_t_category_gen_trans = z_t_category_gen
# #             if self.use_sparsemax:
# #                 z_t_category_gen_trans = sparsemax(torch.log(z_t_category_gen+1e-5))
# #             
# #             if self.transfer_prob:
# #                 if self.block == 'GRU':
# #                     output, h_now = self.trans(z_t_category_gen_trans.view(z_t_category_gen_trans.shape[1], 1, z_t_category_gen_trans.shape[2]), h_now)# p(z_t| z_{t-1})
# #     #                 output, h_now = self.trans(phi_z.view(phi_z.shape[0], 1, phi_z.shape[1]), h_now)# p(z_t| z_{t-1})
# #                 else:
# #                     output, (h_now, c_now) = self.trans(z_t_category_gen_trans.view(z_t_category_gen_trans.shape[1], 1, z_t_category_gen_trans.shape[2]), (h_now, c_now))# p(z_t| z_{t-1})
# #             
# #             else:
# #                 
# #                 phi_z_transfer = torch.t(torch.mm(self.phi_table, torch.t(z_t_category_gen_trans.squeeze(0))))
# #                 
# #                 if self.block == 'GRU':
# #                     output, h_now = self.trans(phi_z_transfer.view(phi_z_transfer.shape[0], 1, phi_z_transfer.shape[1]), h_now)# p(z_t| z_{t-1})
# # #                     output, h_now = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
# #                     
# #                 else:
# # #                     output, (h_now, c_now) = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
# #                     output, (h_now, c_now) = self.trans(phi_z_transfer.view(phi_z_transfer.shape[0], 1, phi_z_transfer.shape[1]), (h_now, c_now))# p(z_t| z_{t-1})
# # 
# # #                 output, (h_now, c_now) = self.trans(phi_z.view(phi_z.shape[0], 1, phi_z.shape[1]), (h_now, c_now))# p(z_t| z_{t-1})
# #             
# 
# #             
# #             h_now = h_now[:,shrinked_x_to_predict_lens > 0]
# # 
# #             if self.latent and self.lstm_latent:
# #                 last_decoded_c_n = last_decoded_c_n[:,shrinked_x_to_predict_lens > 0]
# #                 last_decoded_h_n = last_decoded_h_n[:,shrinked_x_to_predict_lens > 0]
# #             
# #             
# #             if self.use_gate:
# #                 curr_rnn_out = curr_rnn_out[shrinked_x_to_predict_lens > 0,0]
# #                 
# #                 last_h_n = last_h_n[:,shrinked_x_to_predict_lens > 0]
# #             
# #                 last_c_n = last_c_n[:,shrinked_x_to_predict_lens > 0]
# #                 
# #                 last_h_n2 = last_h_n2[:,shrinked_x_to_predict_lens > 0]
# #             
# #                 last_c_n2 = last_c_n2[:,shrinked_x_to_predict_lens > 0]
# #             
# #             if self.block == 'LSTM':
# #                 c_now = c_now[:,shrinked_x_to_predict_lens > 0]
# #             
# #             phi_z = phi_z[shrinked_x_to_predict_lens > 0]
#             
#             curr_x_to_predict_lens -= 1
#              
#             shrinked_x_to_predict_lens -= 1
#             
#             shrinked_x_to_predict_lens = shrinked_x_to_predict_lens[shrinked_x_to_predict_lens > 0]
#             
# #             if t < T_max - 1:            
#             
#             
#              
#             
#             
# #             print(z_representation.shape)
#         
#         final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask))
#         
#         final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask)
#         
#         final_nll_loss = torch.sum(neg_nll_losses)/torch.sum(x_mask)
#         
#         if np.isnan(neg_nll_losses.cpu().detach().numpy()).any():
#             print('here')
#         
#         
#         print('forecasting rmse loss::', final_rmse_loss)
#         
#         print('forecasting mae loss::', final_mae_losses)
#         
#         print('forecasting neg likelihood::', final_nll_loss)
#         
# #         if self.is_missing:
# #             print('forecasting interpolate loss:', torch.norm(interpolated_x*x_mask - x*x_mask))
#         
#         if torch.sum(1-x_to_predict_new_mask) > 0:
#             imputed_loss = torch.sum((torch.abs(origin_x_to_pred - x)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask)
#             imputed_loss2 = torch.sum((torch.abs(origin_x_to_pred - imputed_x2)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask)
#             
#             imputed_mse_loss = torch.sqrt(torch.sum((((origin_x_to_pred - x)**2)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask))
#             
#             imputed_mse_loss2 = torch.sqrt(torch.sum((((origin_x_to_pred - imputed_x2)**2)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask))
#             
#             print('forecasting imputation rmse loss::', imputed_mse_loss)
#             print('forecasting imputation rmse loss 2::', imputed_mse_loss2)
#             
#             print('forecasting imputation mae loss::', imputed_loss)
#             print('forecasting imputation mae loss 2::', imputed_loss2)
#         
#         all_masks = torch.sum(x_mask)
#         
#         rmse_list, mae_list = get_forecasting_res_by_time_steps(x, imputed_x2, x_mask)
#         
#         if not self.evaluate:
#             return final_rmse_loss, torch.sum(x_mask), final_mae_losses, torch.sum(x_mask), final_nll_loss, torch.sum(x_mask), (rmse_list, mae_list, torch.sum(x_mask))
#         else:
#             return imputed_x2*(1-x_mask) + x*x_mask
#     
    def evaluate_forecasting_errors1(self, x, origin_x_to_pred, x_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max_train, prior_cluster_probs, last_gen_probs, last_gen_states, last_time_step, x_to_predict_time_stamps, last_infer_states):


#         rec_losses = torch.zeros((batch_size, T_max), device=x.device)
        T_max = x_to_predict_lens.max().cpu().item()
        
#         if self.is_missing:
#             imputed_x, interpolated_x = self.impute.forward2(x[:,0:T_max,:], x_mask[:,0:T_max,:], T_max)
#             x = imputed_x
        
        rmse_losses = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
        
        mae_losses = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
        
        rmse_losses2 = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
        
        mae_losses2 = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
        
        
#         if self.latent and self.lstm_latent:
#             neg_nll_losses = torch.zeros((x.shape[0], (1+self.x_encoder.bidir)*self.s_dim, T_max), device=x.device)
#             
#             neg_nll_losses2 = torch.zeros((x.shape[0], (1+self.x_encoder.bidir)*self.s_dim, T_max), device=x.device)
#         else:
        neg_nll_losses = torch.zeros((x.shape[0], self.z_dim, T_max), device=x.device)
        neg_nll_losses2 = torch.zeros((x.shape[0], self.z_dim, T_max), device=x.device)
        
        curr_x_to_predict_lens = x_to_predict_lens.clone()
        
#         one_time_step_arr = torch.ones_like(curr_x_to_predict_lens)
        
        shrinked_x_to_predict_lens = x_to_predict_lens.clone() 
        
        imputed_x2 = torch.zeros_like(x)
        
        imputed_x2_2 = torch.zeros_like(x)
        
        
#         last_h_n2 = last_h_n.clone()
#         
#         last_c_n2 = last_c_n.clone()
        
#         first_point_enc_aug = torch.zeros([1, x.shape[0], self.s_dim], device = self.device)
# 
#         sol_y = self.diffeq_solver(first_point_enc_aug, x_to_predict_time_stamps.type(torch.float))

        predicted_time_stamp_list = []
        
        predicted_time_stamp_list.append(last_time_step)
        
        predicted_time_stamp_list.extend(x_to_predict_time_stamps[0].tolist())

        ode_sol = self.trans.z0_diffeq_solver(last_gen_states, torch.tensor(predicted_time_stamp_list, dtype =torch.float))
        
        print(ode_sol.shape)
        
        gen_probs = self.trans.emit_probs(ode_sol[:,:,1:].squeeze(0))
        
        print(gen_probs.shape)
        
        for t in range(T_max):
            
#             print(t)
            
#             if not self.use_sparsemax:
#             if self.use_sparsemax:
#                 #             if not self.use_sparse:
#                 last_gen_probs = self.trans.sparsemax(last_gen_probs)
            
#             print(last_time_step, x_to_predict_time_stamps[0,t])
            
            time_steps = torch.tensor([last_time_step.item(), x_to_predict_time_stamps[0,t].item()])



            
            last_gen_probs, last_gen_states = self.trans.run_odernn_single_step(last_gen_probs[:,curr_x_to_predict_lens>0,:], time_steps, prev_y_state = last_gen_states)
            
            if self.use_gate:
                phi_z, z_representation = self.generate_z(last_gen_probs.squeeze(0), prior_cluster_probs, T_max_train + t, last_infer_states.squeeze(0))
                
                phi_z2, z_representation2 = self.generate_z(gen_probs[:,t], prior_cluster_probs, T_max_train + t, last_infer_states.squeeze(0))
            else:
                phi_z, z_representation = self.generate_z(last_gen_probs.squeeze(0), prior_cluster_probs, T_max_train + t, None)
                
                phi_z2, z_representation2 = self.generate_z(gen_probs[:,t], prior_cluster_probs, T_max_train + t, None)
            
#             print(phi_z.shape)
            
            mean, std, logit_x_t = self.generate_x(phi_z, z_representation)
            
            mean2, std2, logit_x_t2 = self.generate_x(phi_z2, z_representation2)
            
            last_time_step = x_to_predict_time_stamps[0,t]
            
#         if not self.latent:
            imputed_x2[curr_x_to_predict_lens>0,t,:] = logit_x_t
            
            imputed_x2_2[curr_x_to_predict_lens>0,t,:] = logit_x_t2
            
            
            if self.use_gate:
                
                if self.use_mask:
                    
#                     time_steps = torch.tensor([truth_time_steps[k].item(), truth_time_steps[k + self.shift + 1].item()])
                    
                    input_x = torch.cat([logit_x_t, torch.ones_like(logit_x_t)], -1)
                    
                    last_infer_states = self.postnet.run_odernn_single_step(input_x.unsqueeze(0), time_steps, prev_y_state = last_infer_states)
                    
                else:
                    input_x = logit_x_t
                    
                    last_infer_states = self.postnet.run_odernn_single_step(input_x.unsqueeze(0), time_steps, prev_y_state = last_infer_states)
                    
#                     curr_rnn_out, (last_h_n, last_c_n) = self.x_encoder(input_x.view(input_x.shape[0], 1, input_x.shape[1]), torch.ones(logit_x_t.shape[0], device = self.device), init_h = last_h_n, init_c = last_c_n)

            
            
            
            
            
#             else:
#                 if not self.lstm_latent:
#                     logit_x_t = self.x_kernel_decoder(logit_x_t)
#                     
#                     imputed_x2[curr_x_to_predict_lens>0,t,:] = logit_x_t
#                 else:
#                     
# #                     print(t, last_decoded_c_n.shape, logit_x_t.shape)
#                     
#                     logit_x_t, (last_decoded_h_n, last_decoded_c_n)  = self.x_kernel_decoder(logit_x_t.view(logit_x_t.shape[0], 1, logit_x_t.shape[1]), torch.ones(logit_x_t.shape[0], device = self.device), init_h = last_decoded_h_n, init_c = last_decoded_c_n)
#                     
#                     logit_x_t = logit_x_t.squeeze(1)
#                     
#                     imputed_x2[curr_x_to_predict_lens>0,t,:] = logit_x_t

            
#             print(last_h_n.shape, last_c_n.shape, logit_x_t.shape)
#             if self.use_gate:
#                 curr_rnn_out, (last_h_n, last_c_n) = self.x_encoder(logit_x_t.view(logit_x_t.shape[0], 1, logit_x_t.shape[1]), torch.ones(logit_x_t.shape[0], device = self.device), init_h = last_h_n, init_c = last_c_n)
            
#             if self.latent and self.lstm_latent:
                
                
#             rec_loss = (x[:,t,:] - logit_x_t)**2

#             print('print shape')
#             
#             print(logit_x_t.shape)
#             
#             print(x[:,t,:].shape)
#             
#             print(x_mask[:,t,:].shape)
            
#             rec_loss = torch.sqrt(torch.sum((x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])**2, dim = 1))
            
            rec_loss = (x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t*x_mask[curr_x_to_predict_lens>0,t,:])**2
            
            mae_loss = torch.abs(x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t*x_mask[curr_x_to_predict_lens>0,t,:])
            
            rec_loss2 = (x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t2*x_mask[curr_x_to_predict_lens>0,t,:])**2
            
            mae_loss2 = torch.abs(x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t2*x_mask[curr_x_to_predict_lens>0,t,:])
            
            rmse_losses[curr_x_to_predict_lens>0,:,t] = rec_loss
            
            mae_losses[curr_x_to_predict_lens>0,:,t] = mae_loss
            
            rmse_losses2[curr_x_to_predict_lens>0,:,t] = rec_loss2
            
            mae_losses2[curr_x_to_predict_lens>0,:,t] = mae_loss2
            
#             print(mean.shape, std.shape, torch.sum((curr_x_to_predict_lens>0)))
            
#             if not self.latent:
            neg_nll_losses[curr_x_to_predict_lens>0,:,t] = compute_gaussian_probs0(x[curr_x_to_predict_lens>0,t,:], mean, std, x_mask[curr_x_to_predict_lens>0,t,:])
            
            neg_nll_losses2[curr_x_to_predict_lens>0,:,t] = compute_gaussian_probs0(x[curr_x_to_predict_lens>0,t,:], mean2, std2, x_mask[curr_x_to_predict_lens>0,t,:])
#             else:
#                 if not self.lstm_latent:
#                     encoded_x_t = self.x_kernel_encoder(x[curr_x_to_predict_lens>0,t,:])
#                     
#                     neg_nll_losses[curr_x_to_predict_lens>0,:,t] = compute_gaussian_probs0(encoded_x_t, mean, std, torch.ones_like(encoded_x_t))
#                 else:
# #                     encoded_x_t = self.x_encoder(x[curr_x_to_predict_lens>0,t,:])
#                     
#                     x_t = x[curr_x_to_predict_lens>0,t,:]
#                 
#                     curr_rnn_out2, (new_last_h_n2, new_last_c_n2) = self.x_encoder(x_t.view(x_t.shape[0], 1, x_t.shape[1]), torch.ones(x_t.shape[0], device = self.device), init_h = last_h_n2, init_c = last_c_n2)
# 
# #                     curr_rnn_out3, (new_last_h_n3, new_last_c_n3) = self.x_encoder.rnn(x_t.view(x_t.shape[0], 1, x_t.shape[1]), (last_h_n2, last_c_n2))
# #                     
# #                     print(torch.norm(curr_rnn_out2 - curr_rnn_out3))
# #                     
# #                     print(torch.norm(new_last_h_n2 - new_last_h_n3))
#                     
#                     last_h_n2, last_c_n2 = new_last_h_n2, new_last_c_n2
#                     
#                     encoded_x_t = curr_rnn_out2.squeeze(1)
#                     
#                     neg_nll_losses[curr_x_to_predict_lens>0,:,t] = compute_gaussian_probs0(encoded_x_t, mean, std, torch.ones_like(encoded_x_t))
            
#             neg_nll_losses[curr_x_to_predict_lens>0,:,t] = self.compute_reconstruction_loss2(x[curr_x_to_predict_lens>0,t,:], mean[curr_x_to_predict_lens>0,:], std[curr_x_to_predict_lens>0,:], torch.sum((curr_x_to_predict_lens>0)))
            
            
#             print('time stamps::', t)
#             z_t_category_gen_trans = z_t_category_gen
#             if self.use_sparsemax:
#                 z_t_category_gen_trans = sparsemax(torch.log(z_t_category_gen+1e-5))
#             
#             if self.transfer_prob:
#                 if self.block == 'GRU':
#                     output, h_now = self.trans(z_t_category_gen_trans.view(z_t_category_gen_trans.shape[1], 1, z_t_category_gen_trans.shape[2]), h_now)# p(z_t| z_{t-1})
#     #                 output, h_now = self.trans(phi_z.view(phi_z.shape[0], 1, phi_z.shape[1]), h_now)# p(z_t| z_{t-1})
#                 else:
#                     output, (h_now, c_now) = self.trans(z_t_category_gen_trans.view(z_t_category_gen_trans.shape[1], 1, z_t_category_gen_trans.shape[2]), (h_now, c_now))# p(z_t| z_{t-1})
#             
#             else:
#                 
#                 phi_z_transfer = torch.t(torch.mm(self.phi_table, torch.t(z_t_category_gen_trans.squeeze(0))))
#                 
#                 if self.block == 'GRU':
#                     output, h_now = self.trans(phi_z_transfer.view(phi_z_transfer.shape[0], 1, phi_z_transfer.shape[1]), h_now)# p(z_t| z_{t-1})
# #                     output, h_now = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
#                     
#                 else:
# #                     output, (h_now, c_now) = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#                     output, (h_now, c_now) = self.trans(phi_z_transfer.view(phi_z_transfer.shape[0], 1, phi_z_transfer.shape[1]), (h_now, c_now))# p(z_t| z_{t-1})
# 
# #                 output, (h_now, c_now) = self.trans(phi_z.view(phi_z.shape[0], 1, phi_z.shape[1]), (h_now, c_now))# p(z_t| z_{t-1})
#             

#             
#             h_now = h_now[:,shrinked_x_to_predict_lens > 0]
# 
#             if self.latent and self.lstm_latent:
#                 last_decoded_c_n = last_decoded_c_n[:,shrinked_x_to_predict_lens > 0]
#                 last_decoded_h_n = last_decoded_h_n[:,shrinked_x_to_predict_lens > 0]
#             
#             
#             if self.use_gate:
#                 curr_rnn_out = curr_rnn_out[shrinked_x_to_predict_lens > 0,0]
#                 
#                 last_h_n = last_h_n[:,shrinked_x_to_predict_lens > 0]
#             
#                 last_c_n = last_c_n[:,shrinked_x_to_predict_lens > 0]
#                 
#                 last_h_n2 = last_h_n2[:,shrinked_x_to_predict_lens > 0]
#             
#                 last_c_n2 = last_c_n2[:,shrinked_x_to_predict_lens > 0]
#             
#             if self.block == 'LSTM':
#                 c_now = c_now[:,shrinked_x_to_predict_lens > 0]
#             
#             phi_z = phi_z[shrinked_x_to_predict_lens > 0]
            
            curr_x_to_predict_lens -= 1
             
            shrinked_x_to_predict_lens -= 1
            
            shrinked_x_to_predict_lens = shrinked_x_to_predict_lens[shrinked_x_to_predict_lens > 0]
            
#             if t < T_max - 1:            
            
            
             
            
            
#             print(z_representation.shape)
        
        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask)
        
        final_nll_loss = torch.sum(neg_nll_losses)/torch.sum(x_mask)
        
        final_rmse_loss2 = torch.sqrt(torch.sum(rmse_losses2)/torch.sum(x_mask))
        
        final_mae_losses2 = torch.sum(mae_losses2)/torch.sum(x_mask)
        
        final_nll_loss2 = torch.sum(neg_nll_losses2)/torch.sum(x_mask)
        
        if np.isnan(neg_nll_losses.cpu().detach().numpy()).any():
            print('here')
        
        
        print('forecasting rmse loss::', final_rmse_loss)
        
        print('forecasting mae loss::', final_mae_losses)
        
        print('forecasting neg likelihood::', final_nll_loss)

        print('forecasting rmse loss 2::', final_rmse_loss2)
        
        print('forecasting mae loss 2::', final_mae_losses2)
        
        print('forecasting neg likelihood 2::', final_nll_loss2)
        
#         if self.is_missing:
#             print('forecasting interpolate loss:', torch.norm(interpolated_x*x_mask - x*x_mask))
        
        if torch.sum(1-x_to_predict_new_mask) > 0:
            imputed_loss = torch.sum((torch.abs(origin_x_to_pred - x)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask)
            imputed_loss2 = torch.sum((torch.abs(origin_x_to_pred - imputed_x2)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask)
            
            imputed_mse_loss = torch.sqrt(torch.sum((((origin_x_to_pred - x)**2)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask))
            
            imputed_mse_loss2 = torch.sqrt(torch.sum((((origin_x_to_pred - imputed_x2)**2)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask))
            
            print('forecasting imputation rmse loss::', imputed_mse_loss)
            print('forecasting imputation rmse loss 2::', imputed_mse_loss2)
            
            print('forecasting imputation mae loss::', imputed_loss)
            print('forecasting imputation mae loss 2::', imputed_loss2)
        
        all_masks = torch.sum(x_mask)
        
        
        rmse_list, mae_list = get_forecasting_res_by_time_steps(x, imputed_x2, x_mask)
        
        rmse_list2, mae_list2 = get_forecasting_res_by_time_steps(x, imputed_x2_2, x_mask)
        
        if not self.evaluate:
            return (final_rmse_loss, final_rmse_loss2), torch.sum(x_mask), (final_mae_losses, final_mae_losses2), torch.sum(x_mask), (final_nll_loss, final_nll_loss2), torch.sum(x_mask), ((rmse_list, rmse_list2), (mae_list, mae_list2), torch.sum(x_mask))
        else:
            return imputed_x2*(1-x_mask) + x*x_mask
    
    def evaluate_forecasting_errors(self, x, origin_x_to_pred, x_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, h_now, c_now, T_max_train, prior_cluster_probs, curr_rnn_out, last_h_n, last_c_n, last_decoded_h_n, last_decoded_c_n):


#         rec_losses = torch.zeros((batch_size, T_max), device=x.device)
        T_max = x_to_predict_lens.max().cpu().item()
        
        if self.is_missing:
            imputed_x, interpolated_x = self.impute.forward2(x[:,0:T_max,:], x_mask[:,0:T_max,:], T_max)
            x = imputed_x
        
        rmse_losses = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
        
        mae_losses = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
        
#         if self.latent and self.lstm_latent:
#             neg_nll_losses = torch.zeros((x.shape[0], (1+self.x_encoder.bidir)*self.s_dim, T_max), device=x.device)
#         else:
        neg_nll_losses = torch.zeros((x.shape[0], self.z_dim, T_max), device=x.device)
        
        curr_x_to_predict_lens = x_to_predict_lens.clone()
        
#         one_time_step_arr = torch.ones_like(curr_x_to_predict_lens)
        
        shrinked_x_to_predict_lens = x_to_predict_lens.clone() 
        
        imputed_x2 = torch.zeros_like(x)
        
        
        last_h_n2 = last_h_n.clone()
        
        last_c_n2 = last_c_n.clone()
        
        for t in range(T_max):
            
#             print(t)
            
#             if not self.use_sparsemax:
            z_t_category_gen = F.softmax(self.emitter_z(h_now), dim = 2)
            
#             else:
# #                 print(t, torch.sum(shrinked_x_lens > 0))
#                 
#                 if torch.sum(shrinked_x_to_predict_lens > 0) > 0:
#                 
#                     logit_z_t = self.emitter_z(h_now[:,shrinked_x_to_predict_lens > 0,:])
#                     
#                     z_t_category_gen = sparsemax(logit_z_t.view(logit_z_t.shape[1], logit_z_t.shape[2]))
#                 
#                     z_t_category_gen = z_t_category_gen.view(1, z_t_category_gen.shape[0], z_t_category_gen.shape[1])
            
            
            
            phi_z, z_representation = self.generate_z(z_t_category_gen, prior_cluster_probs, T_max_train + t, curr_rnn_out)
            
#             print(phi_z.shape)
            
            mean, std, logit_x_t = self.generate_x(phi_z, z_representation)
            
#             if not self.latent:
            imputed_x2[curr_x_to_predict_lens>0,t,:] = logit_x_t
#             else:
#                 if not self.lstm_latent:
#                     logit_x_t = self.x_kernel_decoder(logit_x_t)
#                     
#                     imputed_x2[curr_x_to_predict_lens>0,t,:] = logit_x_t
#                 else:
#                     
# #                     print(t, last_decoded_c_n.shape, logit_x_t.shape)
#                     
#                     logit_x_t, (last_decoded_h_n, last_decoded_c_n)  = self.x_kernel_decoder(logit_x_t.view(logit_x_t.shape[0], 1, logit_x_t.shape[1]), torch.ones(logit_x_t.shape[0], device = self.device), init_h = last_decoded_h_n, init_c = last_decoded_c_n)
#                     
#                     logit_x_t = logit_x_t.squeeze(1)
#                     
#                     imputed_x2[curr_x_to_predict_lens>0,t,:] = logit_x_t

            
#             print(last_h_n.shape, last_c_n.shape, logit_x_t.shape)
            if self.use_gate:
                curr_rnn_out, (last_h_n, last_c_n) = self.x_encoder(logit_x_t.view(logit_x_t.shape[0], 1, logit_x_t.shape[1]), torch.ones(logit_x_t.shape[0], device = self.device), init_h = last_h_n, init_c = last_c_n)
            
#             if self.latent and self.lstm_latent:
                
                
#             rec_loss = (x[:,t,:] - logit_x_t)**2

#             print('print shape')
#             
#             print(logit_x_t.shape)
#             
#             print(x[:,t,:].shape)
#             
#             print(x_mask[:,t,:].shape)
            
#             rec_loss = torch.sqrt(torch.sum((x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])**2, dim = 1))
            
            rec_loss = (x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t*x_mask[curr_x_to_predict_lens>0,t,:])**2
            
            mae_loss = torch.abs(x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t*x_mask[curr_x_to_predict_lens>0,t,:])
            
            rmse_losses[curr_x_to_predict_lens>0,:,t] = rec_loss
            
            mae_losses[curr_x_to_predict_lens>0,:,t] = mae_loss
            
#             print(mean.shape, std.shape, torch.sum((curr_x_to_predict_lens>0)))
            
#             if not self.latent:
            neg_nll_losses[curr_x_to_predict_lens>0,:,t] = compute_gaussian_probs0(x[curr_x_to_predict_lens>0,t,:], mean, std, x_mask[curr_x_to_predict_lens>0,t,:])
#             else:
#                 if not self.lstm_latent:
#                     encoded_x_t = self.x_kernel_encoder(x[curr_x_to_predict_lens>0,t,:])
#                     
#                     neg_nll_losses[curr_x_to_predict_lens>0,:,t] = compute_gaussian_probs0(encoded_x_t, mean, std, torch.ones_like(encoded_x_t))
#                 else:
# #                     encoded_x_t = self.x_encoder(x[curr_x_to_predict_lens>0,t,:])
#                     
#                     x_t = x[curr_x_to_predict_lens>0,t,:]
#                 
#                     curr_rnn_out2, (new_last_h_n2, new_last_c_n2) = self.x_encoder(x_t.view(x_t.shape[0], 1, x_t.shape[1]), torch.ones(x_t.shape[0], device = self.device), init_h = last_h_n2, init_c = last_c_n2)
# 
# #                     curr_rnn_out3, (new_last_h_n3, new_last_c_n3) = self.x_encoder.rnn(x_t.view(x_t.shape[0], 1, x_t.shape[1]), (last_h_n2, last_c_n2))
# #                     
# #                     print(torch.norm(curr_rnn_out2 - curr_rnn_out3))
# #                     
# #                     print(torch.norm(new_last_h_n2 - new_last_h_n3))
#                     
#                     last_h_n2, last_c_n2 = new_last_h_n2, new_last_c_n2
#                     
#                     encoded_x_t = curr_rnn_out2.squeeze(1)
#                     
#                     neg_nll_losses[curr_x_to_predict_lens>0,:,t] = compute_gaussian_probs0(encoded_x_t, mean, std, torch.ones_like(encoded_x_t))
            
#             neg_nll_losses[curr_x_to_predict_lens>0,:,t] = self.compute_reconstruction_loss2(x[curr_x_to_predict_lens>0,t,:], mean[curr_x_to_predict_lens>0,:], std[curr_x_to_predict_lens>0,:], torch.sum((curr_x_to_predict_lens>0)))
            
            
#             print('time stamps::', t)
            z_t_category_gen_trans = z_t_category_gen
#             if self.use_sparsemax:
#                 z_t_category_gen_trans = sparsemax(torch.log(z_t_category_gen+1e-5))
            
            if self.transfer_prob:
                if self.block == 'GRU':
                    output, h_now = self.trans(z_t_category_gen_trans.view(z_t_category_gen_trans.shape[1], 1, z_t_category_gen_trans.shape[2]), h_now)# p(z_t| z_{t-1})
    #                 output, h_now = self.trans(phi_z.view(phi_z.shape[0], 1, phi_z.shape[1]), h_now)# p(z_t| z_{t-1})
                else:
                    output, (h_now, c_now) = self.trans(z_t_category_gen_trans.view(z_t_category_gen_trans.shape[1], 1, z_t_category_gen_trans.shape[2]), (h_now, c_now))# p(z_t| z_{t-1})
            
            else:
                
                phi_z_transfer = torch.t(torch.mm(self.phi_table, torch.t(z_t_category_gen_trans.squeeze(0))))
                
                if self.block == 'GRU':
                    output, h_now = self.trans(phi_z_transfer.view(phi_z_transfer.shape[0], 1, phi_z_transfer.shape[1]), h_now)# p(z_t| z_{t-1})
#                     output, h_now = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
                    
                else:
#                     output, (h_now, c_now) = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
                    output, (h_now, c_now) = self.trans(phi_z_transfer.view(phi_z_transfer.shape[0], 1, phi_z_transfer.shape[1]), (h_now, c_now))# p(z_t| z_{t-1})

#                 output, (h_now, c_now) = self.trans(phi_z.view(phi_z.shape[0], 1, phi_z.shape[1]), (h_now, c_now))# p(z_t| z_{t-1})
            
            curr_x_to_predict_lens -= 1
            
            shrinked_x_to_predict_lens -= 1
            
            h_now = h_now[:,shrinked_x_to_predict_lens > 0]

#             if self.latent and self.lstm_latent:
#                 last_decoded_c_n = last_decoded_c_n[:,shrinked_x_to_predict_lens > 0]
#                 last_decoded_h_n = last_decoded_h_n[:,shrinked_x_to_predict_lens > 0]
            
            
            if self.use_gate:
                curr_rnn_out = curr_rnn_out[shrinked_x_to_predict_lens > 0,0]
                
                last_h_n = last_h_n[:,shrinked_x_to_predict_lens > 0]
            
                last_c_n = last_c_n[:,shrinked_x_to_predict_lens > 0]
                
                last_h_n2 = last_h_n2[:,shrinked_x_to_predict_lens > 0]
            
                last_c_n2 = last_c_n2[:,shrinked_x_to_predict_lens > 0]
            
            if self.block == 'LSTM':
                c_now = c_now[:,shrinked_x_to_predict_lens > 0]
            
            phi_z = phi_z[shrinked_x_to_predict_lens > 0]
            
            
            
            shrinked_x_to_predict_lens = shrinked_x_to_predict_lens[shrinked_x_to_predict_lens > 0]
            
#             if t < T_max - 1:            
            
            
             
            
            
#             print(z_representation.shape)
        
        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask)
        
        final_nll_loss = torch.sum(neg_nll_losses)/torch.sum(x_mask)
        
        if np.isnan(neg_nll_losses.cpu().detach().numpy()).any():
            print('here')
        
        
        print('forecasting rmse loss::', final_rmse_loss)
        
        print('forecasting mae loss::', final_mae_losses)
        
        print('forecasting neg likelihood::', final_nll_loss)
        
        if self.is_missing:
            print('forecasting interpolate loss:', torch.norm(interpolated_x*x_mask - x*x_mask))
        
        if torch.sum(1-x_to_predict_new_mask) > 0:
            imputed_loss = torch.sum((torch.abs(origin_x_to_pred - x)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask)
            imputed_loss2 = torch.sum((torch.abs(origin_x_to_pred - imputed_x2)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask)
            
            imputed_mse_loss = torch.sqrt(torch.sum((((origin_x_to_pred - x)**2)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask))
            
            imputed_mse_loss2 = torch.sqrt(torch.sum((((origin_x_to_pred - imputed_x2)**2)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask))
            
            print('forecasting imputation rmse loss::', imputed_mse_loss)
            print('forecasting imputation rmse loss 2::', imputed_mse_loss2)
            
            print('forecasting imputation mae loss::', imputed_loss)
            print('forecasting imputation mae loss 2::', imputed_loss2)
        
        all_masks = torch.sum(x_mask)
        
        rmse_list, mae_list = get_forecasting_res_by_time_steps(x, imputed_x2, x_mask)
        
        if not self.evaluate:
            return final_rmse_loss, torch.sum(x_mask), final_mae_losses, torch.sum(x_mask), final_nll_loss, torch.sum(x_mask), (rmse_list, mae_list, torch.sum(x_mask))
        else:
            return imputed_x2*(1-x_mask) + x*x_mask
        
        
#     def evaluate_forecasting_errors2(self, x, batch_size, T_max, h_now, c_now):
# 
# 
#         rec_losses = torch.zeros((batch_size, T_max), device=x.device)
#         
#         for t in range(T_max):
# 
#             z_t_category_gen = F.softmax(self.emitter_z(h_now), dim = 2)
#             
# #             if t < T_max - 1:            
#             
#             phi_z, z_representation = self.generate_z(z_t_category_gen)
#              
#             mean, std, logit_x_t = self.generate_x(phi_z)
# 
#             rec_loss = (x[:,t,:] - logit_x_t)**2
# 
#             
#             rec_losses[:,t] = rec_loss.view(-1)
#             
#             if self.block == 'GRU':
#                 output, (h_now, c_now) = self.trans(phi_z, (h_now, c_now))# p(z_t| z_{t-1})
#             else:
#                 output, h_now = self.trans(phi_z, h_now)# p(z_t| z_{t-1})
#             
#             
#         print('reconstruction mini-batch loss::', rec_losses.view(-1).mean())
# 
#         
    def init_params(self):
        
        for p in self.parameters():
            if len(p.shape) >= 2:
                torch.nn.init.xavier_normal_(p, 1)
    
    
    def compute_phi_table_gap(self):
        
        distance = 0
        
        d = self.centroid_max
        
        
        cluster_diff = torch.norm(self.phi_table.view(self.phi_table.shape[0], 1, self.phi_table.shape[1]) - self.phi_table.view(self.phi_table.shape[0], self.phi_table.shape[1], 1), dim=0)
        
        distance2 = torch.sum(torch.triu(F.relu(d - cluster_diff)**2, diagonal=1))
        
        
        
#         distance_mat = torch.zeros([self.cluster_num, self.cluster_num])
#         for i in range(self.cluster_num):
#             for j in range(self.cluster_num):
#                 if j > i:
#                     
#                     distance_mat[i,j] = (F.relu(d - torch.norm(self.phi_table[:, i] - self.phi_table[:, j])))**2
#                     
#                     distance += distance_mat[i,j]
                
        return distance2
    
    def get_regularization_term(self):
        
        reg_term = 0
        
        for param in self.parameters():
            reg_term += torch.norm(param.view(-1))**2
            
        return reg_term
    
    def train_AE(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, kl_anneal, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_delta_time_stamps, x_to_predict_delta_time_stamps, x_time_stamps, x_to_predict_time_stamps):
#         self.x_encoder.train() # put the RNN back into training mode (i.e. turn on drop-out if applicable)
        
#         rec_loss, kl_loss, first_kl_loss, final_rmse_loss, interpolated_loss, cluster_objs = self.infer2(x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device)
#         loss = (1-self.gaussian_prior_coeff)*rec_loss + kl_anneal*kl_loss + 5e-5*first_kl_loss + self.gaussian_prior_coeff*cluster_objs# + 0.01*interpolated_loss# + 0.001*self.get_regularization_term()#+ 0.001*self.compute_phi_table_gap()# + 0.001*final_entropy_loss#
        
#         print(x_time_stamps)
        
        rec_loss1, kl_loss, first_kl_loss, final_rmse_loss, interpolated_loss, rec_loss2, final_ae_loss = self.infer0(x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_time_stamps, x_to_predict_time_stamps)
        
#         if kl_anneal == 0:
        loss = rec_loss1 + kl_anneal*kl_loss + 5e-5*first_kl_loss + rec_loss2 + 0.0001*interpolated_loss+ (10 + 10* kl_anneal)*final_ae_loss# + 0.01*interpolated_loss# + 0.001*self.get_regularization_term()#+ 0.001*self.compute_phi_table_gap()# + 0.001*final_entropy_loss#
#         else:
#             loss = rec_loss1 + kl_anneal*kl_loss + 5e-5*first_kl_loss + rec_loss2 + (10 + 10* kl_anneal)*final_ae_loss# + 0.01*interpolated_loss# + 0.001*self.get_regularization_term()#+ 0.001*self.compute_phi_table_gap()# + 0.001*final_entropy_loss#
        
        
#         print(torch.norm(rec_loss1 - (1-self.gaussian_prior_coeff)*rec_loss))
#          
#         print(torch.norm(rec_loss2 - self.gaussian_prior_coeff*cluster_objs))
        
         
        self.optimizer.zero_grad()
        
#         rec_loss.backward()
#         
#         kl_loss.backward()
#         
#         first_kl_loss.backward()
#         
#         cluster_objs.backward()
        
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
        self.optimizer.step()
        
        
        
        
        return {'train_loss_AE':loss.item(), 'train_loss_KL':kl_loss.item()}
    
    
#     def get_imputed_samples_test_part(self, x, x_mask, x_lens, x_to_predict, x_to_predict_mask, x_to_predict_lens):
#         
#         
#         T_max = x_lens.max().cpu().item()
#         
#         T_max = x_to_predict_lens.max().cpu().item()
#         
#         if self.is_missing:
#             imputed_x, interpolated_x = self.impute.forward2(x[:,0:T_max,:], x_mask[:,0:T_max,:], T_max)
#             x = imputed_x
#             
#             imputed_x, interpolated_x = self.impute.forward2(x[:,0:T_max,:], x_mask[:,0:T_max,:], T_max)
#             x = imputed_x
#         
#         
#         curr_x_to_predict_lens = x_to_predict_lens.clone()
#         
# #         one_time_step_arr = torch.ones_like(curr_x_to_predict_lens)
#         
#         shrinked_x_to_predict_lens = x_to_predict_lens.clone() 
#         
#         imputed_x = torch.zero_like(x)
#         
#         for t in range(T_max):
#             
# #             print(t)
#             
#             z_t_category_gen = F.softmax(self.emitter_z(h_now), dim = 2)
#             
#             phi_z, z_representation = self.generate_z(z_t_category_gen, t)
#             
#             mean, std, logit_x_t = self.generate_x(phi_z, z_t_category_gen)
# 
#             
# 
# #             rec_loss = (x[:,t,:] - logit_x_t)**2
# 
# #             print('print shape')
# #             
# #             print(logit_x_t.shape)
# #             
# #             print(x[:,t,:].shape)
# #             
# #             print(x_mask[:,t,:].shape)
#             
# #             rec_loss = torch.sqrt(torch.sum((x[:,t,:]*x_mask[:,t,:] - logit_x_t*x_mask[:,t,:])**2, dim = 1))
#             
#             imputed_x[curr_x_to_predict_lens>0,t,:] = logit_x_t
#             
# #             rec_loss = (x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t*x_mask[curr_x_to_predict_lens>0,t,:])**2
# #             
# #             mae_loss = torch.abs(x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t*x_mask[curr_x_to_predict_lens>0,t,:])
# #             
# #             rmse_losses[curr_x_to_predict_lens>0,:,t] = rec_loss
# #             
# #             mae_losses[curr_x_to_predict_lens>0,:,t] = mae_loss
# #             
# # #             print(mean.shape, std.shape, torch.sum((curr_x_to_predict_lens>0)))
# #             
# #             
# #             neg_nll_losses[curr_x_to_predict_lens>0,:,t] = compute_gaussian_probs0(x[curr_x_to_predict_lens>0,t,:], mean, std, x_mask[curr_x_to_predict_lens>0,t,:])
#             
# #             neg_nll_losses[curr_x_to_predict_lens>0,:,t] = self.compute_reconstruction_loss2(x[curr_x_to_predict_lens>0,t,:], mean[curr_x_to_predict_lens>0,:], std[curr_x_to_predict_lens>0,:], torch.sum((curr_x_to_predict_lens>0)))
#             
#             
# #             print('time stamps::', t)
#             
#             if self.block == 'GRU':
#                 output, h_now = self.trans(phi_z.view(phi_z.shape[0], 1, phi_z.shape[1]), h_now)# p(z_t| z_{t-1})
#             else:
#                 output, (h_now, c_now) = self.trans(phi_z.view(phi_z.shape[0], 1, phi_z.shape[1]), (h_now, c_now))# p(z_t| z_{t-1})
#             
#             curr_x_to_predict_lens -= 1
#             
#             shrinked_x_to_predict_lens -= 1
#             
#             h_now = h_now[:,shrinked_x_to_predict_lens > 0]
#             
#             if self.block == 'LSTM':
#                 c_now = c_now[:,shrinked_x_to_predict_lens > 0]
#             
#             phi_z = phi_z[shrinked_x_to_predict_lens > 0]
#             
#             
#             
#             shrinked_x_to_predict_lens = shrinked_x_to_predict_lens[shrinked_x_to_predict_lens > 0]
#     
#     
#         return imputed_x
#     
    
    def get_imputed_samples(self, x, x_mask, x_lens, x_to_predict, x_to_predict_mask, x_to_predict_lens):
        batch_size, _, input_dim = x.size()
        
        T_max = x_lens.max().item()
        
#         if is_GPU:
#             x = x.to(device)
#             
#             x_mask = x_mask.to(device)
#             
#             x_lens = x_lens.to(device)
            
        if self.is_missing:
            x, interpolated_x = self.impute.forward2(x, x_mask, T_max)
        
        
            print('test interpolated loss::', torch.norm(x*x_mask - interpolated_x*x_mask))
        
        
#         if torch.sum(1-x_new_mask) > 0:
#         
#             imputed_loss = torch.sum((torch.abs(origin_x - x)*(1-x_new_mask)))/torch.sum(1-x_new_mask)
#             
#             print('test imputation loss::', imputed_loss)
        
#         x_to_predict = self.impute(x_to_predict, x_to_predict_mask, T_max)
        
#         T_max = x_lens.max()
        h_0 = self.h_0.expand(1, batch_size, self.s_dim).contiguous()
        
        h_prev = self.h_0.expand(1, batch_size, self.h_0.size(0))
        
        c_prev = self.h_0.expand(1, batch_size, self.h_0.size(0))
        
        
        rnn_out,_= self.x_encoder(x, x_lens) # push the observed x's through the rnn;
        
        z_prev = self.z_q_0.expand(batch_size,self.z_q_0.size(0))
        
        curr_x_lens = x_lens.clone()
        
        shrinked_x_lens = x_lens.clone()
        
#         z_prev = self.z_q_0.expand(batch_size,1, self.z_q_0.size(0)) # set z_prev=z_q_0 to setup the recursive conditioning in q(z_t|...)

        last_h_now = torch.zeros_like(h_prev)
        
        last_c_now = torch.zeros_like(c_prev)


        one_time_step = torch.ones_like(curr_x_lens)

        curr_rnn = rnn_out[:,0,:]
        
        all_mean = torch.zeros_like(x)

        all_log_var = torch.zeros_like(x)
        
        
        all_probs = torch.zeros([x.shape[0], x.shape[1], self.cluster_num], device = self.device)


        x_t = 0
        
        x_t_mask = 0
        
        impute_rmse_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
        
        impute_mae_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
        
        imputed_x = torch.zeros_like(x)
        
        imputed_x[:,0] = x[:,0]    

        for t in range(T_max):
#             z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)# p(z_t| z_{t-1})

            z_t, z_t_category_infer, phi_z_infer, z_category_infer_sparse = self.postnet(z_prev, curr_rnn, self.phi_table, t, self.temp) #q(z_t | z_{t-1}, x_{t:T})
    
#                 rec_loss = torch.norm(x[:,t+1,:] - mean)**2/(2*std**2) + torch.log(2*np.pi*std**2)/2
                
#                 rec_loss = torch.bmm(((x[:,t,:]-mean)/(std**2)).view(mean.shape[0],1,mean.shape[1]), (x[:,t,:]-mean).view(mean.shape[0],mean.shape[1],1)).view(-1) + (torch.log((2*np.pi)**x[:,t,:].shape[-1]*torch.prod(std, dim= 1))/2).view(-1) 
                
            
#             z_t, z_t_category_infer, phi_z_infer = self.postnet(z_prev, rnn_out[:,t,:], self.phi_table) #q(z_t | z_{t-1}, x_{t:T})
            
#             kl_div = torch.sum(self.kl_div(torch.transpose(z_t_category_infer, 0, 1), z_t_category_infer),0)
#             
#             kl_states[:,t] = kl_div
            
            
              
            
            if self.block == 'GRU':
                output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0],1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
            else:
                output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1,phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})




            if t >= 1:            
            
                 
                mean, logvar, logit_x_t = self.generate_x(phi_z_infer, z_t)
    
                imputed_x[curr_x_lens > 0,t] = logit_x_t
    
    
                rmse_loss = (x_t*x_t_mask - logit_x_t*x_t_mask)**2
    
    
                all_mean[curr_x_lens > 0,t] = mean
                
                all_log_var[curr_x_lens > 0, t] = logvar
                
                all_probs[curr_x_lens > 0, t] = z_t
#                 rec_loss = torch.norm(x[:,t+1,:] - mean)**2/(2*std**2) + torch.log(2*np.pi*std**2)/2
                
#                 rec_loss = torch.bmm(((x[:,t,:]-mean)/(std**2)).view(mean.shape[0],1,mean.shape[1]), (x[:,t,:]-mean).view(mean.shape[0],mean.shape[1],1)).view(-1) + (torch.log((2*np.pi)**x[:,t,:].shape[-1]*torch.prod(std, dim= 1))/2).view(-1) 
                

            curr_x_lens -= 1
            
            shrinked_x_lens -= 1
            
#             z_t_category_gen = F.softmax(self.emitter_z(h_now), dim = 2)
            
            h_prev = h_now[:, shrinked_x_lens > 0]
            
            if (curr_x_lens == 0).any():
                last_h_now[:, curr_x_lens == 0] = h_now[:, shrinked_x_lens <= 0]
            
            
            
            if self.block == 'LSTM':
                c_prev = c_now[:, shrinked_x_lens > 0]
                if (curr_x_lens == 0).any():
                    last_c_now[:, curr_x_lens == 0] = c_now[:, shrinked_x_lens <= 0]
            
            
#             curr_rnn = rnn_out[curr_x_lens > 0, t]
            
            z_prev = phi_z_infer[shrinked_x_lens > 0]
            
            
            shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]
            
            
            
            
            if t < T_max - 1:            
                x_t = x[curr_x_lens > 0,t+1,:]
#             
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
                
                curr_rnn = rnn_out[curr_x_lens > 0,t+1,:]
                
                
                rmse_loss = (x_t*(1-x_t_mask) - logit_x_t*(1-x_t_mask))**2
                
                mae_loss = torch.abs(x_t*(1-x_t_mask) - logit_x_t*(1-x_t_mask))

                impute_rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
                
                impute_mae_losses[curr_x_lens > 0,:, t-1] = mae_loss

#                 phi_z, z_representation = self.generate_z(z_t_category_gen)
#                  
#                 mean, std, logit_x_t = self.generate_x(phi_z)
#     
#                 rec_loss = torch.norm(x[:,t+1,:] - mean)**2/(2*std**2) + torch.log(2*np.pi*std**2)/2
#     #             kl = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
#     #             kl_states[:,t] = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
#     #             logit_x_t = self.emitter(z_t).contiguous() # p(x_t|z_t)         
#     #             rec_loss = nn.BCEWithLogitsLoss(reduction='none')(logit_x_t.view(-1), x[:,t,:].contiguous().view(-1)).view(batch_size, -1)
#                 rec_losses[:,t+1] = rec_loss.mean(dim=1)             

            
#         x_mask = sequence_mask(x_lens)
#         x_mask = x_mask.gt(0).view(-1)
#         rec_loss = rec_losses.view(-1).masked_select(x_mask).mean()
#         kl_loss = kl_states.view(-1).masked_select(x_mask).mean()

#         rec_loss = rec_losses.view(-1).mean()
#         kl_loss = kl_states.view(-1).mean()
        
        
#         if torch.sum(1-x_to_predict_new_mask) > 0:
#             imputed_loss = torch.sum((torch.abs(origin_x_to_pred - x)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask)
#             print('forecasting imputation loss::', imputed_loss)
        
        self.get_imputed_samples_test_part(x_to_predict, x_to_predict_mask, x_to_predict_lens, h_now, c_now)
        
        final_rmse_loss = torch.sqrt(torch.sum(impute_rmse_losses)/torch.sum(1-x_mask[:,1:,:]))
        
        final_mae_losses = torch.sum(impute_mae_losses)/torch.sum(1-x_mask[:,1:,:])
        
        print('test impute rmse loss::', final_rmse_loss)
        
        print('test impute mae loss::', final_mae_losses)
        
        
        return imputed_x
        

#     def test_samples2(self, x, origin_x, x_mask, x_origin_mask, x_new_mask, x_lens, x_to_predict, origin_x_to_predict, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_delta_time_stamps, x_to_predict_delta_time_stamps, x_time_stamps, x_to_predict_time_stamps):
# #         self.x_encoder.evaluate()
#         
#         batch_size, _, input_dim = x.size()
#         
#         T_max = x_lens.max().item()
#         
#         if is_GPU:
#             x = x.to(device)
#             x_to_predict = x_to_predict.to(device)
#             origin_x = origin_x.to(device)
#             origin_x_to_predict = origin_x_to_predict.to(device)
#             
#             x_mask = x_mask.to(device)
#             
#             x_to_predict_mask = x_to_predict_mask.to(device)
#             
#             x_origin_mask = x_origin_mask.to(device)
#             
#             x_new_mask = x_new_mask.to(device)
#             
#             x_to_predict_origin_mask = x_to_predict_origin_mask.to(device)
#             
#             x_to_predict_new_mask = x_to_predict_new_mask.to(device) 
#             
#             x_lens = x_lens.to(device)
#             
#             x_to_predict_lens = x_to_predict_lens.to(device)
#         
#         if self.is_missing:
#             x, interpolated_x = self.impute.forward2(x, x_mask, T_max)
#         
#         
#             print('test interpolated loss::', torch.norm(x*x_mask - interpolated_x*x_mask))
#         
#         
#         if torch.sum(1-x_new_mask) > 0:
#         
#             imputed_loss = torch.sum((torch.abs(origin_x - x)*(1-x_new_mask)))/torch.sum(1-x_new_mask)
#             
#             print('test imputation loss::', imputed_loss)
#         
# #         x_to_predict = self.impute(x_to_predict, x_to_predict_mask, T_max)
#         
# #         T_max = x_lens.max()
#         h_0 = self.h_0.expand(1, batch_size, self.s_dim).contiguous()
#         
#         h_prev = self.h_0.expand(1, batch_size, self.h_0.size(0))
#         
#         c_prev = self.h_0.expand(1, batch_size, self.h_0.size(0))
#         
#         
#         rnn_out,_= self.x_encoder(x, x_lens) # push the observed x's through the rnn;
#         
#         z_prev = self.z_q_0.expand(batch_size,self.z_q_0.size(0))
#         
#         curr_x_lens = x_lens.clone()
#         
#         shrinked_x_lens = x_lens.clone()
#         
# #         z_prev = self.z_q_0.expand(batch_size,1, self.z_q_0.size(0)) # set z_prev=z_q_0 to setup the recursive conditioning in q(z_t|...)
# 
#         last_h_now = torch.zeros_like(h_prev)
#         
#         last_c_now = torch.zeros_like(c_prev)
# 
# 
#         one_time_step = torch.ones_like(curr_x_lens)
# 
#         curr_rnn = rnn_out[:,0,:]
#         
#         all_mean = torch.zeros_like(x)
# 
#         all_log_var = torch.zeros_like(x)
#         
#         
#         all_probs = torch.zeros([x.shape[0], x.shape[1], self.cluster_num], device = self.device)
# 
# 
#         x_t = 0
#         
#         x_t_mask = 0
#         
#         impute_rmse_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
#         
#         impute_mae_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)   
# 
#         imputed_x2 = torch.zeros_like(x)
#         
#         imputed_x2[:,0] = x[:,0]
# 
#         for t in range(T_max):
# #             z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)# p(z_t| z_{t-1})
# 
#             z_t, z_t_category_infer, phi_z_infer, z_category_infer_sparse = self.postnet(z_prev, curr_rnn, self.phi_table, t, self.temp) #q(z_t | z_{t-1}, x_{t:T})
#     
# #                 rec_loss = torch.norm(x[:,t+1,:] - mean)**2/(2*std**2) + torch.log(2*np.pi*std**2)/2
#                 
# #                 rec_loss = torch.bmm(((x[:,t,:]-mean)/(std**2)).view(mean.shape[0],1,mean.shape[1]), (x[:,t,:]-mean).view(mean.shape[0],mean.shape[1],1)).view(-1) + (torch.log((2*np.pi)**x[:,t,:].shape[-1]*torch.prod(std, dim= 1))/2).view(-1) 
#                 
#             
# #             z_t, z_t_category_infer, phi_z_infer = self.postnet(z_prev, rnn_out[:,t,:], self.phi_table) #q(z_t | z_{t-1}, x_{t:T})
#             
# #             kl_div = torch.sum(self.kl_div(torch.transpose(z_t_category_infer, 0, 1), z_t_category_infer),0)
# #             
# #             kl_states[:,t] = kl_div
#             
#             
#               
#             
#             if self.transfer_prob:
#                 
#                 if self.block == 'GRU':
#     #                 output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
#                     output, h_now = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
#                     
#                 else:
#                     output, (h_now, c_now) = self.trans(z_t.view(z_t.shape[0], 1, z_t.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
# #                 output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#             else:
#                 if self.block == 'GRU':
#                     output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0],1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
#                 else:
#                     output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1,phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
#                     
# 
#             if t >= 1:            
#             
#                  
#                 mean, logvar, logit_x_t = self.generate_x(phi_z_infer, z_t)
#     
#     
#                 rmse_loss = (x_t*x_t_mask - logit_x_t*x_t_mask)**2
#     
#     
#                 imputed_x2[curr_x_lens > 0,t] = logit_x_t
#     
#                 all_mean[curr_x_lens > 0,t] = mean
#                 
#                 all_log_var[curr_x_lens > 0, t] = logvar
#                 
#                 all_probs[curr_x_lens > 0, t] = z_t
#                 
#                 rmse_loss = (x_t*(1-x_t_mask) - logit_x_t*(1-x_t_mask))**2
#                 
#                 mae_loss = torch.abs(x_t*(1-x_t_mask) - logit_x_t*(1-x_t_mask))
# 
#                 impute_rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
#                 
#                 impute_mae_losses[curr_x_lens > 0,:, t-1] = mae_loss
# #                 rec_loss = torch.norm(x[:,t+1,:] - mean)**2/(2*std**2) + torch.log(2*np.pi*std**2)/2
#                 
# #                 rec_loss = torch.bmm(((x[:,t,:]-mean)/(std**2)).view(mean.shape[0],1,mean.shape[1]), (x[:,t,:]-mean).view(mean.shape[0],mean.shape[1],1)).view(-1) + (torch.log((2*np.pi)**x[:,t,:].shape[-1]*torch.prod(std, dim= 1))/2).view(-1) 
#                 
# 
#             curr_x_lens -= 1
#             
#             shrinked_x_lens -= 1
#             
# #             z_t_category_gen = F.softmax(self.emitter_z(h_now), dim = 2)
#             
#             h_prev = h_now[:, shrinked_x_lens > 0]
#             
#             if (curr_x_lens == 0).any():
#                 last_h_now[:, curr_x_lens == 0] = h_now[:, shrinked_x_lens <= 0]
#             
#             
#             
#             if self.block == 'LSTM':
#                 c_prev = c_now[:, shrinked_x_lens > 0]
#                 if (curr_x_lens == 0).any():
#                     last_c_now[:, curr_x_lens == 0] = c_now[:, shrinked_x_lens <= 0]
#             
#             
# #             curr_rnn = rnn_out[curr_x_lens > 0, t]
# 
# 
#             if self.transfer_prob:
#                 z_prev = z_t[shrinked_x_lens > 0,:]
#             else:
#                 z_prev = phi_z_infer[shrinked_x_lens > 0,:]
# #             if self.transfer_prob:
# #                 z_prev = phi_z_infer[shrinked_x_lens > 0]
#             
#             
#             shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]
#             
#             
#             
#             
#             if t < T_max - 1:            
#                 x_t = x[curr_x_lens > 0,t+1,:]
# #             
#                 x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
#                 
#                 curr_rnn = rnn_out[curr_x_lens > 0,t+1,:]
#                 
#                 
# 
# #                 phi_z, z_representation = self.generate_z(z_t_category_gen)
# #                  
# #                 mean, std, logit_x_t = self.generate_x(phi_z)
# #     
# #                 rec_loss = torch.norm(x[:,t+1,:] - mean)**2/(2*std**2) + torch.log(2*np.pi*std**2)/2
# #     #             kl = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
# #     #             kl_states[:,t] = self.kl_div(z_mu, z_logvar, z_prior_mu, z_prior_logvar)
# #     #             logit_x_t = self.emitter(z_t).contiguous() # p(x_t|z_t)         
# #     #             rec_loss = nn.BCEWithLogitsLoss(reduction='none')(logit_x_t.view(-1), x[:,t,:].contiguous().view(-1)).view(batch_size, -1)
# #                 rec_losses[:,t+1] = rec_loss.mean(dim=1)             
# 
#             
# #         x_mask = sequence_mask(x_lens)
# #         x_mask = x_mask.gt(0).view(-1)
# #         rec_loss = rec_losses.view(-1).masked_select(x_mask).mean()
# #         kl_loss = kl_states.view(-1).masked_select(x_mask).mean()
# 
# #         rec_loss = rec_losses.view(-1).mean()
# #         kl_loss = kl_states.view(-1).mean()
#         
#         
# #         if torch.sum(1-x_to_predict_new_mask) > 0:
# #             imputed_loss = torch.sum((torch.abs(origin_x_to_pred - x)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask)
# #             print('forecasting imputation loss::', imputed_loss)
#         
#         if self.is_missing:
#             final_rmse_loss = torch.sqrt(torch.sum(impute_rmse_losses)/torch.sum(1-x_mask[:,1:,:]))
#             
#             final_mae_losses = torch.sum(impute_mae_losses)/torch.sum(1-x_mask[:,1:,:])
#             
#             print('test impute rmse loss::', final_rmse_loss)
#             
#             print('test impute mae loss::', final_mae_losses)
#         
#         
#         if torch.sum(1-x_new_mask) > 0:
#             imputed_loss = torch.sum((torch.abs(origin_x - x)*(1-x_new_mask)))/torch.sum(1-x_new_mask)
#             
#             imputed_loss2 = torch.sum((torch.abs(origin_x - imputed_x2)*(1-x_new_mask)))/torch.sum(1-x_new_mask) 
#             
#             imputed_mse_loss = torch.sqrt(torch.sum((((origin_x - x)**2)*(1-x_new_mask)))/torch.sum(1-x_new_mask))
#             
#             imputed_mse_loss2 = torch.sqrt(torch.sum((((origin_x - imputed_x2)**2)*(1-x_new_mask)))/torch.sum(1-x_new_mask)) 
#             
#             print('training imputation rmse loss::', imputed_mse_loss)
#             
#             print('training imputation rmse loss 2::', imputed_mse_loss2)
#             
#             print('training imputation mae loss::', imputed_loss)
#             
#             print('training imputation mae loss 2::', imputed_loss2)
#         
#         
#         
#         
#         if not self.evaluate:
#             final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count = self.evaluate_forecasting_errors(x_to_predict, origin_x_to_predict, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, last_c_now, T_max)
#         else:
#             imputed_x2 = self.evaluate_forecasting_errors(x_to_predict, origin_x_to_predict, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, last_c_now, T_max)
#         
#         print()
#         
#         if not os.path.exists(data_folder + '/' + output_dir):
#             os.makedirs(data_folder + '/' + output_dir)
#         
#         
#         torch.save(self.phi_table, data_folder + '/' + output_dir + 'cluster_centroids')
#         
#         torch.save(all_mean, data_folder + '/' + output_dir + 'all_mean')
#         
#         torch.save(all_log_var, data_folder + '/' + output_dir + 'all_log_var')
#         
#         torch.save(all_probs, data_folder + '/' + output_dir + 'all_probs')
#         
#         
#         
# #         del x, x_to_predict,origin_x, origin_x_to_predict, x_mask, x_to_predict_mask, x_origin_mask, x_new_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_lens, x_to_predict_lens
#         
#         
#         
#         
#         
#         
#         
#         
#         
#         if not self.evaluate:
#             return final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count
#         else:
#             
#             if torch.sum(1-x_new_mask) > 0:
#                 return imputed_x2, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
#             else:
#                 return imputed_x2, None
#         
#         print('loss::', rec_loss, kl_loss)

        
        
        
        
        
    
    def valid(self, x, x_rev, x_lens):
        self.eval()
        rec_loss, kl_loss = self.infer(x, x_rev, x_lens)
        loss = rec_loss + kl_loss
        return loss
    
#     def generate(self, x, x_rev, x_lens):
#         """
#         generation model p(x_{1:T} | z_{1:T}) p(z_{1:T})
#         """
#         batch_size, _, input_dim = x.size() # number of time steps we need to process in the mini-batch
#         T_max = x_lens.max()
#         h_prev = self.h_0.expand(batch_size, self.h_0.size(0)) # set z_prev=z_0 to setup the recursive conditioning in p(z_t|z_{t-1})
#         
#         z_1_category = torch.ones(self.cluster_num, dtype = torch.float,device = self.device)/self.cluster_num
#         
#         z_now = z_1_category.expand(batch_size, self.z_1_category.size[0]) 
#         
#         generated_x = torch.zeros_like(x, device = self.device)
#         
#         
#         for t in range(1, T_max + 1):
#             
#             if self.block == 'GRU':
#                 out, h_now = self.trans(z_now, h_prev)
#             else:
#                 out, h_now = self.trans(z_now, h_prev)
#             
#             z_category = F.softmax(self.emitter_z(h_now), dim = 2)
#             
#             z_now = z_category
#             
#             phi_z, z_representation = self.generate_z(z_category)
#              
#             _,_,x_t = self.generate_x(phi_z)
#             
#             h_prev = h_now
#             
#             generated_x[:,t] = x_t
#         
#         
#         return generated_x
        
            # sample z_t ~ p(z_t | z_{t-1}) one time step at a time
#             z_t, z_mu, z_logvar = self.trans(z_prev) # p(z_t | z_{t-1})
#             p_x_t = F.sigmoid(self.emitter(z_t))  # compute the probabilities that parameterize the bernoulli likelihood              
#             x_t = torch.bernoulli(p_x_t) #sample observe x_t according to the bernoulli distribution p(x_t|z_t)
#             z_prev = z_t


