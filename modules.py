import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import os
import numpy as np
import random
import sys
from torch.autograd import Variable
import math



parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import SOS_ID, EOS_ID

class MLP(nn.Module):
    def __init__(self, input_size, arch, output_size, activation=nn.ReLU(), batch_norm=True, init_w=0.02, discriminator=False):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.init_w= init_w

        layer_sizes = [input_size] + [int(x) for x in arch.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)            
            if batch_norm and not(discriminator and i==0):# if used as discriminator, then there is no batch norm in the first layer
                bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)
            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], output_size)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, self.init_w)
                layer.bias.data.fill_(0)
            except: pass

class Encoder(nn.Module):
    def __init__(self, embedder, input_size, hidden_size, bidir, n_layers, dropout=0.5, noise_radius=0.2):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.noise_radius=noise_radius
        self.n_layers = n_layers
        self.bidir = bidir
        assert type(self.bidir)==bool
        self.dropout=dropout
        
        self.embedding = embedder # nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, bidirectional=bidir)
        self.init_h = nn.Parameter(torch.randn(self.n_layers*(1+self.bidir), 1, self.hidden_size), requires_grad=True)#learnable h0
        self.init_weights()
        
    def init_weights(self):
        for w in self.rnn.parameters(): # initialize the gate weights with orthogonal
            if w.dim()>1:
                weight_init.orthogonal_(w)
                
    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad
    
    def forward(self, inputs, input_lens=None, init_h=None, noise=False): 
        # init_h: [n_layers*n_dir x batch_size x hid_size]
        if self.embedding is not None:
            inputs=self.embedding(inputs)  # input: [batch_sz x seq_len] -> [batch_sz x seq_len x emb_sz]
        
        batch_size, seq_len, emb_size=inputs.size()
        inputs=F.dropout(inputs, self.dropout, self.training)# dropout
        
        if input_lens is not None:# sort and pack sequence 
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)        
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)
        
        if init_h is None:
            init_h = self.init_h.expand(-1,batch_size,-1).contiguous()# use learnable initial states, expanding along batches
        #self.rnn.flatten_parameters() # time consuming!!
        hids, h_n = self.rnn(inputs, init_h) # hids: [b x seq x (n_dir*hid_sz)]  
                                                  # h_n: [(n_layers*n_dir) x batch_sz x hid_sz] (2=fw&bw)
        if input_lens is not None: # reorder and pad
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)     
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
        h_n = h_n.view(self.n_layers, (1+self.bidir), batch_size, self.hidden_size) #[n_layers x n_dirs x batch_sz x hid_sz]
        h_n = h_n[-1] # get the last layer [n_dirs x batch_sz x hid_sz]
        enc = h_n.transpose(0,1).contiguous().view(batch_size,-1) #[batch_sz x (n_dirs*hid_sz)]
        #if enc.requires_grad:
        #    enc.register_hook(self.store_grad_norm) # store grad norm 
        # norms = torch.norm(enc, 2, 1) # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        # enc = torch.div(enc, norms.unsqueeze(1).expand_as(enc)+1e-5)
        if noise and self.noise_radius > 0:
            gauss_noise = torch.normal(means=torch.zeros(enc.size(), device=inputs.device),std=self.noise_radius)
            enc = enc + gauss_noise
            
        return enc, hids
    
class Encoder_cluster(nn.Module):
#     def __init__(self, embedder, input_size, hidden_size, bidir, n_layers, dropout=0.5, noise_radius=0.2):
    def __init__(self, input_dim, hidden_size, dropout, device, bidir = True, block = 'LSTM', n_layers=1):
        super(Encoder_cluster, self).__init__()
        
        self.hidden_size = hidden_size
#         self.noise_radius=noise_radius
        self.n_layers = n_layers
        self.bidir = bidir
        self.device = device
        assert type(self.bidir)==bool
        self.dropout=dropout
        
#         self.embedding = embedder # nn.Embedding(vocab_size, emb_size)
        self.block = block
        if self.block == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_size, n_layers, batch_first=True, bidirectional=bidir, dropout = self.dropout)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_size, n_layers, batch_first=True, bidirectional=bidir, dropout = self.dropout)
#         self.init_h = nn.Parameter(torch.randn(self.n_layers*(1+self.bidir), 1, self.hidden_size), requires_grad=True)#learnable h0
        self.init_h = torch.zeros([self.n_layers*(1+self.bidir), 1, self.hidden_size], device = self.device)
        
        if self.block == 'LSTM':
            self.init_c = torch.zeros([self.n_layers*(1+self.bidir), 1, self.hidden_size],device = self.device)
#         self.init_weights()
        
    def init_weights(self):
        for w in self.rnn.parameters(): # initialize the gate weights with orthogonal
            if w.dim()>1:
                weight_init.orthogonal_(w)
                
    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad
    
    def forward2(self, inputs, input_lens, rnn_out, exp_last_h_n, exp_last_c_n, init_h=None, init_c = None, noise=False):
        batch_size, seq_len, emb_size=inputs.size()
#         if input_lens is not None:# sort and pack sequence
            
        output = torch.zeros_like(inputs)
        
         
        input_lens_sorted, indices = input_lens.sort(descending=True)
#             print(inputs.shape, input_lens.shape)
        inputs_sorted = inputs.index_select(0, indices)
            
        if init_h is None:
            init_h = self.init_h.expand(-1,batch_size,-1).contiguous()# use learnable initial states, expanding along batches
        if self.block == 'LSTM':
            if init_c is None:            
                init_c = self.init_c.expand(-1,batch_size,-1).contiguous()
        
        
        last_len = None
        
        last_id = None
        
        last_h_n = torch.zeros([(1+self.bidir)*self.n_layers, batch_size, self.hidden_size])
        
        last_c_n = torch.zeros([(1+self.bidir)*self.n_layers, batch_size, self.hidden_size])
        
        
        output_list = torch.zeros([batch_size, seq_len,(1+self.bidir)*self.hidden_size])
        
        for k in range(len(input_lens_sorted)):
            if last_len is None:
                last_len = input_lens_sorted[k]
                last_id = k 
            else:
                if last_len == input_lens_sorted[k]:
                    continue
                else:
                    
                    if self.block == 'LSTM':
                        hids, (h_n, c_n) = self.rnn(inputs_sorted[last_id:k, 0:last_len], (init_h[:,last_id:k,:], init_c[:,last_id:k,:]))
                        last_c_n[:,last_id:k] = c_n
                    else:
                        hids, h_n = self.rnn(inputs_sorted[last_id:k, 0:last_len], init_h[:,last_id:k,:])
                        
                    output_list[last_id:k, 0:last_len] = hids
                    last_h_n[:,last_id:k] = h_n
                    
                    
                    
                    
                    last_id = k
                    last_len = input_lens_sorted[k]
        
        
        
        if self.block == 'LSTM':
            hids, (h_n, c_n) = self.rnn(inputs_sorted[last_id:k+1, 0:last_len], (init_h[:,last_id:k+1,:], init_c[:,last_id:k+1,:]))
            last_c_n[:,last_id:k+1] = c_n
        else:
            hids, h_n = self.rnn(inputs_sorted[last_id:k+1, 0:last_len], init_h[:,last_id:k+1,:])
            
        output_list[last_id:k+1, 0:last_len] = hids
        last_h_n[:,last_id:k+1] = h_n
        
        
        _, inv_indices = indices.sort()
        
        output_hiddens = output_list[inv_indices]
        
        print(torch.norm(output_hiddens[0, 0:input_lens[0]] - rnn_out[0, 0:input_lens[0]]))
        
        print(torch.norm(output_hiddens[-1, 0:input_lens[-1]] - rnn_out[-1, 0:input_lens[-1]]))
        
        print(torch.norm(last_h_n[:,inv_indices] - exp_last_h_n))
        
        print(torch.norm(last_c_n[:,inv_indices] - exp_last_c_n))
        
        return output_hiddens, (last_h_n[:,inv_indices], last_c_n[:, inv_indices])
#             shifted_lens = torch.zeros_like(input_lens_sorted)
#         
#             shifted_lens[0:len(input_lens_sorted) - 1] = input_lens_sorted[1:len(input_lens_sorted)]
#         
#         gaps = shifted_lens - input_lens_sorted
        
        

        
#         for k in range(len(input_lens_sorted)):
                
#         sorted_inputs = inputs[inputs_sorted]
        
        
    def check_hidden_states(self, x, x_lens, init_h, init_c, hids, h_n, c_n):
        
        T_max = x_lens.max()
        
        for i in range(x.shape[0]):
            origin_hids, (curr_h_n, curr_c_n) = self.rnn(x[i, 0:x_lens[i]].view(1, x_lens[i], x.shape[2]), (init_h[:,i,:].view(init_h[:,i,:].shape[0], 1, init_h[:,i,:].shape[1]), init_c[:,i,:].view(init_c[:,i,:].shape[0], 1, init_c[:,i,:].shape[1])))
            
#             print(torch.norm(origin_hids - hids[i]))

            print('lens::', T_max, x_lens[i])
            
            print(torch.norm(curr_h_n.view(-1) - h_n[:,i,:].reshape((-1))))
            
            print(torch.norm(curr_c_n.view(-1) - c_n[:,i,:].reshape((-1))))
            
            
    
    def forward(self, inputs, input_lens=None, init_h=None, init_c = None, noise=False, test = False): 
        # init_h: [n_layers*n_dir x batch_size x hid_size]
#         if self.embedding is not None:
#             inputs=self.embedding(inputs)  # input: [batch_sz x seq_len] -> [batch_sz x seq_len x emb_sz]

#         input_data = inputs.clone()
        origin_inputs = inputs.clone()
        
        batch_size, seq_len, emb_size=inputs.size()
#         inputs=F.dropout(inputs, self.dropout, self.training)# dropout
        
        if input_lens is not None:# sort and pack sequence 
            input_lens_sorted, indices = input_lens.sort(descending=True)
#             print(inputs.shape, input_lens.shape)
            inputs_sorted = inputs.index_select(0, indices)        
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)
            
            if init_h is not None:
#                 origin_init_h = init_h.clone()
                
                init_h = init_h.index_select(1, indices)
                if self.block == 'LSTM':
#                     origin_init_c= init_c.clone()
                    
                    init_c = init_c.index_select(1, indices)
                
#                     origin_hids2, (h_n2, c_n2) = self.rnn(input_data, (origin_init_h, origin_init_c))
        if init_h is None:
            init_h = self.init_h.expand(-1,batch_size,-1).contiguous()# use learnable initial states, expanding along batches
        if self.block == 'LSTM':
            if init_c is None:            
                init_c = self.init_c.expand(-1,batch_size,-1).contiguous()

            origin_hids, (h_n, c_n) = self.rnn(inputs, (init_h, init_c))
            
            
            
        else:
            origin_hids, h_n = self.rnn(inputs, init_h)
            
            
        #self.rnn.flatten_parameters() # time consuming!!
         # hids: [b x seq x (n_dir*hid_sz)]  
                                                  # h_n: [(n_layers*n_dir) x batch_sz x hid_sz] (2=fw&bw)
        if input_lens is not None: # reorder and pad
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(origin_hids, batch_first=True)     
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
            
            if self.block == 'LSTM':
                c_n = c_n.index_select(1, inv_indices)
        
#         if test:
#         
#             lstm_layer = nn.LSTM(self.rnn.input_size, self.rnn.hidden_size, batch_first=True)    
#                     
#             print(input_data.shape, lstm_layer)        
#             
#             full_rnn_decoded_out4, (last_decoded_h_n4, last_decoded_c_n4) = lstm_layer(input_data, (init_h, init_c))
#             
#             print(torch.norm(full_rnn_decoded_out4 - hids))
#             
#             print('here')
#                 self.check_hidden_states(origin_inputs, input_lens, init_h, init_c, hids, h_n, c_n)
#                 
#                 print('here')
            
#         h_n = h_n.view(self.n_layers, (1+self.bidir), batch_size, self.hidden_size) #[n_layers x n_dirs x batch_sz x hid_sz]
#         h_n = h_n[-1] # get the last layer [n_dirs x batch_sz x hid_sz]
#         enc_h = h_n.transpose(0,1).contiguous().view(batch_size,-1) #[batch_sz x (n_dirs*hid_sz)]
#         
#         if self.block == 'LSTM':
#             c_n = c_n.view(self.n_layers, (1+self.bidir), batch_size, self.hidden_size) #[n_layers x n_dirs x batch_sz x hid_sz]
#             c_n = c_n[-1] # get the last layer [n_dirs x batch_sz x hid_sz]
#             enc_c = c_n.transpose(0,1).contiguous().view(batch_size,-1) #[batch_sz x (n_dirs*hid_sz)]
        
        
        #if enc.requires_grad:
        #    enc.register_hook(self.store_grad_norm) # store grad norm 
        # norms = torch.norm(enc, 2, 1) # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        # enc = torch.div(enc, norms.unsqueeze(1).expand_as(enc)+1e-5)
#         if noise and self.noise_radius > 0:
#             gauss_noise = torch.normal(means=torch.zeros(enc.size(), device=inputs.device),std=self.noise_radius)
#             enc = enc + gauss_noise
        if self.block =='LSTM':    
            return hids, (h_n, c_n)
        else:
            return hids, h_n


class GatedTransition2(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in the reference for comparison.
    """
    def __init__(self, z_dim, h_dim, trans_dim):
        super(GatedTransition2, self).__init__()
        self.gate = nn.Sequential( 
            nn.Linear(h_dim, trans_dim),
            nn.ReLU(),
            nn.Linear(trans_dim, z_dim),
            nn.Sigmoid()
        )
        self.proposed_mean = nn.Sequential(
            nn.Linear(h_dim, trans_dim),
            nn.ReLU(),
            nn.Linear(trans_dim, z_dim)
        )           
        
        self.lstm = torch.nn.LSTM(z_dim, h_dim)
        
        self.z_to_mu = nn.Linear(z_dim, z_dim)
        # modify the default initialization of z_to_mu so that it starts out as the identity function
        self.z_to_mu.weight.data = torch.eye(z_dim)
        self.z_to_mu.bias.data = torch.zeros(z_dim)
        self.z_to_logvar = nn.Linear(z_dim, z_dim)
        self.relu = nn.ReLU()

    def forward(self, z_t_1, h_t_1, c_t_1):
        """
        Given the latent `z_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the (diagonal) gaussian distribution `p(z_t | z_{t-1})`
        """        
        gate = self.gate(h_t_1) # compute the gating function
        
        _, (h_t, c_t) = self.lstm(z_t_1.view(1, z_t_1.shape[0], z_t_1.shape[1]).contiguous(), (h_t_1, c_t_1))
        
        
        
        proposed_mean = self.proposed_mean(h_t) # compute the 'proposed mean'
        mu = (1 - gate) * self.z_to_mu(z_t_1) + gate * proposed_mean # compute the scale used to sample z_t, using the proposed mean from
        logvar = self.z_to_logvar(self.relu(proposed_mean)) 
        epsilon = torch.randn(z_t_1.size(), device=z_t_1.device) # sampling z by re-parameterization
        z_t = mu + epsilon * torch.exp(0.5 * logvar)    # [batch_sz x z_sz]
        return z_t, mu.view(mu.shape[1], mu.shape[2]), logvar.view(logvar.shape[1], logvar.shape[2]), h_t, c_t

class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in the reference for comparison.
    """
    def __init__(self, z_dim, trans_dim):
        super(GatedTransition, self).__init__()
        self.gate = nn.Sequential( 
            nn.Linear(z_dim, z_dim),
            nn.Sigmoid()
        )
        self.proposed_mean = nn.Sequential(
            nn.Linear(z_dim,  z_dim)
        )           
        self.z_to_mu = nn.Linear(z_dim, z_dim)
        # modify the default initialization of z_to_mu so that it starts out as the identity function
        self.z_to_mu.weight.data = torch.eye(z_dim)
        self.z_to_mu.bias.data = torch.zeros(z_dim)
        self.z_to_logvar = nn.Linear(z_dim, z_dim)
        self.relu = nn.ReLU()

    def forward(self, z_t_1):
        """
        Given the latent `z_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the (diagonal) gaussian distribution `p(z_t | z_{t-1})`
        """        
        gate = self.gate(z_t_1) # compute the gating function
        proposed_mean = self.proposed_mean(z_t_1) # compute the 'proposed mean'
        mu = (1 - gate) * self.z_to_mu(z_t_1) + gate * proposed_mean # compute the scale used to sample z_t, using the proposed mean from
        logvar = F.softplus(self.z_to_logvar(self.relu(proposed_mean))) 
        epsilon = torch.randn(z_t_1.size(), device=z_t_1.device) # sampling z by re-parameterization
#         z_t = mu + epsilon * torch.exp(0.5 * logvar)    # [batch_sz x z_sz]
        z_t = mu + epsilon * logvar
        return z_t, mu, logvar 




class PostNet_cluster(nn.Module):
    """
    Parameterizes `q(z_t|z_{t-1}, x_{t:T})`, which is the basic building block of the inference (i.e. the variational distribution). 
    The dependence on `x_{t:T}` is through the hidden state of the RNN
    """
    def __init__(self, z_dim, h_dim, cluster_num, dropout, sampling_times, bidirt = True):
        super(PostNet_cluster, self).__init__()
#         self.z_to_h = nn.Sequential(
#             nn.Linear(z_dim, (1+bidirt)*h_dim),
#             nn.Tanh(),
#             nn.Dropout(p = dropout)
#         )
        
#         self.t_thres = t_thres
        
        self.h_to_z = nn.Sequential(
            nn.Linear((1+bidirt)*h_dim + z_dim, cluster_num),
            nn.Dropout(p = dropout)
#             nn.ReLU()
#             nn.Linear(e_dim, cluster_num)
#             nn.Softmax()
            )
        
#         self.use_sprasemax = use_sprasemax
#         
#         self.use_gumbel_softmax = use_gumbel_softmax
        
        self.sampling_times = sampling_times
#         self.h_to_mu = nn.Linear(h_dim, z_dim)
#         self.h_to_logvar = nn.Linear(h_dim, z_dim)
#     def sample_multi_times(self, z_t_prev_category, phi_table, S, h_x):
#         
#         averaged_z_t_category = 0
#         
#         for i in range(S):
#             sampled_z_t_prev = F.gumbel_softmax(torch.log(z_t_prev_category), tau = 0.1, dim = -1)
#             
#             phi_z_t_prev = self.get_z_t_from_samples(sampled_z_t_prev, phi_table)
#             
#             z_t_category = self.gen_z_t_dist_now(phi_z_t_prev, h_x)* sampled_z_t_prev * z_t_prev_category #q(z_t|z_{t-1})
#             
#             averaged_z_t_category += sampled_z_t_prev
#         
#         sampled_z_t_prev = sampled_z_t_prev/S
        
    def gen_z_t_dist_now(self, z_t_1, h_x):
        
#         h_combined = 0.5*(self.z_to_h(z_t_1) + h_x)# combine the rnn hidden state with a transformed version of z_t_1
        h_combined = torch.cat([z_t_1, h_x], -1)
        
#         if not self.use_sprasemax:
            
        z_category = F.softmax(self.h_to_z(h_combined), dim = -1)
        return z_category,z_category 
#         else:
#             z_category = F.softmax(self.h_to_z(h_combined), dim = -1)
#             
#             z_category_sparse = sparsemax(self.h_to_z(h_combined))
#             
#             return z_category, z_category_sparse
    
    def get_z_t_from_samples(self, z_t, phi_table):
        
        return torch.mm(z_t, torch.t(phi_table))
        
        
        
    def forward(self, z_t_1, h_x, phi_table, t, temp=0):
        """
        Given the latent z at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t|z_{t-1}, x_{t:T})`
        """
        
#         sparsemax.device = z_t_1.device
        
        z_category, z_category_sparse = self.gen_z_t_dist_now(z_t_1, h_x)
        
#         if t > self.t_thres:
#          
#             if self.use_gumbel_softmax:
# #                 print(t, 'inference here')
#     #             device = z_category.device
#                   
#                 averaged_z_t = 0
#                   
#                 log_prob = Variable(torch.log(z_category))
#                   
#                 for k in range(self.sampling_times):           
#                     curr_z_t = F.gumbel_softmax(log_prob, tau = 0.05)
#                       
#     #                 curr_z_t = sparsemax(log_prob)
#                       
#                       
#                     averaged_z_t += curr_z_t
#                       
#                     del curr_z_t
#                   
#     #             averaged_z_t = averaged_z_t.to(device)
#                       
#                 z_t = averaged_z_t/self.sampling_times
#                   
#     #             print('diff::', torch.norm(z_t - z_category))
#     #             
#     #             print()
#             else:
#                 z_t = z_category
#                  
#         else:
        z_t = z_category
        
        if len(z_t.shape) == 2:
            phi_z = torch.mm(z_t, torch.t(phi_table))
        else:
            
            phi_table_full = (torch.t(phi_table)).view(1, phi_table.shape[1], phi_table.shape[0])
            
            phi_table_full = phi_table_full.repeat(phi_table.shape[1], 1, 1)
            
            phi_z = torch.bmm(z_t, phi_table_full)
#         mu = self.h_to_mu(h_combined)
#         logvar = self.h_to_logvar(h_combined)
#         std = torch.exp(0.5 * logvar)        
#         epsilon = torch.randn(z_t_1.size(), device=z_t_1.device) # sampling z by re-parameterization
#         z_t = epsilon * std + mu   # [batch_sz x z_sz]
        return z_t, z_category, phi_z, z_category_sparse

class PostNet_cluster_time(nn.Module):
    """
    Parameterizes `q(z_t|z_{t-1}, x_{t:T})`, which is the basic building block of the inference (i.e. the variational distribution). 
    The dependence on `x_{t:T}` is through the hidden state of the RNN
    """
    def __init__(self, z_dim, h_dim, cluster_num, dropout, use_gumbel_softmax, sampling_times, bidirt = True):
        super(PostNet_cluster_time, self).__init__()
        self.z_to_h = nn.Sequential(
            nn.Linear(z_dim+1, (1+bidirt)*h_dim),
            nn.Tanh(),
            nn.Dropout(p = dropout)
        )
        
        self.h_to_z = nn.Sequential(
            nn.Linear((1+bidirt)*h_dim, cluster_num),
            nn.Dropout(p = dropout)
#             nn.ReLU()
#             nn.Linear(e_dim, cluster_num)
#             nn.Softmax()
            )
        
        self.use_gumbel_softmax = use_gumbel_softmax
        
        self.sampling_times = sampling_times
#         self.h_to_mu = nn.Linear(h_dim, z_dim)
#         self.h_to_logvar = nn.Linear(h_dim, z_dim)
#     def sample_multi_times(self, z_t_prev_category, phi_table, S, h_x):
#         
#         averaged_z_t_category = 0
#         
#         for i in range(S):
#             sampled_z_t_prev = F.gumbel_softmax(torch.log(z_t_prev_category), tau = 0.1, dim = -1)
#             
#             phi_z_t_prev = self.get_z_t_from_samples(sampled_z_t_prev, phi_table)
#             
#             z_t_category = self.gen_z_t_dist_now(phi_z_t_prev, h_x)* sampled_z_t_prev * z_t_prev_category #q(z_t|z_{t-1})
#             
#             averaged_z_t_category += sampled_z_t_prev
#         
#         sampled_z_t_prev = sampled_z_t_prev/S
        
    def gen_z_t_dist_now(self, z_t_1, h_x):
        
        h_combined = 0.5*(self.z_to_h(z_t_1) + h_x)# combine the rnn hidden state with a transformed version of z_t_1
        
        z_category = F.softmax(self.h_to_z(h_combined), dim = 1)
        
        return z_category
    
    def get_z_t_from_samples(self, z_t, phi_table):
        
        return torch.mm(z_t, torch.t(phi_table))
        
        
        
    def forward(self, z_t_1, h_x, phi_table):
        """
        Given the latent z at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t|z_{t-1}, x_{t:T})`
        """
        z_category = self.gen_z_t_dist_now(z_t_1, h_x)
        
        if self.use_gumbel_softmax:
            
#             device = z_category.device
            
            averaged_z_t = 0
            
            log_prob = Variable(torch.log(z_category))
            
            for k in range(self.sampling_times):           
                curr_z_t = F.gumbel_softmax(log_prob, tau = 0.1)
                
                averaged_z_t += curr_z_t
                
                del curr_z_t
            
#             averaged_z_t = averaged_z_t.to(device)
                
            z_t = averaged_z_t/self.sampling_times
        else:
            z_t = z_category
        
        phi_z = torch.mm(z_t, torch.t(phi_table))
#         mu = self.h_to_mu(h_combined)
#         logvar = self.h_to_logvar(h_combined)
#         std = torch.exp(0.5 * logvar)        
#         epsilon = torch.randn(z_t_1.size(), device=z_t_1.device) # sampling z by re-parameterization
#         z_t = epsilon * std + mu   # [batch_sz x z_sz]
        return z_t, z_category, phi_z


class PostNet_cluster2(nn.Module):
    """
    Parameterizes `q(z_t|z_{t-1}, x_{t:T})`, which is the basic building block of the inference (i.e. the variational distribution). 
    The dependence on `x_{t:T}` is through the hidden state of the RNN
    """
    def __init__(self, z_dim, h_dim, z_std, dropout):
        super(PostNet_cluster2, self).__init__()
        self.z_to_h = nn.Sequential(
            nn.Linear(z_dim, 2*h_dim),
            nn.Tanh(),
            nn.Dropout(p = dropout)
        )
        
        self.h_to_z_mean = nn.Sequential(
            nn.Linear(2*h_dim, z_dim),
            nn.Dropout(p = dropout)
#             nn.ReLU()
#             nn.Linear(e_dim, cluster_num)
#             nn.Softmax()
            )
        
#         self.z_std = z_std
        self.h_to_z_var = nn.Sequential(
            nn.Linear(2*h_dim, z_dim),
            nn.Dropout(p = dropout)
#             nn.ReLU()
#             nn.Linear(e_dim, cluster_num)
#             nn.Softmax()
            )
        
        
#         self.h_to_mu = nn.Linear(h_dim, z_dim)
#         self.h_to_logvar = nn.Linear(h_dim, z_dim)

    def forward(self, z_t_1, h_x):
        """
        Given the latent z at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t|z_{t-1}, x_{t:T})`
        """
        h_combined = 0.5*(self.z_to_h(z_t_1) + h_x)# combine the rnn hidden state with a transformed version of z_t_1
        
        z_mean = self.h_to_z_mean(h_combined)
        
        z_var = self.h_to_z_var(h_combined)
        
        epsilon = torch.randn(z_t_1.size(), device=z_t_1.device) # sampling z by re-parameterization
        z_t = z_mean + epsilon * torch.exp(0.5 * z_var)    # [batch_sz x z_sz]
        
        
#         z_t = F.gumbel_softmax(z_category)
#         z_t = z_category
#         
#         phi_z = torch.mm(z_t, torch.t(phi_table))
#         mu = self.h_to_mu(h_combined)
#         logvar = self.h_to_logvar(h_combined)
#         std = torch.exp(0.5 * logvar)        
#         epsilon = torch.randn(z_t_1.size(), device=z_t_1.device) # sampling z by re-parameterization
#         z_t = epsilon * std + mu   # [batch_sz x z_sz]
        return z_t, z_mean, z_var
  

class PostNet(nn.Module):
    """
    Parameterizes `q(z_t|z_{t-1}, x_{t:T})`, which is the basic building block of the inference (i.e. the variational distribution). 
    The dependence on `x_{t:T}` is through the hidden state of the RNN
    """
    def __init__(self, z_dim, h_dim):
        super(PostNet, self).__init__()
        self.z_to_h = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.Tanh()
        )
        self.h_to_mu = nn.Linear(h_dim, z_dim)
        self.h_to_logvar = nn.Linear(h_dim, z_dim)

    def forward(self, z_t_1, h_x):
        """
        Given the latent z at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t|z_{t-1}, x_{t:T})`
        """
        h_combined = 0.5*(self.z_to_h(z_t_1) + h_x)# combine the rnn hidden state with a transformed version of z_t_1
        mu = self.h_to_mu(h_combined)
        logvar = self.h_to_logvar(h_combined)
        std = F.softplus(logvar)     
        epsilon = torch.randn(z_t_1.size(), device=z_t_1.device) # sampling z by re-parameterization
        z_t = epsilon * std + mu   # [batch_sz x z_sz]
        return z_t, mu, logvar 
    
    
class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.cuda(), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)
        
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
#         print(self.weight.data)
#         print(self.bias.data)

    def forward(self, input):
#         print(self.filter_square_matrix.mul(self.weight))
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

class GRU_module(nn.Module):
    def __init__(self, input_size, hidden_size, device, num_layers=1, x_mean=0,\
                 bias=True, batch_first=False, bidirectional=False, dropout=0, dropout_type='mloss', return_hidden = False):
        
        super(GRU_module, self).__init__()
        
        self.hidden_size = hidden_size
#         self.noise_radius=noise_radius
        self.n_layers = num_layers
        self.bidir = bidirectional
        self.device = device
        assert type(self.bidir)==bool
        self.dropout=dropout
        
#         self.embedding = embedder # nn.Embedding(vocab_size, emb_size)
        self.rnn = GRUD_cell(input_size, hidden_size, device, self.n_layers, x_mean, bias = True, batch_first=True, bidirectional=self.bidir, dropout = self.dropout)
#         self.init_h = nn.Parameter(torch.randn(self.n_layers*(1+self.bidir), 1, self.hidden_size), requires_grad=True)#learnable h0
        self.init_h = torch.zeros([self.n_layers, 1, self.hidden_size], device = self.device)
        
#         if self.block == 'LSTM':
#         self.init_c = torch.zeros([self.n_layers*(1+self.bidir), 1, self.hidden_size],device = self.device)
    
    def forward2(self, inputs, masks, input_lens, deltas, init_h=None):
        batch_size, seq_len, emb_size=inputs.size()
#         if input_lens is not None:# sort and pack sequence
            
#         output = torch.zeros_like(inputs)
        
         
        input_lens_sorted, indices = input_lens.sort(descending=True)
#             print(inputs.shape, input_lens.shape)
        inputs_sorted = inputs.index_select(0, indices)
        
        mask_sorted = masks.index_select(0, indices)
        
        delta_sorted = deltas.index_select(0, indices)
            
        if init_h is None:
            init_h = self.init_h.expand(-1,batch_size,-1).contiguous()# use learnable initial states, expanding along batches
#         if self.block == 'LSTM':
#             if init_c is None:            
#                 init_c = self.init_c.expand(-1,batch_size,-1).contiguous()
        
        
        
        hids, _ = self.rnn(inputs_sorted, mask_sorted, delta_sorted, init_h)
        
        
        last_len = None
        
        last_id = None
        
        last_h_n = torch.zeros([self.n_layers, batch_size, self.hidden_size], device = self.device)
        
#         last_c_n = torch.zeros([self.n_layers, batch_size, self.hidden_size])
        
        
#         output_list = torch.zeros([batch_size, seq_len,(1+self.bidir)*self.hidden_size])
        
        output_list = torch.zeros([batch_size, seq_len,1*self.hidden_size], device = self.device)
        
        for k in range(len(input_lens_sorted)):
            if last_len is None:
                last_len = input_lens_sorted[k]
                last_id = k 
            else:
                if last_len == input_lens_sorted[k]:
                    continue
                else:
                    
#                     if self.block == 'LSTM':
#                         hids, (h_n, c_n) = self.rnn(inputs_sorted[last_id:k, 0:last_len], (init_h[:,last_id:k,:], init_c[:,last_id:k,:]))
#                         last_c_n[:,last_id:k] = c_n
#                     else:

                    '''Mask, Delta, init_h'''
#                     hids, h_n = self.rnn(inputs_sorted[last_id:k, 0:last_len], mask_sorted[last_id:k, 0:last_len], delta_sorted[last_id:k, 0:last_len], init_h[:,last_id:k,:])
                        
#                     output_list[last_id:k, 0:last_len] = hids
                    last_h_n[:,last_id:k] = hids[last_id:k, last_len - 1]
                    
                    
                    
                    
                    last_id = k
                    last_len = input_lens_sorted[k]
        
        
        
#         if self.block == 'LSTM':
#             hids, (h_n, c_n) = self.rnn(inputs_sorted[last_id:k+1, 0:last_len], (init_h[:,last_id:k+1,:], init_c[:,last_id:k+1,:]))
#             last_c_n[:,last_id:k+1] = c_n
#         else:
#         hids, h_n = self.rnn(inputs_sorted[last_id:k+1, 0:last_len], mask_sorted[last_id:k+1, 0:last_len], delta_sorted[last_id:k+1, 0:last_len], init_h[:,last_id:k+1,:])
            
#         output_list[last_id:k+1, 0:last_len] = hids
        last_h_n[:,last_id:k+1] = hids[last_id:k+1, last_len - 1]
        
        
        _, inv_indices = indices.sort()
        
        output_hiddens = hids[inv_indices]
        
#         print(torch.norm(output_hiddens[0, 0:input_lens[0]] - rnn_out[0, 0:input_lens[0]]))
#         
#         print(torch.norm(output_hiddens[-1, 0:input_lens[-1]] - rnn_out[-1, 0:input_lens[-1]]))
#         
#         print(torch.norm(last_h_n[:,inv_indices] - exp_last_h_n))
#         
#         print(torch.norm(last_c_n[:,inv_indices] - exp_last_c_n))
        
        return output_hiddens, last_h_n[:,inv_indices]#, last_c_n[:, inv_indices])

    
     
    def forward(self, inputs, masks, input_lens, deltas, init_h=None):
        batch_size, seq_len, emb_size=inputs.size()
#         if input_lens is not None:# sort and pack sequence
            
#         output = torch.zeros_like(inputs)
        
         
        input_lens_sorted, indices = input_lens.sort(descending=True)
#             print(inputs.shape, input_lens.shape)
        inputs_sorted = inputs.index_select(0, indices)
        
        mask_sorted = masks.index_select(0, indices)
        
        delta_sorted = deltas.index_select(0, indices)
            
        if init_h is None:
            init_h = self.init_h.expand(-1,batch_size,-1).contiguous()# use learnable initial states, expanding along batches
#         if self.block == 'LSTM':
#             if init_c is None:            
#                 init_c = self.init_c.expand(-1,batch_size,-1).contiguous()
        
        
        last_len = None
        
        last_id = None
        
        last_h_n = torch.zeros([self.n_layers, batch_size, self.hidden_size], device = self.device)
        
#         last_c_n = torch.zeros([self.n_layers, batch_size, self.hidden_size])
        
        
#         output_list = torch.zeros([batch_size, seq_len,(1+self.bidir)*self.hidden_size])
        
        output_list = torch.zeros([batch_size, seq_len,1*self.hidden_size], device = self.device)
        
        for k in range(len(input_lens_sorted)):
            if last_len is None:
                last_len = input_lens_sorted[k]
                last_id = k 
            else:
                if last_len == input_lens_sorted[k]:
                    continue
                else:
                    
#                     if self.block == 'LSTM':
#                         hids, (h_n, c_n) = self.rnn(inputs_sorted[last_id:k, 0:last_len], (init_h[:,last_id:k,:], init_c[:,last_id:k,:]))
#                         last_c_n[:,last_id:k] = c_n
#                     else:

                    '''Mask, Delta, init_h'''
                    hids, h_n = self.rnn(inputs_sorted[last_id:k, 0:last_len], mask_sorted[last_id:k, 0:last_len], delta_sorted[last_id:k, 0:last_len], init_h[:,last_id:k,:])
                        
                    output_list[last_id:k, 0:last_len] = hids
                    last_h_n[:,last_id:k] = h_n
                    
                    
                    
                    
                    last_id = k
                    last_len = input_lens_sorted[k]
        
        
        
#         if self.block == 'LSTM':
#             hids, (h_n, c_n) = self.rnn(inputs_sorted[last_id:k+1, 0:last_len], (init_h[:,last_id:k+1,:], init_c[:,last_id:k+1,:]))
#             last_c_n[:,last_id:k+1] = c_n
#         else:
        hids, h_n = self.rnn(inputs_sorted[last_id:k+1, 0:last_len], mask_sorted[last_id:k+1, 0:last_len], delta_sorted[last_id:k+1, 0:last_len], init_h[:,last_id:k+1,:])
            
        output_list[last_id:k+1, 0:last_len] = hids
        last_h_n[:,last_id:k+1] = h_n
        
        
        _, inv_indices = indices.sort()
        
        output_hiddens = output_list[inv_indices]
        
#         print(torch.norm(output_hiddens[0, 0:input_lens[0]] - rnn_out[0, 0:input_lens[0]]))
#         
#         print(torch.norm(output_hiddens[-1, 0:input_lens[-1]] - rnn_out[-1, 0:input_lens[-1]]))
#         
#         print(torch.norm(last_h_n[:,inv_indices] - exp_last_h_n))
#         
#         print(torch.norm(last_c_n[:,inv_indices] - exp_last_c_n))
        
        return output_hiddens, last_h_n[:,inv_indices]#, last_c_n[:, inv_indices])


class GRUI_cell(nn.Module):
    """
    Implementation of GRUD.
    Inputs: x_mean
            n_smp x 3 x n_channels x len_seq tensor (0: data, 1: mask, 2: deltat)
    """
    def __init__(self, input_size, hidden_size, device, num_layers=1, x_mean=0,\
                 bias=True, batch_first=False, bidirectional=False, dropout=0, dropout_type='mloss', return_hidden = False):

#         use_cuda = torch.cuda.is_available()
#         device = torch.device("cuda:0" if use_cuda else "cpu")
        
        super(GRUI_cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
#         self.output_size = output_size
        self.num_layers = num_layers
        self.return_hidden = return_hidden #controls the output, True if another GRU-D layer follows
        self.device = device

        x_mean = torch.tensor(x_mean, requires_grad = True, device = self.device)
        self.register_buffer('x_mean', x_mean)
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_type = dropout_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        
#         if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
#                 isinstance(dropout, bool):
#             raise ValueError("dropout should be a number in range [0, 1] "
#                              "representing the probability of an element being "
#                              "zeroed")
#         if dropout > 0 and num_layers == 1:
#             warnings.warn("dropout option adds dropout after all but last "
#                           "recurrent layer, so non-zero dropout expects "
#                           "num_layers greater than 1, but got dropout={} and "
#                           "num_layers={}".format(dropout, num_layers))
        
        

        #set up all the operations that are needed in the forward pass
        self.w_dg_x = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.w_dg_h = torch.nn.Linear(input_size, hidden_size, bias = True)

        self.w_mu = torch.nn.Linear(input_size + hidden_size, hidden_size, bias = True)
        
        self.w_r = torch.nn.Linear(input_size + hidden_size, hidden_size, bias = True)
        
        self.w_h = torch.nn.Linear(input_size + hidden_size, hidden_size, bias = True)

#         self.w_xz = torch.nn.Linear(input_size, hidden_size, bias=False)
#         self.w_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
#         self.w_mz = torch.nn.Linear(input_size, hidden_size, bias=True)
# 
#         self.w_xr = torch.nn.Linear(input_size, hidden_size, bias=False)
#         self.w_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
#         self.w_mr = torch.nn.Linear(input_size, hidden_size, bias=False)
#         self.w_xh = torch.nn.Linear(input_size, hidden_size, bias=False)
#         self.w_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
#         self.w_mh = torch.nn.Linear(input_size, hidden_size, bias=True)

#         self.w_hy = torch.nn.Linear(hidden_size, output_size, bias=True)
        
    


        Hidden_State = torch.zeros(self.hidden_size, requires_grad = True, device = self.device)
        #we use buffers because pytorch will take care of pushing them to GPU for us
        self.register_buffer('Hidden_State', Hidden_State)
        self.register_buffer('X_last_obs', torch.zeros(input_size, device = self.device)) #torch.tensor(x_mean) #TODO: what to initialize last observed values with?, also check broadcasting behaviour

    
    #TODO: check usefulness of everything below here, just copied skeleton


        self.reset_parameters()
        
    


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        
        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode == 'LSTM':
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)
    
    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)
    
    



    @property
    def _flat_weights(self):
        return list(self._parameters.values())


    def forward(self, input, Mask, Delta, init_h = None):
        # input.size = (3, 33,49) : num_input or num_hidden, num_layer or step
        #X = torch.squeeze(input[0]) # .size = (33,49)
        #Mask = torch.squeeze(input[1]) # .size = (33,49)
        #Delta = torch.squeeze(input[2]) # .size = (33,49)
#         X = input[:,0,:,:]
#         Mask = input[:,1,:,:]
#         Delta = input[:,2,:,:]
        
        
        X = input
#         Mask = input[:,1,:,:]
#         Delta = input[:,2,:,:]
        

        step_size = X.size(1) # 49
        #print('step size : ', step_size)
        
        output = None
        #h = Hidden_State
        
        if init_h is None:
            h = getattr(self, 'Hidden_State')
        else:
            h = init_h
        #felix - buffer system from newer pytorch version
        x_mean = getattr(self, 'x_mean')
        x_last_obsv = getattr(self, 'X_last_obs')
        

#         device = next(self.parameters()).device
#         output_tensor = torch.empty([X.size()[0], X.size()[1], self.output_size], dtype=X.dtype, device= self.device)
        hidden_tensor = torch.empty(X.size()[0], X.size()[1], self.hidden_size, dtype=X.dtype, device = self.device)

        #iterate over seq
        for timestep in range(X.size()[1]):
            
            #x = torch.squeeze(X[:,layer:layer+1])
            #m = torch.squeeze(Mask[:,layer:layer+1])
            #d = torch.squeeze(Delta[:,layer:layer+1])
#             x = torch.unsqueeze(X[:,timestep,:])
#             m = torch.unsqueeze(Mask[:,timestep,:])
#             d = torch.unsqueeze(Delta[:,timestep,:])
            
            
            x = (X[:,timestep,:]).unsqueeze(0)
            m = (Mask[:,timestep,:]).unsqueeze(0)
            d = (Delta[:,timestep,:]).unsqueeze(0)

            #(4)
            gamma_x = torch.exp(-1* torch.nn.functional.relu( self.w_dg_x(d) ))
#             gamma_h = torch.exp(-1* torch.nn.functional.relu( self.w_dg_h(d) ))

            h_prime = gamma_x*h
            #(5)
            #standard mult handles case correctly, this should work - maybe broadcast x_mean, seems to be taking care of that anyway
            
            #update x_last_obsv
            #print(x.size())
            #print(x_last_obsv.size())
            x_last_obsv = torch.where(m>0,x,x_last_obsv)
            
            
            gate = F.sigmoid(self.w_mu(torch.cat([h_prime, x], -1)))
            
            reset = F.sigmoid(self.w_r(torch.cat([h_prime, x], -1)))
            
            h_bar = F.tanh(self.w_h(torch.cat([h_prime*reset,x], -1)))
            
            h = (1-gate)*h + gate*h_bar
            
            dropout = torch.nn.Dropout(p=self.dropout)
            h = dropout(h)
            
            
#             print(timestep, h)
            #print('after update')
            #print(x_last_obsv)
            
            
            
#             x = m * x + (1 - m) * (gamma_x * x + (1 - gamma_x) * x_mean)
#             x = m * x + (1 - m) * (gamma_x * x_last_obsv + (1 - gamma_x) * x_mean)
# 
#             #(6)
#             if self.dropout == 0:
# 
#                 h = gamma_h*h
#                 
#                 z = F.sigmoid( self.w_xz(x) + self.w_hz(h) + self.w_mz(m))
#                 r = F.sigmoid( self.w_xr(x) + self.w_hr(h) + self.w_mr(m))
# 
#                 h_tilde = F.tanh( self.w_xh(x) + self.w_hh( r*h ) + self.w_mh(m))
# 
# 
#                 h = (1 - z) * h + z * h_tilde
# 
#             #TODO: not adapted yet
#             elif self.dropout_type == 'Moon':
#                 '''
#                 RNNDROP: a novel dropout for rnn in asr(2015)
#                 '''
#                 h = gamma_h * h
# 
#                 z = F.sigmoid((self.w_xz(x) + self.w_hz(h) + self.w_mz(m)))
#                 r = F.sigmoid((self.w_xr(x) + self.w_hr(h) + self.w_mr(m)))
# 
#                 h_tilde = F.tanh((self.w_xh(x) + self.w_hh(r * h) + self.w_mh(m)))
# 
#                 h = (1 - z) * h + z * h_tilde
#                 dropout = torch.nn.Dropout(p=self.dropout)
#                 h = dropout(h)
# 
#             elif self.dropout_type == 'Gal':
#                 '''
#                 A Theoretically grounded application of dropout in recurrent neural networks(2015)
#                 '''
#                 dropout = torch.nn.Dropout(p=self.dropout)
#                 h = dropout(h)
# 
#                 h = gamma_h * h
# 
#                 z = F.sigmoid((self.w_xz(x) + self.w_hz(h) + self.w_mz(m)))
#                 r = F.sigmoid((self.w_xr(x) + self.w_hr(h) + self.w_mr(m)))
#                 h_tilde = F.tanh((self.w_xh(x) + self.w_hh(r * h) + self.w_mh(m)))
# 
#                 h = (1 - z) * h + z * h_tilde
# 
#             elif self.dropout_type == 'mloss':
#                 '''
#                 recurrent dropout without memory loss arXiv 1603.05118
#                 g = h_tilde, p = the probability to not drop a neuron
#                 '''
#                 h = gamma_h*h
#                 z = F.sigmoid( self.w_xz(x) + self.w_hz(h) + self.w_mz(m))
#                 r = F.sigmoid( self.w_xr(x) + self.w_hr(h) + self.w_mr(m))
# 
# 
#                 dropout = torch.nn.Dropout(p=self.dropout)
#                 h_tilde = F.tanh( self.w_xh(x) + self.w_hh( r*h ) + self.w_mh(m))
# 
# 
#                 h = (1 - z) * h + z * h_tilde
#                 h = dropout(h)
#                 #######
# 
#             else:
#                 h = gamma_h * h
# 
#                 z = F.sigmoid((self.w_xz(x) + self.w_hz(h) + self.w_mz(m)))
#                 r = F.sigmoid((self.w_xr(x) + self.w_hr(h) +self.w_mr(m)))
#                 h_tilde = F.tanh((self.w_xh(x) + self.w_hh((r * h)) + self.w_mh(m)))
# 
#                 h = (1 - z) * h + z * h_tilde

            

#             step_output = self.w_hy(h)
#             step_output = torch.sigmoid(step_output)
#             output_tensor[:,timestep,:] = step_output
            hidden_tensor[:,timestep,:] = h
            
        #if self.return_hidden:
            #when i want to stack GRU-Ds, need to put the tensor back together
            #output = torch.stack([hidden_tensor,Mask,Delta], dim=1)
        
#         output = output_tensor, hidden_tensor
        
        output = hidden_tensor
        
        #else:
        #    output = output_tensor
        return output, h




class GRUI_module(nn.Module):
    def __init__(self, input_size, hidden_size, device, num_layers=1, x_mean=0,\
                 bias=True, batch_first=False, bidirectional=False, dropout=0, dropout_type='mloss', return_hidden = False):
        
        super(GRUI_module, self).__init__()
        
        self.hidden_size = hidden_size
#         self.noise_radius=noise_radius
        self.n_layers = num_layers
        self.bidir = bidirectional
        self.device = device
        assert type(self.bidir)==bool
        self.dropout=dropout
        
#         self.embedding = embedder # nn.Embedding(vocab_size, emb_size)
        self.rnn = GRUI_cell(input_size, hidden_size, device, self.n_layers, x_mean, bias = True, batch_first=True, bidirectional=self.bidir, dropout = self.dropout)
#         self.init_h = nn.Parameter(torch.randn(self.n_layers*(1+self.bidir), 1, self.hidden_size), requires_grad=True)#learnable h0
        self.init_h = torch.zeros([self.n_layers, 1, self.hidden_size], device = self.device)
        
#         if self.block == 'LSTM':
#         self.init_c = torch.zeros([self.n_layers*(1+self.bidir), 1, self.hidden_size],device = self.device)
    
    def forward2(self, inputs, masks, input_lens, deltas, init_h=None):
        batch_size, seq_len, emb_size=inputs.size()
#         if input_lens is not None:# sort and pack sequence
            
#         output = torch.zeros_like(inputs)
        
         
        input_lens_sorted, indices = input_lens.sort(descending=True)
#             print(inputs.shape, input_lens.shape)
        inputs_sorted = inputs.index_select(0, indices)
        
        mask_sorted = masks.index_select(0, indices)
        
        delta_sorted = deltas.index_select(0, indices)
            
        if init_h is None:
            init_h = self.init_h.expand(-1, batch_size,-1).contiguous()# use learnable initial states, expanding along batches
#         if self.block == 'LSTM':
#             if init_c is None:            
#                 init_c = self.init_c.expand(-1,batch_size,-1).contiguous()
        
        
        
        hids, _ = self.rnn(inputs_sorted, mask_sorted, delta_sorted, init_h)
        
        
        last_len = None
        
        last_id = None
        
        last_h_n = torch.zeros([self.n_layers, batch_size, self.hidden_size], device = self.device)
        
#         last_c_n = torch.zeros([self.n_layers, batch_size, self.hidden_size])
        
        
#         output_list = torch.zeros([batch_size, seq_len,(1+self.bidir)*self.hidden_size])
        
        output_list = torch.zeros([batch_size, seq_len,1*self.hidden_size], device = self.device)
        
        for k in range(len(input_lens_sorted)):
            if last_len is None:
                last_len = input_lens_sorted[k]
                last_id = k 
            else:
                if last_len == input_lens_sorted[k]:
                    continue
                else:
                    
#                     if self.block == 'LSTM':
#                         hids, (h_n, c_n) = self.rnn(inputs_sorted[last_id:k, 0:last_len], (init_h[:,last_id:k,:], init_c[:,last_id:k,:]))
#                         last_c_n[:,last_id:k] = c_n
#                     else:

                    '''Mask, Delta, init_h'''
#                     hids, h_n = self.rnn(inputs_sorted[last_id:k, 0:last_len], mask_sorted[last_id:k, 0:last_len], delta_sorted[last_id:k, 0:last_len], init_h[:,last_id:k,:])
                        
#                     output_list[last_id:k, 0:last_len] = hids
                    last_h_n[:,last_id:k] = hids[last_id:k, last_len - 1]
                    
                    
                    
                    
                    last_id = k
                    last_len = input_lens_sorted[k]
        
        
        
#         if self.block == 'LSTM':
#             hids, (h_n, c_n) = self.rnn(inputs_sorted[last_id:k+1, 0:last_len], (init_h[:,last_id:k+1,:], init_c[:,last_id:k+1,:]))
#             last_c_n[:,last_id:k+1] = c_n
#         else:
#         hids, h_n = self.rnn(inputs_sorted[last_id:k+1, 0:last_len], mask_sorted[last_id:k+1, 0:last_len], delta_sorted[last_id:k+1, 0:last_len], init_h[:,last_id:k+1,:])
            
#         output_list[last_id:k+1, 0:last_len] = hids
        last_h_n[:,last_id:k+1] = hids[last_id:k+1, last_len - 1]
        
        
        _, inv_indices = indices.sort()
        
        output_hiddens = hids[inv_indices]
        
#         print(torch.norm(output_hiddens[0, 0:input_lens[0]] - rnn_out[0, 0:input_lens[0]]))
#         
#         print(torch.norm(output_hiddens[-1, 0:input_lens[-1]] - rnn_out[-1, 0:input_lens[-1]]))
#         
#         print(torch.norm(last_h_n[:,inv_indices] - exp_last_h_n))
#         
#         print(torch.norm(last_c_n[:,inv_indices] - exp_last_c_n))
        
        return output_hiddens, last_h_n[:,inv_indices]#, last_c_n[:, inv_indices])

    
     
    def forward(self, inputs, masks, input_lens, deltas, init_h=None):
        batch_size, seq_len, emb_size=inputs.size()
#         if input_lens is not None:# sort and pack sequence
            
#         output = torch.zeros_like(inputs)
        
         
        input_lens_sorted, indices = input_lens.sort(descending=True)
#             print(inputs.shape, input_lens.shape)
        inputs_sorted = inputs.index_select(0, indices)
        
        mask_sorted = masks.index_select(0, indices)
        
        delta_sorted = deltas.index_select(0, indices)
            
        if init_h is None:
            init_h = self.init_h.expand(-1,batch_size,-1).contiguous()# use learnable initial states, expanding along batches
#         if self.block == 'LSTM':
#             if init_c is None:            
#                 init_c = self.init_c.expand(-1,batch_size,-1).contiguous()
        
        
        last_len = None
        
        last_id = None
        
        last_h_n = torch.zeros([self.n_layers, batch_size, self.hidden_size], device = self.device)
        
#         last_c_n = torch.zeros([self.n_layers, batch_size, self.hidden_size])
        
        
#         output_list = torch.zeros([batch_size, seq_len,(1+self.bidir)*self.hidden_size])
        
        output_list = torch.zeros([batch_size, seq_len,1*self.hidden_size], device = self.device)
        
        for k in range(len(input_lens_sorted)):
            if last_len is None:
                last_len = input_lens_sorted[k]
                last_id = k 
            else:
                if last_len == input_lens_sorted[k]:
                    continue
                else:
                    
#                     if self.block == 'LSTM':
#                         hids, (h_n, c_n) = self.rnn(inputs_sorted[last_id:k, 0:last_len], (init_h[:,last_id:k,:], init_c[:,last_id:k,:]))
#                         last_c_n[:,last_id:k] = c_n
#                     else:

                    '''Mask, Delta, init_h'''
                    hids, h_n = self.rnn(inputs_sorted[last_id:k, 0:last_len], mask_sorted[last_id:k, 0:last_len], delta_sorted[last_id:k, 0:last_len], init_h[:,last_id:k,:])
                        
                    output_list[last_id:k, 0:last_len] = hids
                    last_h_n[:,last_id:k] = h_n
                    
                    
                    
                    
                    last_id = k
                    last_len = input_lens_sorted[k]
        
        
        
#         if self.block == 'LSTM':
#             hids, (h_n, c_n) = self.rnn(inputs_sorted[last_id:k+1, 0:last_len], (init_h[:,last_id:k+1,:], init_c[:,last_id:k+1,:]))
#             last_c_n[:,last_id:k+1] = c_n
#         else:
        hids, h_n = self.rnn(inputs_sorted[last_id:k+1, 0:last_len], mask_sorted[last_id:k+1, 0:last_len], delta_sorted[last_id:k+1, 0:last_len], init_h[:,last_id:k+1,:])
            
        output_list[last_id:k+1, 0:last_len] = hids
        last_h_n[:,last_id:k+1] = h_n
        
        
        _, inv_indices = indices.sort()
        
        output_hiddens = output_list[inv_indices]
        
#         print(torch.norm(output_hiddens[0, 0:input_lens[0]] - rnn_out[0, 0:input_lens[0]]))
#         
#         print(torch.norm(output_hiddens[-1, 0:input_lens[-1]] - rnn_out[-1, 0:input_lens[-1]]))
#         
#         print(torch.norm(last_h_n[:,inv_indices] - exp_last_h_n))
#         
#         print(torch.norm(last_c_n[:,inv_indices] - exp_last_c_n))
        
        return output_hiddens, last_h_n[:,inv_indices]#, last_c_n[:, inv_indices])


# 
# class GRUI_cell(nn.Module):
#     def __init__(self, latent_dim, input_dim, 
#         update_gate = None,
#         reset_gate = None,
#         new_state_net = None,
#         n_units = 100,
#         device = torch.device("cpu"), use_mask = False, dropout = 0.0):
#         super(GRUI_cell, self).__init__()
# 
#         if update_gate is None:
#             if use_mask:
#                 self.update_gate = nn.Sequential(
# #                    nn.Linear(latent_dim + 2*input_dim, n_units),
# #                    nn.Tanh(),
# #                    nn.Linear(n_units, latent_dim),
#                     nn.Linear(latent_dim + 2*input_dim, latent_dim),
#                     nn.Dropout(p=dropout),
#                    nn.Sigmoid())
#             else:
#                 self.update_gate = nn.Sequential(
# #                    nn.Linear(latent_dim + input_dim, n_units),
# #                    nn.Tanh(),
# #                    nn.Linear(n_units, latent_dim),
#                     nn.Linear(latent_dim + input_dim, latent_dim),
#                     nn.Dropout(p = dropout),
#                    nn.Sigmoid())
# #             utils.init_network_weights(self.update_gate)
#         else: 
#             self.update_gate  = update_gate
# 
#         if reset_gate is None:
#             if use_mask:
#                 self.reset_gate = nn.Sequential(
# #                    nn.Linear(latent_dim + 2*input_dim, n_units),
# #                    nn.Tanh(),
# #                    nn.Linear(n_units, latent_dim),
#                     nn.Linear(latent_dim + 2*input_dim, latent_dim),
#                     nn.Dropout(p = dropout),
#                    nn.Sigmoid())
#             else:
#                 self.reset_gate = nn.Sequential(
# #                    nn.Linear(latent_dim + input_dim, n_units),
# #                    nn.Tanh(),
# #                    nn.Linear(n_units, latent_dim),
#                     nn.Linear(latent_dim + input_dim, latent_dim),
#                     nn.Dropout(p = dropout),
#                    nn.Sigmoid())
# #             utils.init_network_weights(self.reset_gate)
#         else: 
#             self.reset_gate  = reset_gate
# 
#         if new_state_net is None:
#             if use_mask:
#                 self.new_state_net = nn.Sequential(
# #                    nn.Linear(latent_dim + 2*input_dim, n_units),
# #                    nn.Tanh(),
# #                    nn.Linear(n_units, latent_dim)
#                     nn.Linear(latent_dim + 2*input_dim, latent_dim),
#                     nn.Dropout(p = dropout),
#                    )
#             else:
#                 self.new_state_net = nn.Sequential(
# #                    nn.Linear(latent_dim + input_dim, n_units),
# #                    nn.Tanh(),
# #                    nn.Linear(n_units, latent_dim)
#                 nn.Linear(latent_dim + input_dim, latent_dim),
#                 nn.Dropout(p = dropout),
#                    )
# #             utils.init_network_weights(self.new_state_net)
#         else: 
#             self.new_state_net  = new_state_net
# 
#         self.time_decay_layer = nn.Sequential(nn.Linear(input_dim, 1))
# 
# 
#     def forward(self, y_i, x, delta_t_i):
#         
#         
#         1/torch.exp(F.relu(self.time_decay_layer(delta_t_i)))
#         
#         y_concat = torch.cat([y_i, x], -1)
# 
#         update_gate = self.update_gate(y_concat)
#         reset_gate = self.reset_gate(y_concat)
#         
#         
#         concat = y_i * reset_gate
#         
#         concat = torch.cat([concat, x], -1)
#         
#         new_probs = self.new_state_net(concat)
#         
# #         new_state, new_state_std = utils.split_last_dim(self.new_state_net(concat))
# #         new_state_std = new_state_std.abs()
# 
#         new_y_probs = (1-update_gate) * new_probs + update_gate * y_i
# #         new_y_std = (1-update_gate) * new_state_std + update_gate * y_std
# 
#         assert(not torch.isnan(new_y_probs).any())
# 
# #         if masked_update:
# #             # IMPORTANT: assumes that x contains both data and mask
# #             # update only the hidden states for hidden state only if at least one feature is present for the current time point
# #             n_data_dims = x.size(-1)//2
# #             mask = x[:, :, n_data_dims:]
# #             utils.check_mask(x[:, :, :n_data_dims], mask)
# #             
# #             mask = (torch.sum(mask, -1, keepdim = True) > 0).float()
# # 
# #             assert(not torch.isnan(mask).any())
# # 
# #             new_y = mask * new_y + (1-mask) * y_mean
# #             new_y_std = mask * new_y_std + (1-mask) * y_std
# # 
# #             if torch.isnan(new_y).any():
# #                 print("new_y is nan!")
# #                 print(mask)
# #                 print(y_mean)
# #                 print(new_y)
# #                 exit()
# 
# #         new_y_std = new_y_std.abs()
# #         return new_y, new_y_std
#         return new_y_probs


class GRUD_cell(nn.Module):
    """
    Implementation of GRUD.
    Inputs: x_mean
            n_smp x 3 x n_channels x len_seq tensor (0: data, 1: mask, 2: deltat)
    """
    def __init__(self, input_size, hidden_size, device, num_layers=1, x_mean=0,\
                 bias=True, batch_first=False, bidirectional=False, dropout=0, dropout_type='mloss', return_hidden = False):

#         use_cuda = torch.cuda.is_available()
#         device = torch.device("cuda:0" if use_cuda else "cpu")
        
        super(GRUD_cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
#         self.output_size = output_size
        self.num_layers = num_layers
        self.return_hidden = return_hidden #controls the output, True if another GRU-D layer follows
        self.device = device

        x_mean = torch.tensor(x_mean, requires_grad = True, device = self.device)
        self.register_buffer('x_mean', x_mean)
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_type = dropout_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        
#         if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
#                 isinstance(dropout, bool):
#             raise ValueError("dropout should be a number in range [0, 1] "
#                              "representing the probability of an element being "
#                              "zeroed")
#         if dropout > 0 and num_layers == 1:
#             warnings.warn("dropout option adds dropout after all but last "
#                           "recurrent layer, so non-zero dropout expects "
#                           "num_layers greater than 1, but got dropout={} and "
#                           "num_layers={}".format(dropout, num_layers))
        
        

        #set up all the operations that are needed in the forward pass
        self.w_dg_x = torch.nn.Linear(input_size,input_size, bias=True)
        self.w_dg_h = torch.nn.Linear(input_size, hidden_size, bias = True)

        self.w_xz = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mz = torch.nn.Linear(input_size, hidden_size, bias=True)

        self.w_xr = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mr = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_xh = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mh = torch.nn.Linear(input_size, hidden_size, bias=True)

#         self.w_hy = torch.nn.Linear(hidden_size, output_size, bias=True)
        
    


        Hidden_State = torch.zeros(self.hidden_size, requires_grad = True, device = self.device)
        #we use buffers because pytorch will take care of pushing them to GPU for us
        self.register_buffer('Hidden_State', Hidden_State)
        self.register_buffer('X_last_obs', torch.zeros(input_size, device = self.device)) #torch.tensor(x_mean) #TODO: what to initialize last observed values with?, also check broadcasting behaviour

    
    #TODO: check usefulness of everything below here, just copied skeleton


        self.reset_parameters()
        
    


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        
        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode == 'LSTM':
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)
    
    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)
    
    



    @property
    def _flat_weights(self):
        return list(self._parameters.values())


    def forward(self, input, Mask, Delta, init_h = None):
        # input.size = (3, 33,49) : num_input or num_hidden, num_layer or step
        #X = torch.squeeze(input[0]) # .size = (33,49)
        #Mask = torch.squeeze(input[1]) # .size = (33,49)
        #Delta = torch.squeeze(input[2]) # .size = (33,49)
#         X = input[:,0,:,:]
#         Mask = input[:,1,:,:]
#         Delta = input[:,2,:,:]
        
        
        X = input
#         Mask = input[:,1,:,:]
#         Delta = input[:,2,:,:]
        

        step_size = X.size(1) # 49
        #print('step size : ', step_size)
        
        output = None
        #h = Hidden_State
        
        if init_h is None:
            h = getattr(self, 'Hidden_State')
        else:
            h = init_h
        #felix - buffer system from newer pytorch version
        x_mean = getattr(self, 'x_mean')
        x_last_obsv = getattr(self, 'X_last_obs')
        

#         device = next(self.parameters()).device
#         output_tensor = torch.empty([X.size()[0], X.size()[1], self.output_size], dtype=X.dtype, device= self.device)
        hidden_tensor = torch.empty(X.size()[0], X.size()[1], self.hidden_size, dtype=X.dtype, device = self.device)

        #iterate over seq
        for timestep in range(X.size()[1]):
            
            #x = torch.squeeze(X[:,layer:layer+1])
            #m = torch.squeeze(Mask[:,layer:layer+1])
            #d = torch.squeeze(Delta[:,layer:layer+1])
            x = torch.squeeze(X[:,timestep,:])
            m = torch.squeeze(Mask[:,timestep,:])
            d = torch.squeeze(Delta[:,timestep,:])
            

            #(4)
            gamma_x = torch.exp(-1* torch.nn.functional.relu( self.w_dg_x(d) ))
            gamma_h = torch.exp(-1* torch.nn.functional.relu( self.w_dg_h(d) ))


            #(5)
            #standard mult handles case correctly, this should work - maybe broadcast x_mean, seems to be taking care of that anyway
            
            #update x_last_obsv
            #print(x.size())
            #print(x_last_obsv.size())
            x_last_obsv = torch.where(m>0,x,x_last_obsv)
            #print('after update')
            #print(x_last_obsv)
            x = m * x + (1 - m) * (gamma_x * x + (1 - gamma_x) * x_mean)
            x = m * x + (1 - m) * (gamma_x * x_last_obsv + (1 - gamma_x) * x_mean)

            #(6)
            if self.dropout == 0:

                h = gamma_h*h
                
                z = F.sigmoid( self.w_xz(x) + self.w_hz(h) + self.w_mz(m))
                r = F.sigmoid( self.w_xr(x) + self.w_hr(h) + self.w_mr(m))

                h_tilde = F.tanh( self.w_xh(x) + self.w_hh( r*h ) + self.w_mh(m))


                h = (1 - z) * h + z * h_tilde

            #TODO: not adapted yet
            elif self.dropout_type == 'Moon':
                '''
                RNNDROP: a novel dropout for rnn in asr(2015)
                '''
                h = gamma_h * h

                z = F.sigmoid((self.w_xz(x) + self.w_hz(h) + self.w_mz(m)))
                r = F.sigmoid((self.w_xr(x) + self.w_hr(h) + self.w_mr(m)))

                h_tilde = F.tanh((self.w_xh(x) + self.w_hh(r * h) + self.w_mh(m)))

                h = (1 - z) * h + z * h_tilde
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

            elif self.dropout_type == 'Gal':
                '''
                A Theoretically grounded application of dropout in recurrent neural networks(2015)
                '''
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

                h = gamma_h * h

                z = F.sigmoid((self.w_xz(x) + self.w_hz(h) + self.w_mz(m)))
                r = F.sigmoid((self.w_xr(x) + self.w_hr(h) + self.w_mr(m)))
                h_tilde = F.tanh((self.w_xh(x) + self.w_hh(r * h) + self.w_mh(m)))

                h = (1 - z) * h + z * h_tilde

            elif self.dropout_type == 'mloss':
                '''
                recurrent dropout without memory loss arXiv 1603.05118
                g = h_tilde, p = the probability to not drop a neuron
                '''
                h = gamma_h*h
                z = F.sigmoid( self.w_xz(x) + self.w_hz(h) + self.w_mz(m))
                r = F.sigmoid( self.w_xr(x) + self.w_hr(h) + self.w_mr(m))


                dropout = torch.nn.Dropout(p=self.dropout)
                h_tilde = F.tanh( self.w_xh(x) + self.w_hh( r*h ) + self.w_mh(m))


                h = (1 - z) * h + z * h_tilde
                h = dropout(h)
                #######

            else:
                h = gamma_h * h

                z = F.sigmoid((self.w_xz(x) + self.w_hz(h) + self.w_mz(m)))
                r = F.sigmoid((self.w_xr(x) + self.w_hr(h) +self.w_mr(m)))
                h_tilde = F.tanh((self.w_xh(x) + self.w_hh((r * h)) + self.w_mh(m)))

                h = (1 - z) * h + z * h_tilde

            

#             step_output = self.w_hy(h)
#             step_output = torch.sigmoid(step_output)
#             output_tensor[:,timestep,:] = step_output
            hidden_tensor[:,timestep,:] = h
            
        #if self.return_hidden:
            #when i want to stack GRU-Ds, need to put the tensor back together
            #output = torch.stack([hidden_tensor,Mask,Delta], dim=1)
        
#         output = output_tensor, hidden_tensor
        
        output = hidden_tensor
        
        #else:
        #    output = output_tensor
        return output, h











class TimeLSTM_module(nn.Module):
    
    def __init__(self, input_size, hidden_size, dropout, device, bidirectional=False):
        super(TimeLSTM_module, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
#         self.cuda_flag = cuda_flag
        
        
        self.device = device
#         assert type(self.bidir)==bool
        self.dropout=dropout
        
        self.init_h = torch.zeros([self.hidden_size], device = self.device)
        
        self.init_c = torch.zeros([self.hidden_size], device = self.device)
        
        
        self.lstm = TimeLSTMCell(input_size, hidden_size, dropout, device, bidirectional)
        
        
        
#         self.W_all = nn.Linear(hidden_size, hidden_size * 4)
#         self.U_all = nn.Linear(input_size, hidden_size * 4)
#         self.W_d = nn.Linear(hidden_size, hidden_size)
#         self.bidirectional = bidirectional
    def forward(self, inputs, time_stamps, input_lens, init_h=None, init_c=None):
        
        batch_size = inputs.shape[0]
        
        seq_len = inputs.shape[1]
        
        if init_h is None:
            init_h = self.init_h.expand(batch_size, self.init_h.shape[0])
            
        if init_c is None:
            init_c = self.init_c.expand(batch_size, self.init_h.shape[0])
        
        input_lens_sorted, indices = input_lens.sort(descending=True)
#             print(inputs.shape, input_lens.shape)
        inputs_sorted = inputs.index_select(0, indices)
        
        time_stamps_sorted = time_stamps.index_select(0, indices)
        
        
        last_len = None
        
        last_id = None
        
        last_h_n = torch.zeros([batch_size, self.hidden_size], device = self.device)
        
        last_c_n = torch.zeros([batch_size, self.hidden_size], device = self.device)
        
        
#         output_list = torch.zeros([batch_size, seq_len,(1+self.bidir)*self.hidden_size])
        
        output_list = torch.zeros([batch_size, seq_len,1*self.hidden_size], device = self.device)
        
        for k in range(len(input_lens_sorted)):
            if last_len is None:
                last_len = input_lens_sorted[k]
                last_id = k 
            else:
                if last_len == input_lens_sorted[k]:
                    continue
                else:
                    
#                     if self.block == 'LSTM':
#                         hids, (h_n, c_n) = self.rnn(inputs_sorted[last_id:k, 0:last_len], (init_h[:,last_id:k,:], init_c[:,last_id:k,:]))
#                         last_c_n[:,last_id:k] = c_n
#                     else:

                    '''Mask, Delta, init_h'''
                    hids, h_n, c_n = self.lstm(inputs_sorted[last_id:k, 0:last_len], time_stamps_sorted[last_id:k, 0:last_len], init_h[last_id:k,:], init_c[last_id:k,:])
                        
                    output_list[last_id:k, 0:last_len] = hids
                    last_h_n[last_id:k] = h_n
                    last_c_n[last_id:k] = c_n
                    
                    
                    
                    
                    last_id = k
                    last_len = input_lens_sorted[k]
        
        
        
#         if self.block == 'LSTM':
#             hids, (h_n, c_n) = self.rnn(inputs_sorted[last_id:k+1, 0:last_len], (init_h[:,last_id:k+1,:], init_c[:,last_id:k+1,:]))
#             last_c_n[:,last_id:k+1] = c_n
#         else:
        hids, h_n, c_n = self.lstm(inputs_sorted[last_id:k+1, 0:last_len], time_stamps_sorted[last_id:k+1, 0:last_len], init_h[last_id:k+1,:], init_c[last_id:k+1,:])
            
        output_list[last_id:k+1, 0:last_len] = hids
        last_h_n[last_id:k+1] = h_n
        last_c_n[last_id:k+1] = c_n
        
        
        _, inv_indices = indices.sort()
        
        output_hiddens = output_list[inv_indices]
        
#         print(torch.norm(output_hiddens[0, 0:input_lens[0]] - rnn_out[0, 0:input_lens[0]]))
#         
#         print(torch.norm(output_hiddens[-1, 0:input_lens[-1]] - rnn_out[-1, 0:input_lens[-1]]))
#         
#         print(torch.norm(last_h_n[:,inv_indices] - exp_last_h_n))
#         
#         print(torch.norm(last_c_n[:,inv_indices] - exp_last_c_n))
        
        
        
        return output_hiddens, last_h_n[inv_indices], last_c_n[inv_indices]
    
    def forward2(self, inputs, time_stamps, input_lens, init_h=None, init_c=None):
        
        batch_size = inputs.shape[0]
        
        seq_len = inputs.shape[1]
        
        if init_h is None:
            init_h = self.init_h.expand(batch_size, self.init_h.shape[0])
            
        if init_c is None:
            init_c = self.init_c.expand(batch_size, self.init_h.shape[0])
        
        input_lens_sorted, indices = input_lens.sort(descending=True)
#             print(inputs.shape, input_lens.shape)
        inputs_sorted = inputs.index_select(0, indices)
        
        time_stamps_sorted = time_stamps.index_select(0, indices)
        
        
        
        
        h_outputs, c_outputs = self.lstm.forward2(inputs_sorted, time_stamps_sorted, init_h, init_c)
        
        last_len = None
        
        last_id = None
        
        last_h_n = torch.zeros([batch_size, self.hidden_size], device = self.device)
        
        last_c_n = torch.zeros([batch_size, self.hidden_size], device = self.device)
        
        
#         output_list = torch.zeros([batch_size, seq_len,(1+self.bidir)*self.hidden_size])
        
        output_list = torch.zeros([batch_size, seq_len,1*self.hidden_size], device = self.device)
        
        for k in range(len(input_lens_sorted)):
            if last_len is None:
                last_len = input_lens_sorted[k]
                last_id = k 
            else:
                if last_len == input_lens_sorted[k]:
                    continue
                else:
                    
#                     if self.block == 'LSTM':
#                         hids, (h_n, c_n) = self.rnn(inputs_sorted[last_id:k, 0:last_len], (init_h[:,last_id:k,:], init_c[:,last_id:k,:]))
#                         last_c_n[:,last_id:k] = c_n
#                     else:

                    '''Mask, Delta, init_h'''
#                     hids, h_n, c_n = self.lstm(inputs_sorted[last_id:k, 0:last_len], time_stamps_sorted[last_id:k, 0:last_len], init_h[last_id:k,:], init_c[last_id:k,:])
                    
                    
                    
#                     output_list[last_id:k, 0:last_len] = hids
                    last_h_n[last_id:k] = h_outputs[last_id:k,last_len-1]
                    last_c_n[last_id:k] = c_outputs[last_id:k,last_len-1]
                    
                    
                    
                    
                    last_id = k
                    last_len = input_lens_sorted[k]
        
        
        
#         if self.block == 'LSTM':
#             hids, (h_n, c_n) = self.rnn(inputs_sorted[last_id:k+1, 0:last_len], (init_h[:,last_id:k+1,:], init_c[:,last_id:k+1,:]))
#             last_c_n[:,last_id:k+1] = c_n
#         else:
#         hids, h_n, c_n = self.lstm(inputs_sorted[last_id:k+1, 0:last_len], time_stamps_sorted[last_id:k+1, 0:last_len], init_h[last_id:k+1,:], init_c[last_id:k+1,:])
            
        output_list = h_outputs
        last_h_n[last_id:k+1] = h_outputs[last_id:k+1,last_len-1]
        last_c_n[last_id:k+1] = c_outputs[last_id:k+1,last_len-1]
        
        
        _, inv_indices = indices.sort()
        
        output_hiddens = output_list[inv_indices]
        
#         print(torch.norm(output_hiddens[0, 0:input_lens[0]] - rnn_out[0, 0:input_lens[0]]))
#         
#         print(torch.norm(output_hiddens[-1, 0:input_lens[-1]] - rnn_out[-1, 0:input_lens[-1]]))
#         
#         print(torch.norm(last_h_n[:,inv_indices] - exp_last_h_n))
#         
#         print(torch.norm(last_c_n[:,inv_indices] - exp_last_c_n))
        
        
        
        return output_hiddens, last_h_n[inv_indices], last_c_n[inv_indices]
        

class TimeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, device, bidirectional=False):
        # assumes that batch_first is always true
        super(TimeLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(input_size, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional

    def elapse_function(self, delta_t):
        
        return 1.0/(torch.log(np.e + delta_t))


    def forward2(self, inputs, timestamps, init_h=None, init_c=None, reverse=False):
        # inputs: [b, seq, embed]
        # h: [b, hid]
        # c: [b, hid]
        b, seq, embed = inputs.size()
        if init_h is None:
            h = torch.zeros(b, self.hidden_size, requires_grad=False, device = self.device)
        else:
            h = init_h
            
        if init_c is None:
            c = torch.zeros(b, self.hidden_size, requires_grad=False, device = self.device)
        else:
            c = init_c
#         if self.cuda_flag:
#             h = h.cuda()
#             c = c.cuda()
        outputs = []
        
        c_outputs = []
        
        for s in range(seq):
            
#             if torch.sum(torch.isnan(c)) > 0:
#             if s >= 82:
#                 print('here')
            
            c_s1 = F.tanh(self.W_d(c))
            c_s2 = c_s1 * self.elapse_function(timestamps[:, s]).expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = F.sigmoid(f)
            i = F.sigmoid(i)
            o = F.sigmoid(o)
            c_tmp = F.tanh(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * F.tanh(c)
            outputs.append(h)
            c_outputs.append(c)
            
#             if torch.sum(torch.isnan(c)) > 0:
# #             if s >= 82:
#                 print('here')

        if reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, 1)
        
        c_outputs = torch.stack(c_outputs, 1)
        
        return outputs, c_outputs

    def forward(self, inputs, timestamps, init_h=None, init_c=None, reverse=False):
        # inputs: [b, seq, embed]
        # h: [b, hid]
        # c: [b, hid]
        b, seq, embed = inputs.size()
        if init_h is None:
            h = torch.zeros(b, self.hidden_size, requires_grad=False, device = self.device)
        else:
            h = init_h
            
        if init_c is None:
            c = torch.zeros(b, self.hidden_size, requires_grad=False, device = self.device)
        else:
            c = init_c
#         if self.cuda_flag:
#             h = h.cuda()
#             c = c.cuda()
        outputs = []
        for s in range(seq):
            c_s1 = F.tanh(self.W_d(c))
            c_s2 = c_s1 * self.elapse_function(timestamps[:, s]).expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = F.sigmoid(f)
            i = F.sigmoid(i)
            o = F.sigmoid(o)
            c_tmp = F.tanh(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * F.tanh(c)
            outputs.append(h)
        if reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, 1)
        return outputs, h, c
    
    
    
    
    