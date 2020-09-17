

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
import lib.utils as utils
from torch.distributions import Categorical, Normal
import lib.utils as utils
from torch.nn.modules.rnn import LSTM, GRU
from lib.utils import get_device
import torch.nn.functional as F


class GRU_unit_cluster(nn.Module):
	def __init__(self, latent_dim, input_dim, 
		update_gate = None,
		reset_gate = None,
		new_state_net = None,
		n_units = 100,
		device = torch.device("cpu"), use_mask = False, dropout = 0.0):
		super(GRU_unit_cluster, self).__init__()

		if update_gate is None:
			if use_mask:
				self.update_gate = nn.Sequential(
# 				   nn.Linear(latent_dim + 2*input_dim, n_units),
# 				   nn.Tanh(),
# 				   nn.Linear(n_units, latent_dim),
                    nn.Linear(latent_dim + 2*input_dim, latent_dim),
                    nn.Dropout(p=dropout),
				   nn.Sigmoid())
			else:
				self.update_gate = nn.Sequential(
# 				   nn.Linear(latent_dim + input_dim, n_units),
# 				   nn.Tanh(),
# 				   nn.Linear(n_units, latent_dim),
                    nn.Linear(latent_dim + input_dim, latent_dim),
                    nn.Dropout(p = dropout),
				   nn.Sigmoid())
# 			utils.init_network_weights(self.update_gate)
		else: 
			self.update_gate  = update_gate

		if reset_gate is None:
			if use_mask:
				self.reset_gate = nn.Sequential(
# 				   nn.Linear(latent_dim + 2*input_dim, n_units),
# 				   nn.Tanh(),
# 				   nn.Linear(n_units, latent_dim),
                    nn.Linear(latent_dim + 2*input_dim, latent_dim),
                    nn.Dropout(p = dropout),
				   nn.Sigmoid())
			else:
				self.reset_gate = nn.Sequential(
# 				   nn.Linear(latent_dim + input_dim, n_units),
# 				   nn.Tanh(),
# 				   nn.Linear(n_units, latent_dim),
                    nn.Linear(latent_dim + input_dim, latent_dim),
                    nn.Dropout(p = dropout),
				   nn.Sigmoid())
# 			utils.init_network_weights(self.reset_gate)
		else: 
			self.reset_gate  = reset_gate

		if new_state_net is None:
			if use_mask:
				self.new_state_net = nn.Sequential(
# 				   nn.Linear(latent_dim + 2*input_dim, n_units),
# 				   nn.Tanh(),
# 				   nn.Linear(n_units, latent_dim)
                    nn.Linear(latent_dim + 2*input_dim, latent_dim),
                    nn.Dropout(p = dropout),
				   )
			else:
				self.new_state_net = nn.Sequential(
# 				   nn.Linear(latent_dim + input_dim, n_units),
# 				   nn.Tanh(),
# 				   nn.Linear(n_units, latent_dim)
                nn.Linear(latent_dim + input_dim, latent_dim),
                nn.Dropout(p = dropout),
				   )
# 			utils.init_network_weights(self.new_state_net)
		else: 
			self.new_state_net  = new_state_net


	def forward(self, y_i, x):
		y_concat = torch.cat([y_i, x], -1)

		update_gate = self.update_gate(y_concat)
		reset_gate = self.reset_gate(y_concat)
		
		
		concat = y_i * reset_gate
		
		concat = torch.cat([concat, x], -1)
		
		new_probs = self.new_state_net(concat)
		
# 		new_state, new_state_std = utils.split_last_dim(self.new_state_net(concat))
# 		new_state_std = new_state_std.abs()

		new_y_probs = (1-update_gate) * new_probs + update_gate * y_i
# 		new_y_std = (1-update_gate) * new_state_std + update_gate * y_std

		assert(not torch.isnan(new_y_probs).any())

# 		if masked_update:
# 			# IMPORTANT: assumes that x contains both data and mask
# 			# update only the hidden states for hidden state only if at least one feature is present for the current time point
# 			n_data_dims = x.size(-1)//2
# 			mask = x[:, :, n_data_dims:]
# 			utils.check_mask(x[:, :, :n_data_dims], mask)
# 			
# 			mask = (torch.sum(mask, -1, keepdim = True) > 0).float()
# 
# 			assert(not torch.isnan(mask).any())
# 
# 			new_y = mask * new_y + (1-mask) * y_mean
# 			new_y_std = mask * new_y_std + (1-mask) * y_std
# 
# 			if torch.isnan(new_y).any():
# 				print("new_y is nan!")
# 				print(mask)
# 				print(y_mean)
# 				print(new_y)
# 				exit()

# 		new_y_std = new_y_std.abs()
# 		return new_y, new_y_std
		return new_y_probs



class Encoder_z0_RNN(nn.Module):
	def __init__(self, latent_dim, input_dim, lstm_output_size = 20, 
		use_delta_t = True, device = torch.device("cpu")):
		
		super(Encoder_z0_RNN, self).__init__()
	
		self.gru_rnn_output_size = lstm_output_size
		self.latent_dim = latent_dim
		self.input_dim = input_dim
		self.device = device
		self.use_delta_t = use_delta_t

		self.hiddens_to_z0 = nn.Sequential(
		   nn.Linear(self.gru_rnn_output_size, 50),
		   nn.Tanh(),
		   nn.Linear(50, latent_dim * 2),)

# 		utils.init_network_weights(self.hiddens_to_z0)

		input_dim = self.input_dim

		if use_delta_t:
			self.input_dim += 1
		self.gru_rnn = GRU(self.input_dim, self.gru_rnn_output_size).to(device)

	def forward(self, data, time_steps, run_backwards = True):
		# IMPORTANT: assumes that 'data' already has mask concatenated to it 

		# data shape: [n_traj, n_tp, n_dims]
		# shape required for rnn: (seq_len, batch, input_size)
		# t0: not used here
		n_traj = data.size(0)

		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

		data = data.permute(1,0,2) 

		if run_backwards:
			# Look at data in the reverse order: from later points to the first
			data = utils.reverse(data)

		if self.use_delta_t:
			delta_t = time_steps[1:] - time_steps[:-1]
			if run_backwards:
				# we are going backwards in time with
				delta_t = utils.reverse(delta_t)
			# append zero delta t in the end
			delta_t = torch.cat((delta_t, torch.zeros(1).to(self.device)))
			delta_t = delta_t.unsqueeze(1).repeat((1,n_traj)).unsqueeze(-1)
			data = torch.cat((delta_t, data),-1)

		outputs, _ = self.gru_rnn(data)

		# LSTM output shape: (seq_len, batch, num_directions * hidden_size)
		last_output = outputs[-1]

		self.extra_info ={"rnn_outputs": outputs, "time_points": time_steps}

		mean, std = utils.split_last_dim(self.hiddens_to_z0(last_output))
		std = std.abs()

		assert(not torch.isnan(mean).any())
		assert(not torch.isnan(std).any())

		return mean.unsqueeze(0), std.unsqueeze(0)


class Decoder_ODE_RNN_cluster(nn.Module):
	# Derive z0 by running ode backwards.
	# For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
	# Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
	# Continue until we get to z0
	def __init__(self, latent_dim, input_dim, cluster_num, z0_diffeq_solver = None, 
		z0_dim = None, GRU_update = None, 
		n_gru_units = 100, 
		device = torch.device("cpu"), use_sparse = False, dropout=0.0):
		
		super(Decoder_ODE_RNN_cluster, self).__init__()

		if z0_dim is None:
			self.z0_dim = latent_dim
		else:
			self.z0_dim = z0_dim

		self.dropout = dropout

		if GRU_update is None:
			self.GRU_update = GRU_unit_cluster(latent_dim, input_dim, 
				n_units = n_gru_units, 
				device=device, dropout = dropout).to(device)
		else:
			self.GRU_update = GRU_update

		self.z0_diffeq_solver = z0_diffeq_solver
		self.latent_dim = latent_dim
		self.input_dim = input_dim
		self.device = device
		self.cluster_num = cluster_num
		self.use_sparse = use_sparse
		
		self.minimum_step = 0.0
# 		self.sparsemax = Sparsemax(dim=-1, device = self.device)
		
		self.extra_info = None

		self.infer_emitter_z = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
			nn.Linear(latent_dim, self.cluster_num),
			nn.Dropout(p = self.dropout)
		)

		self.infer_transfer_z = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
			nn.Linear(self.cluster_num, latent_dim),
			nn.Dropout(p = self.dropout)
		)
		
		self.decayed_layer = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
			nn.Linear(1, 1),
			nn.Dropout(p = self.dropout),
# 			nn.Sigmoid()
		)


# 		self.transform_z0 = nn.Sequential(
# 		   nn.Linear(latent_dim * 2, 100),
# 		   nn.Tanh(),
# 		   nn.Linear(100, self.z0_dim * 2),)
# 		utils.init_network_weights(self.transform_z0)


	def forward(self, data, time_steps, run_backwards = False, save_info = False, prev_y_states = None):
		# data, time_steps -- observations and their time stamps
		# IMPORTANT: assumes that 'data' already has mask concatenated to it 
		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

		n_traj, n_tp, n_dims = data.size()
		if len(time_steps) == 1:
			prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
# 			prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

			xi = data[:,0,:].unsqueeze(0)

			all_y_i = self.GRU_update(prev_y, xi)
			
			all_y_i = F.softmax(all_y_i.unsqueeze(0), -1)
			
			extra_info = None
		else:
			
			latent_ys, latent_y_states, extra_info = self.run_odernn(
				data, time_steps, run_backwards = run_backwards,
				save_info = save_info, prev_y_state = prev_y_states)

# 		means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
# 		std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)

# 		mean_z0, std_z0 = utils.split_last_dim( self.transform_z0( torch.cat((means_z0, std_z0), -1)))
# 		std_z0 = std_z0.abs()
		if save_info:
			self.extra_info = extra_info

		return latent_ys, latent_y_states


	def update_joint_probs(self, n_traj, joint_probs, t, latent_y_states, delta_t, full_curr_rnn_input = None):
		
# 		n_traj, n_tp, n_dims = data.size()
		if full_curr_rnn_input is None:
			full_curr_rnn_input = torch.zeros((self.cluster_num, n_traj, self.cluster_num), dtype = torch.float, device = self.device)
			
			for k in range(self.cluster_num):
				curr_rnn_input = torch.zeros((n_traj, self.cluster_num), dtype = torch.float, device = self.device)
				curr_rnn_input[:,k] = 1
				full_curr_rnn_input[k] = curr_rnn_input

		
		z_t_category_infer_full = self.emit_probs(latent_y_states[t], full_curr_rnn_input, delta_t, t)
		
		updated_joint_probs = torch.sum(z_t_category_infer_full*torch.t(joint_probs).view(joint_probs.shape[1], joint_probs.shape[0], 1), 0)
		
		joint_probs_sum = torch.sum(updated_joint_probs)
		
# 		print('time::', t)
# 		
# 		joint_probs_sum.backward(retain_graph = True)
		
		return updated_joint_probs
		

	def emit_probs(self, prev_y_state):
		
# 		delta_t = delta_t.to(self.device)
		
# 		if len(prev_y_prob.shape) > 2:
# # 			print(prev_y_state.shape)
# 			
# # 			prev_y_state = prev_y_state.view(1, prev_y_state.shape[0], prev_y_state.shape[1])
# 			prev_y_state = prev_y_state.repeat(prev_y_prob.shape[0], 1,1)
		
# 		if i > 0:
# 			decayed_weight = torch.exp(-(self.decayed_layer(delta_t.view(1,1))))
# 			
# 			decayed_weight = decayed_weight.view(-1)
# 		else:
# 			decayed_weight = 0.5
		
		prev_y_prob = F.softmax(self.infer_emitter_z(prev_y_state), -1)
		
# 		print(torch.sum(prev_y_prob, -1))
		
		return prev_y_prob
	
	def run_odernn_single_step(self, data, time_steps, full_curr_rnn_input = None,
		run_backwards = False, save_info = False, prev_y_state = None):
# 		n_traj, n_tp, n_dims = data.size()
		
		
		
		extra_info = []

		t0 = time_steps[-1]
		if run_backwards:
			t0 = time_steps[0]

# 		device = get_device(data)

# 		prev_y_prob = torch.zeros((1, n_traj, self.cluster_num)).to(self.device)
		
		if prev_y_state is None:
			prev_y_state = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
		
# 		joint_probs = torch.zeros([n_tp, n_traj, self.cluster_num], dtype = torch.float, device = self.device)
		
		
# 		prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)

		prev_t, t_i = time_steps[0],  time_steps[1]

# 		interval_length = time_steps[-1] - time_steps[0]
# 		minimum_step = (interval_length+1) / 500

		#print("minimum step: {}".format(minimum_step))

		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

# 		latent_ys = []
# 		
# 		latent_y_states = []
		
		# Run ODE backwards and combine the y(t) estimates using gating
		time_points_iter = range(0, len(time_steps))
		if run_backwards:
			time_points_iter = reversed(time_points_iter)
		
		yi_ode = self.GRU_update(prev_y_state, data)
		
# 		for i in time_points_iter:
# 		print(t_i, prev_t)
		if (t_i - prev_t) < self.minimum_step:
			time_points = torch.stack((prev_t, t_i))
			inc = self.z0_diffeq_solver.ode_func(prev_t, yi_ode) * (t_i - prev_t)

			assert(not torch.isnan(inc).any())

			ode_sol = yi_ode + inc
			ode_sol = torch.stack((yi_ode, ode_sol), 2).to(self.device)

			assert(not torch.isnan(ode_sol).any())
		else:
			n_intermediate_tp = max(2, ((t_i - prev_t) / self.minimum_step).int())

			time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
			ode_sol = self.z0_diffeq_solver(yi_ode, time_points)

			assert(not torch.isnan(ode_sol).any())

		if torch.mean(ode_sol[:, :, 0, :]  - yi_ode) >= 0.001:
			print("Error: first point of the ODE is not equal to initial value")
			print(torch.mean(ode_sol[:, :, 0, :]  - yi_ode))
			exit()
			#assert(torch.mean(ode_sol[:, :, 0, :]  - prev_y) < 0.001)


			

		prev_y_state = ode_sol[:, :, -1, :]
		xi = data[:,:].unsqueeze(0)
		
		
		
# 			if not self.use_sparse:
# 				y_ode_probs = F.softmax(self.infer_emitter_z(yi_ode), -1)
# 			else:
# 				y_ode_probs = self.sparsemax(self.infer_emitter_z(yi_ode))
		
		
		
# 			if self.use_sparse and i > 0:
# 				prev_y_prob = self.sparsemax(torch.log(prev_y_prob + 1e-5)) 
		
		
		prev_y_prob = self.emit_probs(prev_y_state)
		
		
# 			if i > 0:
# 				decayed_weight = torch.exp(-(self.decayed_layer(t_i - prev_t)))
# 			else:
# 				decayed_weight = 0.5
# 			
# 			
# 			prev_y_prob = self.infer_emitter_z((decayed_weight*self.infer_transfer_z(prev_y_prob) + (1-decayed_weight)*prev_y_state))
		
# 		latent_y_states.append(prev_y_state)
# 			prev_y_state = self.infer_emitter_z(yi)
		
# 			if not self.use_sparse:
# 				prev_y = F.softmax(self.infer_emitter_z(yi), -1)
# 			else:
# 				prev_y = self.sparsemax(self.infer_emitter_z(yi))
# 		prev_t, t_i = time_steps[i],  time_steps[i-1]

# 		latent_ys.append(prev_y_prob)

		if save_info:
			d = {"yi_ode": yi_ode.detach(), #"yi_from_data": yi_from_data,
#  					 "yi": yi.detach(), 
#  					 "yi_std": yi_std.detach(), 
				 "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
			extra_info.append(d)

# 		latent_ys = torch.stack(latent_ys, 1)
# 		
# 		latent_y_states = torch.stack(latent_y_states, 1)

# 		assert(not torch.isnan(yi).any())
# 		assert(not torch.isnan(yi_std).any())
		return prev_y_prob, prev_y_state
# 		return latent_ys, latent_y_states, extra_info
	
	
	def run_odernn(self, data, time_steps, full_curr_rnn_input = None,
		run_backwards = False, save_info = False, prev_y_state = None):
		# IMPORTANT: assumes that 'data' already has mask concatenated to it 


		n_traj, n_tp, n_dims = data.size()
		
		
		
		extra_info = []

		t0 = time_steps[-1]
		if run_backwards:
			t0 = time_steps[0]

# 		device = get_device(data)

# 		prev_y_prob = torch.zeros((1, n_traj, self.cluster_num)).to(self.device)
		
		if prev_y_state is None:
			prev_y_state = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
		
# 		joint_probs = torch.zeros([n_tp, n_traj, self.cluster_num], dtype = torch.float, device = self.device)
		
		
# 		prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)

		prev_t, t_i = time_steps[0],  time_steps[1]

		interval_length = time_steps[-1] - time_steps[0]
# 		minimum_step = (interval_length+1) / 10000
		minimum_step = (time_steps[-1] - time_steps[0])/(len(time_steps)*3)
		self.minimum_step = minimum_step
		#print("minimum step: {}".format(minimum_step))

		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

		latent_ys = []
		
		latent_y_states = []
		
		# Run ODE backwards and combine the y(t) estimates using gating
		time_points_iter = range(0, len(time_steps)-1)
		if run_backwards:
			time_points_iter = reversed(time_points_iter)

		for i in time_points_iter:
			
# 			print(t_i, prev_t)
			xi = data[:,i,:].unsqueeze(0)
			yi_ode = self.GRU_update(prev_y_state, xi)
			
			if (t_i - prev_t) < minimum_step:
				time_points = torch.stack((prev_t, t_i))
				inc = self.z0_diffeq_solver.ode_func(prev_t, yi_ode) * (t_i - prev_t)

				assert(not torch.isnan(inc).any())

				ode_sol = yi_ode + inc
				ode_sol = torch.stack((yi_ode, ode_sol), 2).to(self.device)

				assert(not torch.isnan(ode_sol).any())
			else:
				n_intermediate_tp = max(2, ((t_i - prev_t) / minimum_step).int())

				time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
				ode_sol = self.z0_diffeq_solver(yi_ode, time_points)

				assert(not torch.isnan(ode_sol).any())

			if torch.mean(ode_sol[:, :, 0, :]  - yi_ode) >= 0.001:
				print("Error: first point of the ODE is not equal to initial value")
				print(torch.mean(ode_sol[:, :, 0, :]  - yi_ode))
				exit()
			#assert(torch.mean(ode_sol[:, :, 0, :]  - prev_y) < 0.001)


			

			prev_y_state = ode_sol[:, :, -1, :]
			
			
			
			
# 			if not self.use_sparse:
# 				y_ode_probs = F.softmax(self.infer_emitter_z(yi_ode), -1)
# 			else:
# 				y_ode_probs = self.sparsemax(self.infer_emitter_z(yi_ode))
			
			
			
# 			if self.use_sparse and i > 0:
# 				prev_y_prob = self.sparsemax(torch.log(prev_y_prob + 1e-5)) 
			
			
			prev_y_prob = self.emit_probs(prev_y_state)
			latent_ys.append(prev_y_prob.clone())
			latent_y_states.append(prev_y_state.clone())
# 			if i > 0:
# 				decayed_weight = torch.exp(-(self.decayed_layer(t_i - prev_t)))
# 			else:
# 				decayed_weight = 0.5
# 			
# 			
# 			prev_y_prob = self.infer_emitter_z((decayed_weight*self.infer_transfer_z(prev_y_prob) + (1-decayed_weight)*prev_y_state))
			
			
# 			prev_y_state = self.infer_emitter_z(yi)
			
# 			if not self.use_sparse:
# 				prev_y = F.softmax(self.infer_emitter_z(yi), -1)
# 			else:
# 				prev_y = self.sparsemax(self.infer_emitter_z(yi))
			prev_t, t_i = time_steps[i+1],  time_steps[(i+2)%time_steps.shape[0]]

			

			if save_info:
				d = {"yi_ode": yi_ode.detach(), #"yi_from_data": yi_from_data,
#  					 "yi": yi.detach(), 
#  					 "yi_std": yi_std.detach(), 
					 "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
				extra_info.append(d)

		latent_ys = torch.stack(latent_ys, 1)
		
		latent_y_states = torch.stack(latent_y_states, 1)

# 		assert(not torch.isnan(yi).any())
# 		assert(not torch.isnan(yi_std).any())

		return latent_ys, latent_y_states, extra_info


class Decoder_ODE_RNN_cluster0(nn.Module):
	# Derive z0 by running ode backwards.
	# For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
	# Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
	# Continue until we get to z0
	def __init__(self, latent_dim, input_dim, cluster_num, z0_diffeq_solver = None, 
		z0_dim = None, GRU_update = None, 
		n_gru_units = 100, 
		device = torch.device("cpu"), use_sparse = False, dropout=0.0):
		
		super(Decoder_ODE_RNN_cluster0, self).__init__()

		if z0_dim is None:
			self.z0_dim = latent_dim
		else:
			self.z0_dim = z0_dim

		self.dropout = dropout

		if GRU_update is None:
			self.GRU_update = GRU_unit_cluster(latent_dim, input_dim, 
				n_units = n_gru_units, 
				device=device, dropout = dropout).to(device)
		else:
			self.GRU_update = GRU_update

		self.z0_diffeq_solver = z0_diffeq_solver
		self.latent_dim = latent_dim
		self.input_dim = input_dim
		self.device = device
		self.cluster_num = cluster_num
		self.use_sparse = use_sparse
		
		self.minimum_step = 0.0
# 		self.sparsemax = Sparsemax(dim=-1, device = self.device)
		
		self.extra_info = None

		self.infer_emitter_z = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
			nn.Linear(latent_dim, self.cluster_num),
			nn.Dropout(p = self.dropout)
		)

		self.infer_transfer_z = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
			nn.Linear(self.cluster_num, latent_dim),
			nn.Dropout(p = self.dropout)
		)
		
		self.decayed_layer = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
			nn.Linear(1, 1),
			nn.Dropout(p = self.dropout),
# 			nn.Sigmoid()
		)


# 		self.transform_z0 = nn.Sequential(
# 		   nn.Linear(latent_dim * 2, 100),
# 		   nn.Tanh(),
# 		   nn.Linear(100, self.z0_dim * 2),)
# 		utils.init_network_weights(self.transform_z0)


	def forward(self, data, time_steps, run_backwards = False, save_info = False, prev_y_states = None):
		# data, time_steps -- observations and their time stamps
		# IMPORTANT: assumes that 'data' already has mask concatenated to it 
		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

		n_traj, n_tp, n_dims = data.size()
		if len(time_steps) == 1:
			prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
# 			prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

			xi = data[:,0,:].unsqueeze(0)

			all_y_i = self.GRU_update(prev_y, xi)
			
			all_y_i = F.softmax(all_y_i.unsqueeze(0), -1)
			
			extra_info = None
		else:
			
			latent_ys, latent_y_states, extra_info = self.run_odernn(
				data, time_steps, run_backwards = run_backwards,
				save_info = save_info, prev_y_state = prev_y_states)

# 		means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
# 		std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)

# 		mean_z0, std_z0 = utils.split_last_dim( self.transform_z0( torch.cat((means_z0, std_z0), -1)))
# 		std_z0 = std_z0.abs()
		if save_info:
			self.extra_info = extra_info

		return latent_ys, latent_y_states


	def update_joint_probs(self, n_traj, joint_probs, t, latent_y_states, delta_t, full_curr_rnn_input = None):
		
# 		n_traj, n_tp, n_dims = data.size()
		if full_curr_rnn_input is None:
			full_curr_rnn_input = torch.zeros((self.cluster_num, n_traj, self.cluster_num), dtype = torch.float, device = self.device)
			
			for k in range(self.cluster_num):
				curr_rnn_input = torch.zeros((n_traj, self.cluster_num), dtype = torch.float, device = self.device)
				curr_rnn_input[:,k] = 1
				full_curr_rnn_input[k] = curr_rnn_input

		
		z_t_category_infer_full = self.emit_probs(latent_y_states[t], full_curr_rnn_input, delta_t, t)
		
		updated_joint_probs = torch.sum(z_t_category_infer_full*torch.t(joint_probs).view(joint_probs.shape[1], joint_probs.shape[0], 1), 0)
		
		joint_probs_sum = torch.sum(updated_joint_probs)
		
# 		print('time::', t)
# 		
# 		joint_probs_sum.backward(retain_graph = True)
		
		return updated_joint_probs
		

	def emit_probs(self, prev_y_state):
		
# 		delta_t = delta_t.to(self.device)
		
# 		if len(prev_y_prob.shape) > 2:
# # 			print(prev_y_state.shape)
# 			
# # 			prev_y_state = prev_y_state.view(1, prev_y_state.shape[0], prev_y_state.shape[1])
# 			prev_y_state = prev_y_state.repeat(prev_y_prob.shape[0], 1,1)
		
# 		if i > 0:
# 			decayed_weight = torch.exp(-(self.decayed_layer(delta_t.view(1,1))))
# 			
# 			decayed_weight = decayed_weight.view(-1)
# 		else:
# 			decayed_weight = 0.5
		
		prev_y_prob = F.softmax(self.infer_emitter_z(prev_y_state), -1)
		
# 		print(torch.sum(prev_y_prob, -1))
		
		return prev_y_prob
	
	def run_odernn_single_step(self, data, time_steps, full_curr_rnn_input = None,
		run_backwards = False, save_info = False, prev_y_state = None):
# 		n_traj, n_tp, n_dims = data.size()
		
		
		
		extra_info = []

		t0 = time_steps[-1]
		if run_backwards:
			t0 = time_steps[0]

# 		device = get_device(data)

# 		prev_y_prob = torch.zeros((1, n_traj, self.cluster_num)).to(self.device)
		
		if prev_y_state is None:
			prev_y_state = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
		
# 		joint_probs = torch.zeros([n_tp, n_traj, self.cluster_num], dtype = torch.float, device = self.device)
		
		
# 		prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)

		prev_t, t_i = time_steps[0],  time_steps[1]

# 		interval_length = time_steps[-1] - time_steps[0]
# 		minimum_step = (interval_length+1) / 500

		#print("minimum step: {}".format(minimum_step))

		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

# 		latent_ys = []
# 		
# 		latent_y_states = []
		
		# Run ODE backwards and combine the y(t) estimates using gating
		time_points_iter = range(0, len(time_steps))
		if run_backwards:
			time_points_iter = reversed(time_points_iter)
		
		yi_ode = self.GRU_update(prev_y_state, data)
		
# 		for i in time_points_iter:
# 		print(t_i, prev_t)
		if (t_i - prev_t) < self.minimum_step:
			time_points = torch.stack((prev_t, t_i))
			inc = self.z0_diffeq_solver.ode_func(prev_t, yi_ode) * (t_i - prev_t)

			assert(not torch.isnan(inc).any())

			ode_sol = yi_ode + inc
			ode_sol = torch.stack((yi_ode, ode_sol), 2).to(self.device)

			assert(not torch.isnan(ode_sol).any())
		else:
			n_intermediate_tp = max(2, ((t_i - prev_t) / self.minimum_step).int())

			time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
			ode_sol = self.z0_diffeq_solver(yi_ode, time_points)

			assert(not torch.isnan(ode_sol).any())

		if torch.mean(ode_sol[:, :, 0, :]  - yi_ode) >= 0.001:
			print("Error: first point of the ODE is not equal to initial value")
			print(torch.mean(ode_sol[:, :, 0, :]  - yi_ode))
			exit()
			#assert(torch.mean(ode_sol[:, :, 0, :]  - prev_y) < 0.001)


			

		prev_y_state = ode_sol[:, :, -1, :]
		xi = data[:,:].unsqueeze(0)
		
		
		
# 			if not self.use_sparse:
# 				y_ode_probs = F.softmax(self.infer_emitter_z(yi_ode), -1)
# 			else:
# 				y_ode_probs = self.sparsemax(self.infer_emitter_z(yi_ode))
		
		
		
# 			if self.use_sparse and i > 0:
# 				prev_y_prob = self.sparsemax(torch.log(prev_y_prob + 1e-5)) 
		
		
		prev_y_prob = self.emit_probs(prev_y_state)
		
		
# 			if i > 0:
# 				decayed_weight = torch.exp(-(self.decayed_layer(t_i - prev_t)))
# 			else:
# 				decayed_weight = 0.5
# 			
# 			
# 			prev_y_prob = self.infer_emitter_z((decayed_weight*self.infer_transfer_z(prev_y_prob) + (1-decayed_weight)*prev_y_state))
		
# 		latent_y_states.append(prev_y_state)
# 			prev_y_state = self.infer_emitter_z(yi)
		
# 			if not self.use_sparse:
# 				prev_y = F.softmax(self.infer_emitter_z(yi), -1)
# 			else:
# 				prev_y = self.sparsemax(self.infer_emitter_z(yi))
# 		prev_t, t_i = time_steps[i],  time_steps[i-1]

# 		latent_ys.append(prev_y_prob)

		if save_info:
			d = {"yi_ode": yi_ode.detach(), #"yi_from_data": yi_from_data,
#  					 "yi": yi.detach(), 
#  					 "yi_std": yi_std.detach(), 
				 "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
			extra_info.append(d)

# 		latent_ys = torch.stack(latent_ys, 1)
# 		
# 		latent_y_states = torch.stack(latent_y_states, 1)

# 		assert(not torch.isnan(yi).any())
# 		assert(not torch.isnan(yi_std).any())
		return prev_y_prob, prev_y_state
# 		return latent_ys, latent_y_states, extra_info
	
	
	def run_odernn(self, data, time_steps, full_curr_rnn_input = None,
		run_backwards = False, save_info = False, prev_y_state = None):
		# IMPORTANT: assumes that 'data' already has mask concatenated to it 


		n_traj, n_tp, n_dims = data.size()
		
		
		
		extra_info = []

		t0 = time_steps[-1]
		if run_backwards:
			t0 = time_steps[0]

# 		device = get_device(data)

# 		prev_y_prob = torch.zeros((1, n_traj, self.cluster_num)).to(self.device)
		
		if prev_y_state is None:
			prev_y_state = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
		
# 		joint_probs = torch.zeros([n_tp, n_traj, self.cluster_num], dtype = torch.float, device = self.device)
		
		
# 		prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)

		prev_t, t_i = time_steps[0]-0.01,  time_steps[0]

		interval_length = time_steps[-1] - time_steps[0]
# 		minimum_step = (interval_length+1) / 10000
		minimum_step = (time_steps[-1] - time_steps[0])/(len(time_steps)*3)
		self.minimum_step = minimum_step
		#print("minimum step: {}".format(minimum_step))

		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

		latent_ys = []
		
		latent_y_states = []
		
		# Run ODE backwards and combine the y(t) estimates using gating
		time_points_iter = range(0, len(time_steps)-1)
		if run_backwards:
			time_points_iter = reversed(time_points_iter)

		xi = data[:,0,:].unsqueeze(0)
		yi_ode = self.GRU_update(prev_y_state, xi)
		ode_sol = self.z0_diffeq_solver(yi_ode, time_steps)
# 		n_intermediate_tp = max(2, ((t_i - prev_t) / minimum_step).int())
# 
# 		time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
# 		ode_sol = self.z0_diffeq_solver(yi_ode, time_points)
		latent_y_states2  = torch.transpose(ode_sol, 1, 2)[:, 1:]
		latent_ys2 = self.emit_probs(latent_y_states2)
		
		

# 		for i in time_points_iter:
#  			
# # 			print(t_i, prev_t)
#  
#  			
# 			if (t_i - prev_t) < minimum_step:
# 				time_points = torch.stack((prev_t, t_i))
# 				inc = self.z0_diffeq_solver.ode_func(prev_t, yi_ode) * (t_i - prev_t)
#  
# 				assert(not torch.isnan(inc).any())
#  
# 				ode_sol = yi_ode + inc
# 				ode_sol = torch.stack((yi_ode, ode_sol), 2).to(self.device)
#  
# 				assert(not torch.isnan(ode_sol).any())
# 			else:
# 				n_intermediate_tp = max(2, ((t_i - prev_t) / minimum_step).int())
#  
# 				time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
# 				ode_sol = self.z0_diffeq_solver(yi_ode, time_points)
#  
# 				assert(not torch.isnan(ode_sol).any())
#  
# 			if torch.mean(ode_sol[:, :, 0, :]  - yi_ode) >= 0.001:
# 				print("Error: first point of the ODE is not equal to initial value")
# 				print(torch.mean(ode_sol[:, :, 0, :]  - yi_ode))
# 				exit()
# 			#assert(torch.mean(ode_sol[:, :, 0, :]  - prev_y) < 0.001)
#  
#  
#  			
#  
# 			prev_y_state = ode_sol[:, :, -1, :]
#  			
#  			
#  			
#  			
# # 			if not self.use_sparse:
# # 				y_ode_probs = F.softmax(self.infer_emitter_z(yi_ode), -1)
# # 			else:
# # 				y_ode_probs = self.sparsemax(self.infer_emitter_z(yi_ode))
#  			
#  			
#  			
# # 			if self.use_sparse and i > 0:
# # 				prev_y_prob = self.sparsemax(torch.log(prev_y_prob + 1e-5)) 
#  			
#  			
# 			prev_y_prob = self.emit_probs(prev_y_state)
# 			latent_ys.append(prev_y_prob.clone())
# 			latent_y_states.append(prev_y_state.clone())
# # 			if i > 0:
# # 				decayed_weight = torch.exp(-(self.decayed_layer(t_i - prev_t)))
# # 			else:
# # 				decayed_weight = 0.5
# # 			
# # 			
# # 			prev_y_prob = self.infer_emitter_z((decayed_weight*self.infer_transfer_z(prev_y_prob) + (1-decayed_weight)*prev_y_state))
#  			
#  			
# # 			prev_y_state = self.infer_emitter_z(yi)
#  			
# # 			if not self.use_sparse:
# # 				prev_y = F.softmax(self.infer_emitter_z(yi), -1)
# # 			else:
# # 				prev_y = self.sparsemax(self.infer_emitter_z(yi))
# 			prev_t, t_i = time_steps[i],  time_steps[(i+1)%time_steps.shape[0]]
#  
#  			
#  
# 			if save_info:
# 				d = {"yi_ode": yi_ode.detach(), #"yi_from_data": yi_from_data,
# #  					 "yi": yi.detach(), 
# #  					 "yi_std": yi_std.detach(), 
# 					 "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
# 				extra_info.append(d)
# 
# 		latent_ys = torch.stack(latent_ys, 1)
#  		
# 		latent_y_states = torch.stack(latent_y_states, 1)

# 		assert(not torch.isnan(yi).any())
# 		assert(not torch.isnan(yi_std).any())

		return latent_ys2, latent_y_states2, extra_info


class Decoder_ODE_RNN_cluster2(nn.Module):
	# Derive z0 by running ode backwards.
	# For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
	# Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
	# Continue until we get to z0
	def __init__(self, latent_dim, input_dim, cluster_num, z0_diffeq_solver = None, 
		z0_dim = None, GRU_update = None, 
		n_gru_units = 100, 
		device = torch.device("cpu"), use_sparse = False, dropout=0.0):
		
		super(Decoder_ODE_RNN_cluster2, self).__init__()

		if z0_dim is None:
			self.z0_dim = latent_dim
		else:
			self.z0_dim = z0_dim

		self.dropout = dropout

		if GRU_update is None:
			self.GRU_update = GRU_unit_cluster(latent_dim, input_dim, 
				n_units = n_gru_units, 
				device=device, dropout = dropout).to(device)
		else:
			self.GRU_update = GRU_update

		self.z0_diffeq_solver = z0_diffeq_solver
		self.latent_dim = latent_dim
		self.input_dim = input_dim
		self.device = device
		self.cluster_num = cluster_num
		self.use_sparse = use_sparse
		
		self.minimum_step = 0.0
# 		self.sparsemax = Sparsemax(dim=-1, device = self.device)
		
		self.extra_info = None

		self.infer_emitter_z = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
			nn.Linear(latent_dim, self.cluster_num),
			nn.Dropout(p = self.dropout)
		)

		self.infer_transfer_z = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
			nn.Linear(self.cluster_num, latent_dim),
			nn.Dropout(p = self.dropout)
		)
		
		self.decayed_layer = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
			nn.Linear(1, 1),
			nn.Dropout(p = self.dropout),
# 			nn.Sigmoid()
		)
		
# 		self.


# 		self.transform_z0 = nn.Sequential(
# 		   nn.Linear(latent_dim * 2, 100),
# 		   nn.Tanh(),
# 		   nn.Linear(100, self.z0_dim * 2),)
# 		utils.init_network_weights(self.transform_z0)


	def forward(self, data, time_steps, run_backwards = False, save_info = False, prev_y_states = None):
		# data, time_steps -- observations and their time stamps
		# IMPORTANT: assumes that 'data' already has mask concatenated to it 
		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

		n_traj, n_tp, n_dims = data.size()
		if len(time_steps) == 1:
			prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
# 			prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

			xi = data[:,0,:].unsqueeze(0)

			all_y_i = self.GRU_update(prev_y, xi)
			
			all_y_i = F.softmax(all_y_i.unsqueeze(0), -1)
			
			extra_info = None
		else:
			
			latent_ys, latent_y_states, extra_info = self.run_odernn(
				data, time_steps, run_backwards = run_backwards,
				save_info = save_info, prev_y_state = prev_y_states)

# 		means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
# 		std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)

# 		mean_z0, std_z0 = utils.split_last_dim( self.transform_z0( torch.cat((means_z0, std_z0), -1)))
# 		std_z0 = std_z0.abs()
		if save_info:
			self.extra_info = extra_info

		return latent_ys, latent_y_states


	def update_joint_probs(self, n_traj, joint_probs, t, latent_y_states, delta_t, full_curr_rnn_input = None):
		
# 		n_traj, n_tp, n_dims = data.size()
		if full_curr_rnn_input is None:
			full_curr_rnn_input = torch.zeros((self.cluster_num, n_traj, self.cluster_num), dtype = torch.float, device = self.device)
			
			for k in range(self.cluster_num):
				curr_rnn_input = torch.zeros((n_traj, self.cluster_num), dtype = torch.float, device = self.device)
				curr_rnn_input[:,k] = 1
				full_curr_rnn_input[k] = curr_rnn_input

		
		z_t_category_infer_full = self.emit_probs(latent_y_states[t], full_curr_rnn_input, delta_t, t)
		
		updated_joint_probs = torch.sum(z_t_category_infer_full*torch.t(joint_probs).view(joint_probs.shape[1], joint_probs.shape[0], 1), 0)
		
		joint_probs_sum = torch.sum(updated_joint_probs)
		
# 		print('time::', t)
# 		
# 		joint_probs_sum.backward(retain_graph = True)
		
		return updated_joint_probs
		

	def emit_probs(self, prev_y_state):
		
# 		delta_t = delta_t.to(self.device)
		
# 		if len(prev_y_prob.shape) > 2:
# # 			print(prev_y_state.shape)
# 			
# # 			prev_y_state = prev_y_state.view(1, prev_y_state.shape[0], prev_y_state.shape[1])
# 			prev_y_state = prev_y_state.repeat(prev_y_prob.shape[0], 1,1)
		
# 		if i > 0:
# 			decayed_weight = torch.exp(-(self.decayed_layer(delta_t.view(1,1))))
# 			
# 			decayed_weight = decayed_weight.view(-1)
# 		else:
# 			decayed_weight = 0.5
		
		prev_y_prob = F.softmax(self.infer_emitter_z(prev_y_state), -1)
		
# 		print(torch.sum(prev_y_prob, -1))
		
		return prev_y_prob
	
	def run_odernn_single_step(self, data, time_steps, full_curr_rnn_input = None,
		run_backwards = False, save_info = False, prev_y_state = None):
# 		n_traj, n_tp, n_dims = data.size()
		
		
		
		extra_info = []

		t0 = time_steps[-1]
		if run_backwards:
			t0 = time_steps[0]

# 		device = get_device(data)

# 		prev_y_prob = torch.zeros((1, n_traj, self.cluster_num)).to(self.device)
		
		if prev_y_state is None:
			prev_y_state = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
		
# 		joint_probs = torch.zeros([n_tp, n_traj, self.cluster_num], dtype = torch.float, device = self.device)
		
		
# 		prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)

		prev_t, t_i = time_steps[0],  time_steps[1]

# 		interval_length = time_steps[-1] - time_steps[0]
# 		minimum_step = (interval_length+1) / 500

		#print("minimum step: {}".format(minimum_step))

		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

		curr_x = self.infer_transfer_z(data)

# 		latent_ys = []
# 		
# 		latent_y_states = []
		
		# Run ODE backwards and combine the y(t) estimates using gating
		time_points_iter = range(0, len(time_steps))
		if run_backwards:
			time_points_iter = reversed(time_points_iter)
		
# 		yi_ode = self.GRU_update(prev_y_state, data)
		
# 		for i in time_points_iter:
# 		print(t_i, prev_t)
		if (t_i - prev_t) < self.minimum_step:
			time_points = torch.stack((prev_t, t_i))
			inc = self.z0_diffeq_solver.ode_func(prev_t, curr_x) * (t_i - prev_t)

			assert(not torch.isnan(inc).any())

			ode_sol = curr_x + inc
			ode_sol = torch.stack((curr_x, ode_sol), 2).to(self.device)

			assert(not torch.isnan(ode_sol).any())
		else:
			n_intermediate_tp = max(2, ((t_i - prev_t) / self.minimum_step).int())

			time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
			ode_sol = self.z0_diffeq_solver(curr_x, time_points)

			assert(not torch.isnan(ode_sol).any())

		if torch.mean(ode_sol[:, :, 0, :]  - curr_x) >= 0.001:
			print("Error: first point of the ODE is not equal to initial value")
			print(torch.mean(ode_sol[:, :, 0, :]  - curr_x))
			exit()
			#assert(torch.mean(ode_sol[:, :, 0, :]  - prev_y) < 0.001)


			

		prev_y_state = ode_sol[:, :, -1, :]
		xi = data[:,:].unsqueeze(0)
		
		
		
# 			if not self.use_sparse:
# 				y_ode_probs = F.softmax(self.infer_emitter_z(yi_ode), -1)
# 			else:
# 				y_ode_probs = self.sparsemax(self.infer_emitter_z(yi_ode))
		
		
		
# 			if self.use_sparse and i > 0:
# 				prev_y_prob = self.sparsemax(torch.log(prev_y_prob + 1e-5)) 
		
		
		prev_y_prob = F.softmax(self.infer_emitter_z(prev_y_state), -1)
		
		
# 			if i > 0:
# 				decayed_weight = torch.exp(-(self.decayed_layer(t_i - prev_t)))
# 			else:
# 				decayed_weight = 0.5
# 			
# 			
# 			prev_y_prob = self.infer_emitter_z((decayed_weight*self.infer_transfer_z(prev_y_prob) + (1-decayed_weight)*prev_y_state))
		
# 		latent_y_states.append(prev_y_state)
# 			prev_y_state = self.infer_emitter_z(yi)
		
# 			if not self.use_sparse:
# 				prev_y = F.softmax(self.infer_emitter_z(yi), -1)
# 			else:
# 				prev_y = self.sparsemax(self.infer_emitter_z(yi))
# 		prev_t, t_i = time_steps[i],  time_steps[i-1]

# 		latent_ys.append(prev_y_prob)

		if save_info:
			d = {"yi_ode": data.detach(), #"yi_from_data": yi_from_data,
#  					 "yi": yi.detach(), 
#  					 "yi_std": yi_std.detach(), 
				 "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
			extra_info.append(d)

# 		latent_ys = torch.stack(latent_ys, 1)
# 		
# 		latent_y_states = torch.stack(latent_y_states, 1)

# 		assert(not torch.isnan(yi).any())
# 		assert(not torch.isnan(yi_std).any())
		return prev_y_prob, prev_y_state
# 		return latent_ys, latent_y_states, extra_info
	
	
	def run_odernn(self, data, time_steps, full_curr_rnn_input = None,
		run_backwards = False, save_info = False, prev_y_state = None):
		# IMPORTANT: assumes that 'data' already has mask concatenated to it 


		n_traj, n_tp, n_dims = data.size()
		
		
		
		extra_info = []

		t0 = time_steps[-1]
		if run_backwards:
			t0 = time_steps[0]

# 		device = get_device(data)

# 		prev_y_prob = torch.zeros((1, n_traj, self.cluster_num)).to(self.device)
		
		if prev_y_state is None:
			prev_y_state = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
		
# 		joint_probs = torch.zeros([n_tp, n_traj, self.cluster_num], dtype = torch.float, device = self.device)
		
		
# 		prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)

		prev_t, t_i = time_steps[0]-0.01,  time_steps[0]

		interval_length = time_steps[-1] - time_steps[0]
# 		minimum_step = (interval_length+1) / 10000
		minimum_step = (time_steps[-1] - time_steps[0])/(len(time_steps)*3)
		self.minimum_step = minimum_step
		#print("minimum step: {}".format(minimum_step))

		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

		latent_ys = []
		
		latent_y_states = []
		
		# Run ODE backwards and combine the y(t) estimates using gating
		time_points_iter = range(0, len(time_steps)-1)
		if run_backwards:
			time_points_iter = reversed(time_points_iter)

		for i in time_points_iter:
			
# 			print(t_i, prev_t)
			xi = self.infer_transfer_z(data[:,i,:].unsqueeze(0))
# 			yi_ode = self.GRU_update(prev_y_state, xi)
			
			if (t_i - prev_t) < minimum_step:
				time_points = torch.stack((prev_t, t_i))
				inc = self.z0_diffeq_solver.ode_func(prev_t, xi) * (t_i - prev_t)

				assert(not torch.isnan(inc).any())

				ode_sol = xi + inc
				ode_sol = torch.stack((xi, ode_sol), 2).to(self.device)

				assert(not torch.isnan(ode_sol).any())
			else:
				n_intermediate_tp = max(2, ((t_i - prev_t) / minimum_step).int())

				time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
				ode_sol = self.z0_diffeq_solver(xi, time_points)

				assert(not torch.isnan(ode_sol).any())

			if torch.mean(ode_sol[:, :, 0, :]  - xi) >= 0.001:
				print("Error: first point of the ODE is not equal to initial value")
				print(torch.mean(ode_sol[:, :, 0, :]  - xi))
				exit()
			#assert(torch.mean(ode_sol[:, :, 0, :]  - prev_y) < 0.001)


			

			prev_y_state = ode_sol[:, :, -1, :]
			
			
			
			
# 			if not self.use_sparse:
# 				y_ode_probs = F.softmax(self.infer_emitter_z(yi_ode), -1)
# 			else:
# 				y_ode_probs = self.sparsemax(self.infer_emitter_z(yi_ode))
			
			
			
# 			if self.use_sparse and i > 0:
# 				prev_y_prob = self.sparsemax(torch.log(prev_y_prob + 1e-5)) 
			
			
			prev_y_prob = F.softmax(self.infer_emitter_z(prev_y_state), -1)
			latent_ys.append(prev_y_prob.clone())
			latent_y_states.append(prev_y_state.clone())
# 			if i > 0:
# 				decayed_weight = torch.exp(-(self.decayed_layer(t_i - prev_t)))
# 			else:
# 				decayed_weight = 0.5
# 			
# 			
# 			prev_y_prob = self.infer_emitter_z((decayed_weight*self.infer_transfer_z(prev_y_prob) + (1-decayed_weight)*prev_y_state))
			
			
# 			prev_y_state = self.infer_emitter_z(yi)
			
# 			if not self.use_sparse:
# 				prev_y = F.softmax(self.infer_emitter_z(yi), -1)
# 			else:
# 				prev_y = self.sparsemax(self.infer_emitter_z(yi))
			prev_t, t_i = time_steps[i],  time_steps[(i+1)%time_steps.shape[0]]

			

			if save_info:
				d = {"yi_ode": xi.detach(), #"yi_from_data": yi_from_data,
#  					 "yi": yi.detach(), 
#  					 "yi_std": yi_std.detach(), 
					 "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
				extra_info.append(d)

		latent_ys = torch.stack(latent_ys, 1)
		
		latent_y_states = torch.stack(latent_y_states, 1)

# 		assert(not torch.isnan(yi).any())
# 		assert(not torch.isnan(yi_std).any())

		return latent_ys, latent_y_states, extra_info


class Encoder_z0_ODE_RNN_cluster2(nn.Module):
	# Derive z0 by running ode backwards.
	# For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
	# Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
	# Continue until we get to z0
	def __init__(self, latent_dim, input_dim, cluster_num, z0_diffeq_solver = None, 
		z0_dim = None, GRU_update = None, 
		n_gru_units = 100, 
		device = torch.device("cpu"), use_sparse = False, dropout=0.0, use_mask = False):
		
		super(Encoder_z0_ODE_RNN_cluster2, self).__init__()

		if z0_dim is None:
			self.z0_dim = latent_dim
		else:
			self.z0_dim = z0_dim

		self.dropout = dropout

		if GRU_update is None:
			self.GRU_update = GRU_unit_cluster(latent_dim, input_dim, 
				n_units = n_gru_units, 
				device=device, use_mask = use_mask, dropout = dropout).to(device)
		else:
			self.GRU_update = GRU_update

		self.z0_diffeq_solver = z0_diffeq_solver
		self.latent_dim = latent_dim
		self.input_dim = input_dim
		self.device = device
		self.cluster_num = cluster_num
		self.use_sparse = use_sparse
		self.use_mask = use_mask
		
# 		self.sparsemax = Sparsemax(dim=-1, device = self.device)
		
		self.min_steps = 0.0
		
		self.extra_info = None
				
# 		if self.concat_data:
# 			self.infer_emitter_z = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
# 						nn.Linear(latent_dim + cluster_num, self.cluster_num),
# 						nn.Dropout(p = self.dropout)
# 			)
# 		else:
		self.infer_emitter_z = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
					nn.Linear(latent_dim, self.cluster_num),
					nn.Dropout(p = self.dropout)
		)

		self.infer_transfer_z = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
			nn.Linear(self.cluster_num, latent_dim),
			nn.Dropout(p = self.dropout)
		)
		
		self.decayed_layer = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
			nn.Linear(1, 1),
			nn.Dropout(p = self.dropout),
# 			nn.Sigmoid()
		)


# 		self.transform_z0 = nn.Sequential(
# 		   nn.Linear(latent_dim * 2, 100),
# 		   nn.Tanh(),
# 		   nn.Linear(100, self.z0_dim * 2),)
# 		utils.init_network_weights(self.transform_z0)


	def forward(self, data, time_steps, run_backwards = False, save_info = False):
		# data, time_steps -- observations and their time stamps
		# IMPORTANT: assumes that 'data' already has mask concatenated to it 
		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

		n_traj, n_tp, n_dims = data.size()
		if len(time_steps) == 1:
			prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
# 			prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

			xi = data[:,0,:].unsqueeze(0)

			all_y_i = self.GRU_update(prev_y, xi)
			
			all_y_i = F.softmax(all_y_i.unsqueeze(0), -1)
			
			extra_info = None
		else:
			
			latent_ys, latent_y_states, extra_info = self.run_odernn(
				data, time_steps, run_backwards = run_backwards,
				save_info = save_info)

# 		means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
# 		std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)

# 		mean_z0, std_z0 = utils.split_last_dim( self.transform_z0( torch.cat((means_z0, std_z0), -1)))
# 		std_z0 = std_z0.abs()
		if save_info:
			self.extra_info = extra_info

		return latent_ys, latent_y_states


	def update_joint_probs(self, n_traj, joint_probs, t, latent_y_states, delta_t, full_curr_rnn_input = None):
		
# 		n_traj, n_tp, n_dims = data.size()
		if full_curr_rnn_input is None:
			full_curr_rnn_input = torch.zeros((self.cluster_num, n_traj, self.cluster_num), dtype = torch.float, device = self.device)
			
			for k in range(self.cluster_num):
				curr_rnn_input = torch.zeros((n_traj, self.cluster_num), dtype = torch.float, device = self.device)
				curr_rnn_input[:,k] = 1
				full_curr_rnn_input[k] = curr_rnn_input

		
		z_t_category_infer_full = self.emit_probs(latent_y_states[t], full_curr_rnn_input, delta_t, t)
		
		updated_joint_probs = torch.sum(z_t_category_infer_full*torch.t(joint_probs).view(joint_probs.shape[1], joint_probs.shape[0], 1), 0)
		
		joint_probs_sum = torch.sum(updated_joint_probs)
		
# 		print('time::', t)
# 		
# 		joint_probs_sum.backward(retain_graph = True)
		
		return updated_joint_probs
		

	def emit_probs(self, prev_y_state):
		
		prev_y_prob = F.softmax(self.infer_emitter_z(prev_y_state), -1)
		
# 		print(torch.sum(prev_y_prob, -1))
		
		return prev_y_prob
		
		
# 		delta_t = delta_t.to(self.device)
# 		
# 		if len(prev_y_prob.shape) > 2:
# # 			print(prev_y_state.shape)
# 			
# # 			prev_y_state = prev_y_state.view(1, prev_y_state.shape[0], prev_y_state.shape[1])
# 			prev_y_state = prev_y_state.repeat(prev_y_prob.shape[0], 1,1)
# 		
# 		if i > 0:
# 			decayed_weight = torch.exp(-torch.abs(self.decayed_layer(delta_t.view(1,1))))
# 			
# 			decayed_weight = decayed_weight.view(-1)
# 		else:
# 			decayed_weight = 0.5
# 		
# # 		prev_y_prob = F.softmax(self.infer_emitter_z((decayed_weight*self.infer_transfer_z(prev_y_prob) + (1-decayed_weight)*prev_y_state)), -1)
# 		
# 		
# 		
# # 		prev_y_prob = F.softmax(self.infer_emitter_z((decayed_weight*self.infer_transfer_z(prev_y_prob) + (1-decayed_weight)*prev_y_state)), -1)
# 		
# 		if self.concat_data:
# 			input_z_w = torch.cat([prev_y_prob, prev_y_state], -1)
# 			prev_y_prob = F.softmax(self.infer_emitter_z(input_z_w), -1)
# 		else:
# 			prev_y_prob = F.softmax(self.infer_emitter_z((decayed_weight*self.infer_transfer_z(prev_y_prob) + (1-decayed_weight)*prev_y_state)), -1)
# 		
# 		
# 		
# # 			prev_y_prob = F.softmax(self.infer_emitter_z((decayed_weight*self.infer_transfer_z(prev_y_prob) + (1-decayed_weight)*prev_y_state)), -1)
# 			
# 		
# # 		print(torch.sum(prev_y_prob, -1))
# 		
# 		return prev_y_prob
		
	def run_odernn(self, data, time_steps, full_curr_rnn_input = None,
		run_backwards = False, save_info = False):
		# IMPORTANT: assumes that 'data' already has mask concatenated to it 


		n_traj, n_tp, n_dims = data.size()
		
		
		
		extra_info = []

		t0 = time_steps[-1]
		if run_backwards:
			t0 = time_steps[0]

# 		device = get_device(data)

		prev_y_prob = torch.zeros((1, n_traj, self.cluster_num)).to(self.device)
		
		prev_y_state = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
		
		joint_probs = torch.zeros([n_tp, n_traj, self.cluster_num], dtype = torch.float, device = self.device)
		
		
# 		prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)
	
		if not run_backwards:
			prev_t, t_i = time_steps[0] - 0.01,  time_steps[0]
		else:
			prev_t, t_i = time_steps[-1],  time_steps[-1] + 0.01

		interval_length = time_steps[-1] - time_steps[0]
		minimum_step = (time_steps[-1] - time_steps[0])/(len(time_steps)*3)

		self.min_steps = minimum_step

		#print("minimum step: {}".format(minimum_step))

		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

		latent_ys = []
		
		latent_y_states = []
		
		# Run ODE backwards and combine the y(t) estimates using gating
		time_points_iter = range(0, len(time_steps))
		if run_backwards:
			time_points_iter = reversed(time_points_iter)

		for i in time_points_iter:
			
# 			print(i, t_i, prev_t)
			if (t_i - prev_t) < minimum_step:
				time_points = torch.stack((prev_t, t_i))
				inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y_state) * (t_i - prev_t)

				assert(not torch.isnan(inc).any())

				ode_sol = prev_y_state + inc
				ode_sol = torch.stack((prev_y_state, ode_sol), 2).to(self.device)

				assert(not torch.isnan(ode_sol).any())
			else:
				n_intermediate_tp = max(2, ((t_i-prev_t) / minimum_step).int())

				time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
				ode_sol = self.z0_diffeq_solver(prev_y_state, time_points)

				assert(not torch.isnan(ode_sol).any())

			if torch.mean(ode_sol[:, :, 0, :]  - prev_y_state) >= 0.001:
				print("Error: first point of the ODE is not equal to initial value")
				print(torch.mean(ode_sol[:, :, 0, :]  - prev_y_state))
				exit()
			#assert(torch.mean(ode_sol[:, :, 0, :]  - prev_y) < 0.001)


			

			yi_ode = ode_sol[:, :, -1, :]
			xi = data[:,i,:].unsqueeze(0)
# 			mask_i = mask[:,i,:].unsqueeze(0)
# 			
# 			if self.use_mask:
# 				xi = torch.cat([xi, mask_i], -1)
			
# 			if not self.use_sparse:
# 				y_ode_probs = F.softmax(self.infer_emitter_z(yi_ode), -1)
# 			else:
# 				y_ode_probs = self.sparsemax(self.infer_emitter_z(yi_ode))
			
			prev_y_state = self.GRU_update(yi_ode, xi)
			
# 			if self.use_sparse:
# 				prev_y_prob = self.sparsemax(torch.log(prev_y_prob + 1e-5)) 
			
			
			prev_y_prob = self.emit_probs(prev_y_state)
			
			
# 			if i > 0:
# 				decayed_weight = torch.exp(-(self.decayed_layer(t_i - prev_t)))
# 			else:
# 				decayed_weight = 0.5
# 			
# 			
# 			prev_y_prob = self.infer_emitter_z((decayed_weight*self.infer_transfer_z(prev_y_prob) + (1-decayed_weight)*prev_y_state))
			
			latent_y_states.append(prev_y_state.clone())
# 			prev_y_state = self.infer_emitter_z(yi)
			
# 			if not self.use_sparse:
# 				prev_y = F.softmax(self.infer_emitter_z(yi), -1)
# 			else:
# 				prev_y = self.sparsemax(self.infer_emitter_z(yi))
			if not run_backwards:
				prev_t, t_i = time_steps[i],  time_steps[(i+1)%time_steps.shape[0]]
			else:
				prev_t, t_i = time_steps[(i-1)],  time_steps[i]

			latent_ys.append(prev_y_prob.clone())

			if save_info:
				d = {"yi_ode": yi_ode.detach(), #"yi_from_data": yi_from_data,
#  					 "yi": yi.detach(), 
#  					 "yi_std": yi_std.detach(), 
					 "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
				extra_info.append(d)

		latent_ys = torch.stack(latent_ys, 1)
		
		latent_y_states = torch.stack(latent_y_states, 1)

# 		assert(not torch.isnan(yi).any())
# 		assert(not torch.isnan(yi_std).any())

		return latent_ys, latent_y_states, extra_info


class Encoder_z0_ODE_RNN_cluster(nn.Module):
	# Derive z0 by running ode backwards.
	# For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
	# Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
	# Continue until we get to z0
	def __init__(self, latent_dim, input_dim, cluster_num, z0_diffeq_solver = None, 
		z0_dim = None, GRU_update = None, 
		n_gru_units = 100, 
		device = torch.device("cpu"), use_sparse = False, dropout=0.0, use_mask = False):
		
		super(Encoder_z0_ODE_RNN_cluster, self).__init__()

		if z0_dim is None:
			self.z0_dim = latent_dim
		else:
			self.z0_dim = z0_dim

		self.dropout = dropout

		if GRU_update is None:
			self.GRU_update = GRU_unit_cluster(latent_dim, input_dim, 
				n_units = n_gru_units, 
				device=device, use_mask = use_mask, dropout = dropout).to(device)
		else:
			self.GRU_update = GRU_update

		self.z0_diffeq_solver = z0_diffeq_solver
		self.latent_dim = latent_dim
		self.input_dim = input_dim
		self.device = device
		self.cluster_num = cluster_num
		self.use_sparse = use_sparse
		self.use_mask = use_mask
		
# 		self.sparsemax = Sparsemax(dim=-1, device = self.device)
		
		self.min_steps = 0.0
		
		self.extra_info = None
		
		self.concat_data = True
		
		if self.concat_data:
			self.infer_emitter_z = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
						nn.Linear(latent_dim + cluster_num, self.cluster_num),
						nn.Dropout(p = self.dropout)
			)
		else:
			self.infer_emitter_z = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
						nn.Linear(latent_dim, self.cluster_num),
						nn.Dropout(p = self.dropout)
			)

		self.infer_transfer_z = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
			nn.Linear(self.cluster_num, latent_dim),
			nn.Dropout(p = self.dropout)
		)
		
		self.decayed_layer = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
			nn.Linear(1, 1),
			nn.Dropout(p = self.dropout),
# 			nn.Sigmoid()
		)


# 		self.transform_z0 = nn.Sequential(
# 		   nn.Linear(latent_dim * 2, 100),
# 		   nn.Tanh(),
# 		   nn.Linear(100, self.z0_dim * 2),)
# 		utils.init_network_weights(self.transform_z0)


	def forward(self, data, time_steps, run_backwards = False, save_info = False):
		# data, time_steps -- observations and their time stamps
		# IMPORTANT: assumes that 'data' already has mask concatenated to it 
		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

		n_traj, n_tp, n_dims = data.size()
		if len(time_steps) == 1:
			prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
# 			prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

			xi = data[:,0,:].unsqueeze(0)

			all_y_i = self.GRU_update(prev_y, xi)
			
			all_y_i = F.softmax(all_y_i.unsqueeze(0), -1)
			
			extra_info = None
		else:
			
			latent_ys, latent_y_states, extra_info = self.run_odernn(
				data, time_steps, run_backwards = run_backwards,
				save_info = save_info)

# 		means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
# 		std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)

# 		mean_z0, std_z0 = utils.split_last_dim( self.transform_z0( torch.cat((means_z0, std_z0), -1)))
# 		std_z0 = std_z0.abs()
		if save_info:
			self.extra_info = extra_info

		return latent_ys, latent_y_states


	def update_joint_probs(self, n_traj, joint_probs, t, latent_y_states, delta_t, full_curr_rnn_input = None):
		
# 		n_traj, n_tp, n_dims = data.size()
		if full_curr_rnn_input is None:
			full_curr_rnn_input = torch.zeros((self.cluster_num, n_traj, self.cluster_num), dtype = torch.float, device = self.device)
			
			for k in range(self.cluster_num):
				curr_rnn_input = torch.zeros((n_traj, self.cluster_num), dtype = torch.float, device = self.device)
				curr_rnn_input[:,k] = 1
				full_curr_rnn_input[k] = curr_rnn_input

		
		z_t_category_infer_full = self.emit_probs(latent_y_states[t], full_curr_rnn_input, delta_t, t)
		
		updated_joint_probs = torch.sum(z_t_category_infer_full*torch.t(joint_probs).view(joint_probs.shape[1], joint_probs.shape[0], 1), 0)
		
		joint_probs_sum = torch.sum(updated_joint_probs)
		
# 		print('time::', t)
# 		
# 		joint_probs_sum.backward(retain_graph = True)
		
		return updated_joint_probs
		

	def emit_probs(self, prev_y_state, prev_y_prob, delta_t, i):
		
		delta_t = delta_t.to(self.device)
		
		if len(prev_y_prob.shape) > 2:
# 			print(prev_y_state.shape)
			
# 			prev_y_state = prev_y_state.view(1, prev_y_state.shape[0], prev_y_state.shape[1])
			prev_y_state = prev_y_state.repeat(prev_y_prob.shape[0], 1,1)
		
		if i > 0:
			decayed_weight = torch.exp(-torch.abs(self.decayed_layer(delta_t.view(1,1))))
			
			decayed_weight = decayed_weight.view(-1)
		else:
			decayed_weight = 0.5
		
# 		prev_y_prob = F.softmax(self.infer_emitter_z((decayed_weight*self.infer_transfer_z(prev_y_prob) + (1-decayed_weight)*prev_y_state)), -1)
		
		
		
# 		prev_y_prob = F.softmax(self.infer_emitter_z((decayed_weight*self.infer_transfer_z(prev_y_prob) + (1-decayed_weight)*prev_y_state)), -1)
		
		if self.concat_data:
			input_z_w = torch.cat([prev_y_prob, prev_y_state], -1)
			prev_y_prob = F.softmax(self.infer_emitter_z(input_z_w), -1)
		else:
			prev_y_prob = F.softmax(self.infer_emitter_z((decayed_weight*self.infer_transfer_z(prev_y_prob) + (1-decayed_weight)*prev_y_state)), -1)
		
		
		
# 			prev_y_prob = F.softmax(self.infer_emitter_z((decayed_weight*self.infer_transfer_z(prev_y_prob) + (1-decayed_weight)*prev_y_state)), -1)
			
		
# 		print(torch.sum(prev_y_prob, -1))
		
		return prev_y_prob
	
	
	def run_odernn_single_step(self, data, time_steps, full_curr_rnn_input = None,
		run_backwards = False, save_info = False, prev_y_state = None):
# 		n_traj, n_tp, n_dims = data.size()
		
		
		
		extra_info = []

		t0 = time_steps[-1]
		if run_backwards:
			t0 = time_steps[0]

# 		device = get_device(data)

# 		prev_y_prob = torch.zeros((1, n_traj, self.cluster_num)).to(self.device)
		n_traj = data.size()[1]
		if prev_y_state is None:
			prev_y_state = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
		
# 		joint_probs = torch.zeros([n_tp, n_traj, self.cluster_num], dtype = torch.float, device = self.device)
		
		
# 		prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)

		prev_t, t_i = time_steps[0],  time_steps[1]

# 		interval_length = time_steps[-1] - time_steps[0]
# 		minimum_step = (interval_length+1) / 500

		#print("minimum step: {}".format(minimum_step))

		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

# 		latent_ys = []
# 		
# 		latent_y_states = []
		
		# Run ODE backwards and combine the y(t) estimates using gating
		time_points_iter = range(0, len(time_steps))
		if run_backwards:
			time_points_iter = reversed(time_points_iter)
		
		
		
# 		for i in time_points_iter:
# 		print(t_i, prev_t)
		if (t_i - prev_t) < self.min_steps:
			time_points = torch.stack((prev_t, t_i))
			inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y_state) * (t_i - prev_t)

			assert(not torch.isnan(inc).any())

			ode_sol = prev_y_state + inc
			ode_sol = torch.stack((prev_y_state, ode_sol), 2).to(self.device)

			assert(not torch.isnan(ode_sol).any())
		else:
			n_intermediate_tp = max(2, ((t_i - prev_t) / self.min_steps).int())

			time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
			ode_sol = self.z0_diffeq_solver(prev_y_state, time_points)

			assert(not torch.isnan(ode_sol).any())

		if torch.mean(ode_sol[:, :, 0, :]  - prev_y_state) >= 0.001:
			print("Error: first point of the ODE is not equal to initial value")
			print(torch.mean(ode_sol[:, :, 0, :]  - prev_y_state))
			exit()
			#assert(torch.mean(ode_sol[:, :, 0, :]  - prev_y) < 0.001)


		

		yi_ode = ode_sol[:, :, -1, :]
		
		prev_y_state = self.GRU_update(yi_ode, data)
		
		xi = data[:,:].unsqueeze(0)
		
		
		
# 			if not self.use_sparse:
# 				y_ode_probs = F.softmax(self.infer_emitter_z(yi_ode), -1)
# 			else:
# 				y_ode_probs = self.sparsemax(self.infer_emitter_z(yi_ode))
		
		
		
# 			if self.use_sparse and i > 0:
# 				prev_y_prob = self.sparsemax(torch.log(prev_y_prob + 1e-5)) 
		
		
# 		prev_y_prob = self.emit_probs(prev_y_state)
		
		
# 			if i > 0:
# 				decayed_weight = torch.exp(-(self.decayed_layer(t_i - prev_t)))
# 			else:
# 				decayed_weight = 0.5
# 			
# 			
# 			prev_y_prob = self.infer_emitter_z((decayed_weight*self.infer_transfer_z(prev_y_prob) + (1-decayed_weight)*prev_y_state))
		
# 		latent_y_states.append(prev_y_state)
# 			prev_y_state = self.infer_emitter_z(yi)
		
# 			if not self.use_sparse:
# 				prev_y = F.softmax(self.infer_emitter_z(yi), -1)
# 			else:
# 				prev_y = self.sparsemax(self.infer_emitter_z(yi))
# 		prev_t, t_i = time_steps[i],  time_steps[i-1]

# 		latent_ys.append(prev_y_prob)

		if save_info:
			d = {"yi_ode": yi_ode.detach(), #"yi_from_data": yi_from_data,
#  					 "yi": yi.detach(), 
#  					 "yi_std": yi_std.detach(), 
				 "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
			extra_info.append(d)

# 		latent_ys = torch.stack(latent_ys, 1)
# 		
# 		latent_y_states = torch.stack(latent_y_states, 1)

# 		assert(not torch.isnan(yi).any())
# 		assert(not torch.isnan(yi_std).any())
		return prev_y_state
	
	def run_odernn(self, data, time_steps, full_curr_rnn_input = None,
		run_backwards = False, save_info = False, exp_y_states=None, exp_y_probs=None):
		# IMPORTANT: assumes that 'data' already has mask concatenated to it 


		n_traj, n_tp, n_dims = data.size()
		
		
		
		extra_info = []

		t0 = time_steps[-1]
		if run_backwards:
			t0 = time_steps[0]

# 		device = get_device(data)

		prev_y_prob = torch.zeros((1, n_traj, self.cluster_num)).to(self.device)
		
		prev_y_state = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
		
		joint_probs = torch.zeros([n_tp, n_traj, self.cluster_num], dtype = torch.float, device = self.device)
		
		
# 		prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)
	
		if not run_backwards:
			prev_t, t_i = time_steps[0] - 0.01,  time_steps[0]
		else:
			prev_t, t_i = time_steps[-1],  time_steps[-1] + 0.01

		interval_length = time_steps[-1] - time_steps[0]
		minimum_step = (time_steps[-1] - time_steps[0])/(len(time_steps)*0.5)

		self.min_steps = minimum_step

		#print("minimum step: {}".format(minimum_step))

		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

		latent_ys = []
		
# 		latent_ys0 = []
		
		latent_y_states = []
		
		# Run ODE backwards and combine the y(t) estimates using gating
		time_points_iter = range(0, len(time_steps))
		if run_backwards:
			time_points_iter = reversed(time_points_iter)
			
		first_ys = 0
		
		first_y_state = 0	
		
		count = 0
		
		for i in time_points_iter:
			
# 			print(i, t_i, prev_t)
			if (t_i - prev_t) < minimum_step:
				time_points = torch.stack((prev_t, t_i))
				inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y_state) * (t_i - prev_t)

				assert(not torch.isnan(inc).any())

				ode_sol = prev_y_state + inc
				ode_sol = torch.stack((prev_y_state, ode_sol), 2).to(self.device)

				assert(not torch.isnan(ode_sol).any())
			else:
				n_intermediate_tp = max(2, ((t_i-prev_t) / minimum_step).int())

				time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
				ode_sol = self.z0_diffeq_solver(prev_y_state, time_points)

				assert(not torch.isnan(ode_sol).any())

			if torch.mean(ode_sol[:, :, 0, :]  - prev_y_state) >= 0.001:
				print("Error: first point of the ODE is not equal to initial value")
				print(torch.mean(ode_sol[:, :, 0, :]  - prev_y_state))
				exit()
			#assert(torch.mean(ode_sol[:, :, 0, :]  - prev_y) < 0.001)


			

			yi_ode = ode_sol[:, :, -1, :]
			xi = data[:,i,:].unsqueeze(0)
# 			mask_i = mask[:,i,:].unsqueeze(0)
# 			
# 			if self.use_mask:
# 				xi = torch.cat([xi, mask_i], -1)
			
# 			if not self.use_sparse:
# 				y_ode_probs = F.softmax(self.infer_emitter_z(yi_ode), -1)
# 			else:
# 				y_ode_probs = self.sparsemax(self.infer_emitter_z(yi_ode))
			
			prev_y_state = self.GRU_update(yi_ode, xi)
			
			if exp_y_states is not None:
				print(torch.norm(exp_y_states[:, count] - prev_y_state))
# 			if i > 0:
# 				decayed_weight = torch.exp(-(self.decayed_layer(t_i - prev_t)))
# 			else:
# 				decayed_weight = 0.5
# 			
# 			

# 			prev_y_prob = self.emit_probs(prev_y_state, prev_y_prob, t_i - prev_t, i)

# 			if i == 0:
# 				first_ys = prev_y_prob.clone()
# 				
# 				first_y_state = prev_y_state.clone()
# 			
# 			latent_ys0.append(prev_y_prob.clone())

# 			prev_y_prob = self.infer_emitter_z((decayed_weight*self.infer_transfer_z(prev_y_prob) + (1-decayed_weight)*prev_y_state))
			
			latent_y_states.append(prev_y_state.clone())
# 			prev_y_state = self.infer_emitter_z(yi)
			
# 			if not self.use_sparse:
# 				prev_y = F.softmax(self.infer_emitter_z(yi), -1)
# 			else:
# 				prev_y = self.sparsemax(self.infer_emitter_z(yi))
			if not run_backwards:
				prev_t, t_i = time_steps[i],  time_steps[(i+1)%time_steps.shape[0]]
			else:
				prev_t, t_i = time_steps[(i-1)],  time_steps[i]

# 			latent_ys.append(prev_y_prob.clone())

			if save_info:
				d = {"yi_ode": yi_ode.detach(), #"yi_from_data": yi_from_data,
#  					 "yi": yi.detach(), 
#  					 "yi_std": yi_std.detach(), 
					 "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
				extra_info.append(d)
				
			count += 1

		
		latent_y_states = torch.stack(latent_y_states, 1)
		
		prev_t, t_i = time_steps[0] - 0.01,  time_steps[0]
		
		if run_backwards:
			latent_y_states = torch.flip(latent_y_states, [1])
		prev_y_prob = torch.zeros((1, n_traj, self.cluster_num)).to(self.device)
		for t in range(latent_y_states.shape[1]):
			
# 			print(prev_t, t_i)
			
			prev_y_state = latent_y_states[:,t]
			
# 			if self.use_sparse:
# 				prev_y_prob = self.sparsemax(torch.log(prev_y_prob + 1e-5)) 
			
			
			curr_prob = self.emit_probs(prev_y_state, prev_y_prob, t_i - prev_t, t)
			
# 			exp_prev_y_prob = latent_ys0[t]
# 			
# 			print(torch.norm(first_ys - curr_prob))
# 			
# 			print(torch.norm(first_ys - exp_prev_y_prob))
# 			
# 			print(torch.norm(first_y_state - prev_y_state))
# 			
# 			print(torch.norm(curr_prob - exp_prev_y_prob))
			
			prev_y_prob = curr_prob
			
			latent_ys.append(prev_y_prob.clone())
			
			prev_t, t_i = time_steps[t],  time_steps[(t+1)%time_steps.shape[0]]
			
		latent_ys = torch.stack(latent_ys, 1)

# 		assert(not torch.isnan(yi).any())
# 		assert(not torch.isnan(yi_std).any())

		return latent_ys, latent_y_states, extra_info



class Decoder(nn.Module):
	def __init__(self, latent_dim, input_dim):
		super(Decoder, self).__init__()
		# decode data from latent space where we are solving an ODE back to the data space

		decoder = nn.Sequential(
		   nn.Linear(latent_dim, input_dim),)

		utils.init_network_weights(decoder)	
		self.decoder = decoder

	def forward(self, data):
		return self.decoder(data)


