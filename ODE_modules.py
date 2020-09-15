'''
Created on Jul 24, 2020

'''

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.encoder_decoder import *
from lib.likelihood_eval import *

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.modules.rnn import GRUCell, LSTMCell, RNNCellBase

from torch.distributions.normal import Normal
from torch.distributions import Independent
from torch.nn.parameter import Parameter

from torchdiffeq import odeint as odeint

from lib.utils import *
def create_classifier(z0_dim, n_labels):
    return nn.Sequential(
            nn.Linear(z0_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, n_labels),)


class Baseline(nn.Module):
    def __init__(self, input_dim, latent_dim, device, 
        obsrv_std = 0.01, use_binary_classif = False,
        classif_per_tp = False,
        use_poisson_proc = False,
        linear_classifier = False,
        n_labels = 1,
        train_classif_w_reconstr = False):
        super(Baseline, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_labels = n_labels

        self.obsrv_std = torch.Tensor([obsrv_std]).to(device)
        self.device = device

        self.use_binary_classif = use_binary_classif
        self.classif_per_tp = classif_per_tp
        self.use_poisson_proc = use_poisson_proc
        self.linear_classifier = linear_classifier
        self.train_classif_w_reconstr = train_classif_w_reconstr

        z0_dim = latent_dim
        if use_poisson_proc:
            z0_dim += latent_dim

        if use_binary_classif: 
            if linear_classifier:
                self.classifier = nn.Sequential(
                    nn.Linear(z0_dim, n_labels))
            else:
                self.classifier = create_classifier(z0_dim, n_labels)
            utils.init_network_weights(self.classifier)


    def get_gaussian_likelihood(self, truth, pred_y, mask = None):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute likelihood of the data under the predictions
        log_density_data = masked_gaussian_log_density(pred_y, truth, 
            obsrv_std = self.obsrv_std, mask = mask)
        log_density_data = log_density_data.permute(1,0)

        # Compute the total density
        # Take mean over n_traj_samples
        log_density = torch.mean(log_density_data, 0)

        # shape: [n_traj]
        return log_density


    def get_mse(self, truth, pred_y, mask = None):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute likelihood of the data under the predictions
        log_density_data = compute_mse(pred_y, truth, mask = mask)
        # shape: [1]
        return torch.mean(log_density_data)


    def compute_all_losses(self, batch_dict,
        n_tp_to_sample = None, n_traj_samples = 1, kl_coef = 1.):

        # Condition on subsampled points
        # Make predictions for all the points
        pred_x, info = self.get_reconstruction(batch_dict["tp_to_predict"], 
            batch_dict["observed_data"], batch_dict["observed_tp"], 
            mask = batch_dict["observed_mask"], n_traj_samples = n_traj_samples,
            mode = batch_dict["mode"])

        # Compute likelihood of all the points
        likelihood = self.get_gaussian_likelihood(batch_dict["data_to_predict"], pred_x,
            mask = batch_dict["mask_predicted_data"])

        mse = self.get_mse(batch_dict["data_to_predict"], pred_x,
            mask = batch_dict["mask_predicted_data"])

        ################################
        # Compute CE loss for binary classification on Physionet
        # Use only last attribute -- mortatility in the hospital 
        device = batch_dict["data_to_predict"].device
        ce_loss = torch.Tensor([0.]).to(device)
        
        if (batch_dict["labels"] is not None) and self.use_binary_classif:
            if (batch_dict["labels"].size(-1) == 1) or (len(batch_dict["labels"].size()) == 1):
                ce_loss = compute_binary_CE_loss(
                    info["label_predictions"], 
                    batch_dict["labels"])
            else:
                ce_loss = compute_multiclass_CE_loss(
                    info["label_predictions"], 
                    batch_dict["labels"],
                    mask = batch_dict["mask_predicted_data"])

            if torch.isnan(ce_loss):
                print("label pred")
                print(info["label_predictions"])
                print("labels")
                print( batch_dict["labels"])
                raise Exception("CE loss is Nan!")

        pois_log_likelihood = torch.Tensor([0.]).to(get_device(batch_dict["data_to_predict"]))
        if self.use_poisson_proc:
            pois_log_likelihood = compute_poisson_proc_likelihood(
                batch_dict["data_to_predict"], pred_x, 
                info, mask = batch_dict["mask_predicted_data"])
            # Take mean over n_traj
            pois_log_likelihood = torch.mean(pois_log_likelihood, 1)

        loss = - torch.mean(likelihood)

        if self.use_poisson_proc:
            loss = loss - 0.1 * pois_log_likelihood 

        if self.use_binary_classif:
            if self.train_classif_w_reconstr:
                loss = loss +  ce_loss * 100
            else:
                loss =  ce_loss

        # Take mean over the number of samples in a batch
        results = {}
        results["loss"] = torch.mean(loss)
        results["likelihood"] = torch.mean(likelihood).detach()
        results["mse"] = torch.mean(mse).detach()
        results["pois_likelihood"] = torch.mean(pois_log_likelihood).detach()
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["kl"] = 0.
        results["kl_first_p"] =  0.
        results["std_first_p"] = 0.

        if batch_dict["labels"] is not None and self.use_binary_classif:
            results["label_predictions"] = info["label_predictions"].detach()
        return results



class VAE_Baseline(nn.Module):
    def __init__(self, input_dim, latent_dim, 
        z0_prior, device,
        obsrv_std = 0.01, 
        use_binary_classif = False,
        classif_per_tp = False,
        use_poisson_proc = False,
        linear_classifier = False,
        n_labels = 1,
        train_classif_w_reconstr = False):

        super(VAE_Baseline, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        self.n_labels = n_labels

        self.obsrv_std = torch.Tensor([obsrv_std]).to(device)

        self.z0_prior = z0_prior
        self.use_binary_classif = use_binary_classif
        self.classif_per_tp = classif_per_tp
        self.use_poisson_proc = use_poisson_proc
        self.linear_classifier = linear_classifier
        self.train_classif_w_reconstr = train_classif_w_reconstr

        z0_dim = latent_dim
        if use_poisson_proc:
            z0_dim += latent_dim

        if use_binary_classif: 
            if linear_classifier:
                self.classifier = nn.Sequential(
                    nn.Linear(z0_dim, n_labels))
            else:
                self.classifier = create_classifier(z0_dim, n_labels)
            utils.init_network_weights(self.classifier)


    def get_gaussian_likelihood(self, truth, pred_y, mask = None):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        n_traj, n_tp, n_dim = truth.size()

        # Compute likelihood of the data under the predictions
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
        
        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)
        log_density_data = masked_gaussian_log_density(pred_y, truth_repeated, 
            obsrv_std = self.obsrv_std, mask = mask)
        log_density_data = log_density_data.permute(1,0)
        log_density = torch.mean(log_density_data, 1)

        # shape: [n_traj_samples]
        return log_density


    def get_mse(self, truth, pred_y, mask = None):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        n_traj, n_tp, n_dim = truth.size()

        # Compute likelihood of the data under the predictions
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
        
        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute likelihood of the data under the predictions
        log_density_data = compute_mse(pred_y, truth_repeated, mask = mask)
        # shape: [1]
        return log_density_data

    def get_mae(self, truth, pred_y, mask = None):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        n_traj, n_tp, n_dim = truth.size()

        # Compute likelihood of the data under the predictions
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
        
        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute likelihood of the data under the predictions
        log_density_data = compute_mae(pred_y, truth_repeated, mask = mask)
        # shape: [1]
        return log_density_data


    def compute_all_losses(self, batch_dict, n_traj_samples = 1, kl_coef = 1.):
        # Condition on subsampled points
        # Make predictions for all the points
        pred_y, info = self.get_reconstruction(batch_dict["tp_to_predict"], 
            batch_dict["observed_data"], batch_dict["observed_tp"], 
            mask = batch_dict["observed_mask"], n_traj_samples = n_traj_samples,
            mode = batch_dict["mode"])

        #print("get_reconstruction done -- computing likelihood")
        fp_mu, fp_std, fp_enc = info["first_point"]
        fp_std = fp_std.abs()
        fp_distr = Normal(fp_mu, fp_std)

        assert(torch.sum(fp_std < 0) == 0.)

        kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

        if torch.isnan(kldiv_z0).any():
            print(fp_mu)
            print(fp_std)
            raise Exception("kldiv_z0 is Nan!")

        # Mean over number of latent dimensions
        # kldiv_z0 shape: [n_traj_samples, n_traj, n_latent_dims] if prior is a mixture of gaussians (KL is estimated)
        # kldiv_z0 shape: [1, n_traj, n_latent_dims] if prior is a standard gaussian (KL is computed exactly)
        # shape after: [n_traj_samples]
        
#         kldiv_z0 = kldiv_z0[kldiv_z0 != np.Inf]
        
        kldiv_z0 = torch.mean(kldiv_z0, (1,2))

        # Compute likelihood of all the points
#         rec_likelihood = self.get_gaussian_likelihood(
#             batch_dict["data_to_predict"], pred_y,
#             mask = batch_dict["mask_predicted_data"])

#         mse = self.get_mse(
#             batch_dict["data_to_predict"], pred_y,
#             mask = batch_dict["mask_predicted_data"])
        
        mse = (batch_dict["data_to_predict"] - pred_y)**2
        
#         mse_loss = torch.sqrt(torch.sum(mse*batch_dict["mask_predicted_data"]).detach()/(torch.sum(batch_dict["mask_predicted_data"]).detach()))
        
#         mae = self.get_mae(
#             batch_dict["data_to_predict"], pred_y,
#             mask = batch_dict["mask_predicted_data"])
        
        
        mae = torch.abs(batch_dict["data_to_predict"] - pred_y)
        
        
        l2_norm_loss = (batch_dict["data_to_predict"].view([1,batch_dict["data_to_predict"].shape[0],batch_dict["data_to_predict"].shape[1],batch_dict["data_to_predict"].shape[2]]) - pred_y)**2 
        
        negll = compute_gaussian_probs0(batch_dict["data_to_predict"], pred_y, 2*torch.log(self.obsrv_std), batch_dict["mask_predicted_data"])
        
        negll_loss = torch.sum(negll*batch_dict["mask_predicted_data"])/torch.sum(batch_dict["mask_predicted_data"])

        pois_log_likelihood = torch.Tensor([0.]).to(get_device(batch_dict["data_to_predict"]))
        if self.use_poisson_proc:
            pois_log_likelihood = compute_poisson_proc_likelihood(
                batch_dict["data_to_predict"], pred_y, 
                info, mask = batch_dict["mask_predicted_data"])
            # Take mean over n_traj
            pois_log_likelihood = torch.mean(pois_log_likelihood, 1)

        ################################
        # Compute CE loss for binary classification on Physionet
        device = get_device(batch_dict["data_to_predict"])
        ce_loss = torch.Tensor([0.]).to(device)
        if (batch_dict["labels"] is not None) and self.use_binary_classif:

            if (batch_dict["labels"].size(-1) == 1) or (len(batch_dict["labels"].size()) == 1):
                ce_loss = compute_binary_CE_loss(
                    info["label_predictions"], 
                    batch_dict["labels"])
            else:
                ce_loss = compute_multiclass_CE_loss(
                    info["label_predictions"], 
                    batch_dict["labels"],
                    mask = batch_dict["mask_predicted_data"])

        rec_likelihood = 0

        # IWAE loss
        loss = - torch.logsumexp(rec_likelihood -  kl_coef * kldiv_z0,0) + negll_loss
        if torch.isnan(loss):
            loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0,0) + negll_loss
            
        if self.use_poisson_proc:
            loss = loss - 0.1 * pois_log_likelihood 

        if self.use_binary_classif:
            if self.train_classif_w_reconstr:
                loss = loss +  ce_loss * 100
            else:
                loss =  ce_loss

        results = {}
        results["loss"] = torch.mean(loss)
#         results["likelihood"] = torch.mean(rec_likelihood).detach()
        results["likelihood_res"] = torch.sum(negll*batch_dict["mask_predicted_data"]).detach()/torch.sum(batch_dict["mask_predicted_data"]).detach()
        results["rmse"] = torch.sqrt(torch.sum(mse*batch_dict["mask_predicted_data"]).detach()/(torch.sum(batch_dict["mask_predicted_data"]).detach()))
        results["mae"] = torch.sum(mae*batch_dict["mask_predicted_data"]).detach()/(torch.sum(batch_dict["mask_predicted_data"]).detach())
        results["pois_likelihood"] = torch.mean(pois_log_likelihood).detach()
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["kl_first_p"] =  torch.mean(kldiv_z0).detach()
        results["std_first_p"] = torch.mean(fp_std).detach()
        results["l2_norm_loss"] = l2_norm_loss.view(-1).mean()
        results["predicted_data"] = pred_y

        if batch_dict["labels"] is not None and self.use_binary_classif:
            results["label_predictions"] = info["label_predictions"].detach()
        
        
#         print('loss::', results["l2_norm_loss"], results["kl_first_p"])
        
        return results


class ODE_RNN(Baseline):
    def __init__(self, input_dim, latent_dim, device = torch.device("cpu"),
        z0_diffeq_solver = None, n_gru_units = 100,  n_units = 100,
        concat_mask = False, obsrv_std = 0.1, use_binary_classif = False,
        classif_per_tp = False, n_labels = 1, train_classif_w_reconstr = False):

        Baseline.__init__(self, input_dim, latent_dim, device = device, 
            obsrv_std = obsrv_std, use_binary_classif = use_binary_classif,
            classif_per_tp = classif_per_tp,
            n_labels = n_labels,
            train_classif_w_reconstr = train_classif_w_reconstr)

        ode_rnn_encoder_dim = latent_dim
    
        self.ode_gru = Encoder_z0_ODE_RNN( 
            latent_dim = ode_rnn_encoder_dim, 
            input_dim = (input_dim) * 2, # input and the mask
            z0_diffeq_solver = z0_diffeq_solver, 
            n_gru_units = n_gru_units, 
            device = device).to(device)

        self.z0_diffeq_solver = z0_diffeq_solver

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, input_dim),)

        utils.init_network_weights(self.decoder)


    def get_reconstruction(self, time_steps_to_predict, data, truth_time_steps, 
        mask = None, n_traj_samples = None, mode = None):

        if (len(truth_time_steps) != len(time_steps_to_predict)) or (torch.sum(time_steps_to_predict - truth_time_steps) != 0):
            raise Exception("Extrapolation mode not implemented for ODE-RNN")

        # time_steps_to_predict and truth_time_steps should be the same 
        assert(len(truth_time_steps) == len(time_steps_to_predict))
        assert(mask is not None)
        
        data_and_mask = data
        if mask is not None:
            data_and_mask = torch.cat([data, mask],-1)

        _, _, latent_ys, _ = self.ode_gru.run_odernn(
            data_and_mask, truth_time_steps, run_backwards = False)
        
        latent_ys = latent_ys.permute(0,2,1,3)
        last_hidden = latent_ys[:,:,-1,:]

            #assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)

        outputs = self.decoder(latent_ys)
        # Shift outputs for computing the loss -- we should compare the first output to the second data point, etc.
        first_point = data[:,0,:]
        outputs = utils.shift_outputs(outputs, first_point)

        extra_info = {"first_point": (latent_ys[:,:,-1,:], 0.0, latent_ys[:,:,-1,:])}

        if self.use_binary_classif:
            if self.classif_per_tp:
                extra_info["label_predictions"] = self.classifier(latent_ys)
            else:
                extra_info["label_predictions"] = self.classifier(last_hidden).squeeze(-1)

        # outputs shape: [n_traj_samples, n_traj, n_tp, n_dims]
        return outputs, extra_info

class ODEFunc(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_func_net, device = torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc, self).__init__()

        self.input_dim = input_dim
        self.device = device

        utils.init_network_weights(ode_func_net)
        self.gradient_net = ode_func_net

    def forward(self, t_local, y, backwards = False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        return self.gradient_net(y)

    def sample_next_point_from_prior(self, t_local, y):
        """
        t_local: current time point
        y: value at the current time point
        """
        return self.get_ode_gradient_nn(t_local, y)

#####################################################################################################

class ODEFunc_w_Poisson(ODEFunc):
    
    def __init__(self, input_dim, latent_dim, ode_func_net,
        lambda_net, device = torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc_w_Poisson, self).__init__(input_dim, latent_dim, ode_func_net, device)

        self.latent_ode = ODEFunc(input_dim = input_dim, 
            latent_dim = latent_dim, 
            ode_func_net = ode_func_net,
            device = device)

        self.latent_dim = latent_dim
        self.lambda_net = lambda_net
        # The computation of poisson likelihood can become numerically unstable. 
        #The integral lambda(t) dt can take large values. In fact, it is equal to the expected number of events on the interval [0,T]
        #Exponent of lambda can also take large values
        # So we divide lambda by the constant and then multiply the integral of lambda by the constant
        self.const_for_lambda = torch.Tensor([100.]).to(device)

    def extract_poisson_rate(self, augmented, final_result = True):
        y, log_lambdas, int_lambda = None, None, None

        assert(augmented.size(-1) == self.latent_dim + self.input_dim)        
        latent_lam_dim = self.latent_dim // 2

        if len(augmented.size()) == 3:
            int_lambda  = augmented[:,:,-self.input_dim:] 
            y_latent_lam = augmented[:,:,:-self.input_dim]

            log_lambdas  = self.lambda_net(y_latent_lam[:,:,-latent_lam_dim:])
            y = y_latent_lam[:,:,:-latent_lam_dim]

        elif len(augmented.size()) == 4:
            int_lambda  = augmented[:,:,:,-self.input_dim:]
            y_latent_lam = augmented[:,:,:,:-self.input_dim]

            log_lambdas  = self.lambda_net(y_latent_lam[:,:,:,-latent_lam_dim:])
            y = y_latent_lam[:,:,:,:-latent_lam_dim]

        # Multiply the intergral over lambda by a constant 
        # only when we have finished the integral computation (i.e. this is not a call in get_ode_gradient_nn)
        if final_result:
            int_lambda = int_lambda * self.const_for_lambda
            
        # Latents for performing reconstruction (y) have the same size as latent poisson rate (log_lambdas)
        assert(y.size(-1) == latent_lam_dim)

        return y, log_lambdas, int_lambda, y_latent_lam


    def get_ode_gradient_nn(self, t_local, augmented):
        y, log_lam, int_lambda, y_latent_lam = self.extract_poisson_rate(augmented, final_result = False)
        dydt_dldt = self.latent_ode(t_local, y_latent_lam)

        log_lam = log_lam - torch.log(self.const_for_lambda)
        return torch.cat((dydt_dldt, torch.exp(log_lam)),-1)


class LatentODE(VAE_Baseline):
    def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver, 
        z0_prior, device, obsrv_std = None, 
        use_binary_classif = False, use_poisson_proc = False,
        linear_classifier = False,
        classif_per_tp = False,
        n_labels = 1,
        train_classif_w_reconstr = False):

        super(LatentODE, self).__init__(
            input_dim = input_dim, latent_dim = latent_dim, 
            z0_prior = z0_prior, 
            device = device, obsrv_std = obsrv_std, 
            use_binary_classif = use_binary_classif,
            classif_per_tp = classif_per_tp, 
            linear_classifier = linear_classifier,
            use_poisson_proc = use_poisson_proc,
            n_labels = n_labels,
            train_classif_w_reconstr = train_classif_w_reconstr)

        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver
        self.decoder = decoder
        self.use_poisson_proc = use_poisson_proc

    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, 
        mask = None, n_traj_samples = 1, run_backwards = True, mode = None):

        if isinstance(self.encoder_z0, Encoder_z0_ODE_RNN) or \
            isinstance(self.encoder_z0, Encoder_z0_RNN):

            truth_w_mask = truth
            if mask is not None:
                truth_w_mask = torch.cat((truth, mask), -1)
            first_point_mu, first_point_std = self.encoder_z0(
                truth_w_mask, truth_time_steps, run_backwards = run_backwards)

            means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
            sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
            first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)

        else:
            raise Exception("Unknown encoder type {}".format(type(self.encoder_z0).__name__))
        
        first_point_std = first_point_std.abs()
        assert(torch.sum(first_point_std < 0) == 0.)

        if self.use_poisson_proc:
            n_traj_samples, n_traj, n_dims = first_point_enc.size()
            # append a vector of zeros to compute the integral of lambda
            zeros = torch.zeros([n_traj_samples, n_traj,self.input_dim]).to(get_device(truth))
            first_point_enc_aug = torch.cat((first_point_enc, zeros), -1)
            means_z0_aug = torch.cat((means_z0, zeros), -1)
        else:
            first_point_enc_aug = first_point_enc
            means_z0_aug = means_z0
            
        assert(not torch.isnan(time_steps_to_predict).any())
        assert(not torch.isnan(first_point_enc).any())
        assert(not torch.isnan(first_point_enc_aug).any())

        # Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
        sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)

        if self.use_poisson_proc:
            sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

            assert(torch.sum(int_lambda[:,:,0,:]) == 0.)
            assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)

        pred_x = self.decoder(sol_y)

        all_extra_info = {
            "first_point": (first_point_mu, first_point_std, first_point_enc),
            "latent_traj": sol_y.detach()
        }

        if self.use_poisson_proc:
            # intergral of lambda from the last step of ODE Solver
            all_extra_info["int_lambda"] = int_lambda[:,:,-1,:]
            all_extra_info["log_lambda_y"] = log_lambda_y

        if self.use_binary_classif:
            if self.classif_per_tp:
                all_extra_info["label_predictions"] = self.classifier(sol_y)
            else:
                all_extra_info["label_predictions"] = self.classifier(first_point_enc).squeeze(-1)

        return pred_x, all_extra_info


    def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples = 1):
        # input_dim = starting_point.size()[-1]
        # starting_point = starting_point.view(1,1,input_dim)

        # Sample z0 from prior
        starting_point_enc = self.z0_prior.sample([n_traj_samples, 1, self.latent_dim]).squeeze(-1)

        starting_point_enc_aug = starting_point_enc
        if self.use_poisson_proc:
            n_traj_samples, n_traj, n_dims = starting_point_enc.size()
            # append a vector of zeros to compute the integral of lambda
            zeros = torch.zeros(n_traj_samples, n_traj,self.input_dim).to(self.device)
            starting_point_enc_aug = torch.cat((starting_point_enc, zeros), -1)

        sol_y = self.diffeq_solver.sample_traj_from_prior(starting_point_enc_aug, time_steps_to_predict, 
            n_traj_samples = 3)

        if self.use_poisson_proc:
            sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)
        
        return self.decoder(sol_y)
    
    
class DiffeqSolver(nn.Module):
    def __init__(self, input_dim, ode_func, method, latents, 
            odeint_rtol = 1e-4, odeint_atol = 1e-5, device = torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.latents = latents        
        self.device = device
        self.ode_func = ode_func

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict, backwards = False):
        """
        # Decode the trajectory through ODE Solver
        """
        n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
        n_dims = first_point.size()[-1]

        pred_y = odeint(self.ode_func, first_point, time_steps_to_predict, 
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
        pred_y = pred_y.permute(1,2,0,3)

        assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
        assert(pred_y.size()[0] == n_traj_samples)
        assert(pred_y.size()[1] == n_traj)

        return pred_y

    def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict, 
        n_traj_samples = 1):
        """
        # Decode the trajectory through ODE Solver using samples from the prior

        time_steps_to_predict: time steps at which we want to sample the new trajectory
        """
        func = self.ode_func.sample_next_point_from_prior

        pred_y = odeint(func, starting_point_enc, time_steps_to_predict, 
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
        # shape: [n_traj_samples, n_traj, n_tp, n_dim]
        pred_y = pred_y.permute(1,2,0,3)
        return pred_y

