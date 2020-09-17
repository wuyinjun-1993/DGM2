

def config_DGM2_O():
    conf = {
 
# Model Arguments
    'd': 4,
    'input_dim': 88,
    'z_dim':3,
    's_dim': 10,
    'h_dim': 10,
    'e_dim': 10,
    'emission_z_dim':10,
    'emission_x_dim':10,
    'trans_dim':20,
    'rnn_dim':60,
    'cluster_num':20,
    'temp':1.0, # softmax temperature (lower --> more discrete)
    'dropout':0.0, # dropout applied to layers (0 = no dropout)
    'x_std': 0.1,
    'latent_x_std': 0.05,
    'sampling_times': 10,
    'use_gumbel': False,
    'temp_min':0.1,
    'temp_init':0.5,
    'temp_anneal':0.01,    
# Training Arguments
    'batch_size':20,
    'epochs':500, # maximum number of epochs
    'min_epochs':2, # minimum number of epochs to train for
 
    'use_gate': False,
    'lr':1e-2, # autoencoder learning rate
    'beta1':0.9, # beta1 for adam
    'beta2':0.99,
    'clip_norm':20.0,  # gradient clipping, max norm       
    'weight_decay':2.0,
    'anneal_epochs':1000,
    'min_anneal':0.1,
    }
    return conf 

def config_DGM2_L():
    conf = {
 
# Model Arguments
    'd': 4,
    'input_dim': 88,
    'z_dim':3,
    's_dim': 10,
    'h_dim': 10,
    'e_dim': 10,
    'emission_z_dim':5,
    'emission_x_dim':5,
    'trans_dim':20,
    'rnn_dim':60,
    'cluster_num':20,
    'temp':1.0, # softmax temperature (lower --> more discrete)
    'dropout':0.05, # dropout applied to layers (0 = no dropout)
    'x_std': 0.1,
    'latent_x_std': 0.05,
    'sampling_times': 10,
    'use_gumbel': False,
    'temp_min':0.1,
    'temp_init':0.5,
    'temp_anneal':0.01,    
# Training Arguments
    'batch_size':20,
    'epochs':500, # maximum number of epochs
    'min_epochs':2, # minimum number of epochs to train for
 
    'use_gate': False,
    'lr':1e-2, # autoencoder learning rate
    'beta1':0.9, # beta1 for adam
    'beta2':0.99,
    'clip_norm':20.0,  # gradient clipping, max norm       
    'weight_decay':2.0,
    'anneal_epochs':1000,
    'min_anneal':0.1,
    }
    return conf 

'''mimic3 data'''
# def config_DHMM_cluster():
#     conf = {
#  
# # Model Arguments
#     'd': 4,
#     'input_dim': 88,
#     'z_dim':10,
#     's_dim': 10,
#     'h_dim': 10,
#     'e_dim': 30,
#     'emission_z_dim':10,
#     'emission_x_dim':10,
#     'trans_dim':20,
#     'rnn_dim':60,
#     'cluster_num':50,
#     'temp':1.0, # softmax temperature (lower --> more discrete)
#     'dropout':0.2, # dropout applied to layers (0 = no dropout)
#     'x_std': 0.1,
#     'sampling_times': 10,
#     'use_gumbel': False,
# # Training Arguments
#     'batch_size':20,
#     'epochs':500, # maximum number of epochs
#     'min_epochs':2, # minimum number of epochs to train for
#  
#     'lr':1e-2, # autoencoder learning rate
#     'beta1':0.9, # beta1 for adam
#     'beta2':0.99,
#     'clip_norm':20.0,  # gradient clipping, max norm       
#     'weight_decay':2.0,
#     'anneal_epochs':1000,
#     'min_anneal':0.1,
#     }
#     return conf 

# def config_DHMM_cluster():
#     conf = {
#  
# # Model Arguments
#     'input_dim': 88,
#     'z_dim':100,
#     's_dim': 100,
#     'h_dim': 100,
#     'e_dim': 100,
#     'emission_z_dim':50,
#     'emission_x_dim':100,
#     'trans_dim':20,
#     'rnn_dim':60,
#     'cluster_num':1000,
#     'temp':1.0, # softmax temperature (lower --> more discrete)
#     'dropout':0, # dropout applied to layers (0 = no dropout)
#     'x_std': 0.01,
#     'd': 2,
# # Training Arguments
#     'batch_size':1000,
#     'epochs':5000, # maximum number of epochs
#     'min_epochs':2, # minimum number of epochs to train for
#  
#     'lr':5e-3, # autoencoder learning rate
#     'beta1':0.9, # beta1 for adam
#     'beta2':0.99,
#     'clip_norm':20.0,  # gradient clipping, max norm       
#     'weight_decay':2.0,
#     'anneal_epochs':1000,
#     'min_anneal':0.1,
#     }
#     return conf 



# def config_DHMM_cluster_tlstm():
#     conf = {
#  
# # Model Arguments
#     'd': 4,
#     'input_dim': 88,
#     'z_dim':10,
#     's_dim': 10,
#     'h_dim': 10,
#     'e_dim': 10,
#     'emission_z_dim':50,
#     'emission_x_dim':20,
#     'trans_dim':20,
#     'rnn_dim':60,
#     'cluster_num':50,
#     'temp':1.0, # softmax temperature (lower --> more discrete)
#     'dropout':0.0, # dropout applied to layers (0 = no dropout)
#     'x_std': 0.1,
#     'sampling_times': 10,
#     'use_gumbel': False,
# # Training Arguments
#     'batch_size':20,
#     'epochs':500, # maximum number of epochs
#     'min_epochs':2, # minimum number of epochs to train for
#  
#     'lr':1e-2, # autoencoder learning rate
#     'beta1':0.9, # beta1 for adam
#     'beta2':0.99,
#     'clip_norm':20.0,  # gradient clipping, max norm       
#     'weight_decay':2.0,
#     'anneal_epochs':1000,
#     'min_anneal':0.1,
#     }
#     return conf 
# 
# 
# 
# 
# def config_DHMM_cluster2():
#     conf = {
#  
# # Model Arguments
#     'input_dim': 88,
#     'z_dim':10,
#     's_dim': 10,
#     'h_dim': 10,
#     'e_dim': 30,
#     'emission_z_dim':50,
#     'emission_x_dim':100,
#     'trans_dim':20,
#     'rnn_dim':60,
#     'cluster_num':30,
#     'temp':1.0, # softmax temperature (lower --> more discrete)
#     'dropout':0, # dropout applied to layers (0 = no dropout)
#     'x_std': 0.1,
#     'z_std': 0.1,
#     'd': 2,
# # Training Arguments
#     'batch_size':20,
#     'epochs':5000, # maximum number of epochs
#     'min_epochs':2, # minimum number of epochs to train for
#  
#     'lr':1e-2, # autoencoder learning rate
#     'beta1':0.9, # beta1 for adam
#     'beta2':0.99,
#     'clip_norm':20.0,  # gradient clipping, max norm       
#     'weight_decay':2.0,
#     'anneal_epochs':1000,
#     'min_anneal':0.1,
#     }
#     return conf 
# 
# 
# def config_DHMM_cluster3():
#     conf = {
#  
# # Model Arguments
#     'input_dim': 88,
#     'z_dim':10,
#     's_dim': 10,
#     'h_dim': 10,
#     'e_dim': 50,
#     'emission_z_dim':50,
#     'emission_x_dim':100,
#     'trans_dim':20,
#     'rnn_dim':60,
#     'cluster_num':30,
#     'temp':1.0, # softmax temperature (lower --> more discrete)
#     'dropout':0, # dropout applied to layers (0 = no dropout)
#     'x_std': 0.1,
#     'z_std': 0.1,
#     'd': 2,
# # Training Arguments
#     'batch_size':20,
#     'epochs':5000, # maximum number of epochs
#     'min_epochs':2, # minimum number of epochs to train for
#  
#     'lr':1e-2, # autoencoder learning rate
#     'beta1':0.9, # beta1 for adam
#     'beta2':0.99,
#     'clip_norm':20.0,  # gradient clipping, max norm       
#     'weight_decay':2.0,
#     'anneal_epochs':1000,
#     'min_anneal':0.1,
#     }
#     return conf 
# 
# 
# def config_DHMM_cluster4():
#     conf = {
#  
# # Model Arguments
#     'input_dim': 88,
#     'z_dim':10,
#     's_dim': 10,
#     'h_dim': 10,
#     'e_dim': 50,
#     'emission_z_dim':50,
#     'emission_x_dim':100,
#     'trans_dim':20,
#     'rnn_dim':60,
#     'cluster_num':30,
#     'temp':1.0, # softmax temperature (lower --> more discrete)
#     'dropout':0, # dropout applied to layers (0 = no dropout)
#     'x_std': 0.1,
#     'z_std': 0.1,
#     'd': 2,
# # Training Arguments
#     'batch_size':20,
#     'epochs':5000, # maximum number of epochs
#     'min_epochs':2, # minimum number of epochs to train for
#  
#     'lr':1e-2, # autoencoder learning rate
#     'beta1':0.9, # beta1 for adam
#     'beta2':0.99,
#     'clip_norm':20.0,  # gradient clipping, max norm       
#     'weight_decay':2.0,
#     'anneal_epochs':1000,
#     'min_anneal':0.1,
#     }
#     return conf 

