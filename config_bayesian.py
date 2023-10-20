############### Configuration file for Bayesian ###############
layer_type = 'bbb'  # 'bbb' or 'lrt'
activation_type = 'relu'  # 'softplus' or 'relu'
priors={
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}

n_epochs = 40
lr_start = 0.00005
num_workers = 4
valid_size = 0.0
batch_size = 32
train_ens = 15
valid_ens = 1
beta_type = 0.1  # 'Blundell', 'Standard', etc. Use float for const value
augmentation = True # augmentation for the POCUS data set 
