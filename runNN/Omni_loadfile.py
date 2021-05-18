import numpy as np
import energyflow as ef

import pickle

import OmniFold.omnifold as of
from energyflow.archs import PFN
from energyflow.archs import DNN

percval = '50Perc'
massval = '16000'

fi = np.load('/data0/wmccorma/'+percval+'Sig_2mil_'+massval+'MeV.npz', allow_pickle=True)

X_gen = fi['X_gen']
X_det = fi['X_det']
Y_gen = fi['Y_gen']
Y_det = fi['Y_det']

print('shapes')
print(X_det.shape)
print(X_gen.shape)

itnum = 60

Phi_sizes = [100, 100, 256]
F_sizes = [100, 100, 100]

det_args = {'input_dim': 4, 'Phi_sizes': Phi_sizes, 'F_sizes': F_sizes, 'patience': 10, 'filepath': '/data0/wmccorma/'+percval+'_'+massval+'_Step1_{}', 'save_weights_only': False, 'modelcheck_opts': {'save_best_only': True, 'verbose': 1}}
mc_args = {'input_dim': 4, 'Phi_sizes': Phi_sizes, 'F_sizes': F_sizes, 'patience': 10, 'filepath': '/data0/wmccorma/'+percval+'_'+massval+'_Step2_{}', 'save_weights_only': False, 'modelcheck_opts': {'save_best_only': True, 'verbose': 1}}

fitargs = {'batch_size': 10000, 'epochs': 120, 'verbose': 1}

# reweight the sim and data to have the same total weight to begin with
ndata, nsim = np.count_nonzero(Y_det[:,1]), np.count_nonzero(Y_det[:,0])
wdata = np.ones(ndata)
winit = ndata/nsim*np.ones(nsim)

multifold_ws = of.omnifold(X_gen, Y_gen, X_det, Y_det, wdata, winit, (ef.archs.efn.PFN, det_args), (ef.archs.efn.PFN, mc_args), fitargs, val=0.2, it=itnum, trw_ind=-2, weights_filename='/data0/wmccorma/Official_'+percval+'_'+massval)
