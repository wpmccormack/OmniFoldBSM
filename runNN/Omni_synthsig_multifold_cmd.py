import numpy as np
import energyflow as ef

import pickle

import OmniFold.omnifold as of
from energyflow.archs import PFN
from energyflow.archs import DNN

import sys
print('Arg list: ', str(sys.argv))

#percval = '40Perc'
#massval = '8000'
percval = str(sys.argv[1])
massval = str(sys.argv[2])

fi = np.load('/data0/wmccorma/'+percval+'Sig_'+massval+'MeV_synthsig_multifold.npz', allow_pickle=True)

avg_gen_glob = fi['X_gen_glob']
avg_gen_glob = (avg_gen_glob - np.mean(avg_gen_glob, axis=0))/np.std(avg_gen_glob, axis=0)
avg_det_glob = fi['X_det_glob']
avg_det_glob = (avg_det_glob - np.mean(avg_det_glob, axis=0))/np.std(avg_det_glob, axis=0)

X_gen = avg_gen_glob
X_det = avg_det_glob
Y_gen = fi['Y_gen']
Y_det = fi['Y_det']

itnum = 30

Phi_sizes = [200, 200, 256]
F_sizes = [200, 200, 200]

model_layer_sizes = [100, 100, 100]

#det_args = {'input_dim': 4, 'Phi_sizes': Phi_sizes, 'F_sizes': F_sizes, 'num_global_features': 3, 'patience': 10, 'filepath': '/data0/wmccorma/'+percval+'_'+massval+'_synthsig_Step1_{}', 'save_weights_only': False, 'modelcheck_opts': {'save_best_only': True, 'verbose': 1}}
#mc_args = {'input_dim': 4, 'Phi_sizes': Phi_sizes, 'F_sizes': F_sizes, 'num_global_features': 3, 'patience': 10, 'filepath': '/data0/wmccorma/'+percval+'_'+massval+'_synthsig_Step2_{}', 'save_weights_only': False, 'modelcheck_opts': {'save_best_only': True, 'verbose': 1}}

det_args = {'input_dim': len(X_gen[0]), 'dense_sizes': model_layer_sizes, 'patience': 10, 'filepath': '/data0/wmccorma/'+percval+'_'+massval+'_synthsig_multifold_Step1_{}', 'save_weights_only': False, 'modelcheck_opts': {'save_best_only': True, 'verbose': 1}}

mc_args = {'input_dim': len(X_gen[0]), 'dense_sizes': model_layer_sizes, 'patience': 10, 'filepath': '/data0/wmccorma/'+percval+'_'+massval+'_synthsig_multifold_Step2_{}', 'save_weights_only': False, 'modelcheck_opts': {'save_best_only': True, 'verbose': 1}}

fitargs = {'batch_size': 500, 'epochs': 20, 'verbose': 1}
#fitargs = {'batch_size': 10000, 'epochs': 120, 'restore_best_weights':True, 'verbose': 1}

# reweight the sim and data to have the same total weight to begin with
ndata, nsim = np.count_nonzero(Y_det[:,1]), np.count_nonzero(Y_det[:,0])
wdata = np.ones(ndata)
winit = ndata/nsim*np.ones(nsim)

multifold_ws = of.omnifold(X_gen, Y_gen, X_det, Y_det, wdata, winit, (ef.archs.dnn.DNN, det_args), (ef.archs.dnn.DNN, mc_args), fitargs, val=0.2, it=itnum, trw_ind=-2, weights_filename='/data0/wmccorma/Official_synthsig_multifold_'+percval+'_perc_'+massval+'_mass')
