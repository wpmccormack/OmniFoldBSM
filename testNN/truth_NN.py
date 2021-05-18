import numpy as np
import energyflow as ef

import OmniFold.omnifold_glob as of
from energyflow.archs import PFN
from energyflow.archs import DNN

fi = np.load('16000_SLIMTRUTH_MULTIFOLD.npz', allow_pickle=True)

X_det = fi['X_det'][:]
Y_det = fi['Y_det'][:]

print('shapes')
print(X_det.shape)
print(Y_det.shape)

Phi_sizes = [100, 100, 256]
F_sizes = [100, 100, 100]

model_layer_sizes = [100, 100, 100]

latent_dropout = .2
F_dropouts = .2

#Phi_acts = ['relu', 'tanh',  'sigmoid']
#F_acts = ['relu', 'tanh',  'sigmoid']
Phi_acts = 'relu'
F_acts = 'relu'

Phi_k_inits = 'ones'
F_k_inits = 'ones'

Phi_l2_regs = .05
F_l2_regs = .05

det_args = {'input_dim': 11, 'dense_sizes': model_layer_sizes, 'patience': 10, 'filepath': '/data0/wmccorma/16000_SD_150_Feb_3_multifold.h5', 'save_weights_only': False, 'modelcheck_opts': {'save_best_only': True, 'verbose': 1}}

#det_args = {'input_dim': len(X_gen[0]), 'dense_sizes': model_layer_sizes, 'patience': 10, 'filepath': '/data0/wmccorma/'+percval+'_'+massval+'_synthsig_multifold_Step1_{}', 'save_weights_only': False, 'modelcheck_opts': {'save_best_only': True, 'verbose': 1}}

fitargs = {'batch_size': 1000, 'epochs': 45, 'verbose': 1}

# reweight the sim and data to have the same total weight to begin with                                                                            
ndata, nsim = np.count_nonzero(Y_det[:,1]), np.count_nonzero(Y_det[:,0])
wdata = np.ones(ndata)
winit = ndata/nsim*np.ones(nsim)

model_det = ef.archs.dnn.DNN(**det_args)

perm_det = np.random.permutation(len(X_det))

train_frac = int(.7*len(X_det))

X = X_det[perm_det[:train_frac]]
Y = Y_det[perm_det[:train_frac]]

X_val = X_det[perm_det[train_frac:]]
Y_val = Y_det[perm_det[train_frac:]]

w = np.concatenate((wdata, winit))
w_train, w_val = w[perm_det[:train_frac]], w[perm_det[train_frac:]]

hist = model_det.fit(X, Y, sample_weight=w_train, **fitargs, validation_data=(X_val, Y_val, w_val))

model_json = model_det.to_json()
with open("/data0/wmccorma/16000_SB_45_Feb_3_multifold.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_det.save_weights("/data0/wmccorma/16000_SB_45_Feb_3_multifold_labels.h5")
print("Saved model to disk")

preds = model_det.predict(X_det, batch_size = 10000)

np.savez('16000_SB_45_Feb_3_multifold_preds.npz', **{'preds': preds})

import sklearn

sklearn.metrics.roc_auc_score(Y_det, preds, sample_weight=w)
