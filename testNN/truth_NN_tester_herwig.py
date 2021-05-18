import numpy as np
import energyflow as ef

import OmniFold.omnifold as of
from energyflow.archs import PFN
from energyflow.archs import DNN

from tensorflow.keras.models import model_from_json
# load json and create model
json_file = open('/data0/wmccorma/16000_SB_150_Jan_27.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/data0/wmccorma/16000_SB_150_Jan_27_labels.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

fi = np.load('16000_SLIMTRUTH.npz', allow_pickle=True)
X_det = fi['X_det'][:]
Y_det = fi['Y_det'][:]

fi2 = np.load('/data0/wmccorma/0PercSig_5MeV_Herwig_DataWithBSM_synthsig.npz', allow_pickle=True)

X_det2 = fi2['X_detT_part'][:200000]
X_det3 = fi2['X_gen_part'][:260000]

Y_det2 = ef.utils.to_categorical(np.concatenate((np.ones(2000), np.zeros(198000))))

preds = loaded_model.predict(X_det, batch_size = 10000)

preds2 = loaded_model.predict(X_det2, batch_size = 10000)

preds3 = loaded_model.predict(X_det3, batch_size = 10000)

import sklearn

fpr, tpr, th = sklearn.metrics.roc_curve(Y_det[:, 1], preds[:, 1])
roc_auc = sklearn.metrics.auc(fpr, tpr)
print(roc_auc)

fpr2, tpr2, th2 = sklearn.metrics.roc_curve(Y_det2[:, 1], preds2[:, 1])
roc_auc2 = sklearn.metrics.auc(fpr2, tpr2)
print(roc_auc2)


#print(th)
#print(np.divide(tpr[:],np.sqrt(fpr[:])))

sif = np.divide(tpr[:],np.sqrt(fpr[:]))

sif2 = np.divide(tpr2[:],np.sqrt(fpr2[:]))

np.savez('16000MeV_0Perc_SB_150_Jan_27_preds_HerwigDATA.npz', **{'preds_half_true': preds, 'fpr_half_true': fpr, 'tpr_half_true': tpr, 'th_half_true': th, 'sif_half_true': sif, 'preds_data': preds2, 'fpr_data': fpr, 'tpr_data': tpr, 'th_data': th, 'sif_data': sif, 'preds_gen': preds3})
