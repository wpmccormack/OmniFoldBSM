import numpy as np
import energyflow as ef

import sys
print('Arg list: ', str(sys.argv))

def pad_events(events, val=0, max_length=None):
    event_lengths = []
    for i in range(len(events)):
        if(len(events[i])>0):
            event_lengths.append(events[i].shape[0])
        else:
            event_lengths.append(0)
    #event_lengths = [event.shape[0] for event in events]
    if max_length is None:
        max_length = max(event_lengths)
    output = []
    for i in range(len(events)):
        if(len(events[i])>0):
            output.append( np.vstack((events[i], val*np.ones((max_length - len(events[i]), 4)))) )
        else:
            output.append(val*np.ones((max_length, 4)))
    output = np.asarray(output)
    return output
#return np.asarray([np.vstack((event, val*np.ones((max_length - ev_len, event.shape[1]))))
#                       for event,ev_len in zip(events, event_lengths)])

def get_max_length(events):
    c = 0
    for i in range(len(events)):
        if(len(events[i])>0):
            if(events[i].shape[0] > c):
                c = events[i].shape[0]
    return c

print("/data0/users/pkomiske/OmniFoldSearch/HZa"+str(sys.argv[1])+"MeV_ZJet.pickle")

#a = [1,2]
#c = [3,4]

a = np.load("/data0/users/pkomiske/OmniFoldSearch/Pythia21_ZJet.pickle", allow_pickle=True)

bkg_sim = a['sim_particles'][:]
bkg_gen = a['gen_particles'][:]

del a

#print(a.files)
#b = np.load("/data0/users/pkomiske/OmniFoldSearch/Pythia21_ZJet_Obs.npz")
#print(b.files)

#c = np.load("/data0/users/pkomiske/OmniFoldSearch/HZa16000MeV_ZJet.pickle", allow_pickle=True)
c = np.load("/data0/users/pkomiske/OmniFoldSearch/HZa"+str(sys.argv[1])+"MeV_ZJet.pickle", allow_pickle=True)

sig_sim = c['sim_particles'][:]
sig_gen = c['gen_particles'][:]

del c

#print(c.files)
#d = np.load("/data0/users/pkomiske/OmniFoldSearch/HZa16000MeV_ZJet_Obs.npz")
#print(d.files)

# how many iterations of the unfolding process
itnum = 6

#obs_multifold = ['Mass', 'Mult', 'Width', 'Tau21', 'zg', 'SDMass']
#obs_multifold = ['Mass', 'Mult', 'zg', 'SDMass']
#print(c['sim_particles'][0])
#print(abs(np.sum(c['sim_particles'][0])))
#print(c['gen_particles'][20771])
#for i in range(10):
#    print(c['sim_particles'][i])

#print(len(c['sim_particles']), len(c['gen_particles']))
#print(c['sim_particles'][12])
#print(c['gen_particles'][12])
#np.delete(c['sim_particles'],12)
#np.delete(c['gen_particles'],12)
#print(len(c['sim_particles']), len(c['gen_particles']))
#print(c['sim_particles'][11])
#print(c['sim_particles'][13])

"""
nozeros = False
passnum = 0
while(not nozeros):
    #print('pass ', passnum)
    lenval = len(sig_sim)
    for i in range(len(sig_sim)):
        #if( (abs(np.sum(sig_sim[i])) < 0.001) or (abs(np.sum(sig_gen[i])) < 0.001) ):
        if( (abs(np.sum(sig_gen[i])) < 0.001) ):
            #print('found empty event', i, np.sum(sig_sim[i]), ' ', np.sum(sig_gen[i]))
            #print(sig_sim[i])
            #print(sig_gen[i])
            sig_sim = np.delete(sig_sim,i,0)
            sig_gen = np.delete(sig_gen,i,0)
            break
    if( i == (lenval-1)):
        nozeros = True


nozeros = False
passnum = 0
while(not nozeros):
    #print('pass ', passnum)
    lenval = len(bkg_sim)
    for i in range(len(bkg_sim)):
        #if( (abs(np.sum(bkg_sim[i])) < 0.001) or (abs(np.sum(bkg_gen[i])) < 0.001) ):
        if( (abs(np.sum(bkg_gen[i])) < 0.001) ):
            #print('found empty event', i)
            bkg_sim = np.delete(bkg_sim,i,0)
            bkg_gen = np.delete(bkg_gen,i,0)
            break
    if( i == (lenval-1)):
        nozeros = True
"""
numdel = 0
i=0
#for i in range(len(sig_sim)):

while(i<len(sig_sim)):
    #if( (abs(np.sum(sig_sim[i])) < 0.001) or (abs(np.sum(sig_gen[i])) < 0.001) ):
    if( (abs(np.sum(sig_gen[i])) < 0.001) or (abs(np.sum(sig_sim[i])) < 0.001) ):
        #print('found empty event', i, np.sum(sig_sim[i]), ' ', np.sum(sig_gen[i]))
        #print(sig_sim[i])
        #print(sig_gen[i])
        #print(i, sig_gen[i])
        sig_sim = np.delete(sig_sim,i,0)
        sig_gen = np.delete(sig_gen,i,0)
        numdel+=1
    else:
        i+=1
print(numdel)
i=0
numdel = 0
while(i<len(bkg_sim)):
    #if( (abs(np.sum(bkg_sim[i])) < 0.001) or (abs(np.sum(bkg_gen[i])) < 0.001) ):
    if( (abs(np.sum(bkg_gen[i])) < 0.001) or (abs(np.sum(bkg_sim[i])) < 0.001) ):
        #print('found empty event', i, np.sum(bkg_sim[i]), ' ', np.sum(bkg_gen[i]))
        #print(bkg_sim[i])
        #print(bkg_gen[i])
        #print(i, bkg_gen[i])
        #print(bkg_sim[i])
        bkg_sim = np.delete(bkg_sim,i,0)
        bkg_gen = np.delete(bkg_gen,i,0)
        numdel+=1
    else:
        i+=1
print(numdel)

nevs = 200000
#nbkg = 10000

perc = 200000*float(sys.argv[2])*.01
print(int(perc))
iperc = int(perc)


nbkg = 200000 - iperc
nsig = iperc


#data = np.concatenate([c['sim_particles'][:nsig],a['sim_particles'][nevs:nevs+nbkg]])
#sim = a['sim_particles'][:nevs]
#gen = a['gen_particles'][:nevs]

print('sig_sim len ', len(sig_sim), ' bkg_sim len ', len(bkg_sim))
print('sig_gen len ', len(sig_gen), ' bkg_gen len ', len(bkg_gen))

data = np.concatenate([sig_sim[:nsig],bkg_sim[nevs:nevs+nbkg]])
sim = bkg_sim[:nevs]
gen = bkg_gen[:nevs]

dataT = np.concatenate([sig_gen[:nsig],bkg_gen[nevs:nevs+nbkg]])
simT = bkg_gen[:nevs]

print(data[12])

sim_data_max_length = max(get_max_length(sim), get_max_length(data))
simT_dataT_max_length = max(get_max_length(simT), get_max_length(dataT))

print(sim_data_max_length)

gen, sim = pad_events(gen), pad_events(sim, max_length=sim_data_max_length)
data = pad_events(data, max_length=sim_data_max_length)

simT = pad_events(simT, max_length=simT_dataT_max_length)
dataT = pad_events(dataT, max_length=simT_dataT_max_length)

#print(data[12])

global X_det, Y_det
X_det = (np.concatenate((data, sim), axis=0))
Y_det = ef.utils.to_categorical(np.concatenate((np.ones(len(data)), np.zeros(len(sim)))))

#print('X_det[12] ', X_det[12])

global X_detT, Y_detT
X_detT = (np.concatenate((dataT, simT), axis=0))
Y_detT = ef.utils.to_categorical(np.concatenate((np.ones(len(dataT)), np.zeros(len(simT)))))

global X_gen, Y_gen
X_gen = (np.concatenate((gen, gen)))
Y_gen = ef.utils.to_categorical(np.concatenate((np.ones(len(gen)), np.zeros(len(gen)))))

for x in X_det:
    #print(x)
    if(x[0][0] <= 0):
        continue
    mask = x[:,0] > 0
    yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
    x[mask,1:3] -= yphi_avg
    #print(x[:,0].sum())
    x[mask,0] /= x[:,0].sum()
    #print('done')
ef.utils.remap_pids(X_det, pid_i=3)

#print('X_det[12] again ', X_det[12])

numnull = 0
for i in range(len(X_det)):
    if(abs(X_det[i][0][0]) < 0.00001):
        numnull+=1
        #print(X_det[i])

print(numnull)
        
for x in X_gen:
    if(x[0][0] <= 0):
        continue
    mask = x[:,0] > 0
    yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
    x[mask,1:3] -= yphi_avg
    x[mask,0] /= x[:,0].sum()
    
ef.utils.remap_pids(X_gen, pid_i=3)


for x in X_detT:
    if(x[0][0] <= 0):
        continue
    mask = x[:,0] > 0
    yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
    x[mask,1:3] -= yphi_avg
    x[mask,0] /= x[:,0].sum()

print(X_det.shape, X_gen.shape, X_detT.shape)
    
ef.utils.remap_pids(X_detT, pid_i=3)
    
percname = int(float(sys.argv[2])*100.)

np.savez('/data0/users/wmccorma/'+str(percname)+'PercSig_small_'+str(sys.argv[1])+'MeV.npz', **{'X_det': X_det, 'Y_det': Y_det, 'X_gen': X_gen, 'Y_gen': Y_gen, 'X_detT': X_detT, 'Y_detT': Y_detT})
