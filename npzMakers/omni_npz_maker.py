import numpy as np
import energyflow as ef

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

a = np.load("/data0/users/pkomiske/OmniFoldSearch/Pythia21_ZJet.pickle", allow_pickle=True)
#print(a.files)
#b = np.load("/data0/users/pkomiske/OmniFoldSearch/Pythia21_ZJet_Obs.npz")
#print(b.files)

c = np.load("/data0/users/pkomiske/OmniFoldSearch/HZa16000MeV_ZJet.pickle", allow_pickle=True)
#print(c.files)
#d = np.load("/data0/users/pkomiske/OmniFoldSearch/HZa16000MeV_ZJet_Obs.npz")
#print(d.files)

# how many iterations of the unfolding process
itnum = 6

#obs_multifold = ['Mass', 'Mult', 'Width', 'Tau21', 'zg', 'SDMass']
#obs_multifold = ['Mass', 'Mult', 'zg', 'SDMass']



nevs = 1000000
#nbkg = 10000
nbkg = 950000
nsig = 50000


data = np.concatenate([c['sim_particles'][:nsig],a['sim_particles'][nevs:nevs+nbkg]])
sim = a['sim_particles'][:nevs]
gen = a['gen_particles'][:nevs]


sim_data_max_length = max(get_max_length(sim), get_max_length(data))

print(sim_data_max_length)

gen, sim = pad_events(gen), pad_events(sim, max_length=sim_data_max_length)
data = pad_events(data, max_length=sim_data_max_length)

global X_det, Y_det
X_det = (np.concatenate((data, sim), axis=0))
Y_det = ef.utils.to_categorical(np.concatenate((np.ones(len(data)), np.zeros(len(sim)))))

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

for x in X_gen:
    if(x[0][0] <= 0):
        continue
    mask = x[:,0] > 0
    yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
    x[mask,1:3] -= yphi_avg
    x[mask,0] /= x[:,0].sum()
    
ef.utils.remap_pids(X_gen, pid_i=3)


np.savez('5PercSig_1mil.npz', **{'X_det': X_det, 'Y_det': Y_det, 'X_gen': X_gen, 'Y_gen': Y_gen})
