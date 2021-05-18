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


a = np.load("/data0/users/pkomiske/OmniFoldSearch/Pythia21_ZJet.pickle", allow_pickle=True)

bkg_sim = a['sim_particles'][:600000]
bkg_gen = a['gen_particles'][:600000]

del a

a = np.load("/data0/users/pkomiske/OmniFoldSearch/Pythia21_ZJet.npz")

sjetsa = a['sim_jets'][:600000]
gjetsa = a['gen_jets'][:600000]
gZsa = a['gen_Zs'][:600000]

del a

b = np.load("/data0/users/pkomiske/OmniFoldSearch/Pythia21_ZJet_Obs.npz")
smultsb = b['sim_mults'][:600000]
gmultsb = b['gen_mults'][:600000]

del b

sig_sim = []
sig_gen = []

sjetsc = []
gjetsc = []
gZsc = []

smultsd = []
gmultsd = []

masses = ['500', '1000', '2000', '4000', '8000', '16000']

for m in masses:
    c = np.load("/data0/users/pkomiske/OmniFoldSearch/HZa"+m+"MeV_not125_ZJet.pickle", allow_pickle=True)
    
    sig_sim.append(c['sim_particles'][:])
    sig_gen.append(c['gen_particles'][:])
    
    del c
    
    c = np.load("/data0/users/pkomiske/OmniFoldSearch/HZa"+m+"MeV_not125_ZJet.npz")

    sjetsc.append(c['sim_jets'][:])
    gjetsc.append(c['gen_jets'][:])
    gZsc.append(c['gen_Zs'][:])

    del c
    
    d = np.load("/data0/users/pkomiske/OmniFoldSearch/HZa"+m+"MeV_not125_ZJet_Obs.npz")

    smultsd.append(d['sim_mults'][:])
    gmultsd.append(d['gen_mults'][:])

    del d

for m in range(len(masses)):
    numdel = 0
    i=0
    while(i<len(sig_sim[m])):
        if( (abs(np.sum(sig_gen[m][i])) < 0.001) or (abs(np.sum(sig_sim[m][i])) < 0.001) or (smultsd[m][i] == 0) or (gmultsd[m][i] == 0)):
            sig_sim[m] = np.delete(sig_sim[m],i,0)
            sig_gen[m] = np.delete(sig_gen[m],i,0)
            sjetsc[m] = np.delete(sjetsc[m],i,0)
            gjetsc[m] = np.delete(gjetsc[m],i,0)
            gZsc[m] = np.delete(gZsc[m],i,0)
            smultsd[m] = np.delete(smultsd[m],i,0)
            gmultsd[m] = np.delete(gmultsd[m],i,0)
            numdel+=1
        else:
            i+=1
    print(numdel)

i=0
numdel = 0
while(i<len(bkg_sim)):
    if( (abs(np.sum(bkg_gen[i])) < 0.001) or (abs(np.sum(bkg_sim[i])) < 0.001) or (smultsb[i] == 0) or (gmultsb[i] == 0)):
        bkg_sim = np.delete(bkg_sim,i,0)
        bkg_gen = np.delete(bkg_gen,i,0)
        sjetsa = np.delete(sjetsa,i,0)
        gjetsa = np.delete(gjetsa,i,0)
        gZsa = np.delete(gZsa,i,0)
        smultsb = np.delete(smultsb,i,0)
        gmultsb = np.delete(gmultsb,i,0)
        numdel+=1
    else:
        i+=1
print(numdel)

invmaT = []
print(len(gjetsa))
for i in range(len(gjetsa)):
    ej = np.sqrt(gjetsa[i][3]*gjetsa[i][3] + gjetsa[i][0]*np.cosh(gjetsa[i][1])*gjetsa[i][0]*np.cosh(gjetsa[i][1]))
    ez = np.sqrt(90.*90. + gZsa[i][0]*np.cosh(gZsa[i][1])*gZsa[i][0]*np.cosh(gZsa[i][1]))
    eb = ej+ez
    pxb = gjetsa[i][0]*np.cos(gjetsa[i][2]) + gZsa[i][0]*np.cos(gZsa[i][2])
    pyb = gjetsa[i][0]*np.sin(gjetsa[i][2]) + gZsa[i][0]*np.sin(gZsa[i][2])
    pzb = gjetsa[i][0]*np.sinh(gjetsa[i][1]) + gZsa[i][0]*np.sinh(gZsa[i][1])
    MB = np.sqrt(eb*eb - pxb*pxb - pyb*pyb - pzb*pzb)
    invmaT.append([MB,gjetsa[i][3],gmultsb[i]])

invmcT = []
for m in range(len(masses)):
    invmcTtmp = []
    print(len(gjetsc[m]))
    for i in range(len(gjetsc[m])):
        ej = np.sqrt(gjetsc[m][i][3]*gjetsc[m][i][3] + gjetsc[m][i][0]*np.cosh(gjetsc[m][i][1])*gjetsc[m][i][0]*np.cosh(gjetsc[m][i][1]))
        ez = np.sqrt(90.*90. + gZsc[m][i][0]*np.cosh(gZsc[m][i][1])*gZsc[m][i][0]*np.cosh(gZsc[m][i][1]))
        eb = ej+ez
        pxb = gjetsc[m][i][0]*np.cos(gjetsc[m][i][2]) + gZsc[m][i][0]*np.cos(gZsc[m][i][2])
        pyb = gjetsc[m][i][0]*np.sin(gjetsc[m][i][2]) + gZsc[m][i][0]*np.sin(gZsc[m][i][2])
        pzb = gjetsc[m][i][0]*np.sinh(gjetsc[m][i][1]) + gZsc[m][i][0]*np.sinh(gZsc[m][i][1])
        MB = np.sqrt(eb*eb - pxb*pxb - pyb*pyb - pzb*pzb)
        invmcTtmp.append([MB,gjetsc[m][i][3],gmultsd[m][i]])
    invmcT.append(invmcTtmp)
    
    
invma = []
print(len(sjetsa))
for i in range(len(sjetsa)):
    ej = np.sqrt(sjetsa[i][3]*sjetsa[i][3] + sjetsa[i][0]*np.cosh(sjetsa[i][1])*sjetsa[i][0]*np.cosh(sjetsa[i][1]))
    ez = np.sqrt(90.*90. + gZsa[i][0]*np.cosh(gZsa[i][1])*gZsa[i][0]*np.cosh(gZsa[i][1]))
    eb = ej+ez
    pxb = sjetsa[i][0]*np.cos(sjetsa[i][2]) + gZsa[i][0]*np.cos(gZsa[i][2])
    pyb = sjetsa[i][0]*np.sin(sjetsa[i][2]) + gZsa[i][0]*np.sin(gZsa[i][2])
    pzb = sjetsa[i][0]*np.sinh(sjetsa[i][1]) + gZsa[i][0]*np.sinh(gZsa[i][1])
    MB = np.sqrt(eb*eb - pxb*pxb - pyb*pyb - pzb*pzb)
    invma.append([MB,sjetsa[i][3],smultsb[i]])

invmc = []
for m in range(len(masses)):
    invmctmp = []
    print(len(sjetsc[m]))
    for i in range(len(sjetsc[m])):
        ej = np.sqrt(sjetsc[m][i][3]*sjetsc[m][i][3] + sjetsc[m][i][0]*np.cosh(sjetsc[m][i][1])*sjetsc[m][i][0]*np.cosh(sjetsc[m][i][1]))
        ez = np.sqrt(90.*90. + gZsc[m][i][0]*np.cosh(gZsc[m][i][1])*gZsc[m][i][0]*np.cosh(gZsc[m][i][1]))
        eb = ej+ez
        pxb = sjetsc[m][i][0]*np.cos(sjetsc[m][i][2]) + gZsc[m][i][0]*np.cos(gZsc[m][i][2])
        pyb = sjetsc[m][i][0]*np.sin(sjetsc[m][i][2]) + gZsc[m][i][0]*np.sin(gZsc[m][i][2])
        pzb = sjetsc[m][i][0]*np.sinh(sjetsc[m][i][1]) + gZsc[m][i][0]*np.sinh(gZsc[m][i][1])
        MB = np.sqrt(eb*eb - pxb*pxb - pyb*pyb - pzb*pzb)
        invmctmp.append([MB,sjetsc[m][i][3],smultsd[m][i]])
    invmc.append(invmctmp)
    


nevs = 200000

perc = 200000*float(sys.argv[2])*.01
print(int(perc))
iperc = int(perc)


nbkg = 200000 - iperc
nsig = iperc

begin = 10000

data = np.concatenate([sig_sim[int(sys.argv[1])][:nsig],bkg_sim[nevs:nevs+nbkg]])
sim = np.concatenate([bkg_sim[:nevs]])
gen = np.concatenate([bkg_gen[:nevs]])

dataT = np.concatenate([sig_gen[int(sys.argv[1])][:nsig],bkg_gen[nevs:nevs+nbkg]])
simT = np.concatenate([ bkg_gen[:nevs]])


Mdata = np.concatenate([invmc[int(sys.argv[1])][:nsig],invma[nevs:nevs+nbkg]])
Msim = np.concatenate([invma[:nevs]])
Mgen = np.concatenate([invmaT[:nevs]])

MdataT = np.concatenate([invmcT[int(sys.argv[1])][:nsig],invmaT[nevs:nevs+nbkg]])
MsimT = np.concatenate([invmaT[:nevs]])



sim_data_max_length = max(get_max_length(sim), get_max_length(data))
simT_dataT_max_length = max(get_max_length(simT), get_max_length(dataT))

print(sim_data_max_length)
print(simT_dataT_max_length)

gen, sim = pad_events(gen), pad_events(sim, max_length=sim_data_max_length)
data = pad_events(data, max_length=sim_data_max_length)

simT = pad_events(simT, max_length=simT_dataT_max_length)
dataT = pad_events(dataT, max_length=simT_dataT_max_length)



global tmpX_det, Y_det
tmpX_det = (np.concatenate((data, sim), axis=0))
Y_det = ef.utils.to_categorical(np.concatenate((np.ones(len(data)), np.zeros(len(sim)))))

#print('X_det[12] ', X_det[12])

global tmpX_detT, Y_detT
tmpX_detT = (np.concatenate((dataT, simT), axis=0))
Y_detT = ef.utils.to_categorical(np.concatenate((np.ones(len(dataT)), np.zeros(len(simT)))))

global tmpX_gen, Y_gen
tmpX_gen = (np.concatenate((gen, gen)))
Y_gen = ef.utils.to_categorical(np.concatenate((np.ones(len(gen)), np.zeros(len(gen)))))

for x in tmpX_det:
    if(x[0][0] <= 0):
        continue
    mask = x[:,0] > 0
    yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
    x[mask,1:3] -= yphi_avg
    x[mask,0] /= x[:,0].sum()
ef.utils.remap_pids(tmpX_det, pid_i=3)

        
for x in tmpX_gen:
    if(x[0][0] <= 0):
        continue
    mask = x[:,0] > 0
    yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
    x[mask,1:3] -= yphi_avg
    x[mask,0] /= x[:,0].sum()
    
ef.utils.remap_pids(tmpX_gen, pid_i=3)


for x in tmpX_detT:
    if(x[0][0] <= 0):
        continue
    mask = x[:,0] > 0
    yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
    x[mask,1:3] -= yphi_avg
    x[mask,0] /= x[:,0].sum()

ef.utils.remap_pids(tmpX_detT, pid_i=3)
    
print(tmpX_det.shape, tmpX_gen.shape, tmpX_detT.shape)


global MtmpX_det
MtmpX_det = (np.concatenate((Mdata, Msim), axis=0))

global MtmpX_detT
MtmpX_detT = (np.concatenate((MdataT, MsimT), axis=0))

global MtmpX_gen
MtmpX_gen = (np.concatenate((Mgen, Mgen)))



X_det = [tmpX_det, MtmpX_det]
X_detT = [tmpX_detT, MtmpX_detT]
X_gen = [tmpX_gen, MtmpX_gen]


percname = int(float(sys.argv[2]))

#np.savez('/data0/users/wmccorma/'+str(percname)+'PercSig_'+str(sys.argv[1])+'MeV_synthsig.npz', **{'X_det': X_det, 'Y_det': Y_det, 'X_gen': X_gen, 'Y_gen': Y_gen, 'X_detT': X_detT, 'Y_detT': Y_detT})
np.savez('/data0/users/wmccorma/'+str(percname)+'PercSig_'+str(sys.argv[1])+'MeV_smOnly_not125.npz', **{'X_det_part': tmpX_det, 'Y_det': Y_det, 'X_gen_part': tmpX_gen, 'Y_gen': Y_gen, 'X_detT_part': tmpX_detT, 'Y_detT': Y_detT, 'X_det_glob': MtmpX_det, 'X_gen_glob': MtmpX_gen, 'X_detT_glob': MtmpX_detT})
