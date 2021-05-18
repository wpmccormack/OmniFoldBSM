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

snsubsb = b['sim_nsubs'][:600000]
gnsubsb = b['gen_nsubs'][:600000]

ssdmsb = b['sim_sdms'][:600000]
gsdmsb = b['gen_sdms'][:600000]

szgsb = b['sim_zgs'][:600000]
gzgsb = b['gen_zgs'][:600000]

sn95sb = b['sim_n95s'][:600000]
gn95sb = b['gen_n95s'][:600000]


del b

sig_sim = []
sig_gen = []

sjetsc = []
gjetsc = []
gZsc = []

smultsd = []
gmultsd = []

snsubsd = []
gnsubsd = []

ssdmsd = []
gsdmsd = []

szgsd = []
gzgsd = []

sn95sd = []
gn95sd = []

masses = ['500', '1000', '2000', '4000', '8000', '16000']

maxlen = 50000

for m in masses:
    c = np.load("/data0/users/pkomiske/OmniFoldSearch/HZa"+m+"MeV_ZJet.pickle", allow_pickle=True)
    
    sig_sim.append(c['sim_particles'][:maxlen])
    sig_gen.append(c['gen_particles'][:maxlen])
    
    del c
    
    c = np.load("/data0/users/pkomiske/OmniFoldSearch/HZa"+m+"MeV_ZJet.npz")

    sjetsc.append(c['sim_jets'][:maxlen])
    gjetsc.append(c['gen_jets'][:maxlen])
    gZsc.append(c['gen_Zs'][:maxlen])

    del c
    
    d = np.load("/data0/users/pkomiske/OmniFoldSearch/HZa"+m+"MeV_ZJet_Obs.npz")

    smultsd.append(d['sim_mults'][:maxlen])
    gmultsd.append(d['gen_mults'][:maxlen])

    snsubsd.append(d['sim_nsubs'][:maxlen])
    gnsubsd.append(d['gen_nsubs'][:maxlen])

    ssdmsd.append(d['sim_sdms'][:maxlen])
    gsdmsd.append(d['gen_sdms'][:maxlen])

    szgsd.append(d['sim_zgs'][:maxlen])
    gzgsd.append(d['gen_zgs'][:maxlen])

    sn95sd.append(d['sim_n95s'][:maxlen])
    gn95sd.append(d['gen_n95s'][:maxlen])
    
    del d


for m in masses:
    c = np.load("/data0/users/pkomiske/OmniFoldSearch/HZa"+m+"MeV_not125_ZJet.pickle", allow_pickle=True)
    
    sig_sim.append(c['sim_particles'][:maxlen])
    sig_gen.append(c['gen_particles'][:maxlen])
    
    del c
    
    c = np.load("/data0/users/pkomiske/OmniFoldSearch/HZa"+m+"MeV_not125_ZJet.npz")

    sjetsc.append(c['sim_jets'][:maxlen])
    gjetsc.append(c['gen_jets'][:maxlen])
    gZsc.append(c['gen_Zs'][:maxlen])

    del c
    
    d = np.load("/data0/users/pkomiske/OmniFoldSearch/HZa"+m+"MeV_not125_ZJet_Obs.npz")

    smultsd.append(d['sim_mults'][:maxlen])
    gmultsd.append(d['gen_mults'][:maxlen])

    snsubsd.append(d['sim_nsubs'][:maxlen])
    gnsubsd.append(d['gen_nsubs'][:maxlen])

    ssdmsd.append(d['sim_sdms'][:maxlen])
    gsdmsd.append(d['gen_sdms'][:maxlen])

    szgsd.append(d['sim_zgs'][:maxlen])
    gzgsd.append(d['gen_zgs'][:maxlen])

    sn95sd.append(d['sim_n95s'][:maxlen])
    gn95sd.append(d['gen_n95s'][:maxlen])
    
    del d

for m in range(2*len(masses)):
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
            snsubsd[m] = np.delete(snsubsd[m],i,0)
            gnsubsd[m] = np.delete(gnsubsd[m],i,0)
            ssdmsd[m] = np.delete(ssdmsd[m],i,0)
            gsdmsd[m] = np.delete(gsdmsd[m],i,0)
            szgsd[m] = np.delete(szgsd[m],i,0)
            gzgsd[m] = np.delete(gzgsd[m],i,0)
            sn95sd[m] = np.delete(sn95sd[m],i,0)
            gn95sd[m] = np.delete(gn95sd[m],i,0)
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
        snsubsb = np.delete(snsubsb,i,0)
        gnsubsb = np.delete(gnsubsb,i,0)
        ssdmsb = np.delete(ssdmsb,i,0)
        gsdmsb = np.delete(gsdmsb,i,0)
        szgsb = np.delete(szgsb,i,0)
        gzgsb = np.delete(gzgsb,i,0)
        sn95sb = np.delete(sn95sb,i,0)
        gn95sb = np.delete(gn95sb,i,0)
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
    invmaT.append([MB,gjetsa[i][3],gmultsb[i],gjetsa[i][0],gZsa[i][0],gnsubsb[i][0],gnsubsb[i][1],gnsubsb[i][2],gsdmsb[i][0],gzgsb[i][0],gn95sb[i]])

invmcT = []
for m in range(2*len(masses)):
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
        invmcTtmp.append([MB,gjetsc[m][i][3],gmultsd[m][i],gjetsc[m][i][0],gZsc[m][i][0],gnsubsd[m][i][0],gnsubsd[m][i][1],gnsubsd[m][i][2],gsdmsd[m][i][0],gzgsd[m][i][0],gn95sd[m][i]])
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
    invma.append([MB,sjetsa[i][3],smultsb[i],sjetsa[i][0],gZsa[i][0],snsubsb[i][0],snsubsb[i][1],snsubsb[i][2],ssdmsb[i][0],szgsb[i][0],sn95sb[i]])

invmc = []
for m in range(2*len(masses)):
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
        invmctmp.append([MB,sjetsc[m][i][3],smultsd[m][i],sjetsc[m][i][0],gZsc[m][i][0],snsubsd[m][i][0],snsubsd[m][i][1],snsubsd[m][i][2],ssdmsd[m][i][0],szgsd[m][i][0],sn95sd[m][i]])
    invmc.append(invmctmp)
    


nevs = 200000

perc = 200000*float(sys.argv[2])*.01
print(int(perc))
iperc = int(perc)


nbkg = 200000 - iperc
nsig = iperc

begin = 10000

#data = np.concatenate([sig_sim[int(sys.argv[1])][:nsig],bkg_sim[nevs:nevs+nbkg]])
#sim = np.concatenate([sig_sim[0][-begin:], sig_sim[1][-begin:], sig_sim[2][-begin:], sig_sim[3][-begin:], sig_sim[4][-begin:], sig_sim[5][-begin:], bkg_sim[:nevs]])
#gen = np.concatenate([sig_gen[0][-begin:], sig_gen[1][-begin:], sig_gen[2][-begin:], sig_gen[3][-begin:], sig_gen[4][-begin:], sig_gen[5][-begin:], bkg_gen[:nevs]])

#dataT = np.concatenate([sig_gen[int(sys.argv[1])][:nsig],bkg_gen[nevs:nevs+nbkg]])
#simT = np.concatenate([sig_gen[0][-begin:], sig_gen[1][-begin:], sig_gen[2][-begin:], sig_gen[3][-begin:], sig_gen[4][-begin:], sig_gen[5][-begin:], bkg_gen[:nevs]])


Mdata = np.concatenate([invmc[int(sys.argv[1])+6][:nsig],invma[nevs:nevs+nbkg]])
Msim = np.concatenate([invmc[0][-begin:], invmc[1][-begin:], invmc[2][-begin:], invmc[3][-begin:], invmc[4][-begin:], invmc[5][-begin:], invma[:nevs]])
Mgen = np.concatenate([invmcT[0][-begin:], invmcT[1][-begin:], invmcT[2][-begin:], invmcT[3][-begin:], invmcT[4][-begin:], invmcT[5][-begin:], invmaT[:nevs]])

MdataT = np.concatenate([invmcT[int(sys.argv[1])+6][:nsig],invmaT[nevs:nevs+nbkg]])
MsimT = np.concatenate([invmcT[0][-begin:], invmcT[1][-begin:], invmcT[2][-begin:], invmcT[3][-begin:], invmcT[4][-begin:], invmcT[5][-begin:], invmaT[:nevs]])



#sim_data_max_length = max(get_max_length(sim), get_max_length(data))
#simT_dataT_max_length = max(get_max_length(simT), get_max_length(dataT))

#print(sim_data_max_length)
#print(simT_dataT_max_length)

#gen, sim = pad_events(gen), pad_events(sim, max_length=sim_data_max_length)
#data = pad_events(data, max_length=sim_data_max_length)

#simT = pad_events(simT, max_length=simT_dataT_max_length)
#dataT = pad_events(dataT, max_length=simT_dataT_max_length)



global tmpX_det, Y_det
#tmpX_det = (np.concatenate((data, sim), axis=0))
Y_det = ef.utils.to_categorical(np.concatenate((np.ones(len(Mdata)), np.zeros(len(Msim)))))

#print('X_det[12] ', X_det[12])

global tmpX_detT, Y_detT
#tmpX_detT = (np.concatenate((dataT, simT), axis=0))
Y_detT = ef.utils.to_categorical(np.concatenate((np.ones(len(MdataT)), np.zeros(len(MsimT)))))

global tmpX_gen, Y_gen
#tmpX_gen = (np.concatenate((gen, gen)))
Y_gen = ef.utils.to_categorical(np.concatenate((np.ones(len(Mgen)), np.zeros(len(Mgen)))))
"""
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
"""

global MtmpX_det
MtmpX_det = (np.concatenate((Mdata, Msim), axis=0))

global MtmpX_detT
MtmpX_detT = (np.concatenate((MdataT, MsimT), axis=0))

global MtmpX_gen
MtmpX_gen = (np.concatenate((Mgen, Mgen)))


#X_det = [tmpX_det, MtmpX_det]
#X_detT = [tmpX_detT, MtmpX_detT]
#X_gen = [tmpX_gen, MtmpX_gen]


percname = int(float(sys.argv[2]))


np.savez('/data0/users/wmccorma/'+str(percname)+'PercSig_'+str(sys.argv[1])+'MeV_synthsigOnly125_multifold_not125_corrData.npz', **{'Y_det': Y_det, 'Y_gen': Y_gen, 'Y_detT': Y_detT, 'X_det_glob': MtmpX_det, 'X_gen_glob': MtmpX_gen, 'X_detT_glob': MtmpX_detT})
