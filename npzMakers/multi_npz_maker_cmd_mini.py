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

#a = np.load("/data0/users/pkomiske/OmniFoldSearch/Pythia21_ZJet.pickle", allow_pickle=True)
a = np.load("/data0/users/pkomiske/OmniFoldSearch/Pythia21_ZJet.npz")

sjetsa = a['sim_jets'][:500000]
gjetsa = a['gen_jets'][:500000]
gZsa = a['gen_Zs'][:500000]

del a

b = np.load("/data0/users/pkomiske/OmniFoldSearch/Pythia21_ZJet_Obs.npz")
smultsb = b['sim_mults'][:500000]
gmultsb = b['gen_mults'][:500000]

del b

#c = np.load("/data0/users/pkomiske/OmniFoldSearch/HZa16000MeV_ZJet.pickle", allow_pickle=True)
c = np.load("/data0/users/pkomiske/OmniFoldSearch/HZa"+str(sys.argv[1])+"MeV_ZJet.npz")

sjetsc = c['sim_jets'][:]
gjetsc = c['gen_jets'][:]
gZsc = c['gen_Zs'][:]

del c

d = np.load("/data0/users/pkomiske/OmniFoldSearch/HZa"+str(sys.argv[1])+"MeV_ZJet_Obs.npz")

smultsd = d['sim_mults'][:]
gmultsd = d['gen_mults'][:]

del d


numdel = 0
i=0
while(i<len(sjetsa)):
    if( (smultsb[i] == 0) or (gmultsb[i] == 0) ):
        sjetsa = np.delete(sjetsa,i,0)
        gjetsa = np.delete(gjetsa,i,0)
        gZsa = np.delete(gZsa,i,0)
        smultsb = np.delete(smultsb,i,0)
        gmultsb = np.delete(gmultsb,i,0)
        numdel+=1
    else:
        i+=1
print(numdel)
numdel = 0

i=0
while(i<len(sjetsc)):
    if( (smultsd[i] == 0) or (gmultsd[i] == 0) ):
        sjetsc = np.delete(sjetsc,i,0)
        gjetsc = np.delete(gjetsc,i,0)
        gZsc = np.delete(gZsc,i,0)
        smultsd = np.delete(smultsd,i,0)
        gmultsd = np.delete(gmultsd,i,0)
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
    invmaT.append([MB,gjetsa[i][3]])

invmcT = []
print(len(gjetsc))
for i in range(len(gjetsc)):
    ej = np.sqrt(gjetsc[i][3]*gjetsc[i][3] + gjetsc[i][0]*np.cosh(gjetsc[i][1])*gjetsc[i][0]*np.cosh(gjetsc[i][1]))
    ez = np.sqrt(90.*90. + gZsc[i][0]*np.cosh(gZsc[i][1])*gZsc[i][0]*np.cosh(gZsc[i][1]))
    eb = ej+ez
    pxb = gjetsc[i][0]*np.cos(gjetsc[i][2]) + gZsc[i][0]*np.cos(gZsc[i][2])
    pyb = gjetsc[i][0]*np.sin(gjetsc[i][2]) + gZsc[i][0]*np.sin(gZsc[i][2])
    pzb = gjetsc[i][0]*np.sinh(gjetsc[i][1]) + gZsc[i][0]*np.sinh(gZsc[i][1])
    MB = np.sqrt(eb*eb - pxb*pxb - pyb*pyb - pzb*pzb)
    invmcT.append([MB,gjetsc[i][3]])


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
    invma.append([MB,sjetsa[i][3]])

invmc = []
print(len(sjetsc))
for i in range(len(sjetsc)):
    ej = np.sqrt(sjetsc[i][3]*sjetsc[i][3] + sjetsc[i][0]*np.cosh(sjetsc[i][1])*sjetsc[i][0]*np.cosh(sjetsc[i][1]))
    ez = np.sqrt(90.*90. + gZsc[i][0]*np.cosh(gZsc[i][1])*gZsc[i][0]*np.cosh(gZsc[i][1]))
    eb = ej+ez
    pxb = sjetsc[i][0]*np.cos(sjetsc[i][2]) + gZsc[i][0]*np.cos(gZsc[i][2])
    pyb = sjetsc[i][0]*np.sin(sjetsc[i][2]) + gZsc[i][0]*np.sin(gZsc[i][2])
    pzb = sjetsc[i][0]*np.sinh(sjetsc[i][1]) + gZsc[i][0]*np.sinh(gZsc[i][1])
    MB = np.sqrt(eb*eb - pxb*pxb - pyb*pyb - pzb*pzb)
    invmc.append([MB,sjetsc[i][3]])

nevs = 200000

perc = 200000*float(sys.argv[2])*.01
print(int(perc))
iperc = int(perc)


nbkg = 200000 - iperc
nsig = iperc


data = np.concatenate([invmc[:nsig],invma[nevs:nevs+nbkg]])
sim = invma[:nevs]
gen = invmaT[:nevs]

dataT = np.concatenate([invmcT[:nsig],invmaT[nevs:nevs+nbkg]])
simT = invmaT[:nevs]

print(data[12])

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
    
percname = int(float(sys.argv[2])*100.)

np.savez('/data0/users/wmccorma/'+str(percname)+'PercSig_ONLY_INVMASS_'+str(sys.argv[1])+'MeV.npz', **{'X_det': X_det, 'Y_det': Y_det, 'X_gen': X_gen, 'Y_gen': Y_gen, 'X_detT': X_detT, 'Y_detT': Y_detT})
