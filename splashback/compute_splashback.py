import os

import numpy as np
import matplotlib.pyplot as plt

from colossus.halo import splashback
from colossus.cosmology import cosmology
import plotparams
plotparams.buba()

my_cosmo = {'flat': True, 'H0': 67.74, 'Om0': 0.3089, 'Ob0': 0.0486, 'sigma8': 0.815903, 'ns': 0.9667}
cosmo = cosmology.setCosmology('TNG_cosmo', my_cosmo)
#cosmo = cosmology.setCosmology('planck15')
G_N = 4.3009e-3 #pc/Msun (km/s)^2

#sim_type = 'MTNG'; fp_dm = 'fp'
sim_type = 'MTNG'; fp_dm = 'dm'
#sim_type = 'TNG'; fp_dm = 'dm'
#sim_type = 'TNG'; fp_dm = 'fp'
tng_dir_dic = {'TNG': "/mnt/gosling1/boryanah/TNG300/", 'MTNG': "/mnt/alan1/boryanah/MTNG/"}
Lbox_dic = {'TNG': 205., 'MTNG': 500.}
data_dir_dic = {'TNG': "data_tng/", 'MTNG': "data_mtng/"}
tng_dir_dic['MTNG'] = "/mnt/alan1/boryanah/MTNG/dm_arepo/"

# simulation params
tng_dir = tng_dir_dic[sim_type]
Lbox = Lbox_dic[sim_type]
data_dir = data_dir_dic[sim_type]
mdef = '200m'

#snapshot0 = 179; z0 = 1.0;
snapshot0 = 264; z0 = 0.0;

if (fp_dm == 'fp' or "arepo" in tng_dir) and sim_type == 'MTNG':
    snapshot0 += 0    
elif fp_dm == 'dm' and sim_type == 'MTNG':
    snapshot0 += 5

if sim_type == 'MTNG':
    Group_M_Mean200 = np.load(tng_dir+f'data_{fp_dm:s}/Group_M_Mean200_{fp_dm:s}_{snapshot0:d}.npy')*1.e10
    Group_R_Mean200 = np.load(tng_dir+f'data_{fp_dm:s}/Group_R_Mean200_{fp_dm:s}_{snapshot0:d}.npy')*1.e3
    #GroupGamma = np.load(tng_dir+f'data_{fp_dm:s}/GroupGamma_{snapshot0:d}_{snapshot1:d}_{fp_dm:s}.npy')
    GroupGamma = np.load(tng_dir+f'data_{fp_dm:s}/GroupGamma_{fp_dm:s}_{snapshot0:d}.npy')
    #Group_V_Mean200 = np.sqrt(G_N*Group_M_Mean200/(Group_R_Mean200*1.e6))
    #Group_V_Mean200[Group_R_Mean200 == 0.] = 0.
else:
    Group_M_Mean200 = np.load(tng_dir+f'Group_M_Mean200_{fp_dm:s}_{snapshot0:d}.npy')*1.e10 # Msun/h
    Group_R_Mean200 = np.load(tng_dir+f'Group_R_Mean200_{fp_dm:s}_{snapshot0:d}.npy') # ckpc/h
    GroupGamma = np.load(tng_dir+f'GroupGamma_{snapshot0:d}_{snapshot1:d}_{fp_dm:s}.npy')
M0 = Group_M_Mean200
R0 = Group_R_Mean200*1./(1+z0) # physical
Gamma = GroupGamma

#print("Gamma = ", Gamma[:100])
#Gamma = (np.log(M0)-np.log(M1))/(np.log(a0)-np.log(a1)) #a0 = 1./(1+z0) #a1 = 1./(1+z1)

print(R0.min())
R0[R0 < 1.e-8] = 1.e-8
print(R0.min())

#Rsp, Msp, mask = splashback.splashbackRadius(z0, mdef, R=R0, M=None, c=None, Gamma=Gamma, model='diemer20', statistic='median', rspdef='sp-apr-p90')#, rspdef='percentile95')#, c_model='diemer19', profile='nfw') # rsp-apr-p75
Rsp, Msp, mask = splashback.splashbackRadius(z0, mdef, R=R0, M=None, c=None, Gamma=Gamma, model='diemer20', statistic='median', rspdef='sp-apr-p90')#, rspdef='percentile95')#, c_model='diemer19', profile='nfw') # rsp-apr-p75
print("if empty, bad")
print(Rsp, Msp, mask)

Rsplash = np.zeros(len(M0))
Msplash = np.zeros(len(M0))
Rsplash[mask] = Rsp
Msplash[mask] = Msp
if sim_type == 'MTNG':
    np.save(tng_dir+f'data_{fp_dm:s}/Group_R_Splash_{fp_dm:s}_{snapshot0:d}.npy', Rsplash)
    np.save(tng_dir+f'data_{fp_dm:s}/Group_M_Splash_{fp_dm:s}_{snapshot0:d}.npy', Msplash)
else:
    np.save(tng_dir+f'Group_R_{fp_dm:s}_{snapshot0:d}.npy', Rsplash)
    np.save(tng_dir+f'Group_M_{fp_dm:s}_{snapshot0:d}.npy', Msplash)
