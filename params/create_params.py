import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interpn
from numba import njit
import numpy.linalg as la

from tools import numba_tsc_3D
import plotparams
plotparams.buba()

sim_type = "MTNG"; fp_dm = 'fp' # no choice
#sim_type = "MTNG"; fp_dm = 'dm' # no choice
#sim_type = "TNG"; fp_dm = 'dm' # actual choice
tng_dir_dic = {'TNG': "/mnt/gosling1/boryanah/TNG300/", 'MTNG': "/mnt/alan1/boryanah/MTNG/"}
Lbox_dic = {'TNG': 205., 'MTNG': 500.}
pos_part_dic = {'TNG': "parts_position_tng300-2_99.npy", 'MTNG': "data_parts/pos_down_1000_snap_179_fp.npy"}
#void_dic = {'TNG': "../finders/tracers.SVF_recen_ovl0.5", 'MTNG': "../visualize/data/pos_down_10000_snap_179_fp.SVF_recen_ovl0.5"}
data_dir_dic = {'TNG': "data_tng/", 'MTNG': "data_mtng/"}
N_dim_dic = {'TNG': 512, 'MTNG':512}
sim_name_dic = {'TNG': "tng300-2", 'MTNG': "mtng"}

#pos_part = np.load(tng_dir+"parts_position_tng300-3_99.npy")/1000.
tng_dir = tng_dir_dic[sim_type]
Lbox = Lbox_dic[sim_type]
data_dir = data_dir_dic[sim_type]
N_dim = N_dim_dic[sim_type]
#pos_parts = np.load(tng_dir+pos_part_dic[sim_type])
sim_name = sim_name_dic[sim_type]

'''
creating
GroupShear_R[1,2,5]_fp_179.npy
GroupEnv_R[1,2,5]_fp_179.npy
GroupMarkedEnv_R[1,2,5]_s0.25_p2_fp_179.npy
'''

if fp_dm == 'fp': snapshot = 179 # both dm and fp?
else: snapshot = 184 # both dm and fp?
G_N = 4.3009e-3 #pc/Msun (km/s)^2

SubhaloGrNr = np.load(tng_dir+f'data_{fp_dm:s}/SubhaloGroupNr_{fp_dm:s}_{snapshot:d}.npy')
unique_sub_grnr, firsts = np.unique(SubhaloGrNr, return_index=True)

SubhaloVmax = np.load(tng_dir+f'data_{fp_dm:s}/SubhaloVmax_{fp_dm:s}_{snapshot:d}.npy')
SubhaloSpin = np.load(tng_dir+f'data_{fp_dm:s}/SubhaloSpin_{fp_dm:s}_{snapshot:d}.npy')
SubhaloSpin = np.sqrt(np.sum(SubhaloSpin**2, axis=1)) # kpc, km/s
SubhaloVelDisp = np.load(tng_dir+f'data_{fp_dm:s}/SubhaloVelDisp_{fp_dm:s}_{snapshot:d}.npy')
SubhaloVmaxRad = np.load(tng_dir+f'data_{fp_dm:s}/SubhaloVmaxRad_{fp_dm:s}_{snapshot:d}.npy')
SubhaloHalfmassRad = np.load(tng_dir+f'data_{fp_dm:s}/SubhaloHalfmassRadType_{fp_dm:s}_{snapshot:d}.npy')[:, 1]

Group_M_Crit200 = np.load(tng_dir+f'data_{fp_dm:s}/Group_M_TopHat200_{fp_dm:s}_{snapshot:d}.npy')*1.e10
Group_R_Crit200 = np.load(tng_dir+f'data_{fp_dm:s}/Group_R_TopHat200_{fp_dm:s}_{snapshot:d}.npy')
Group_V_Crit200 = np.sqrt(G_N*Group_M_Crit200/(Group_R_Crit200*1.e6))
Group_V_Crit200[Group_R_Crit200 == 0.] = 0.

Group_Vmax = np.zeros(len(Group_M_Crit200))
GroupVelDisp = np.zeros(len(Group_M_Crit200))
GroupVmaxRad = np.zeros(len(Group_M_Crit200))
GroupHalfmassRad = np.zeros(len(Group_M_Crit200))
GroupSpin = np.zeros(len(Group_M_Crit200))

Group_Vmax[unique_sub_grnr] = SubhaloVmax[firsts]
GroupVmaxRad[unique_sub_grnr] = SubhaloVmaxRad[firsts]
GroupVelDisp[unique_sub_grnr] = SubhaloVelDisp[firsts]
GroupSpin[unique_sub_grnr] = SubhaloSpin[firsts]
GroupHalfmassRad[unique_sub_grnr] = SubhaloHalfmassRad[firsts]

GroupConc = (Group_Vmax/Group_V_Crit200)
GroupConcRad = (Group_R_Crit200/GroupHalfmassRad)
GroupSpin /= np.sqrt(2.)*(Group_V_Crit200*Group_R_Crit200*1.e3)
GroupVirial = GroupVelDisp**2*GroupVmaxRad
#GroupHalfmassRad

GroupConc[Group_V_Crit200 == 0.] = 0.
GroupConcRad[GroupHalfmassRad == 0.] = 0.
GroupSpin[(Group_V_Crit200 == 0.) | (Group_R_Crit200 == 0.)] = 0.

def g(c):
    return 1./(np.log(1. + c) - c/(1. + c))
GroupPotential = GroupConcRad*g(GroupConcRad)*Group_V_Crit200**2
GroupPotential[np.isinf(GroupPotential)] = 0.
GroupPotential[np.isnan(GroupPotential)] = 0.
GroupPotential[GroupPotential == 0.] = 0.
print("min max avg", GroupPotential.min(), GroupPotential.max(), np.mean(GroupPotential))

mbins = np.logspace(10, 15, 31)
mbinc = (mbins[1:]+mbins[:-1])*.5
hist, _ = np.histogram(Group_M_Crit200, bins=mbins)

hist_w, _ = np.histogram(Group_M_Crit200, bins=mbins, weights=GroupConc)
print(hist_w, hist)
hist_w /= hist
hist_w[hist == 0.] = 0.
plt.figure(figsize=(9, 7))
plt.plot(mbinc, hist_w, label='Conc')
plt.xscale('log')
plt.legend()
plt.xlim([1.e10, 1.e15])
#plt.show()

hist_w, _ = np.histogram(Group_M_Crit200, bins=mbins, weights=GroupConcRad)
hist_w /= hist
hist_w[hist == 0.] = 0.
plt.figure(figsize=(9, 7))
plt.plot(mbinc, hist_w, label='ConcRad')
plt.xscale('log')
plt.legend()
plt.xlim([1.e10, 1.e15])
#plt.show()

hist_w, _ = np.histogram(Group_M_Crit200, bins=mbins, weights=GroupSpin)
hist_w /= hist
hist_w[hist == 0.] = 0.
plt.figure(figsize=(9, 7))
plt.plot(mbinc, hist_w, label='Spin')
plt.xscale('log')
plt.legend()
plt.xlim([1.e10, 1.e15])
#plt.show()

hist_w, _ = np.histogram(Group_M_Crit200, bins=mbins, weights=GroupVirial)
hist_w /= hist
hist_w[hist == 0.] = 0.
plt.figure(figsize=(9, 7))
plt.plot(mbinc, hist_w, label='Virial')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlim([1.e10, 1.e15])
plt.show()

np.save(tng_dir+f'data_{fp_dm:s}/GroupPotential_{fp_dm:s}_{snapshot:d}.npy', GroupPotential)
np.save(tng_dir+f'data_{fp_dm:s}/GroupVirial_{fp_dm:s}_{snapshot:d}.npy', GroupVirial)
np.save(tng_dir+f'data_{fp_dm:s}/GroupVelDisp_{fp_dm:s}_{snapshot:d}.npy', GroupVelDisp)
np.save(tng_dir+f'data_{fp_dm:s}/GroupConc_{fp_dm:s}_{snapshot:d}.npy', GroupConc)
np.save(tng_dir+f'data_{fp_dm:s}/GroupConcRad_{fp_dm:s}_{snapshot:d}.npy', GroupConcRad)
