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

sim_type = "MTNG";
#sim_type = "TNG";
tng_dir_dic = {'TNG': "/mnt/gosling1/boryanah/TNG300/", 'MTNG': "/mnt/alan1/boryanah/MTNG/dm_arepo/"}
tng_dir = tng_dir_dic[sim_type]

z_ints = [0., 0.5, 1.0, 1.5]
z_ints = [0., 0.5, 1.5]
z_ints = [0.0, 1.0]

zs = [0., 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]
snaps = [264, 237, 214, 179, 151, 129, 94, 80, 69, 51]
z_dict = {}
for i in range(len(zs)):
    key = f"{zs[i]:.3f}"
    z_dict[key] = snaps[i]

def g(c):
    return 1./(np.log(1. + c) - c/(1. + c))
    
G_N = 4.3009e-3 #pc/Msun (km/s)^2

for z_int in z_ints:
    #for fp_dm in ['fp', 'dm']:
    for fp_dm in ['dm']:
        print("redshift = ", z_int, fp_dm)
        if fp_dm == 'fp' or "arepo" in tng_dir: snapshot = z_dict[f"{z_int:.3f}"] # both dm and fp?
        else: snapshot = z_dict[f"{z_int:.3f}"]+5 #print("need a dic for dm"); exit()#snapshot = 184 # both dm and fp?

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

        print(SubhaloGrNr.shape, SubhaloVmax.shape, Group_M_Crit200.shape, unique_sub_grnr.max())

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

        GroupPotential = GroupConcRad*g(GroupConcRad)*Group_V_Crit200**2
        GroupPotential[np.isinf(GroupPotential)] = 0.
        GroupPotential[np.isnan(GroupPotential)] = 0.
        GroupPotential[GroupPotential == 0.] = 0.
        print("min max avg", GroupPotential.min(), GroupPotential.max(), np.mean(GroupPotential))

        plot = False
        if plot:
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
        np.save(tng_dir+f'data_{fp_dm:s}/GroupHalfmassRad_{fp_dm:s}_{snapshot:d}.npy', GroupHalfmassRad)
        np.save(tng_dir+f'data_{fp_dm:s}/GroupVelDispSqR_{fp_dm:s}_{snapshot:d}.npy', GroupVelDisp**2*GroupHalfmassRad)
        np.save(tng_dir+f'data_{fp_dm:s}/GroupConc_{fp_dm:s}_{snapshot:d}.npy', GroupConc)
        np.save(tng_dir+f'data_{fp_dm:s}/GroupConcRad_{fp_dm:s}_{snapshot:d}.npy', GroupConcRad)
