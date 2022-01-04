import os
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import plotparams
plotparams.buba()

# simulation specs
tng_dir = "/mnt/gosling1/boryanah/TNG300/"
Lbox = 205.

# define constants
star_mass = 0.000743736*1.e10
min_thresh = 5000.*star_mass

# load subhalo fields
SubhaloGrNr = np.load(tng_dir+"SubhaloGrNr_fp.npy")
SubhaloMstar = np.load(tng_dir+"SubhaloMassType_fp.npy")[:,4]*1.e10
Group_M_TopHat200 = np.load(tng_dir+"Group_M_TopHat200_fp.npy")*1.e10

# mark specs
Rs = [1, 2, 5]
r_outers = [3, 5, 7]
delta_s = 0.25
p = 2
#marks = ['GroupEnv', 'GroupAnnEnv', 'GroupMarkedEnv', 'GroupShear']
marks = ['GroupShear']
fnames = []
for mark in marks:
    if 'GroupEnv' == mark:
        for R in Rs:
            fnames.append(f"GroupEnv_R{R:d}")
    elif 'GroupShear' == mark:
        for R in Rs:
            fnames.append(f"GroupShear_R{R:d}")
    elif 'GroupAnnEnv' == mark:
        for r_outer in r_outers:
            fnames.append(f"GroupAnnEnv_R{r_outer:d}")
    elif 'GroupMarkedEnv' == mark:
        for R in Rs:
            fnames.append(f"GroupMarkedEnv_R{R:d}_s{delta_s:.2f}_p{p:d}")

for fname in fnames:
    # load mark
    GroupMark = np.load(tng_dir+fname+"_fp.npy").astype(np.float32)

    # galaxy positions and marks
    inds = np.arange(len(SubhaloGrNr), dtype=int)
    inds = inds[SubhaloMstar > min_thresh]
    mark_gals = GroupMark[SubhaloGrNr[inds]]
    hmass_gals = Group_M_TopHat200[SubhaloGrNr[inds]]
    print("number of galaxies = ", len(inds))
    print("min and max mark = ", np.min(mark_gals), np.max(mark_gals))
    print("min and max mark = ", np.min(GroupMark), np.max(GroupMark))


    # jdfkdjf
    bins = np.logspace(11, 15, 31)
    binc = (bins[1:] + bins[:-1])/2.
    median, _, _ = stats.binned_statistic(Group_M_TopHat200, GroupMark, statistic='median', bins=bins)
    median_gals, _, _ = stats.binned_statistic(hmass_gals, mark_gals, statistic='median', bins=bins)
    
    # draw scatter plot between galaxies and environment
    # corrfunc
    #plt.scatter(Group_M_TopHat200, GroupMark, alpha=0.4, s=0.1)
    #plt.scatter(hmass_gals, mark_gals, alpha=0.8, s=10, marker='*')


    #plt.plot(binc, median, label='all')
    #plt.plot(binc, median_gals, label='gals')
    plt.plot(binc, np.ones(len(median)), 'k--')
    plt.plot(binc, median_gals/median, label='gals')
    plt.xlim([1.e11, 1.e15])
    #plt.ylim([1.e-3, np.max(GroupMark)])
    plt.ylim([0.5, 1.5])
    plt.xscale('log')
    #plt.yscale('log')
    plt.legend()
    plt.savefig("figs/scatter_"+fname+"_gals.png")
    plt.close()
    #plt.show()


