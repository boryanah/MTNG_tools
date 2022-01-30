import os

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
snapshot = 179
str_snap = f"_{snapshot:d}"
Lbox = 500. # Mpc/h
#gal_types = ['LRG', 'ELG']
gal_types = ['ELG']
#gal_type = 'ELG'
#gal_type = 'LRG'
n_gals = ['5.9e-04', '7.4e-04', '9.7e-04', '2.0e-03']
n_gals = ['5.9e-04']

# load other halo properties
GroupPos_fp = np.load(tng_dir+'data_fp/GroupPos_fp'+str_snap+'.npy')
GroupEnv_fp = np.load(tng_dir+'data_fp/GroupEnv_R2_fp'+str_snap+'.npy')
GrMcrit_fp = np.load(tng_dir+'data_fp/Group_M_TopHat200_fp'+str_snap+'.npy')*1.e10
SubhaloSFR = np.load(tng_dir+f"data_fp/SubhaloSFR_fp_{snapshot:d}.npy")
SubhaloMstar = np.load(tng_dir+f"data_fp/SubhaloMassType_fp_{snapshot:d}.npy")[:, 4]*1.e10
SubhaloGrNr = np.load(tng_dir+f"data_fp/SubhaloGroupNr_fp_{snapshot:d}.npy")

# max halo mass
print("max halo mass = %.1e"%GrMcrit_fp.max())

# identify central subhalos
_, sub_inds_cent = np.unique(SubhaloGrNr, return_index=True)

for n_gal in n_gals:

    for gal_type in gal_types:
    
        # indices of the galaxies
        index = np.load(f"/home/boryanah/MTNG/selection/data/index_{gal_type:s}_{n_gal:s}_{snapshot:d}.npy")

        # which galaxies are centrals
        index_cent = np.intersect1d(index, sub_inds_cent)

        # galaxy properties
        grnr_gal = SubhaloGrNr[index]
        grnr_cent_gal = SubhaloGrNr[index_cent]
        #mass_gal = GrMcrit_fp[grnr_gal]
        #mass_cent_gal = GrMcrit_fp[grnr_cent_gal]

        # count unique halo repetitions
        grnr_gal_uni, cts = np.unique(grnr_gal, return_counts=True)
        count_halo = np.zeros(len(GrMcrit_fp), dtype=int)
        count_halo[grnr_gal_uni] = cts
        grnr_cent_gal_uni, cts = np.unique(grnr_cent_gal, return_counts=True)
        count_cent_halo = np.zeros(len(GrMcrit_fp), dtype=int)
        count_cent_halo[grnr_cent_gal_uni] = cts

        want_save = True
        if want_save:
            np.save(tng_dir+f"data_fp/GroupCount{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.npy", count_halo)
            np.save(tng_dir+f"data_fp/GroupCentsCount{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.npy", count_cent_halo)
            #quit()

        # define mass bins
        mbins = np.logspace(11, 15, 41)
        mbinc = (mbins[1:]+mbins[:-1]) * 0.5

        # histograms
        #hist_gal, _ = np.histogram(mass_gal, bins=mbins)
        #hist_cent_gal, _ = np.histogram(mass_cent_gal, bins=mbins)
        hist_gal, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=count_halo)
        hist_cent_gal, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=count_cent_halo)
        hist_halo, _ = np.histogram(GrMcrit_fp, bins=mbins)
        hod_cent_gal = hist_cent_gal/hist_halo
        hod_gal = hist_gal/hist_halo
        hod_sat_gal = hod_gal-hod_cent_gal

        std, _, _ = stats.binned_statistic(GrMcrit_fp, count_halo-count_cent_halo, statistic='std', bins=mbins)
        poisson = np.sqrt(hod_sat_gal)
        poiss_up, poiss_dw = hod_sat_gal + poisson, hod_sat_gal - poisson
        print("std = ", std)
        print("poisson = ", poisson)
        print("percentage difference = ", 100.*(std-poisson)/std)
        
        plt.plot(mbinc, hod_gal, color='black', ls='-')
        plt.plot(mbinc, hod_cent_gal, color='dodgerblue', ls='--')
        #plt.plot(mbinc, hod_sat_gal, color='dodgerblue', ls='--')
        plt.errorbar(mbinc, hod_sat_gal, yerr=std, capsize=4, color='dodgerblue', ls='-')
        plt.fill_between(mbinc, poiss_up, poiss_dw, color='black', alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(f"figs/hod_{gal_type:s}.png")
        plt.show()
