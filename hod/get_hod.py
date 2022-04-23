"""
Plot HOD and Poisson noise
"""
import os

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

# colors
#hexcolors_bright = ['#0099BB','#0077BB','#33BBEE','#EE3377','#EE7733','#BBBBBB','#CC3311']
hexcolors_bright = ['#CC3311','#0077BB','#EE7733','#BBBBBB','#33BBEE','#EE3377','#0099BB']

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
#gal_types = ['LRG', 'ELG']
gal_types = ['ELG']
n_gal = '2.0e-03' #['7.0e-04', '2.0e-03']
#snapshots = [179, 264]
snapshots = [264]
#zs = [1., 0.]
zs = [0.]

# definitions for the axes
left, width = 0.14, 0.85#0.1, 0.65
bottom, height = 0.1, 0.25#0.2#65
spacing = 0.005

rect_scatter = [left, bottom + (height + spacing), width, 0.6]
rect_histx = [left, bottom, width, height]

# start with a rectangular Figure
plt.figure(figsize=(9, 10))
ax_scatter = plt.axes(rect_scatter)
#(direction='in', top=True, right=True)
ax_histx = plt.axes(rect_histx)

counter = 0
for i, snapshot in enumerate(snapshots):
    z = zs[i]
    z_label = f"z = {z:.1f}"
    
    # load other halo properties
    GroupPos_fp = np.load(tng_dir+f'data_fp/GroupPos_fp_{snapshot:d}.npy')
    GrMcrit_fp = np.load(tng_dir+f'data_fp/Group_M_TopHat200_fp_{snapshot:d}.npy')*1.e10
    SubhaloSFR = np.load(tng_dir+f"data_fp/SubhaloSFR_fp_{snapshot:d}.npy")
    SubhaloMstar = np.load(tng_dir+f"data_fp/SubhaloMassType_fp_{snapshot:d}.npy")[:, 4]*1.e10
    SubhaloGrNr = np.load(tng_dir+f"data_fp/SubhaloGroupNr_fp_{snapshot:d}.npy")

    # max halo mass
    print("max halo mass = %.1e"%GrMcrit_fp.max())

    # identify central subhalos
    _, sub_inds_cent = np.unique(SubhaloGrNr, return_index=True)
    
    for gal_type in gal_types:
        gal_label = "{\\rm "+f"{gal_type}s"+"}"
    
        # indices of the galaxies
        index = np.load(f"/home/boryanah/MTNG/selection/data/index_{gal_type:s}_{n_gal}_{snapshot:d}.npy")

        # which galaxies are centrals
        index_cent = np.intersect1d(index, sub_inds_cent)

        # galaxy properties
        grnr_gal = SubhaloGrNr[index]
        grnr_cent_gal = SubhaloGrNr[index_cent]

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

        # define mass bins
        mbins = np.logspace(11, 15, 41)
        mbinc = (mbins[1:]+mbins[:-1]) * 0.5

        # satellite counts
        count_sats_halo = count_halo - count_cent_halo

        # probability
        want_probs = True
        if want_probs:
            # whether a halo has both a central and a satellite or just a satellite
            choice_anysat = ((count_sats_halo > 0)).astype(int)
            choice_acent = ((count_cent_halo == 1)).astype(int)
            choice_anysat_acent = ((count_sats_halo > 0) & (count_cent_halo == 1)).astype(int)
            choice_nosat_acent = ((count_sats_halo == 0) & (count_cent_halo == 1)).astype(int)
            choice_anysat_nocent = ((count_sats_halo > 0) & (count_cent_halo == 0)).astype(int)
            choice_nosat_nocent = ((count_sats_halo == 0) & (count_cent_halo == 0)).astype(int)

            hist_anysat, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=choice_anysat)
            hist_acent, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=choice_acent)
            hist_anysat_acent, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=choice_anysat_acent)
            hist_nosat_acent, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=choice_nosat_acent)
            hist_anysat_nocent, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=choice_anysat_nocent)
            hist_nosat_nocent, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=choice_nosat_nocent)
            hist_norm, _ = np.histogram(GrMcrit_fp, bins=mbins)

            # probability that a halo at a given mass bin holds blah
            prob_anysat = hist_anysat/hist_norm
            prob_acent = hist_acent/hist_norm
            prob_anysat_acent = hist_anysat_acent/hist_norm
            prob_nosat_acent = hist_nosat_acent/hist_norm
            prob_anysat_nocent = hist_anysat_nocent/hist_norm
            prob_nosat_nocent = hist_nosat_nocent/hist_norm

            np.savez(f"data/{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.npz", mbinc=mbinc, prob_acent=prob_acent, prob_anysat=prob_anysat, prob_acent_given_anysat=prob_anysat_acent/prob_anysat, prob_anysat_given_acent=prob_anysat_acent/prob_acent)

            quit()
            plt.figure(1, figsize=(9, 7))
            plt.plot(mbinc, prob_acent)
            plt.xscale('log')
            plt.savefig(f"acent_{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.png")
            plt.close()

            plt.figure(1, figsize=(9, 7))
            plt.plot(mbinc, prob_anysat)
            plt.xscale('log')
            plt.savefig(f"anysat_{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.png")
            plt.close()

            plt.figure(1, figsize=(9, 7))
            plt.plot(mbinc, prob_anysat_nocent/prob_anysat)
            plt.xscale('log')
            plt.savefig(f"anysat_nocent_by_anysat_{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.png")
            plt.close()

            plt.figure(1, figsize=(9, 7))
            plt.plot(mbinc, prob_anysat_acent/prob_anysat)
            plt.xscale('log')
            plt.savefig(f"anysat_acent_by_anysat_{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.png")
            plt.close()

            plt.figure(1, figsize=(9, 7))
            plt.plot(mbinc, prob_anysat_acent/prob_acent)
            plt.xscale('log')
            plt.savefig(f"anysat_acent_by_acent_{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.png")
            plt.close()

            plt.figure(1, figsize=(9, 7))
            plt.plot(mbinc, prob_nosat_acent/prob_acent)
            plt.xscale('log')
            plt.savefig(f"nosat_acent_by_acent_{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.png")
            plt.close()

            quit()


        # histograms
        hist_gal, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=count_halo)
        hist_cent_gal, _ = np.histogram(GrMcrit_fp, bins=mbins, weights=count_cent_halo)
        hist_halo, _ = np.histogram(GrMcrit_fp, bins=mbins)
        hod_cent_gal = hist_cent_gal/hist_halo
        hod_gal = hist_gal/hist_halo
        hod_sat_gal = hod_gal-hod_cent_gal

        std, _, _ = stats.binned_statistic(GrMcrit_fp, count_halo-count_cent_halo, statistic='std', bins=mbins)
        #std, _, _ = stats.binned_statistic(GrMcrit_fp, count_sats_halo*(count_sats_halo-1), statistic='mean', bins=mbins); std = np.sqrt(std) # alternative definition
        poisson = np.sqrt(hod_sat_gal)
        poiss_up, poiss_dw = hod_sat_gal + poisson, hod_sat_gal - poisson
        np.savez(f"data/hod_{n_gal}_{gal_type}_{snapshot}.npz", hod_cent_gal=hod_cent_gal, mbinc=mbinc, hod_sat_gal=hod_sat_gal, std=std, poisson=poisson)
        
        print("std = ", std)
        print("poisson = ", poisson)
        print("percentage difference = ", 100.*(std-poisson)/std)
        
        #plt.plot(mbinc, hod_gal, color='black', ls='-')
        #plt.plot(mbinc, hod_cent_gal, color=hexcolors_bright[counter], ls='--')
        #plt.errorbar(mbinc, hod_sat_gal, yerr=std, capsize=4, color=hexcolors_bright[counter], ls='-')
        #plt.fill_between(mbinc, poiss_up, poiss_dw, color='black', alpha=0.3)

        ax_scatter.plot(mbinc, hod_cent_gal, color=hexcolors_bright[counter], ls='-', lw=1.5, label=rf"${z_label}, \ {gal_label}$")
        ax_scatter.plot(mbinc, hod_sat_gal, color=hexcolors_bright[counter], ls='-', lw=2.5)

        #ax_histx.plot(mbinc, std, color=hexcolors_bright[counter], ls='-', lw=2.5) # og
        #ax_histx.plot(mbinc, poisson, color=hexcolors_bright[counter], ls='--', lw=2.5) # og
        # TESTING
        ax_histx.axhline(y=1, color='black', ls='--', lw=1.5)
        ax_histx.plot(mbinc, std/poisson, color=hexcolors_bright[counter], ls='-', lw=2.5)
        
        counter += 1
ax_scatter.set_xscale('log')
ax_scatter.set_yscale('log')
ax_scatter.legend()
#ax_scatter.set_xlabel(r'$M_{\rm halo} \ [M_\odot/h]$')
ax_scatter.set_ylabel(r'$\langle N_{\rm gal} \rangle$')
#ax_histx.plot([], [], color='black', ls='-', label='Simulation') # og
#ax_histx.plot([], [], color='black', ls='--', label='Poisson') # og
#ax_histx.legend() # og
ymin, ymax = ax_histx.get_ylim()
ymin = np.floor(ymin*10.)/10.
ymax = np.ceil(ymax*10.)/10.
ax_histx.set_yticks(np.arange(ymin, ymax, 0.1))
ax_histx.grid(color='gray', linestyle='-', linewidth=1.)
ax_histx.minorticks_on()    
ax_histx.set_xlabel(r'$M_{\rm halo} \ [M_\odot/h]$')
#ax_histx.set_ylabel(r'${\rm Std}[\langle N_{\rm sat} \rangle]$') # og
ax_histx.set_ylabel(r'${\rm Std}[\langle N_{\rm sat} \rangle]/\sqrt{\langle N_{\rm sat} \rangle}$') # TESTING
ax_histx.set_ylim([0.77, 1.23]) # TESTING
ax_histx.set_xscale('log')
#ax_histx.set_yscale('log') # og
plt.savefig(f"figs/hod_{n_gal}.png")
plt.show()
