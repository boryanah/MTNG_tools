import os

import numpy as np
import matplotlib.pyplot as plt

tng_dir = "/mnt/alan1/boryanah/MTNG/"

zs = [0., 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]
snaps = [264, 237, 214, 179, 151, 129, 94, 80, 69, 51]
z_dict = {}
for i in range(len(zs)):
    key = f"{zs[i]:.3f}"
    z_dict[key] = snaps[i]

#n_gals = [2.e-3, 7.e-4]
n_gals = [2.e-3]
#z_ints = [0., 0.5, 1.0, 1.5]
#z_ints = [0., 1.0]
z_ints = [1.]

want_show = False
count = 0
#z = np.float(sys.argv[1])
for z_int in z_ints:
    for n_gal in n_gals:
        snapshot = z_dict[f"{z_int:.3f}"]  #snapshot = 179
        str_snap = f"_{snapshot:d}"
        Lbox = 500. # Mpc/h

        print("z = ", z_int)

        # load other halo properties
        GroupPos_fp = np.load(tng_dir+'data_fp/GroupPos_fp'+str_snap+'.npy')
        #GroupEnv_fp = np.load(tng_dir+'data_fp/GroupEnv_R2_fp'+str_snap+'.npy')
        GrMcrit_fp = np.load(tng_dir+'data_fp/Group_M_TopHat200_fp'+str_snap+'.npy')*1.e10
        SubhaloSFR = np.load(tng_dir+f"data_fp/SubhaloSFR_fp_{snapshot:d}.npy")
        SubhaloGrNr = np.load(tng_dir+f"data_fp/SubhaloGroupNr_fp_{snapshot:d}.npy")
        SubhaloMstar = np.load(tng_dir+f"data_fp/SubhaloMassType_fp_{snapshot:d}.npy")[:, 4]*1.e10
        
        # specific star formation
        SubhalosSFR = SubhaloSFR/SubhaloMstar

        # 7.4e-4
        #mstar_thresh = 8.44e9
        #ssfr_thresh = 8e-10
        #mstar_lrg = 7.2e10 # fyi

        # 5.9e-4
        #mstar_thresh = 9e9
        #ssfr_thresh = 8.5e-10
        #mstar_lrg = 8.4e10 # fyi

        # 7.0e-4
        if np.isclose(n_gal, 7.e-4):
            if np.isclose(0., z_int):
                mstar_thresh = 4.9e9
                ssfr_thresh = 2.9e-10
                mstar_lrg = 1.1e11 # fyi
            elif np.isclose(0.5, z_int):
                mstar_thresh = 5.9e9
                ssfr_thresh = 5.5e-10
                mstar_lrg = 9.4e10 # fyi
            elif np.isclose(1., z_int):
                mstar_thresh = 8.3e9
                ssfr_thresh = 8.2e-10
                mstar_lrg = 7.5e10 # fyi
            elif np.isclose(1.5, z_int):
                mstar_thresh = 1.31e10
                ssfr_thresh = 9.9e-10
                mstar_lrg = 5.8e+10 # fyi

        # 9.7e-4
        #mstar_thresh = 7.9e9
        #ssfr_thresh = 7.4e-10
        #mstar_lrg = 6.0e10 # fyi

        # 2.0e-3
        if np.isclose(n_gal, 2.e-3):
            if np.isclose(0., z_int):
                mstar_thresh = 4.8e9
                ssfr_thresh = 2.e-10
                mstar_lrg = 5.1e10 # fyi
            elif np.isclose(0.5, z_int):
                mstar_thresh = 4.5e9
                ssfr_thresh = 4.e-10
                mstar_lrg = 4.2e10 # fyi
            elif np.isclose(1., z_int):
                # want to elevate ssfr threshold, so need to bring down mstar
                #mstar_thresh = 6.e9 # og
                #ssfr_thresh = 6e-10 # og
                mstar_thresh = 2.5e9 # TEST
                ssfr_thresh = 8e-10 # TEST
                mstar_lrg = 3.3e10 # fyi
            elif np.isclose(1.5, z_int):
                mstar_thresh = 7.7e9
                ssfr_thresh = 7.7e-10
                mstar_lrg = 2.5e+10 # fyi

        index_ELG = np.arange(len(SubhaloMstar), dtype=int)
        index_ELG = index_ELG[(SubhaloMstar > mstar_thresh) & (SubhalosSFR > ssfr_thresh)]
        # og
        #index_LRG = np.argsort(SubhaloMstar)[::-1][:len(index_ELG)]
        i_sort = np.argsort(SubhaloMstar)[::-1]
        index_LRG = (i_sort[(SubhalosSFR[i_sort] <= ssfr_thresh)])[:len(index_ELG)]
        n_LRG = len(index_LRG)/Lbox**3
        n_ELG = len(index_ELG)/Lbox**3

        # order the indices of LRGs by their parent index
        index_LRG = index_LRG[np.argsort(SubhaloGrNr[index_LRG])]

        print("number of ELG galaxies = ", len(index_ELG))
        print(f"number density of ELG galaxies = {n_ELG:.1e}")
        print(f"number density of LRG galaxies = {n_LRG:.1e}")
        print("min mass LRG = %.1e" %SubhaloMstar[index_LRG].min())

        np.save(f"data/index_ELG_{n_ELG:.1e}_{snapshot:d}.npy", index_ELG)
        np.save(f"data/index_LRG_{n_LRG:.1e}_{snapshot:d}.npy", index_LRG)
        # reduce number of things to be plotted
        choice = SubhaloMstar > 5.e9
        
        count += 1
        if want_show:
        
            plt.figure(count+1, figsize=(9, 7))
            plt.scatter(SubhaloMstar[choice], SubhalosSFR[choice], s=4, color="gray", alpha=0.1, marker="*")
            plt.scatter(SubhaloMstar[index_LRG], SubhalosSFR[index_LRG], s=50, color="orangered", marker="*", alpha=0.2)
            plt.scatter(SubhaloMstar[index_ELG], SubhalosSFR[index_ELG], s=50, color="dodgerblue", marker="*", alpha=0.2)
            plt.gca().axvline(x=mstar_thresh, color='k', ls='--')
            plt.gca().axhline(y=ssfr_thresh, color='k', ls='--')
            plt.ylim([1.e-14, 1.e-7])
            plt.xscale('log')
            plt.yscale('log')
            
plt.show()
