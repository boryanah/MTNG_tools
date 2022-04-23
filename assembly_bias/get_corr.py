"""
Compute the correlation function of prediction and true for different halo properties
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import Corrfunc

from utils import get_jack_corr
import plotparams
plotparams.buba()

np.random.seed(100) 

zs = [0., 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]
snaps = [264, 237, 214, 179, 151, 129, 94, 80, 69, 51]
z_dict = {}
for i in range(len(zs)):
    z_dict[snaps[i]] = zs[i]

hexcolors_bright = ['#0077BB','#33BBEE','#0099BB','#EE7733','#CC3311','#EE3377','#BBBBBB']
greysafecols = ['#809BC8', '#FF6666', '#FFCC66', '#64C204']

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
Lbox = 500. # Mpc/h
gal_type = sys.argv[1] # 'LRG' # 'RLG'
fit_type = sys.argv[2] # 'ramp' # 'plane'
#fun_types = ['tanh', 'erf', 'gd', 'abs', 'arctan', 'linear']
fun_types = ['linear']
fun_type_sats = 'linear'
fp_dm = 'fp'
#fp_dm = 'dm'
mode = 'all' #'all' # fitting once for all # 'bins' fitting in bins
if gal_type == 'ELG':
    want_drad = True
else:
    want_drad = False
drad_str = "_drad" if want_drad else ""
want_pseudo = False
pseudo_str = "_pseudo" if want_pseudo else ""
want_vrad = False
vrad_str = "_vrad" if want_vrad else ""
want_fixocc = False
fixocc_str = "_fixocc" if want_fixocc else ""
want_splash = False
splash_str = "_splash" if want_splash else ""
if len(sys.argv) > 3:
    n_gal = sys.argv[3]
else:
    n_gal = '2.0e-03' # '7.0e-04'
if len(sys.argv) > 4:
    snapshot_fp = int(sys.argv[4])
    if fp_dm == 'dm':
        offset = 5
    elif fp_dm == 'fp':
        offset = 0
    snapshot = snapshot_fp + offset
    redshift = z_dict[snapshot]
else:
    snapshot_fp = 179;
    if fp_dm == 'dm':
        offset = 5
    elif fp_dm == 'fp':
        offset = 0
    snapshot = snapshot_fp + offset
    redshift = 1.
    
print(f"{gal_type}_{fit_type}_{vrad_str}_{splash_str}_{pseudo_str}_{drad_str}_{fixocc_str}_{fp_dm}_{snapshot:d}_{n_gal}")

# WORKS AS LONG AS YOU DON'T ADD MORE THAN ONE REPEATED QUANTITY
new_params = ['GroupVelAni', 'SubhaloMass_peak']
#params = ['GroupConc', 'Group_M_Crit200_peak', 'GroupGamma', 'GroupVelDispSqR', 'GroupShearAdapt', 'GroupEnvAdapt', 'GroupEnv_R1.5', 'GroupShear_R1.5', 'GroupConcRad', 'GroupVirial', 'GroupSnap_peak', 'GroupVelDisp', 'GroupPotential', 'Group_M_Splash', 'Group_R_Splash', 'GroupNsubs', 'GroupSnap_peak', 'GroupMarkedEnv_R2.0_s0.25_p2', 'GroupHalfmassRad']
params = ['GroupConc', 'Group_M_Crit200_peak', 'GroupShearAdapt', 'GroupEnvAdapt', 'Group_R_Splash', 'GroupNsubs']
#params = []
n_combos = len(params)*(len(params)-1)//2
if 'ramp' == fit_type:
    secondaries = params.copy()
    tertiaries = ['None']
    
    if len(new_params) > 0:
        secondaries = new_params.copy()
    print(secondaries)
else:
    secondaries = []
    tertiaries = []

    if len(new_params) > 0:
        for i_param in range(len(params)):
            for j_param in range(len(new_params)):
                if params[i_param] == new_params[j_param]: continue

                if params[i_param] < new_params[j_param]:
                    secondaries.append(params[i_param])
                    tertiaries.append(new_params[j_param])
                    print(params[i_param], new_params[j_param])
                else:
                    secondaries.append(new_params[j_param])
                    tertiaries.append(params[i_param])
                    print(new_params[j_param], params[i_param])
    else:
        for i_param in range(len(params)):
            for j_param in range(len(params)):
                if i_param <= j_param: continue

                if params[i_param] < params[j_param]:
                    secondaries.append(params[i_param])
                    tertiaries.append(params[j_param])
                    print(params[i_param], params[j_param])
                else:
                    secondaries.append(params[j_param])
                    tertiaries.append(params[i_param])
                    print(params[j_param], params[i_param])
print("combos = ", len(secondaries))
if fit_type == 'ramp':
    secondaries.append('None')

# load other halo properties
SubhaloPos = np.load(tng_dir+f'data_{fp_dm}/SubhaloPos_{fp_dm}_{snapshot:d}.npy')
SubhaloGrNr = np.load(tng_dir+f'data_{fp_dm}/SubhaloGroupNr_{fp_dm}_{snapshot:d}.npy')
GroupPos = np.load(tng_dir+f'data_{fp_dm}/GroupPos_{fp_dm}_{snapshot:d}.npy')
GrMcrit = np.load(tng_dir+f'data_{fp_dm}/Group_M_TopHat200_{fp_dm}_{snapshot:d}.npy')*1.e10
SubhaloPos_fp = np.load(tng_dir+f'data_fp/SubhaloPos_fp_{snapshot_fp:d}.npy')
SubhaloGrNr_fp = np.load(tng_dir+f'data_fp/SubhaloGroupNr_fp_{snapshot_fp:d}.npy')
GroupPos_fp = np.load(tng_dir+f'data_fp/GroupPos_fp_{snapshot_fp:d}.npy')
GrMcrit_fp = np.load(tng_dir+f'data_fp/Group_M_TopHat200_fp_{snapshot_fp:d}.npy')*1.e10

# indices of the galaxies
index = np.load(f"/home/boryanah/MTNG/selection/data/index_{gal_type:s}_{n_gal:s}_{snapshot_fp:d}.npy")

# identify central subhalos
_, sub_inds_cent = np.unique(SubhaloGrNr_fp, return_index=True)

# which galaxies are centrals
index_cent = np.intersect1d(index, sub_inds_cent)
index_sats = index[~np.in1d(index, index_cent)]
np.random.shuffle(index_cent)
np.random.shuffle(index_sats)

rbins = np.logspace(-1, 1.5, 31)
drbin = rbins[1:] - rbins[:-1]
rbinc = (rbins[1:] + rbins[:-1])/2.
np.save(f"{gal_type:s}/rbinc.npy", rbinc)

for i in range(len(fun_types)):
    fun_type = fun_types[i]
    
    for i_pair in range(len(secondaries)):
        secondary = secondaries[i_pair]
        if fit_type == 'plane':
            tertiary = tertiaries[i_pair]
        else:
            tertiary = 'None'

        print("param pair = ", i_pair, secondary, tertiary)
        # directory of ramp and plane
        if fit_type == 'plane':
            print(f"{gal_type:s}/pos_pred_all_{fun_type_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
            print(f"{gal_type:s}/pos_pred_all_{fun_type:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
            if mode == 'bins':
                pos_sats_pred = np.load(f"{gal_type:s}/pos_pred_{fun_type_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                pos_cent_pred = np.load(f"{gal_type:s}/pos_pred_{fun_type:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
            elif mode == 'all':
                pos_sats_pred = np.load(f"{gal_type:s}/pos_pred_all_{fun_type_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                pos_cent_pred = np.load(f"{gal_type:s}/pos_pred_all_{fun_type:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                ind_sats_pred = np.load(f"{gal_type:s}/ind_pred_all_{fun_type_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                ind_cent_pred = np.load(f"{gal_type:s}/ind_pred_all_{fun_type:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
        else:
            if mode == 'bins':            
                pos_sats_pred = np.load(f"{gal_type:s}/pos_pred_{fun_type_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                pos_cent_pred = np.load(f"{gal_type:s}/pos_pred_{fun_type:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
            elif mode == 'all':
                pos_sats_pred = np.load(f"{gal_type:s}/pos_pred_all_{fun_type_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                pos_cent_pred = np.load(f"{gal_type:s}/pos_pred_all_{fun_type:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                ind_sats_pred = np.load(f"{gal_type:s}/ind_pred_all_{fun_type_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                ind_cent_pred = np.load(f"{gal_type:s}/ind_pred_all_{fun_type:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")


        #pos_cent_pred = GroupPos[ind_cent_pred]
        #pos_sats_pred = GroupPos[ind_sats_pred]

        """
        pos_sats_pred -= GroupPos[ind_sats_pred]
        pos_sats_pred[pos_sats_pred > Lbox/2.] -= Lbox
        pos_sats_pred[pos_sats_pred < -Lbox/2.] += Lbox
        pos_sats_pred = np.sqrt(np.sum(pos_sats_pred**2, axis=1))
        pos_sats_pred = np.sort(pos_sats_pred)[::-1]
        rbins = np.logspace(-2, 1, 21)
        rbinc = (rbins[1:] + rbins[:-1])*.5
        hist, _ = np.histogram(pos_sats_pred, bins=rbins)
        plt.figure(figsize=(9, 7))
        plt.plot(rbinc, hist)
        """

        # this is acutally because we already have downsampled the prediction in the initial step, so it's guaranteed that here the truth will be less 
        pos_cent_true = SubhaloPos_fp[index_cent[:len(pos_cent_pred)]]
        pos_sats_true = SubhaloPos_fp[index_sats[:len(pos_sats_pred)]]
        #pos_sats_true = GroupPos_fp[SubhaloGrNr_fp[index_sats[:len(pos_sats_pred)]]]
        #pos_cent_true = GroupPos_fp[SubhaloGrNr_fp[index_cent[:len(pos_cent_pred)]]]

        """
        pos_sats_true -= GroupPos_fp[SubhaloGrNr_fp[index_sats[:len(pos_sats_pred)]]]
        pos_sats_true[pos_sats_true > Lbox/2.] -= Lbox
        pos_sats_true[pos_sats_true < -Lbox/2.] += Lbox
        pos_sats_true = np.sqrt(np.sum(pos_sats_true**2, axis=1))
        pos_sats_true = np.sort(pos_sats_true)[::-1]
        hist, _ = np.histogram(pos_sats_true, bins=rbins)
        plt.plot(rbinc, hist)
        plt.show()
        """

        pos_pred = np.vstack((pos_cent_pred, pos_sats_pred))
        pos_true = np.vstack((pos_cent_true, pos_sats_true))
        pos_true %= Lbox
        pos_pred %= Lbox
        print("nans =", np.sum(np.isnan(pos_pred)))
        w_pred = np.ones(pos_pred.shape[0], dtype=pos_pred.dtype)
        w_true = np.ones(pos_true.shape[0], dtype=pos_true.dtype)
        print("true and fake difference = ", len(w_pred)-len(w_true))
        print("fake number = ", len(w_pred))

        # N_dim should maybe be 5
        rat_mean, rat_err, corr_shuff_mean, corr_shuff_err, corr_true_mean, corr_true_err, _ = get_jack_corr(pos_true, w_true, pos_pred, w_pred, Lbox, N_dim=3, bins=rbins)
        if secondary == 'None':
            np.save(f"{gal_type:s}/corr_mean_{n_gal}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{fp_dm:s}_{snapshot:d}.npy", corr_true_mean)
            np.save(f"{gal_type:s}/corr_shuff_mean_{n_gal}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{fp_dm:s}_{snapshot:d}.npy", corr_shuff_mean)
            np.save(f"{gal_type:s}/corr_rat_shuff_mean_{n_gal}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{fp_dm:s}_{snapshot:d}.npy", rat_mean)
            np.save(f"{gal_type:s}/corr_rat_shuff_err_{n_gal}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{fp_dm:s}_{snapshot:d}.npy", rat_err)

            #mbinc = np.array([12.0, 12.5, 13.0, 13.5, 14.0])
            #dbins = np.ones(len(mbinc))*0.5
            mbinc = np.array([12.5, 13.5])
            dbins = np.ones(len(mbinc))*1. # TESTING
            
            rbins = np.logspace(-1, 1.5, 10)
            rbinc = (rbins[1:]+rbins[:-1])/2.

            # fix GrMcrit (maybe not needed)
            mcrit_cent_pred = GrMcrit[ind_cent_pred]
            mcrit_cent_true = GrMcrit_fp[SubhaloGrNr_fp[index_cent[:len(pos_cent_pred)]]]
            mcrit_sats_pred = GrMcrit[ind_sats_pred]
            mcrit_sats_true = GrMcrit_fp[SubhaloGrNr_fp[index_sats[:len(pos_sats_pred)]]]
            
            for j in range(len(mbinc)):
                # choice for centrals
                choice_cent_true = (10.**(mbinc[j]-dbins[j]/2.) < mcrit_cent_true) & (10.**(mbinc[j]+dbins[j]/2.) >= mcrit_cent_true)
                choice_cent_pred = (10.**(mbinc[j]-dbins[j]/2.) < mcrit_cent_pred) & (10.**(mbinc[j]+dbins[j]/2.) >= mcrit_cent_pred)
                if (np.sum(choice_cent_true) == 0) or (np.sum(choice_cent_pred) == 0):
                    skip_cent = True
                else:
                    skip_cent = False
                w_cent_true = np.ones(np.sum(choice_cent_true), dtype=pos_cent_true.dtype)
                w_cent_pred = np.ones(np.sum(choice_cent_pred), dtype=pos_cent_pred.dtype)
                p_cent_true = pos_cent_true[choice_cent_true]%Lbox
                p_cent_pred = pos_cent_pred[choice_cent_pred]%Lbox
                
                # choice for satellites
                choice_sats_true = (10.**(mbinc[j]-dbins[j]/2.) < mcrit_sats_true) & (10.**(mbinc[j]+dbins[j]/2.) >= mcrit_sats_true)
                choice_sats_pred = (10.**(mbinc[j]-dbins[j]/2.) < mcrit_sats_pred) & (10.**(mbinc[j]+dbins[j]/2.) >= mcrit_sats_pred)
                if (np.sum(choice_sats_true) == 0) or (np.sum(choice_sats_pred) == 0):
                    skip_sats = True
                else:
                    skip_sats = False
                w_sats_true = np.ones(np.sum(choice_sats_true), dtype=pos_sats_true.dtype)
                w_sats_pred = np.ones(np.sum(choice_sats_pred), dtype=pos_sats_pred.dtype)
                p_sats_true = pos_sats_true[choice_sats_true]%Lbox
                p_sats_pred = pos_sats_pred[choice_sats_pred]%Lbox

                # choice for all galaxies
                p_gals_true = np.vstack((p_cent_true, p_sats_true))
                p_gals_pred = np.vstack((p_cent_pred, p_sats_pred))
                w_gals_true = np.hstack((w_cent_true, w_sats_true))
                w_gals_pred = np.hstack((w_cent_pred, w_sats_pred))
                if (len(w_gals_true) == 0) or (len(w_gals_pred) == 0):
                    skip_gals = True
                else:
                    skip_gals = False
                
                # all (gals+sent)
                if skip_gals:
                    rat_mean, rat_err, corr_shuff_mean, corr_shuff_err, corr_true_mean, corr_true_err, _ = tuple(np.split(np.zeros((len(rbinc), 7)), 7, axis=1))
                    print("skip gals in bin = ", j)
                else:
                    rat_mean, rat_err, corr_shuff_mean, corr_shuff_err, corr_true_mean, corr_true_err, _ = get_jack_corr(p_gals_true, w_gals_true, p_gals_pred, w_gals_pred, Lbox, N_dim=3, bins=rbins)
                
                np.savez(f'{gal_type:s}/{gal_type:s}_corr_bin_{j:d}_gals{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npz', mean=rat_mean, err=rat_err, shuff_mean=corr_shuff_mean, shuff_err=corr_shuff_err, true_mean=corr_true_mean, true_err=corr_true_err, binc=rbinc, logm=mbinc[j])
                
                # centrals
                if skip_cent:
                    rat_mean, rat_err, corr_shuff_mean, corr_shuff_err, corr_true_mean, corr_true_err, _ = tuple(np.split(np.zeros((len(rbinc), 7)), 7, axis=1))
                    print("skip cent in bin = ", j)
                else:
                    rat_mean, rat_err, corr_shuff_mean, corr_shuff_err, corr_true_mean, corr_true_err, _ = get_jack_corr(p_cent_true, w_cent_true, p_cent_pred, w_cent_pred, Lbox, N_dim=3, bins=rbins)
                
                np.savez(f'{gal_type:s}/{gal_type:s}_corr_bin_{j:d}_cent{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npz', mean=rat_mean, err=rat_err, shuff_mean=corr_shuff_mean, shuff_err=corr_shuff_err, true_mean=corr_true_mean, true_err=corr_true_err, binc=rbinc, logm=mbinc[j])
                
                # satellites
                if skip_sats:
                    rat_mean, rat_err, corr_shuff_mean, corr_shuff_err, corr_true_mean, corr_true_err, _ = tuple(np.split(np.zeros((len(rbinc), 7)), 7, axis=1))
                    print("skip sats in bin = ", j)
                else:
                    rat_mean, rat_err, corr_shuff_mean, corr_shuff_err, corr_true_mean, corr_true_err, _ = get_jack_corr(p_sats_true, w_sats_true, p_sats_pred, w_sats_pred, Lbox, N_dim=3, bins=rbins)
                
                np.savez(f'{gal_type:s}/{gal_type:s}_corr_bin_{j:d}_sats{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npz', mean=rat_mean, err=rat_err, shuff_mean=corr_shuff_mean, shuff_err=corr_shuff_err, true_mean=corr_true_mean, true_err=corr_true_err, binc=rbinc, logm=mbinc[j])


            quit()

        
        # TESTING plotting
        want_test = False
        if want_test:
            # remove
            # for testing quickly
            plt.figure(figsize=(9, 7))
            plt.plot(rbinc, np.ones(len(rbinc)), 'k--')
            plt.errorbar(rbinc, corr_true_mean*rbinc**2, yerr=corr_true_err*rbinc**2, ls='-', capsize=4, color='black', label='True')
            plt.errorbar(rbinc, corr_shuff_mean*rbinc**2, yerr=corr_shuff_err*rbinc**2, ls='-', capsize=4, color='dodgerblue', label='Predicted')
            plt.xscale('log')
            plt.savefig(f'figs/corr_{fun_type:s}_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{snapshot:d}.png')

            plt.figure(figsize=(9, 7))
            plt.plot(rbinc, np.ones(len(rbinc)), 'k--')
            plt.errorbar(rbinc, rat_mean, yerr=rat_err, ls='-', capsize=4, color='dodgerblue', label='Predicted')
            plt.xscale('log')
            plt.ylabel(r'$\xi_{\rm pred}(r)/\xi_{\rm true}(r)$')
            plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
            plt.ylim([0.8, 1.2])
            text = f'{secondary:s}_{tertiary:s}_{vrad_lab:s}_{splash_lab:s}'
            text = ' '.join(text.split('_'))
            plt.text(x=0.03, y=0.1, s=text, transform=plt.gca().transAxes)

            plt.savefig(f'figs/corr_{fun_type:s}_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{snapshot:d}.png')
            plt.close()
            #plt.show()
            quit()

        if fit_type == 'plane':
            if mode == 'bins':
                np.save(f"{gal_type:s}/corr_rat_mean_{fun_type:s}_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy", rat_mean)
                np.save(f"{gal_type:s}/corr_rat_err_{fun_type:s}_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy", rat_err)
            elif mode == 'all':
                np.save(f"{gal_type:s}/corr_rat_mean_all_{fun_type:s}_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy", rat_mean)
                np.save(f"{gal_type:s}/corr_rat_err_all_{fun_type:s}_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy", rat_err)
        else:
            if mode == 'bins':
                np.save(f"{gal_type:s}/corr_rat_mean_{fit_type:s}_{secondary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy", rat_mean)
                np.save(f"{gal_type:s}/corr_rat_err_{fit_type:s}_{secondary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy", rat_err)
            elif mode == 'all':
                np.save(f"{gal_type:s}/corr_rat_mean_all_{fun_type:s}_{secondary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy", rat_mean)
                np.save(f"{gal_type:s}/corr_rat_err_all_{fun_type:s}_{secondary:s}{vrad_str:s}{splash_str:s}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy", rat_err)
