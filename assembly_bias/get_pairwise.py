"""
Compute the correlation function of prediction and true for different halo properties
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import get_jack_pair
import plotparams
plotparams.buba()

from numba_2pcf.cf import numba_pairwise_vel

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
gal_type = sys.argv[1] # 'LRG' # 'ELG'
fit_type = sys.argv[2] # 'ramp' # 'plane'
#fun_types = ['tanh', 'erf', 'gd', 'abs', 'arctan', 'linear']
fun_types = ['linear']
fun_type_sats = 'linear'
fp_dm = 'fp'
#fp_dm = 'dm'
mode = 'all' # fitting once for all # 'bins' fitting in bins
want_drad = int(sys.argv[5])
drad_str = "_drad" if want_drad else ""
want_pseudo = False
pseudo_str = "_pseudo" if want_pseudo else ""
want_vrad = int(sys.argv[6])
vrad_str = "_vrad" if want_vrad else ""
want_fixocc = int(sys.argv[7])
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
    redshift = z_dict[snapshot_fp]
else:
    snapshot_fp = 179;
    if fp_dm == 'dm':
        offset = 5
    elif fp_dm == 'fp':
        offset = 0
    snapshot = snapshot_fp + offset
    redshift = 1.
print(f"{gal_type}_{fit_type}_{vrad_str}_{splash_str}_{snapshot:d}_{n_gal}")

#params = ['GroupConc', 'Group_M_Crit200_peak', 'GroupGamma', 'GroupVelDispSqR', 'GroupShearAdapt', 'GroupEnvAdapt', 'GroupMarkedEnv_s0.25_p2']#, 'GroupGamma'] # 'GroupConcRad'
params = []
n_combos = len(params)*(len(params)-1)//2
if 'ramp' == fit_type:
    secondaries = params.copy()
    tertiaries = ['None']
else:
    secondaries = []
    tertiaries = []
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
#secondaries = ['None']
#secondaries = ['GroupEnv_R2']
#secondaries = ['GroupGamma']
#secondaries = ['GroupShear_R2']
#secondaries = ['GroupConc']
#secondaries = ['GroupVelDispSqR']
#secondaries = ['GroupVelAni']
#secondaries = ['Group_M_Crit200_peak']
#secondaries = ['GroupEnvAdapt']
#secondaries = ['GroupHalfmassRad']
#secondaries = ['GroupPotentialCen']
#secondaries = ['GroupPotentialFoF']
#secondaries = ['GroupPotential_TopHat200']
#tertiaries = ['GroupConc']

# load other halo properties
SubhaloPos = np.load(tng_dir+f'data_fp/SubhaloPos_fp_{snapshot_fp:d}.npy')
SubhaloVel = np.load(tng_dir+f'data_fp/SubhaloVel_fp_{snapshot_fp:d}.npy')
SubhaloGrNr = np.load(tng_dir+f'data_fp/SubhaloGroupNr_fp_{snapshot_fp:d}.npy')
GroupPos = np.load(tng_dir+f'data_fp/GroupPos_fp_{snapshot_fp:d}.npy')
GroupCount = np.load(tng_dir+f"data_fp/GroupCount{gal_type:s}_{n_gal:s}_fp_{snapshot_fp:d}.npy")
GroupCountCent = np.load(tng_dir+f"data_fp/GroupCentsCount{gal_type:s}_{n_gal:s}_fp_{snapshot_fp:d}.npy")
GroupCountSats = GroupCount-GroupCountCent
GrMcrit = np.load(tng_dir+f'data_fp/Group_M_TopHat200_fp_{snapshot_fp:d}.npy')*1.e10
index_halo = np.arange(len(GrMcrit), dtype=int)

# indices of the galaxies
index = np.load(f"/home/boryanah/MTNG/selection/data/index_{gal_type:s}_{n_gal:s}_{snapshot_fp:d}.npy")

# identify central subhalos
_, sub_inds_cent = np.unique(SubhaloGrNr, return_index=True)

# which galaxies are centrals
index_cent = np.intersect1d(index, sub_inds_cent)
index_sats = index[~np.in1d(index, index_cent)]
np.random.shuffle(index_cent)
np.random.shuffle(index_sats)

#rbins = np.linspace(0., 100., 51)
rbins = np.linspace(0., 20., 16)
drbin = rbins[1:] - rbins[:-1]
rbinc = (rbins[1:]+rbins[:-1])/2.
np.save(f"{gal_type:s}/pair_rbinc.npy", rbinc)

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
            print(f"{gal_type:s}/pos_pred_all_{fun_type_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{fp_dm:s}_{snapshot:d}.npy")
            print(f"{gal_type:s}/pos_pred_all_{fun_type:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{fp_dm:s}_{snapshot:d}.npy")
            if mode == 'bins':
                pos_sats_pred = np.load(f"{gal_type:s}/pos_pred_{fun_type_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                pos_cent_pred = np.load(f"{gal_type:s}/pos_pred_{fun_type:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                vel_sats_pred = np.load(f"{gal_type:s}/vel_pred_{fun_type_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                vel_cent_pred = np.load(f"{gal_type:s}/vel_pred_{fun_type:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
            elif mode == 'all':
                pos_sats_pred = np.load(f"{gal_type:s}/pos_pred_all_{fun_type_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                pos_cent_pred = np.load(f"{gal_type:s}/pos_pred_all_{fun_type:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                vel_sats_pred = np.load(f"{gal_type:s}/vel_pred_all_{fun_type_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                vel_cent_pred = np.load(f"{gal_type:s}/vel_pred_all_{fun_type:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                ind_sats_pred = np.load(f"{gal_type:s}/ind_pred_all_{fun_type_sats:s}_sats_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                ind_cent_pred = np.load(f"{gal_type:s}/ind_pred_all_{fun_type:s}_cent_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
        else:
            if mode == 'bins':            
                pos_sats_pred = np.load(f"{gal_type:s}/pos_pred_{fun_type_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                pos_cent_pred = np.load(f"{gal_type:s}/pos_pred_{fun_type:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                vel_sats_pred = np.load(f"{gal_type:s}/vel_pred_{fun_type_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                vel_cent_pred = np.load(f"{gal_type:s}/vel_pred_{fun_type:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
            elif mode == 'all':
                pos_sats_pred = np.load(f"{gal_type:s}/pos_pred_all_{fun_type_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                pos_cent_pred = np.load(f"{gal_type:s}/pos_pred_all_{fun_type:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                vel_sats_pred = np.load(f"{gal_type:s}/vel_pred_all_{fun_type_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                vel_cent_pred = np.load(f"{gal_type:s}/vel_pred_all_{fun_type:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                ind_sats_pred = np.load(f"{gal_type:s}/ind_pred_all_{fun_type_sats:s}_sats_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                ind_cent_pred = np.load(f"{gal_type:s}/ind_pred_all_{fun_type:s}_cent_{secondary:s}{vrad_str:s}{splash_str:s}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
        
        pos_pred = np.vstack((pos_cent_pred, pos_sats_pred))
        vel_pred = np.vstack((vel_cent_pred, vel_sats_pred))

        # this is acutally because we already have downsampled the prediction in the initial step, so it's guaranteed that here the truth will be less 
        pos_cent_true = SubhaloPos[index_cent[:len(pos_cent_pred)]]
        pos_sats_true = SubhaloPos[index_sats[:len(pos_sats_pred)]]
        vel_cent_true = SubhaloVel[index_cent[:len(pos_cent_pred)]]
        vel_sats_true = SubhaloVel[index_sats[:len(pos_sats_pred)]]
        
        pos_true = np.vstack((pos_cent_true, pos_sats_true))
        vel_true = np.vstack((vel_cent_true, vel_sats_true))

        pos_true %= Lbox
        pos_pred %= Lbox
        v_true = vel_true[:, 2]
        v_pred = vel_pred[:, 2]

        # N_dim should maybe be 5
        rat_mean, rat_err, pair_shuff_mean, pair_shuff_err, pair_true_mean, pair_true_err, _ = get_jack_pair(pos_true, vel_true, pos_pred, vel_pred, Lbox, N_dim=3, bins=rbins)
        if secondary == 'None':
            np.save(f"{gal_type:s}/pair_mean_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{fp_dm:s}_{snapshot:d}.npy", pair_true_mean)
            np.save(f"{gal_type:s}/pair_shuff_mean_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{fp_dm:s}_{snapshot:d}.npy", pair_shuff_mean)
            np.save(f"{gal_type:s}/pair_rat_shuff_mean_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{fp_dm:s}_{snapshot:d}.npy", rat_mean)
            np.save(f"{gal_type:s}/pair_rat_shuff_err_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{fp_dm:s}_{snapshot:d}.npy", rat_err)
            quit()

        want_plot = False
        if want_plot:           
            # for testing quickly
            plt.figure(figsize=(9, 7))
            plt.plot(rbinc, np.ones(len(rbinc)), 'k--')
            plt.errorbar(rbinc, pair_true_mean, yerr=pair_true_err, ls='-', capsize=4, color='black', label='True')
            plt.errorbar(rbinc, pair_shuff_mean, yerr=pair_shuff_err, ls='-', capsize=4, color='dodgerblue', label='Predicted')
            plt.ylabel(r'$\hat p(r)$')
            plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
            plt.xscale('log')
            plt.savefig(f'figs/try_pair_{fun_type:s}_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{snapshot:d}.png')
            plt.close()

            plt.figure(figsize=(9, 7))
            plt.plot(rbinc[1:], np.ones(len(rbinc))[1:], 'k--')
            plt.errorbar(rbinc[1:], rat_mean[1:], yerr=rat_err[1:], ls='-', capsize=4, color='dodgerblue', label='Predicted')
            plt.ylabel(r'$\hat p_{\rm pred}(r)/\hat p_{\rm true}(r)$')
            plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
            plt.ylim([0.8, 1.2])
            plt.xscale('log')
            text = f'{secondary:s}_{tertiary:s}_{vrad_lab:s}_{splash_lab:s}'
            text = ' '.join(text.split('_'))
            plt.text(x=0.03, y=0.1, s=text, transform=plt.gca().transAxes)

            plt.savefig(f'figs/pair_{fun_type:s}_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{snapshot:d}.png')
            #plt.show()
            plt.close()
            quit()


        if fit_type == 'plane':
            if mode == 'bins':
                np.save(f"{gal_type:s}/pair_rat_mean_{fun_type:s}_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy", rat_mean)
                np.save(f"{gal_type:s}/pair_rat_err_{fun_type:s}_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy", rat_err)
            elif mode == 'all':
                np.save(f"{gal_type:s}/pair_rat_mean_all_{fun_type:s}_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy", rat_mean)
                np.save(f"{gal_type:s}/pair_rat_err_all_{fun_type:s}_{secondary:s}_{tertiary:s}{vrad_str:s}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy", rat_err)
        else:
            if mode == 'bins':
                np.save(f"{gal_type:s}/pair_rat_mean_{fit_type:s}_{secondary:s}{vrad_str:s}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy", rat_mean)
                np.save(f"{gal_type:s}/pair_rat_err_{fit_type:s}_{secondary:s}{vrad_str:s}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy", rat_err)
            elif mode == 'all':
                np.save(f"{gal_type:s}/pair_rat_mean_all_{fun_type:s}_{secondary:s}{vrad_str:s}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy", rat_mean)
                np.save(f"{gal_type:s}/pair_rat_err_all_{fun_type:s}_{secondary:s}{vrad_str:s}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy", rat_err)
