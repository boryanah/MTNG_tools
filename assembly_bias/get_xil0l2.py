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
from astropy.cosmology import FlatLambdaCDM

from utils import get_jack_xil0l2, get_xil0l2
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
tng_dir = "/mnt/alan1/boryanah/MTNG/dm_arepo/" # TESTING
#tng_dir = "/mnt/alan1/boryanah/MTNG/"
tng_dir_fp = "/mnt/alan1/boryanah/MTNG/"
Lbox = 500. # Mpc/h
gal_type = sys.argv[1]
fit_type = sys.argv[2]
fun_types = ['linear']
fun_type_sats = 'linear'
#fp_dm = 'fp'
fp_dm = 'dm'
if "arepo" in tng_dir:
    tng_dir_fp = tng_dir
#mode = 'bins' # fitting in bins
mode = 'all' # fitting once for all
if gal_type == 'ELG':
    want_drad = False #True TESTING og is 1
    want_cond = False #True
    want_pseudo = False #True 
else:
    want_drad = False
    want_cond = False
    want_pseudo = False
drad_str = "_drad" if want_drad else ""
cond_str = "_cond" if want_cond else ""
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
    if fp_dm == 'dm' and "arepo" not in tng_dir:
        offset = 5
    else:
        offset = 0
    snapshot = snapshot_fp + offset
    redshift = z_dict[snapshot_fp]
else:
    snapshot_fp = 179;
    if fp_dm == 'dm' and "arepo" not in tng_dir:
        offset = 5
    else:
        offset = 0
    snapshot = snapshot_fp + offset
    redshift = 1.
print(f"{gal_type}_{fit_type}_{vrad_str}_{splash_str}_{pseudo_str}_{drad_str}_{fixocc_str}_{cond_str}_{fp_dm}_{snapshot:d}_{n_gal}")

# set up cosmology
h = 0.6774
cosmo = FlatLambdaCDM(H0=h*100, Om0=0.3089, Tcmb0=2.725)
H_z = cosmo.H(redshift).value
print("H(z) = ", H_z)

params = ['GroupConc', 'SubhaloMass_peak', 'GroupShearAdapt', 'GroupEnvAdapt', 'Group_R_Splash', 'GroupVelAni']
#params = ['GroupVelAni']
#params = []
n_combos = len(params)*(len(params)-1)//2
print("combos = ", n_combos)
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

# load other halo properties
SubhaloPos_fp = np.load(tng_dir_fp+f'data_{fp_dm}/SubhaloPos_{fp_dm}_{snapshot_fp:d}.npy')
SubhaloVel_fp = np.load(tng_dir_fp+f'data_{fp_dm}/SubhaloVel_{fp_dm}_{snapshot_fp:d}.npy')
SubhaloGrNr_fp = np.load(tng_dir_fp+f'data_{fp_dm}/SubhaloGroupNr_{fp_dm}_{snapshot_fp:d}.npy')

# indices of the galaxies
if 'dm_arepo' in tng_dir:
    index = np.load(f"/home/boryanah/MTNG/selection/data/index_{gal_type:s}_{n_gal:s}_{snapshot:d}_dm_arepo.npy")
else:
    index = np.load(f"/home/boryanah/MTNG/selection/data/index_{gal_type:s}_{n_gal:s}_{snapshot:d}.npy")

# identify central subhalos
_, sub_inds_cent = np.unique(SubhaloGrNr_fp, return_index=True)

# which galaxies are centrals
index_cent = np.intersect1d(index, sub_inds_cent)
index_sats = index[~np.in1d(index, index_cent)]
np.random.shuffle(index_cent)
np.random.shuffle(index_sats)

rbins = np.logspace(-1, 1.5, 17)
drbin = rbins[1:] - rbins[:-1]
rbinc = (rbins[1:]+rbins[:-1])/2.
np.save(f"{gal_type}/xil0l2_rbinc.npy", rbinc)

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
            print(f"{gal_type}/pos_pred_all_{fun_type_sats}_sats_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
            print(f"{gal_type}/pos_pred_all_{fun_type}_cent_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
            if mode == 'bins':
                pos_sats_pred = np.load(f"{gal_type}/pos_pred_{fun_type_sats}_sats_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                pos_cent_pred = np.load(f"{gal_type}/pos_pred_{fun_type}_cent_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_sats_pred = np.load(f"{gal_type}/vel_pred_{fun_type_sats}_sats_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_cent_pred = np.load(f"{gal_type}/vel_pred_{fun_type}_cent_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
            elif mode == 'all':
                pos_sats_pred = np.load(f"{gal_type}/pos_pred_all_{fun_type_sats}_sats_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                pos_cent_pred = np.load(f"{gal_type}/pos_pred_all_{fun_type}_cent_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_sats_pred = np.load(f"{gal_type}/vel_pred_all_{fun_type_sats}_sats_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_cent_pred = np.load(f"{gal_type}/vel_pred_all_{fun_type}_cent_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                ind_sats_pred = np.load(f"{gal_type}/ind_pred_all_{fun_type_sats}_sats_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                ind_cent_pred = np.load(f"{gal_type}/ind_pred_all_{fun_type}_cent_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
        else:
            print(f"{gal_type}/pos_pred_all_{fun_type_sats}_sats_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
            if mode == 'bins':            
                pos_sats_pred = np.load(f"{gal_type}/pos_pred_{fun_type_sats}_sats_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                pos_cent_pred = np.load(f"{gal_type}/pos_pred_{fun_type}_cent_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_sats_pred = np.load(f"{gal_type}/vel_pred_{fun_type_sats}_sats_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_cent_pred = np.load(f"{gal_type}/vel_pred_{fun_type}_cent_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
            elif mode == 'all':
                pos_sats_pred = np.load(f"{gal_type}/pos_pred_all_{fun_type_sats}_sats_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                pos_cent_pred = np.load(f"{gal_type}/pos_pred_all_{fun_type}_cent_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_sats_pred = np.load(f"{gal_type}/vel_pred_all_{fun_type_sats}_sats_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_cent_pred = np.load(f"{gal_type}/vel_pred_all_{fun_type}_cent_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                ind_sats_pred = np.load(f"{gal_type}/ind_pred_all_{fun_type_sats}_sats_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                ind_cent_pred = np.load(f"{gal_type}/ind_pred_all_{fun_type}_cent_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")

        
        pos_pred = np.vstack((pos_cent_pred, pos_sats_pred))
        vel_pred = np.vstack((vel_cent_pred, vel_sats_pred))
        
        pos_cent_true = SubhaloPos_fp[index_cent[:len(pos_cent_pred)]]
        pos_sats_true = SubhaloPos_fp[index_sats[:len(pos_sats_pred)]]
        vel_cent_true = SubhaloVel_fp[index_cent[:len(pos_cent_pred)]]
        vel_sats_true = SubhaloVel_fp[index_sats[:len(pos_sats_pred)]]

        # TESTING new now when we do dm arepo
        print(len(pos_cent_pred), len(pos_cent_true), len(pos_sats_pred), len(pos_sats_true))
        tmp_inds = np.arange(len(pos_cent_pred))
        np.random.shuffle(tmp_inds)
        tmp_inds = tmp_inds[:len(pos_cent_true)]
        pos_cent_pred = pos_cent_pred[tmp_inds]
        vel_cent_pred = vel_cent_pred[tmp_inds]
        tmp_inds = np.arange(len(pos_sats_pred))
        np.random.shuffle(tmp_inds)
        tmp_inds = tmp_inds[:len(pos_sats_true)]
        pos_sats_pred = pos_sats_pred[tmp_inds]
        vel_sats_pred = vel_sats_pred[tmp_inds]
        del tmp_inds
        print("after removing = ", len(pos_cent_pred), len(pos_cent_true), len(pos_sats_pred), len(pos_sats_true))
        pos_pred = np.vstack((pos_cent_pred, pos_sats_pred))
        vel_pred = np.vstack((vel_cent_pred, vel_sats_pred))
        
        pos_true = np.vstack((pos_cent_true, pos_sats_true))
        vel_true = np.vstack((vel_cent_true, vel_sats_true))
        print("nans = ", np.sum(np.isnan(pos_pred)))
        print("nans = ", np.sum(np.isnan(pos_true)))
        print("nans = ", np.sum(np.isnan(vel_pred)))
        print("nans = ", np.sum(np.isnan(vel_true)))
        
        # adding RSDs
        z_extra_true = vel_true[:, 2]*(1.+redshift)/H_z*h
        z_extra_pred = vel_pred[:, 2]*(1.+redshift)/H_z*h
        print(z_extra_true.min(), z_extra_true.max(), np.mean(np.abs(z_extra_true)), np.median(np.abs(z_extra_true)))
        print(z_extra_pred.min(), z_extra_pred.max(), np.mean(np.abs(z_extra_pred)), np.median(np.abs(z_extra_pred)))
        pos_true[:, 2] += z_extra_true # Mpc/h
        pos_pred[:, 2] += z_extra_pred # Mpc/h
        
        pos_true %= Lbox
        pos_pred %= Lbox
        w_pred = np.ones(pos_pred.shape[0], dtype=pos_pred.dtype)
        w_true = np.ones(pos_true.shape[0], dtype=pos_true.dtype)
        print("true and fake difference = ", len(w_pred)-len(w_true))
        print("fake number = ", len(w_pred))

        pos_true = pos_true.astype(np.float32)
        pos_pred = pos_pred.astype(np.float32)
        
        # N_dim should maybe be 5
        ratl0_mean, ratl0_err, ratl2_mean, ratl2_err, xil0_pred_mean, xil0_pred_err, xil0_true_mean, xil0_true_err, xil2_pred_mean, xil2_pred_err, xil2_true_mean, xil2_true_err, _ = get_jack_xil0l2(pos_true, pos_pred, Lbox, N_dim=3, bins=rbins)
        if secondary == 'None':
            np.save(f"{gal_type}/xil0_mean_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{fp_dm}_{snapshot:d}.npy", xil0_true_mean)
            np.save(f"{gal_type}/xil0_shuff_mean_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{fp_dm}_{snapshot:d}.npy", xil0_pred_mean)
            np.save(f"{gal_type}/ratl0_shuff_mean_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{fp_dm}_{snapshot:d}.npy", ratl0_mean)
            np.save(f"{gal_type}/ratl0_shuff_err_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{fp_dm}_{snapshot:d}.npy", ratl0_err)
            np.save(f"{gal_type}/xil2_mean_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{fp_dm}_{snapshot:d}.npy", xil2_true_mean)
            np.save(f"{gal_type}/xil2_shuff_mean_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{fp_dm}_{snapshot:d}.npy", xil2_pred_mean)
            np.save(f"{gal_type}/ratl2_shuff_mean_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{fp_dm}_{snapshot:d}.npy", ratl2_mean)
            np.save(f"{gal_type}/ratl2_shuff_err_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{fp_dm}_{snapshot:d}.npy", ratl2_err)
            quit()

        want_plot = False
        if want_plot:
            # for testing quickly
            plt.figure(figsize=(9, 7))
            plt.plot(rbinc, np.ones(len(rbinc)), 'k--')
            plt.errorbar(rbinc, xil0_true_mean*rbinc**2, yerr=xil0_true_err*rbinc**2, ls='-', capsize=4, color='black', label='True l0')
            plt.errorbar(rbinc, xil0_pred_mean*rbinc**2, yerr=xil0_pred_err*rbinc**2, ls='-', capsize=4, color='dodgerblue', label='Predicted l0')
            plt.errorbar(rbinc, xil2_true_mean*rbinc**2, yerr=xil2_true_err*rbinc**2, ls='--', capsize=4, color='black', label='True l2')
            plt.errorbar(rbinc, xil2_pred_mean*rbinc**2, yerr=xil2_pred_err*rbinc**2, ls='--', capsize=4, color='dodgerblue', label='Predicted l2')
            plt.xscale('log')
            plt.ylabel(r'$\xi(r) r^2$')
            plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
            plt.legend()
            plt.savefig(f'figs/xil0l2_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{snapshot:d}.png')
            plt.close()
        
            plt.figure(figsize=(9, 7))
            plt.plot(rbinc, np.ones(len(rbinc)), 'k--')
            plt.errorbar(rbinc, ratl0_mean, yerr=ratl0_err, ls='-', capsize=4, color='dodgerblue')
            plt.errorbar(rbinc, ratl2_mean, yerr=ratl2_err, ls='--', capsize=4, color='dodgerblue')
            plt.ylabel(r'$\xi_{\rm pred}/\xi_{\rm true}$')
            plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
            plt.ylim([0.6, 1.4])
            plt.xscale('log')
            text = f'{secondary}_{tertiary}_{vrad_lab}_{splash_lab}'
            text = ' '.join(text.split('_'))
            plt.text(x=0.03, y=0.1, s=text, transform=plt.gca().transAxes)
        
            plt.savefig(f'figs/xil0l2_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{snapshot:d}.png')
            plt.close()#plt.show()
            quit()
        

        if fit_type == 'plane':
            if mode == 'bins':
                np.save(f"{gal_type}/ratl0_mean_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl0_mean)
                np.save(f"{gal_type}/ratl0_err_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl0_err)
                np.save(f"{gal_type}/ratl2_mean_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl2_mean)
                np.save(f"{gal_type}/ratl2_err_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl2_err)
            elif mode == 'all':
                np.save(f"{gal_type}/ratl0_mean_all_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl0_mean)
                np.save(f"{gal_type}/ratl0_err_all_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl0_err)
                np.save(f"{gal_type}/ratl2_mean_all_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl2_mean)
                np.save(f"{gal_type}/ratl2_err_all_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl2_err)

        else:
            if mode == 'bins':
                np.save(f"{gal_type}/ratl0_mean_{fit_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl0_mean)
                np.save(f"{gal_type}/ratl0_err_{fit_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl0_err)
                np.save(f"{gal_type}/ratl2_mean_{fit_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl2_mean)
                np.save(f"{gal_type}/ratl2_err_{fit_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl2_err)
            elif mode == 'all':
                np.save(f"{gal_type}/ratl0_mean_all_{fun_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl0_mean)
                np.save(f"{gal_type}/ratl0_err_all_{fun_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl0_err)
                np.save(f"{gal_type}/ratl2_mean_all_{fun_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl2_mean)
                np.save(f"{gal_type}/ratl2_err_all_{fun_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}{cond_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl2_err)
                
