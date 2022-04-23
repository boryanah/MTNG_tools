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
tng_dir = "/mnt/alan1/boryanah/MTNG/"
Lbox = 500. # Mpc/h
gal_type = sys.argv[1]
fit_type = sys.argv[2]
fun_types = ['linear']
fun_type_sats = 'linear'
fp_dm = 'fp'
#fp_dm = 'dm'
#mode = 'bins' # fitting in bins
mode = 'all' # fitting once for all
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
print(f"{gal_type}_{fit_type}_{vrad_str}_{splash_str}_{pseudo_str}_{drad_str}_{fixocc_str}_{fp_dm}_{snapshot:d}_{n_gal}")

# set up cosmology
h = 0.6774
cosmo = FlatLambdaCDM(H0=h*100, Om0=0.3089, Tcmb0=2.725)
H_z = cosmo.H(redshift).value
print("H(z) = ", H_z)

#params = ['GroupConc', 'Group_M_Crit200_peak', 'GroupGamma', 'GroupVelDispSqR', 'GroupShearAdapt', 'GroupEnvAdapt', 'GroupMarkedEnv_s0.25_p2']#, 'GroupGamma'] # 'GroupConcRad'
params = []
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
#secondaries = ['None']
#secondaries = ['GroupEnv_R2'] # TESTING
#secondaries = ['GroupPotential']
#secondaries = ['GroupVelDisp']
#secondaries = ['GroupConc']
#secondaries = ['GroupVelDisp']
#tertiaries = ['GroupEnv_R2']
#tertiaries = ['GroupConcRad']
#tertiaries = ['GroupConc']
#tertiaries = ['GroupShear_R2']

# load other halo properties
SubhaloPos = np.load(tng_dir+f'data_fp/SubhaloPos_fp_{snapshot_fp:d}.npy')
SubhaloVel = np.load(tng_dir+f'data_fp/SubhaloVel_fp_{snapshot_fp:d}.npy')
SubhaloGrNr = np.load(tng_dir+f'data_fp/SubhaloGroupNr_fp_{snapshot_fp:d}.npy')
GroupPos = np.load(tng_dir+f'data_fp/GroupPos_fp_{snapshot_fp:d}.npy')
GroupVel = np.load(tng_dir+f'data_fp/GroupVel_fp_{snapshot_fp:d}.npy')
GroupVelDisp = np.load(tng_dir+f'data_fp/GroupVelDisp_fp_{snapshot_fp:d}.npy')
GroupCount = np.load(tng_dir+f"data_fp/GroupCount{gal_type}_{n_gal}_fp_{snapshot_fp:d}.npy")
GroupCountCent = np.load(tng_dir+f"data_fp/GroupCentsCount{gal_type}_{n_gal}_fp_{snapshot_fp:d}.npy")
GroupCountSats = GroupCount-GroupCountCent
GrMcrit = np.load(tng_dir+f'data_fp/Group_M_TopHat200_fp_{snapshot_fp:d}.npy')*1.e10
index_halo = np.arange(len(GrMcrit), dtype=int)

# indices of the galaxies
index = np.load(f"/home/boryanah/MTNG/selection/data/index_{gal_type}_{n_gal}_{snapshot_fp:d}.npy")

# identify central subhalos
_, sub_inds_cent = np.unique(SubhaloGrNr, return_index=True)

# which galaxies are centrals
index_cent = np.intersect1d(index, sub_inds_cent)
index_sats = index[~np.in1d(index, index_cent)]
np.random.shuffle(index_cent)
np.random.shuffle(index_sats)

rbins = np.logspace(-1, 1.5, 31)
drbin = rbins[1:] - rbins[:-1]
rbinc = (rbins[1:]+rbins[:-1])/2.
np.save(f"{gal_type}/rbinc.npy", rbinc)

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
            print(f"{gal_type}/pos_pred_all_{fun_type_sats}_sats_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
            print(f"{gal_type}/pos_pred_all_{fun_type}_cent_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
            if mode == 'bins':
                pos_sats_pred = np.load(f"{gal_type}/pos_pred_{fun_type_sats}_sats_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                pos_cent_pred = np.load(f"{gal_type}/pos_pred_{fun_type}_cent_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_sats_pred = np.load(f"{gal_type}/vel_pred_{fun_type_sats}_sats_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_cent_pred = np.load(f"{gal_type}/vel_pred_{fun_type}_cent_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
            elif mode == 'all':
                pos_sats_pred = np.load(f"{gal_type}/pos_pred_all_{fun_type_sats}_sats_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                pos_cent_pred = np.load(f"{gal_type}/pos_pred_all_{fun_type}_cent_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_sats_pred = np.load(f"{gal_type}/vel_pred_all_{fun_type_sats}_sats_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_cent_pred = np.load(f"{gal_type}/vel_pred_all_{fun_type}_cent_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                ind_sats_pred = np.load(f"{gal_type}/ind_pred_all_{fun_type_sats}_sats_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                ind_cent_pred = np.load(f"{gal_type}/ind_pred_all_{fun_type}_cent_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
        else:
            if mode == 'bins':            
                pos_sats_pred = np.load(f"{gal_type}/pos_pred_{fun_type_sats}_sats_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                pos_cent_pred = np.load(f"{gal_type}/pos_pred_{fun_type}_cent_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_sats_pred = np.load(f"{gal_type}/vel_pred_{fun_type_sats}_sats_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_cent_pred = np.load(f"{gal_type}/vel_pred_{fun_type}_cent_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
            elif mode == 'all':
                pos_sats_pred = np.load(f"{gal_type}/pos_pred_all_{fun_type_sats}_sats_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                pos_cent_pred = np.load(f"{gal_type}/pos_pred_all_{fun_type}_cent_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_sats_pred = np.load(f"{gal_type}/vel_pred_all_{fun_type_sats}_sats_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                vel_cent_pred = np.load(f"{gal_type}/vel_pred_all_{fun_type}_cent_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                ind_sats_pred = np.load(f"{gal_type}/ind_pred_all_{fun_type_sats}_sats_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")
                ind_cent_pred = np.load(f"{gal_type}/ind_pred_all_{fun_type}_cent_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy")

        
        pos_pred = np.vstack((pos_cent_pred, pos_sats_pred))
        vel_pred = np.vstack((vel_cent_pred, vel_sats_pred))
        
        pos_cent_true = SubhaloPos[index_cent[:len(pos_cent_pred)]]
        pos_sats_true = SubhaloPos[index_sats[:len(pos_sats_pred)]]
        vel_cent_true = SubhaloVel[index_cent[:len(pos_cent_pred)]]
        vel_sats_true = SubhaloVel[index_sats[:len(pos_sats_pred)]]

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
            np.save(f"{gal_type}/xil0_mean_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{fp_dm}_{snapshot:d}.npy", xil0_true_mean)
            np.save(f"{gal_type}/xil0_shuff_mean_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{fp_dm}_{snapshot:d}.npy", xil0_pred_mean)
            np.save(f"{gal_type}/ratl0_shuff_mean_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{fp_dm}_{snapshot:d}.npy", ratl0_mean)
            np.save(f"{gal_type}/ratl0_shuff_err_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{fp_dm}_{snapshot:d}.npy", ratl0_err)
            np.save(f"{gal_type}/xil2_mean_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{fp_dm}_{snapshot:d}.npy", xil2_true_mean)
            np.save(f"{gal_type}/xil2_shuff_mean_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{fp_dm}_{snapshot:d}.npy", xil2_pred_mean)
            np.save(f"{gal_type}/ratl2_shuff_mean_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{fp_dm}_{snapshot:d}.npy", ratl2_mean)
            np.save(f"{gal_type}/ratl2_shuff_err_{n_gal}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{fp_dm}_{snapshot:d}.npy", ratl2_err)
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
            plt.savefig(f'figs/xil0l2_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{snapshot:d}.png')
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
        
            plt.savefig(f'figs/xil0l2_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{snapshot:d}.png')
            plt.close()#plt.show()
            quit()
        

        if fit_type == 'plane':
            if mode == 'bins':
                np.save(f"{gal_type}/ratl0_mean_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl0_mean)
                np.save(f"{gal_type}/ratl0_err_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl0_err)
                np.save(f"{gal_type}/ratl2_mean_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl2_mean)
                np.save(f"{gal_type}/ratl2_err_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl2_err)
            elif mode == 'all':
                np.save(f"{gal_type}/ratl0_mean_all_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl0_mean)
                np.save(f"{gal_type}/ratl0_err_all_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl0_err)
                np.save(f"{gal_type}/ratl2_mean_all_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl2_mean)
                np.save(f"{gal_type}/ratl2_err_all_{fun_type}_{secondary}_{tertiary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl2_err)

        else:
            if mode == 'bins':
                np.save(f"{gal_type}/ratl0_mean_{fit_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl0_mean)
                np.save(f"{gal_type}/ratl0_err_{fit_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl0_err)
                np.save(f"{gal_type}/ratl2_mean_{fit_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl2_mean)
                np.save(f"{gal_type}/ratl2_err_{fit_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl2_err)
            elif mode == 'all':
                np.save(f"{gal_type}/ratl0_mean_all_{fun_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl0_mean)
                np.save(f"{gal_type}/ratl0_err_all_{fun_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl0_err)
                np.save(f"{gal_type}/ratl2_mean_all_{fun_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl2_mean)
                np.save(f"{gal_type}/ratl2_err_all_{fun_type}_{secondary}{vrad_str}{splash_str}{pseudo_str}{drad_str}{fixocc_str}_{n_gal}_{fp_dm}_{snapshot:d}.npy", ratl2_err)
                
quit()

# mass bins  # notice slightly lower upper limit cause few halos
mbins = np.logspace(10, 14, 41)
print("number of halos above the last mass bin = ", np.sum(mbins[-1] < GrMcrit))
choice = (mbins[-1] >= GrMcrit) & (mbins[0] < GrMcrit)
                
GroupCountShuff = GroupCount.copy()
GroupCountCentShuff = GroupCountCent.copy()
GroupCountSatsShuff = GroupCountSats.copy()
for i in range(len(mbins)-1):
    choice = ((mbins[i]) < GrMcrit) & ((mbins[i+1]) >= GrMcrit)
    nhalo = np.sum(choice)
    if nhalo == 0: continue
    
    cts = GroupCount[choice]
    cts_cent = GroupCountCent[choice]
    cts_sats = GroupCountSats[choice]
    ngal = np.sum(cts)
    if ngal == 0: continue
    
    pos = GroupPos[choice]
    cts = cts.astype(pos.dtype)
    cts_sats = cts_sats.astype(pos.dtype)
    cts_cent = cts_cent.astype(pos.dtype)
    cts_shuff = cts.copy()
    cts_sats_shuff = cts_sats.copy()
    cts_cent_shuff = cts_cent.copy()
    np.random.shuffle(cts_shuff)
    np.random.shuffle(cts_sats_shuff)
    np.random.shuffle(cts_cent_shuff)

    GroupCountShuff[choice] = cts_shuff
    GroupCountCentShuff[choice] = cts_cent_shuff
    GroupCountSatsShuff[choice] = cts_sats_shuff

# get the new downsampled numbers # not needed because number is preserved
GroupCountCentCopy, GroupCountCentPred = (GroupCountCent, GroupCountCentShuff)
GroupCountSatsCopy, GroupCountSatsPred = (GroupCountSats, GroupCountSatsShuff)

# it is a bit expensive, but let's get the subhalo positions for both predicted and true
index_halo_copy = index_halo.copy()
num_sats_pred = GroupCountSatsPred[GroupCountSatsPred > 0]
hid_sats_pred = index_halo_copy[GroupCountSatsPred > 0]
index_halo_copy = index_halo.copy()
num_sats_true = GroupCountSatsCopy[GroupCountSatsCopy > 0]
hid_sats_true = index_halo_copy[GroupCountSatsCopy > 0]

# positions of the central and satellite galaxies
pos_cent_pred = GroupPos[GroupCountCentPred > 0]
pos_cent_true = GroupPos[GroupCountCentCopy > 0]
pos_sats_pred = get_pos_sats(hid_sats_pred, num_sats_pred)
pos_sats_true = get_pos_sats(hid_sats_true, num_sats_true) # not a problem with this
pos_pred = np.vstack((pos_cent_pred, pos_sats_pred))
pos_true = np.vstack((pos_cent_true, pos_sats_true))
w_pred = np.ones(pos_pred.shape[0], dtype=pos_pred.dtype)
w_true = np.ones(pos_true.shape[0], dtype=pos_true.dtype)

# N_dim should maybe be 5
rat_mean, rat_err, corr_shuff_mean, corr_shuff_err, corr_true_mean, corr_true_err, _ = get_jack_corr(pos_true, w_true, pos_pred, w_pred, Lbox, N_dim=3, bins=rbins)

'''
# remove
plt.figure(figsize=(9, 7))
plt.plot(rbinc, np.ones(len(rbinc)), 'k--')
plt.errorbar(rbinc, corr_true_mean*rbinc**2, yerr=corr_true_err*rbinc**2, ls='-', capsize=4, color='black', label='True')
plt.errorbar(rbinc, corr_shuff_mean*rbinc**2, yerr=corr_shuff_err*rbinc**2, ls='-', capsize=4, color='dodgerblue', label='Predicted')
plt.xscale('log')
#plt.savefig(f'figs/corr_{fun_type}_{secondary}_{tertiary}_{snapshot:d}.png')

plt.figure(figsize=(9, 7))
plt.plot(rbinc, np.ones(len(rbinc)), 'k--')
plt.errorbar(rbinc, rat_mean, yerr=rat_err, ls='-', capsize=4, color='dodgerblue', label='Predicted')
plt.xscale('log')
#plt.savefig(f'figs/corr_{fun_type}_{secondary}_{tertiary}_{snapshot:d}.png')
plt.show()
quit()
'''

np.save(f"{gal_type}/corr_rat_mean_shuff_fp_{snapshot:d}.npy", rat_mean)
np.save(f"{gal_type}/corr_rat_err_shuff_fp_{snapshot:d}.npy", rat_err)
np.save(f"{gal_type}/corr_mean_shuff_fp_{snapshot:d}.npy", corr_shuff_mean)
np.save(f"{gal_type}/corr_err_shuff_fp_{snapshot:d}.npy", corr_shuff_err)
np.save(f"{gal_type}/corr_mean_true_fp_{snapshot:d}.npy", corr_true_mean)
np.save(f"{gal_type}/corr_err_true_fp_{snapshot:d}.npy", corr_true_err)

quit()

'''
# centrals and satellies
xi = Corrfunc.theory.xi(Lbox, 16, rbins, pos[cts > 0, 0], pos[cts > 0, 1], pos[cts > 0, 2], weights=cts[cts > 0], weight_type="pair_product")['xi']
xi_shuff = Corrfunc.theory.xi(Lbox, 16, rbins, pos[cts_shuff > 0, 0], pos[cts_shuff > 0, 1], pos[cts_shuff > 0, 2], weights=cts_shuff[cts_shuff > 0], weight_type="pair_product")['xi']
plt.plot(rbinc, xi_shuff/xi, color=hexcolors_bright[i], label=r'$\log M = %.1f$'%mbinc[i])

rat_mean, rat_err, corr_shuff_mean, corr_shuff_err, corr_true_mean, corr_true_err, _ = get_jack_corr(pos[cts > 0], cts[cts > 0], pos[cts_shuff > 0], cts_shuff[cts_shuff > 0], Lbox, N_dim=5, bins=rbins)

plt.errorbar(rbinc*(1.+0.03*i), rat_mean, yerr=rat_err, ls='-', capsize=4, color=hexcolors_bright[i], label=r'$\log M = %.1f$'%mbinc[i])
'''
'''
# centrals
rat_mean, rat_err, corr_shuff_mean, corr_shuff_err, corr_true_mean, corr_true_err, _ = get_jack_corr(pos[cts_cent > 0], cts[cts_cent > 0], pos[cts_cent_shuff > 0], cts_cent_shuff[cts_cent_shuff > 0], Lbox, N_dim=5, bins=rbins)

#np.savez(f'{gal_type}/{gal_type}_corr_bin_{i:d}_cent_fp_{snapshot:d}.npz', mean=rat_mean, err=rat_err, shuff_mean=corr_shuff_mean, shuff_err=corr_shuff_err, true_mean=corr_true_mean, true_err=corr_true_err, binc=rbinc, logm=mbinc[i])
    
# satellites
rat_mean, rat_err, corr_shuff_mean, corr_shuff_err, corr_true_mean, corr_true_err, _ = get_jack_corr(pos[cts_sats > 0], cts[cts_sats > 0], pos[cts_sats_shuff > 0], cts_sats_shuff[cts_sats_shuff > 0], Lbox, N_dim=5, bins=rbins)

#np.savez(f'{gal_type}/{gal_type}_corr_bin_{i:d}_sats_fp_{snapshot:d}.npz', mean=rat_mean, err=rat_err, shuff_mean=corr_shuff_mean, shuff_err=corr_shuff_err, true_mean=corr_true_mean, true_err=corr_true_err, binc=rbinc, logm=mbinc[i])
'''

"""
per weird idea
GroupCountCopy = GroupCount.copy()

GroupCountPred = GroupCountSatsPred+GroupCountCentPred
diff = np.abs(np.sum(GroupCountPred) - np.sum(GroupCount))
print("difference = ", diff)
if np.sum(GroupCountPred) < np.sum(GroupCount):
GroupCountChange = GroupCountCopy.copy()
else:
GroupCountChange = GroupCountPred.copy()
index_halo = np.arange(len(GroupCountChange), dtype=int)
index_halo = index_halo[GroupCountChange > 0]
count = GroupCountChange[GroupCountChange > 0].astype(float)
count /= np.sum(count)
#index_all = np.repeat(index_halo, count)
samples = np.random.choice(index_halo, diff, replace=False, p=count)
GroupCountChange[samples] -= 1
if np.sum(GroupCountPred) < np.sum(GroupCountCopy):
GroupCountCopy = GroupCountChange
else:
GroupCountPred = GroupCountChange

print("predicted and true total number of galaxies", np.sum(GroupCountPred), np.sum(GroupCountCopy))
GroupCountPred = GroupCountPred.astype(GroupPos_fp.dtype)
GroupCountCopy = GroupCountCopy.astype(GroupPos_fp.dtype)
"""
