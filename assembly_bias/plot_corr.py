"""
Plot in a large plot all the galaxy assembly bias for all halo properties
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

from matplotlib.ticker import (MultipleLocator, LogitLocator, FormatStrFormatter, AutoMinorLocator)
# parameter names
#fixticks, fixlegend, moveparam name up

zs = [0., 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]
snaps = [264, 237, 214, 179, 151, 129, 94, 80, 69, 51]
z_dict = {}
for i in range(len(zs)):
    z_dict[snaps[i]] = zs[i]

# colors
hexcolors_bright = ['#CC3311','#0077BB','#EE7733','limegreen','#BBBBBB','#33BBEE','#EE3377','#0099BB']
greysafecols = ['#809BC8', 'black', '#FF6666', '#FFCC66', '#64C204']
# 0077BB is dark blue; EE7733 is orange; EE3377 is cyclamen; 33BBEE is blue; CC3311 is brick; 0099BB is dark green-blue; BBBBBB is silver

# simulation parameters
#tng_dir = "/mnt/alan1/boryanah/MTNG/"
tng_dir = "/mnt/alan1/boryanah/MTNG/dm_arepo/"
Lbox = 500. # Mpc/h
gal_types = ['LRG', 'ELG']
ignore_type = ''#'LRG'
fit_type = 'ramp' # 'ramp' # 'plane'
#fun_types = ['tanh', 'erf', 'gd', 'abs', 'arctan', 'linear']
fun_type = 'linear'
fun_type_sats = 'linear'
#fp_dm = 'fp'
fp_dm = 'dm'
mode = 'all' #'all' # fitting once for all # 'bins' fitting in bins
n_gal = '2.0e-03'# '2.0e-03' # '7.0e-04'
p1, p2 = n_gal.split('e-0')
snapshots = [179, 264]
zs = [1., 0.]
cs = ['gold', 'goldenrod']
als = [0.3, 0.3]

#params = ['GroupConc', 'Group_M_Crit200_peak', 'GroupGamma', 'GroupVelDispSqR', 'GroupShearAdapt', 'GroupEnvAdapt', 'GroupEnv_R1.5', 'GroupShear_R1.5', 'GroupConcRad', 'GroupVirial', 'GroupSnap_peak', 'GroupVelDisp', 'GroupPotential', 'Group_M_Splash', 'Group_R_Splash', 'GroupNsubs', 'GroupSnap_peak', 'GroupHalfmassRad']#, 'GroupMarkedEnv_R2.0_s0.25_p2'
#params = ['GroupConc', 'Group_M_Crit200_peak', 'GroupShearAdapt', 'GroupEnvAdapt', 'Group_R_Splash', 'GroupNsubs']
params = ['GroupConc', 'SubhaloMass_peak', 'GroupShearAdapt', 'GroupEnvAdapt', 'Group_R_Splash', 'GroupVelAni']
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

if fit_type == 'plane':
    #nrow = 3
    #figsize=(22, 5*2.)
    #lists = [0, 5, 10]
    #number = 10 # last row
    nrow = 5
    figsize=(8.5*2., 20)
    lists = [0, 3, 6, 9, 12]
    number = 12 # last row
else:
    #figsize=(10, 12) # og
    #lists = [0, 2, 4]
    #number = 4 # last row
    nrow = 2
    figsize=(16, 8)
    lists = [0, 3]
    number = 3

fig, axes = plt.subplots(nrow, len(secondaries)//nrow, figsize=figsize)

counter = 0
for i, snapshot in enumerate(snapshots):
    z = zs[i]
    z_label = f"z = {z:.1f}"
    if fp_dm == 'dm' and "arepo" not in tng_dir:
        snapshot_dm = snapshot+5
    else:
        snapshot_dm = snapshot
    
    for gal_type in gal_types:
        gal_label = "{\\rm "+f"{gal_type}s"+"}"

        if gal_type == ignore_type:
            counter += 1
            continue
        
        # load correlation function
        if gal_type == 'ELG':
            drad_str = ""
            cond_str = ""
            pseudo_str = ""
            #drad_str = "_drad"
            #cond_str = "_cond"
            #pseudo_str = "_pseudo"
        else:
            drad_str = ""
            cond_str = ""
            pseudo_str = ""

        rat_mean = np.load(f"{gal_type:s}/corr_rat_shuff_mean_{n_gal}{pseudo_str}{drad_str}{cond_str}_{fp_dm}_{snapshot:d}.npy")
        rat_err = np.load(f"{gal_type:s}/corr_rat_shuff_err_{n_gal}{pseudo_str}{drad_str}{cond_str}_{fp_dm}_{snapshot:d}.npy")
        rbinc = np.load(f"{gal_type:s}/rbinc.npy")

        for i_pair in range(len(secondaries)):
            secondary = secondaries[i_pair]
            if fit_type == 'plane':
                tertiary = tertiaries[i_pair]

            if fit_type == 'plane':
                label = r"$\texttt{%s}, \ \texttt{%s}$"%(('\_'.join(secondary.split('_'))).split('Group')[-1].split('Subhalo')[-1], ('\_'.join(tertiary.split('_'))).split('Group')[-1].split('Subhalo')[-1])
                
            else:
                label = r"$\texttt{%s}$"%(('\_'.join(secondary.split('_'))).split('Group')[-1].split('Subhalo')[-1])


            plt.subplot(nrow, len(secondaries)//nrow, i_pair+1)
            plt.axhline(y=1., color='k', ls='--')

            if fit_type == 'plane':
                if mode == 'bins':
                    rat_mean_this = np.load(f"{gal_type:s}/corr_rat_mean_{fun_type:s}_{secondary:s}_{tertiary:s}{pseudo_str}{drad_str}{cond_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                    rat_err_this = np.load(f"{gal_type:s}/corr_rat_err_{fun_type:s}_{secondary:s}_{tertiary:s}{pseudo_str}{drad_str}{cond_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                elif mode == 'all':
                    rat_mean_this = np.load(f"{gal_type:s}/corr_rat_mean_all_{fun_type:s}_{secondary:s}_{tertiary:s}{pseudo_str}{drad_str}{cond_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                    rat_err_this = np.load(f"{gal_type:s}/corr_rat_err_all_{fun_type:s}_{secondary:s}_{tertiary:s}{pseudo_str}{drad_str}{cond_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
            else:
                if mode == 'bins':
                    rat_mean_this = np.load(f"{gal_type:s}/corr_rat_mean_{fun_type:s}_{secondary:s}{pseudo_str}{drad_str}{cond_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                    rat_err_this = np.load(f"{gal_type:s}/corr_rat_err_{fun_type:s}_{secondary:s}{pseudo_str}{drad_str}{cond_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                elif mode == 'all':
                    rat_mean_this = np.load(f"{gal_type:s}/corr_rat_mean_all_{fun_type:s}_{secondary:s}{pseudo_str}{drad_str}{cond_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")
                    rat_err_this = np.load(f"{gal_type:s}/corr_rat_err_all_{fun_type:s}_{secondary:s}{pseudo_str}{drad_str}{cond_str}_{n_gal}_{fp_dm:s}_{snapshot:d}.npy")

            # turn into shilver shaded
            #plt.errorbar(rbinc, rat_mean, ls='-', capsize=4, lw=1.5, color=cs[i])
            plt.errorbar(rbinc, rat_mean, ls=':', capsize=4, lw=1.5, color=hexcolors_bright[counter])
            #plt.fill_between(rbinc, rat_mean+rat_err, rat_mean-rat_err, color=cs[i], alpha=als[i])
            plt.fill_between(rbinc, rat_mean+rat_err, rat_mean-rat_err, color=hexcolors_bright[counter], alpha=als[i])

            if i_pair == len(secondaries)-1 and gal_type == 'LRG':
                plt.errorbar(rbinc, rat_mean_this, yerr=rat_err_this, ls='-', capsize=4, color=hexcolors_bright[counter], label=rf"${z_label}, \ {gal_label}$")
            if i_pair == len(secondaries)-2 and gal_type == 'ELG':
                plt.errorbar(rbinc, rat_mean_this, yerr=rat_err_this, ls='-', capsize=4, color=hexcolors_bright[counter], label=rf"${z_label}, \ {gal_label}$")
            else:
                plt.errorbar(rbinc, rat_mean_this, yerr=rat_err_this, ls='-', capsize=4, color=hexcolors_bright[counter])

            if counter == 0:
                if fit_type == 'ramp':
                    plt.text(x=0.36, y=0.83, s=label, transform=plt.gca().transAxes, fontsize=18)
                else:
                    plt.text(x=0.2, y=0.83, s=label, transform=plt.gca().transAxes, fontsize=18)

            plt.xscale('log')
            plt.ylim([0.75, 1.18])
            ymin, ymax = plt.gca().get_ylim()
            ymin = np.floor(ymin*10.)/10.
            ymax = np.ceil(ymax*10.)/10.
            plt.gca().set_yticks(np.arange(ymin, ymax, 0.1))
            #plt.gca().tick_params(axis='x', which='minor', top=True, direction='in', length=2)
            #plt.gca().xaxis.set_minor_locator(LogitLocator(30))
            #plt.gca().minorticks_on() # gives to both
            #plt.xlim([0.1, 40.]) # og
            plt.xlim([1.0, 30.])
            if i_pair >= number:
                plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
            if i_pair in lists:
                plt.ylabel(r'$\xi_{\rm pred}/\xi_{\rm true}$')
            
            if i_pair not in lists:
                plt.gca().axes.yaxis.set_ticks([])
            if i_pair < number:
                plt.gca().axes.xaxis.set_ticks([])
            
        counter += 1

plt.subplot(nrow, len(secondaries)//nrow, len(secondaries)//nrow)
label = r"$n_{\rm gal} = %s \times 10^{-%s}$"%(p1, p2)
plt.text(0.45, 0.1, s=label, transform=plt.gca().transAxes, fontsize=20)

plt.subplot(nrow, len(secondaries)//nrow, len(secondaries))
plt.legend(ncol=1, fontsize=16, loc='lower right', frameon=False)

plt.subplot(nrow, len(secondaries)//nrow, len(secondaries)-1)
plt.legend(ncol=1, fontsize=16, loc='lower right', frameon=False)

plt.savefig(f"figs/corr_{fit_type}_{n_gal}_{fp_dm:s}.pdf", bbox_inches='tight', pad_inches=0.)
#plt.close()
plt.show()




