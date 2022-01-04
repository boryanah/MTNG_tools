import os

import numpy as np
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

"""
# matplotlib settings
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times'],'size':18})
rc('text', usetex=True)
"""

hexcolors_bright = ['#0077BB','#33BBEE','#0099BB','#EE7733','#CC3311','#EE3377','#BBBBBB']
greysafecols = ['#809BC8', 'black', '#FF6666', '#FFCC66', '#64C204']

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
#fp_dm = 'fp'; snapshot = 179; snapshot_fp = 179
fp_dm = 'dm'; snapshot = 184; snapshot_fp = 179
#gal_type = 'ELG'
gal_type = 'LRG'
fit_type = 'ramp'
#fit_type = 'plane'
#fun_types = ['tanh', 'erf', 'gd', 'abs', 'arctan', 'linear']
#fun_types = ['erf', 'linear']
fun_types = ['linear']
#mode = 'bins'
mode = 'all'

# GroupConc or Rad
params = ['GroupVirial', 'GroupConcRad', 'GroupVelDisp', 'GroupShear_R2', 'GroupEnv_R2', 'GroupMarkedEnv_R2_s0.25_p2']
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

rat_mean = np.load(f"{gal_type:s}/corr_rat_mean_shuff_fp_{snapshot_fp:d}.npy")
rat_err = np.load(f"{gal_type:s}/corr_rat_err_shuff_fp_{snapshot_fp:d}.npy")
rbinc = np.load(f"{gal_type:s}/rbinc.npy")

if fit_type == 'plane':
    figsize=(18, 5.5*2.)
    lists = [0, 5, 10]
    number = 10
else:
    figsize=(10, 12)
    lists = [0, 2, 4]
    number = 4

for fun_type in fun_types:
    fig, axes = plt.subplots(3, len(secondaries)//3, figsize=figsize)
    for i_pair in range(len(secondaries)):
        secondary = secondaries[i_pair]
        if fit_type == 'plane':
            tertiary = tertiaries[i_pair]

        if fit_type == 'plane':
            label = r"$\texttt{%s}, \ \texttt{%s}$"%(('\_'.join(secondary.split('_'))).split('Group')[-1], ('\_'.join(tertiary.split('_'))).split('Group')[-1])
        else:
            label = r"$\texttt{%s}$"%(('\_'.join(secondary.split('_'))).split('Group')[-1])


        plt.subplot(3, len(secondaries)//3, i_pair+1)
        plt.plot(rbinc, np.ones(len(rbinc)), 'k--')

        if fit_type == 'plane':
            if mode == 'bins':
                rat_mean_this = np.load(f"{gal_type:s}/corr_rat_mean_{fun_type:s}_{secondary:s}_{tertiary:s}_{fp_dm:s}_{snapshot:d}.npy")
                rat_err_this = np.load(f"{gal_type:s}/corr_rat_err_{fun_type:s}_{secondary:s}_{tertiary:s}_{fp_dm:s}_{snapshot:d}.npy")
            elif mode == 'all':
                rat_mean_this = np.load(f"{gal_type:s}/corr_rat_mean_all_{fun_type:s}_{secondary:s}_{tertiary:s}_{fp_dm:s}_{snapshot:d}.npy")
                rat_err_this = np.load(f"{gal_type:s}/corr_rat_err_all_{fun_type:s}_{secondary:s}_{tertiary:s}_{fp_dm:s}_{snapshot:d}.npy")
        else:
            if mode == 'bins':
                rat_mean_this = np.load(f"{gal_type:s}/corr_rat_mean_{fun_type:s}_{secondary:s}_{fp_dm:s}_{snapshot:d}.npy")
                rat_err_this = np.load(f"{gal_type:s}/corr_rat_err_{fun_type:s}_{secondary:s}_{fp_dm:s}_{snapshot:d}.npy")
            elif mode == 'all':
                rat_mean_this = np.load(f"{gal_type:s}/corr_rat_mean_all_{fun_type:s}_{secondary:s}_{fp_dm:s}_{snapshot:d}.npy")
                rat_err_this = np.load(f"{gal_type:s}/corr_rat_err_all_{fun_type:s}_{secondary:s}_{fp_dm:s}_{snapshot:d}.npy")                
        plt.errorbar(rbinc, rat_mean, yerr=rat_err, ls='-', capsize=4, color='red')#, label='Shuffled')
        plt.errorbar(rbinc*(1.+0.03), rat_mean_this, yerr=rat_err_this, ls='-', lw=1.5, capsize=4, color=greysafecols[0], label=label)


        plt.legend(fontsize=14)
        #plt.text(x=0.6, y=0.1, s="$\log M = %.1f$"%logm, transform=plt.gca().transAxes)

        plt.xscale('log')
        plt.ylim([0.5, 1.5])
        #plt.xlim([1, 50.]) # og
        plt.xlim([0.1, 50.])
        if i_pair >= number:
            plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
        if i_pair in lists:
            plt.ylabel(r'${\rm Ratio}$')
        if i_pair not in lists:
            plt.gca().axes.yaxis.set_ticks([])
        if i_pair < number:
            plt.gca().axes.xaxis.set_ticks([])

    if mode == 'bins':
        plt.savefig(f"figs/{gal_type:s}_corr_pred_{fit_type:s}_{fun_type:s}_{fp_dm:s}_{snapshot:d}.png", bbox_inches='tight', pad_inches=0.)
    else:
        plt.savefig(f"figs/{gal_type:s}_corr_pred_all_{fit_type:s}_{fun_type:s}_{fp_dm:s}_{snapshot:d}.png", bbox_inches='tight', pad_inches=0.)
    #plt.close()
    plt.show()
