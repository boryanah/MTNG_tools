import numpy as np
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

gal_type = 'LRG'

colors=['m', 'r', 'orange', 'y', 'g', 'aquamarine', 'c', 'b', 'purple', 'black']
color = [0., 0.5, 0.5]
cent_sats = 'sats'
#cent_sats = 'cent'
snapshot = 179
fit_type = 'plane'
gal_type = 'LRG'
if cent_sats == 'cent':
    fun_types = ['tanh', 'erf', 'gd', 'abs', 'arctan', 'linear']
else:
    fun_types = ['linear']

params = ['GroupVirial', 'GroupConc', 'GroupVelDisp', 'GroupShear_R2', 'GroupEnv_R2', 'GroupMarkedEnv_R2_s0.25_p2']
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
#secondaries = ['GroupEnv_R2']
#tertiaries = ['GroupConc']
#tertiaries = ['GroupShear_R2']

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

        a_arr = np.load(f"{gal_type:s}/a_arr_{cent_sats:s}_{fun_type:s}_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy")
        b_arr = np.load(f"{gal_type:s}/b_arr_{cent_sats:s}_{fun_type:s}_{secondary:s}_{tertiary:s}_fp_{snapshot:d}.npy")

        mbinc = np.load(f"{gal_type:s}/mbinc.npy")
        if cent_sats == 'sats':
            choice = (mbinc > 3.e12)
        else:
            choice = (mbinc > 1.e12)

        ylabel = r"$\texttt{%s}$"%(('\_'.join(secondary.split('_'))).split('Group')[-1])
        xlabel = r"$\texttt{%s}$"%(('\_'.join(tertiary.split('_'))).split('Group')[-1])

        plt.subplot(3, len(secondaries)//3, i_pair+1)
        plt.plot(a_arr[choice], b_arr[choice], color='dodgerblue', label=label)

        plt.legend(fontsize=14)

        plt.axvline(x=0., color='k', ls='--')
        plt.axhline(y=0., color='k', ls='--')
        plt.text(a_arr[choice][0], b_arr[choice][0], f'{mbinc[choice][0]:.1e}', fontsize=12)
        plt.text(a_arr[choice][-1], b_arr[choice][-1], f'{mbinc[choice][-1]:.1e}', fontsize=12)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        '''
        if i_pair >= number:
            plt.xlabel(xlabel)
        if i_pair in lists:
            plt.ylabel(ylabel)
        '''
        if i_pair not in lists:
            plt.gca().axes.yaxis.set_ticks([])
        if i_pair < number:
            plt.gca().axes.xaxis.set_ticks([])
        if cent_sats == 'sats':
            plt.xlim([-3., 3.])
            plt.ylim([-3., 3.])
        else:
            plt.xlim([-3., 3.])
            plt.ylim([-3., 3.])
        #plt.legend(ncol=4)
        #plt.show()

    plt.savefig(f"figs_ab/{gal_type:s}_{fun_type:s}_{cent_sats:s}_fp_{snapshot:d}.png")
    plt.close()
    plt.show()
