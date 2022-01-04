import numpy as np
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()


ks = [1., 2., 4., 8.]
ls = ["-", "--", ":"]
lw = [2., 2.5, 3.5]
cols = ['#0077BB','#33BBEE','#0099BB','#EE7733','#CC3311','#EE3377','#BBBBBB']
labels = [r"${\rm 1NN}$", r"${\rm 2NN}$", r"${\rm 4NN}$", r"${\rm 8NN}$"]

tng_dir = "/mnt/alan1/boryanah/MTNG/"
fp_dm = 'fp';snapshot = 179
fp_dm = 'dm';snapshot = 184
#gal_type = 'ELG'
gal_type = 'LRG'
fun_types = ['linear']
fun_type_sats = 'linear'
fit_type = 'ramp'
#fit_type = 'plane'

#data_mean = np.load(f"data/true_mean_data.npy")
#data_err = np.load(f"data/true_err_data.npy")
binc = np.load(f"{gal_type:s}/binc.npy")

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

if fit_type == 'plane':
    figsize=(18, 5.5*2.)
    lists = [0, 5, 10]
    number = 10
else:
    figsize=(10, 12)
    lists = [0, 2, 4]
    number = 4
    

for i in range(len(fun_types)):
    fun_type = fun_types[i]    

    fig, axes = plt.subplots(3, len(secondaries)//3, figsize=figsize)
    
    for i_pair in range(len(secondaries)):
        secondary = secondaries[i_pair]
        if fit_type == 'plane':
            tertiary = tertiaries[i_pair]
        else:
            tertiary = 'None'
            
        if fit_type == 'plane':
            label = r"$\texttt{%s}, \ \texttt{%s}$"%(('\_'.join(secondary.split('_'))).split('Group')[-1], ('\_'.join(tertiary.split('_'))).split('Group')[-1])
        else:
            label = r"$\texttt{%s}$"%(('\_'.join(secondary.split('_'))).split('Group')[-1])

        if fit_type == 'plane':
            other = fit_type+'_'+secondary+"_"+tertiary
        else:
            other = fit_type+'_'+secondary

        if fit_type == 'plane':
            data_rat_mean = np.load(f"{gal_type:s}/kNN_rat_mean_all_{fun_type:s}_{secondary:s}_{tertiary:s}_{fp_dm:s}_{snapshot:d}.npy")
            data_rat_err = np.load(f"{gal_type:s}/kNN_rat_err_all_{fun_type:s}_{secondary:s}_{tertiary:s}_{fp_dm:s}_{snapshot:d}.npy")
        else:
            data_rat_mean = np.load(f"{gal_type:s}/kNN_rat_mean_all_{fun_type:s}_{secondary:s}_{fp_dm:s}_{snapshot:d}.npy")
            data_rat_err = np.load(f"{gal_type:s}/kNN_rat_err_all_{fun_type:s}_{secondary:s}_{fp_dm:s}_{snapshot:d}.npy")

        # plot
        plt.subplot(3, len(secondaries)//3, i_pair+1)
        plt.plot(np.linspace(0, 20, 2), np.ones(2), 'k--', linewidth=2.)

        for i in range(len(ks)):
            hist_rat_mean = data_rat_mean[:, i]
            hist_rat_err = data_rat_err[:, i]

            kNN = ks[i]
            
            plt.errorbar(binc[:, i]*(1+i*0.005)/kNN**(1./3), hist_rat_mean, capsize=4, yerr=hist_rat_err, fmt='o', color=cols[i], label=label)

            if i == 0:
                plt.legend(fontsize=14)

        if i_pair >= number:
            plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
        if i_pair in lists:
            plt.ylabel(r'{\rm Ratio}')
        if i_pair not in lists:
            plt.gca().axes.yaxis.set_ticks([])
        if i_pair < number:
            plt.gca().axes.xaxis.set_ticks([])
        plt.xscale('log')
        plt.ylim([0.96, 1.04])
        plt.xlim([2.8, 32])


    plt.savefig(f"figs/kNN_{gal_type:s}_{fit_type:s}_{fun_type:s}_{fp_dm:s}_{snapshot:d}.png", bbox_inches='tight', pad_inches=0.03)
    plt.show()
    plt.close()
