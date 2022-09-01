import numpy as np
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

snapshot = 179
ignore_str = ""
boundaries = ['r200m', 'rsplash']
labels = ['Virial radius', 'Splashback']
colors = ['#33BBEE','#EE7733','#CC3311','#0099BB','#0077BB','#EE3377','#BBBBBB']

# colors
hexcolors_bright = ['#CC3311','#0077BB','#EE7733','limegreen','#BBBBBB','#33BBEE','#EE3377','#0099BB']
# 0077BB is dark blue; EE7733 is orange; EE3377 is cyclamen; 33BBEE is blue; CC3311 is brick; 0099BB is dark green-blue; BBBBBB is silver

# simulation parametes
fp_dm = 'fp'
gal_types = ['LRG', 'ELG']
n_gal = '2.0e-03' #['7.0e-04', '2.0e-03']
p1, p2 = n_gal.split('e-0')
snapshots = [179]#, 264]
zs = [1., 0.]


plt.figure(figsize=(9, 7))
counter = 0
for i, snapshot in enumerate(snapshots):
    z = zs[i]
    z_label = f"z = {z:.1f}"
    if fp_dm == 'dm':
        snapshot_dm = snapshot+5
    else:
        snapshot_dm = snapshot
    
    for gal_type in gal_types:
        gal_label = "{\\rm "+f"{gal_type}s"+"}"

        # load correlation function    
        data_corr = np.load(f"data/corr_{gal_type}_rsplash_{snapshot:d}{ignore_str}.npz")
        rbinc = data_corr['rbinc']
        rat_mean_s = data_corr['rat_mean']
        rat_err_s = data_corr['rat_err']
        #corr_shuff_mean = data_corr['corr_shuff_mean']
        #corr_shuff_err = data_corr['corr_shuff_err']
        #corr_true_mean = data_corr['corr_true_mean']
        #corr_true_err = data_corr['corr_true_err']

        data_corr = np.load(f"data/corr_{gal_type}_r200m_{snapshot:d}{ignore_str}.npz")
        rbinc = data_corr['rbinc']
        rat_mean_r = data_corr['rat_mean']
        rat_err_r = data_corr['rat_err']
        
        #plt.errorbar(rbinc*(1.+0.05*counter), rat_mean_s, yerr=rat_err_s, capsize=4, color=hexcolors_bright[counter], ls='-', label=rf"${z_label}, \ {gal_label}$")
        #plt.errorbar(rbinc*(1.-0.05*counter), rat_mean_r, yerr=rat_err_r, capsize=4, color=hexcolors_bright[counter], ls='--')
        if gal_type == 'ELG':
            plt.errorbar(rbinc, rat_mean_s, capsize=4, color=hexcolors_bright[counter], ls='-', label=rf"${z_label}, \ {gal_label}$")
            plt.errorbar(rbinc, rat_mean_r, capsize=4, color=hexcolors_bright[counter], ls='--')
        else:
            plt.errorbar(rbinc*(1.+0.05*(counter+1)), rat_mean_s, yerr=rat_err_s, capsize=4, color=hexcolors_bright[counter], ls='-', label=rf"${z_label}, \ {gal_label}$")
            plt.errorbar(rbinc*(1.-0.05*counter), rat_mean_r, yerr=rat_err_r, capsize=4, color=hexcolors_bright[counter], ls='--')

        counter += 1
plt.plot([], [], color='black', ls='-', label='Splashback')
plt.plot([], [], color='black', ls='--', label='Vir. radius')
label = r"$n_{\rm gal} = %s \times 10^{-%s}$"%(p1, p2)
plt.text(0.1, 0.63, s=label, transform=plt.gca().transAxes)
plt.axhline(y=1, color='k', ls='--')
#plt.ylim([0.75, 1.25])
xmin, xmax = plt.gca().get_xlim()
#xmin = 0.1
xmin = 1.
plt.xlim([xmin, xmax])
plt.ylim([0.75, 1.25])
plt.legend(ncol=2, fontsize=22)
plt.xscale('log')
plt.ylabel(r'$\xi_{\rm pred}/\xi_{\rm true}$')
plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
plt.savefig(f'figs/corr_rat_{snapshot:d}{ignore_str}.pdf', bbox_inches='tight', pad_inches=0.)
plt.show()
