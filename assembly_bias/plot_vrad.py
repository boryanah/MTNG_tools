import numpy as np
import matplotlib.pyplot as plt

import Corrfunc
import plotparams
plotparams.buba()

# colors
hexcolors_bright = ['#CC3311','#0077BB','#EE7733','limegreen','#BBBBBB','#33BBEE','#EE3377','#0099BB']
# 0077BB is dark blue; EE7733 is orange; EE3377 is cyclamen; 33BBEE is blue; CC3311 is brick; 0099BB is dark green-blue; BBBBBB is silver

# simulation parametes
fp_dm = 'fp'
gal_types = ['LRG', 'ELG']
n_gal = '2.0e-03' #['7.0e-04', '2.0e-03']
p1, p2 = n_gal.split('e-0')
snapshots = [179]#, 264]
zs = [1.]#, 0.]

# start with a rectangular Figure
plt.figure(figsize=(9, 7))

"""
# definitions for the axes
left, width = 0.14, 0.85#0.1, 0.65
bottom, height = 0.1, 0.25#0.2#65
spacing = 0.005
rect_scatter = [left, bottom + (height + spacing), width, 0.6]
rect_histx = [left, bottom, width, height]
ax_scatter = plt.axes(rect_scatter)
ax_histx = plt.axes(rect_histx)
"""

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
        #corr_true_mean = np.load(f"{gal_type:s}/corr_mean_{n_gal}_{fp_dm:s}_{snapshot_dm:d}.npy") 
        #corr_shuff_mean = np.load(f"{gal_type:s}/corr_shuff_mean_{n_gal}_{fp_dm:s}_{snapshot_dm:d}.npy")
        #rat_shuff_mean = np.load(f"{gal_type:s}/corr_rat_shuff_mean_{n_gal}_{fp_dm:s}_{snapshot_dm:d}.npy")
        #rat_shuff_mean_np = np.load(f"{gal_type:s}/corr_rat_shuff_mean_{n_gal}_vrad_{fp_dm:s}_{snapshot_dm:d}.npy")
        
        #rat_shuff_mean = np.load(f"{gal_type:s}/ratl2_shuff_mean_{n_gal}_{fp_dm:s}_{snapshot_dm:d}.npy") # xil2
        #rat_shuff_mean_np = np.load(f"{gal_type:s}/ratl2_shuff_mean_{n_gal}_vrad_{fp_dm:s}_{snapshot_dm:d}.npy") # xil2
        #rbinc = np.load(f"{gal_type:s}/rbinc.npy") # xil2
        
        rat_shuff_mean = np.load(f"{gal_type:s}/pair_rat_shuff_mean_{n_gal}_{fp_dm:s}_{snapshot_dm:d}.npy") # pair
        rat_shuff_mean_np = np.load(f"{gal_type:s}/pair_rat_shuff_mean_{n_gal}_vrad_{fp_dm:s}_{snapshot_dm:d}.npy") # pair
        rbinc = np.load(f"{gal_type:s}/pair_rbinc.npy") # pair
        

        plt.plot(rbinc, rat_shuff_mean_np, color=hexcolors_bright[counter], ls='-', label=rf"${z_label}, \ {gal_label}$")
        plt.plot(rbinc, rat_shuff_mean, color=hexcolors_bright[counter], ls='--')

        counter += 1

plt.plot([], [], color='black', ls='-', label=r'$v_{\rm rad}, \ {\rm sampled}$')
plt.plot([], [], color='black', ls='--', label=r'$v_{\rm rad}, \ {\rm random}$')
label = r"$n_{\rm gal} = %s \times 10^{-%s}$"%(p1, p2)
plt.text(0.6, 0.1, s=label, transform=plt.gca().transAxes)
plt.axhline(y=1, color='black', ls='--')
#plt.xlim([rbinc[0], rbinc[-1]])
xmin, xmax = plt.gca().get_xlim()
#plt.xlim([0.3, 10.])
#plt.ylim([0.85, 1.1])
#plt.ylim([0.75, 1.25])
plt.ylim([0.5, 1.25])
plt.legend(ncol=2, fontsize=22)
plt.xscale('log')
plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
plt.ylabel(r'$\hat p_{\rm pred}(r) / \hat p_{\rm true}(r)$') # pair
#plt.ylabel(r'$\xi_{\ell=2, \ {\rm pred}}(r) / \xi_{\ell=2, \ {\rm true}}(r)$') # xil2
plt.savefig(f"figs/vrad_{n_gal}.pdf", bbox_inches='tight', pad_inches=0.)
plt.show()

