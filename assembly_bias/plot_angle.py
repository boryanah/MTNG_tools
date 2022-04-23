import matplotlib.pyplot as plt
import numpy as np

import plotparams
plotparams.buba()

# colors
hexcolors_bright = ['#CC3311','#0077BB','#EE7733','#BBBBBB','#33BBEE','#EE3377','#0099BB']

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
gal_types = ['LRG', 'ELG']
n_gal = '2.0e-03' #['7.0e-04', '2.0e-03']
p1, p2 = n_gal.split('e-0')
fp_dm = 'fp'
snapshots = [179]#, 264]
zs = [1.]#, 0.]

counter = 0
plt.figure(figsize=(9, 7))
for i, snapshot in enumerate(snapshots):
    z = zs[i]
    z_label = f"z = {z:.1f}"
    if fp_dm == 'dm':
        snapshot_dm = snapshot+5
    else:
        snapshot_dm = snapshot
    
    for gal_type in gal_types:
        gal_label = "{\\rm "+f"{gal_type}s"+"}"
        
        data = np.load(f"data/{gal_type}_angle.npz")
        hist_LRG = data['hist_ang'].astype(np.float32)
        hist_LRG /= np.sum(hist_LRG)
        if gal_type == 'ELG':
            hist_rand = data['hist_rand'].astype(np.float32)
            hist_rand /= np.sum(hist_rand)
        binc = data['binc']

        plt.plot(binc, hist_LRG, color=hexcolors_bright[counter], ls='-', label=rf"${z_label}, \ {gal_label}$")
        if gal_type == 'ELG':
            plt.plot(binc, hist_rand, color='black', label=r"${\rm Random \ distn}$")
        counter += 1
plt.legend(fontsize=22)
plt.xlim([0., 180.])
plt.xlabel(r"$\cos^{-1} ({\hat r}_1 \cdot {\hat r}_n) \ [{\rm deg.}]$")
#plt.ylabel('PDF')
label = r"$n_{\rm gal} = %s \times 10^{-%s}$"%(p1, p2)
plt.text(0.1, 0.83, s=label, transform=plt.gca().transAxes)
plt.savefig("figs/angle.png")
plt.show()

quit()
plt.figure(figsize=(9, 7))
plt.title("Angle wrt first satellite")
plt.plot(np.cos(binc*np.pi/180.), hist_LRG, label='LRG')
plt.plot(np.cos(binc*np.pi/180.), hist_ELG, label='ELG')
plt.plot(np.cos(binc*np.pi/180.), hist_rand, label='random')
plt.legend()
#plt.xlim([0., 180.])
#plt.xlim([0., 1.])
plt.xlabel('cos(angle)')
plt.ylabel('Number')
plt.savefig("angle.png")
plt.show()
