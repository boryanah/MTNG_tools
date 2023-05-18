import matplotlib.pyplot as plt
import numpy as np

import plotparams
plotparams.buba()

# colors
hexcolors_bright = ['#CC3311','#0077BB','#EE7733','limegreen','#BBBBBB','#33BBEE','#EE3377','#0099BB']

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
gal_types = ['LRG', 'ELG']
n_gal = '2.0e-03' #['7.0e-04', '2.0e-03']
p1, p2 = n_gal.split('e-0')
fp_dm = 'fp'
snapshots = [179]#, 264]
zs = [1.]#, 0.]

# kinda ugly
data = np.load(f"data/subs_angle.npz")
hist_rand = data['hist_ang'].astype(np.float32)
hist_rand /= np.sum(hist_rand)

# definitions for the axes
left, width = 0.14, 0.85#0.1, 0.65
bottom, height = 0.1, 0.25#0.2#65
spacing = 0.005

rect_scatter = [left, bottom + (height + spacing), width, 0.6]
rect_histx = [left, bottom, width, height]

# start with a rectangular Figure
plt.figure(figsize=(9, 10))
ax_scatter = plt.axes(rect_scatter)
#(direction='in', top=True, right=True)
ax_histx = plt.axes(rect_histx)

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
        
        data = np.load(f"data/{gal_type}_angle.npz")
        hist_LRG = data['hist_ang'].astype(np.float32)
        hist_LRG /= np.sum(hist_LRG)
        if gal_type == 'ELG':
            #hist_rand = data['hist_rand'].astype(np.float32) # og
            #hist_rand /= np.sum(hist_rand) # og
            #plt.plot(binc, hist_rand, color='black', ls='-', label=r"${\rm Random}$")
            
            # https://www.mariushobbhahn.com/2021-05-20-Change_of_variable/ change of var
            #y = np.cos(binc*np.pi/180.) # angle
            #plt.plot(y, hist_rand/np.sqrt(1.-y**2.)*180./np.pi, color='black', ls='-', label=rf"${z_label}, \ {{\rm Subhalos}}$")
            ax_scatter.plot(binc, hist_rand, color='black', ls='-', label=rf"${z_label}, \ {{\rm Subhalos}}$") # cosine
        binc = data['binc']

        print(np.sum(hist_LRG))
        print(binc.shape, binc[:2], binc[-2:])#148551

        #y = np.cos(binc*np.pi/180.) # angle
        #plt.plot(y, hist_LRG/np.sqrt(1.-y**2.)*180./np.pi, color=hexcolors_bright[counter], ls='-', label=rf"${z_label}, \ {gal_label}$") # or nothing
        ax_scatter.plot(binc, hist_LRG, color=hexcolors_bright[counter], ls='-', label=rf"${z_label}, \ {gal_label}$") # cosine
        ax_histx.plot(binc, hist_LRG/hist_rand, color=hexcolors_bright[counter]) 
        if gal_type == 'ELG':
            #plt.plot(np.cos(binc*np.pi/180.), hist_rand, color='black', label=r"${\rm Random \ distn}$")

            cth = (2.*np.random.rand(250000)-1.)
            ang = np.arccos(cth)*180./np.pi
            bs = np.linspace(-0.98, 0.98, 101)
            bc = (bs[1:]+bs[:-1])*.5

            #hist, _ = np.histogram(cth, bins=bs)#, density=True)
            #hist = hist.astype(np.float32)
            #hist /= (np.sum(hist))
            #plt.plot(bc, hist, color='green', label=r"${\rm Random \ cosine \ ready}$")
            hist, _ = np.histogram(np.cos(ang*np.pi/180.), bins=bs)
            hist = hist/np.sum(hist)
            ax_scatter.plot(bc, hist, color='black', lw=1.5, label=r"${\rm Isotropic \ distn}$")
            #ax_scatter.axhline(y=0.01, color='black', ls='--', lw=1.5, label=r"${\rm Isotropic \ distn}$")
            ax_histx.axhline(y=1, color='black', ls='--')
            
        counter += 1
ax_scatter.legend(fontsize=22, loc="upper right")
#plt.xlim([0., 180.])
ax_scatter.set_xlim([1., -1.])
ax_scatter.set_xticklabels([])
ax_histx.set_xlim([1., -1.])
#plt.xlabel(r"$\cos^{-1} ({\hat r}_1 \cdot {\hat r}_n) \ [{\rm deg.}]$")
ax_histx.set_xlabel(r"${\hat r}_1 \cdot {\hat r}_n$")
ax_scatter.set_ylabel(r'${\rm PDF(X)}$')
ax_histx.set_ylabel(r'${\rm PDF(X)/PDF(Subhalos)}$')
ax_scatter.set_yscale('log')
label = r"$n_{\rm gal} = %s \times 10^{-%s}$"%(p1, p2)
#plt.text(0.1, 0.83, s=label, transform=plt.gca().transAxes)
ax_scatter.text(0.1, 0.83, s=label, transform=ax_scatter.transAxes)
plt.savefig("figs/angle.pdf", bbox_inches='tight', pad_inches=0.)
plt.show()
