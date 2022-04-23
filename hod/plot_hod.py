"""
Plot HOD and Poisson noise
"""
import os

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

# colors
#hexcolors_bright = ['#0099BB','#0077BB','#33BBEE','#EE3377','#EE7733','#BBBBBB','#CC3311']
hexcolors_bright = ['#CC3311','#0077BB','#EE7733','#BBBBBB','#33BBEE','#EE3377','#0099BB']

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
gal_types = ['LRG', 'ELG']
n_gal = '2.0e-03' #['7.0e-04', '2.0e-03']

p1, p2 = n_gal.split('e-0')
snapshots = [179, 264]
zs = [1., 0.]

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
    print(z_label)
    for gal_type in gal_types:
        gal_label = "{\\rm "+f"{gal_type}s"+"}"
        print(gal_label)
        data = np.load(f"data/hod_{n_gal}_{gal_type}_{snapshot}.npz")
        hod_cent_gal = data['hod_cent_gal']
        mbinc = data['mbinc']
        hod_sat_gal = data['hod_sat_gal']
        std = data['std']
        poisson = data['poisson']

        alpha  = (poisson/std)**2
        print(alpha)
        print(mbinc)
                
        print("std = ", std)
        print("poisson = ", poisson)
        print("percentage difference = ", 100.*(std-poisson)/std)
        
        #plt.plot(mbinc, hod_gal, color='black', ls='-')
        #plt.plot(mbinc, hod_cent_gal, color=hexcolors_bright[counter], ls='--')
        #plt.errorbar(mbinc, hod_sat_gal, yerr=std, capsize=4, color=hexcolors_bright[counter], ls='-')
        #plt.fill_between(mbinc, poiss_up, poiss_dw, color='black', alpha=0.3)

        ax_scatter.plot(mbinc, hod_cent_gal, color=hexcolors_bright[counter], ls='-', lw=3.5, label=rf"${z_label}, \ {gal_label}$")
        ax_scatter.plot(mbinc, hod_sat_gal, color=hexcolors_bright[counter], ls='-', lw=2.5)

        #ax_histx.plot(mbinc, std, color=hexcolors_bright[counter], ls='-', lw=2.5) # og
        #ax_histx.plot(mbinc, poisson, color=hexcolors_bright[counter], ls='--', lw=2.5) # og
        # TESTING
        ax_histx.axhline(y=1, color='black', ls='--', lw=3.5)
        ax_histx.plot(mbinc, std/poisson, color=hexcolors_bright[counter], ls='-', lw=2.5)
        
        counter += 1
label = r"$n_{\rm gal} = %s \times 10^{-%s}$"%(p1, p2)
ax_scatter.set_xscale('log')
ax_scatter.set_yscale('log')
ax_scatter.legend()
ax_scatter.text(0.1, 0.8, s=label, transform=ax_scatter.transAxes)
#ax_scatter.set_xlabel(r'$M_{\rm halo} \ [M_\odot/h]$')
ax_scatter.set_ylabel(r'$\langle N_{\rm gal} \rangle$')
#ax_histx.plot([], [], color='black', ls='-', label='Simulation') # og
#ax_histx.plot([], [], color='black', ls='--', label='Poisson') # og
#ax_histx.legend() # og
ymin, ymax = ax_histx.get_ylim()
ymin = np.floor(ymin*10.)/10.
ymax = np.ceil(ymax*10.)/10.
ax_histx.set_yticks(np.arange(ymin, ymax, 0.1))
ax_histx.grid(color='gray', linestyle='-', linewidth=1.)
ax_histx.minorticks_on()    
ax_histx.set_xlabel(r'$M_{\rm halo} \ [M_\odot/h]$')
#ax_histx.set_ylabel(r'${\rm Std}[\langle N_{\rm sat} \rangle]$') # og
ax_histx.set_ylabel(r'$\sqrt{{\rm Var}[N_{\rm sat}]}/\sqrt{\langle N_{\rm sat} \rangle}$') # TESTING
ax_histx.set_ylim([0.77, 1.23]) # TESTING
ax_histx.set_xscale('log')
#ax_histx.set_yscale('log') # og
plt.savefig(f"figs/hod_{n_gal}.png", bbox_inches='tight', pad_inches=0.)
plt.show()
