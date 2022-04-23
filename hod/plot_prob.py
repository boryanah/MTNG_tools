import numpy as np
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

hexcolors_bright = ['#CC3311','#0077BB','#EE7733','#BBBBBB','#33BBEE','#EE3377','#0099BB']

gal_type = 'ELG'
n_gal = '2.0e-03' #['7.0e-04', '2.0e-03']
snapshots = [179, 264]
gal_types = ['LRG', 'ELG']
snapshots = [179, 264]
zs = [1., 0.]

# labels
label_prob_acent = r'$P({\rm 1 \ cent})$'
label_prob_anysat = r'$P({\rm >0 \ sat})$'
label_prob_acent_given_anysat = r'$P({\rm 1 \ cent}|{\rm >0 \ sat})$'
label_prob_anysat_given_acent = r'$P({\rm >0 \ sat}|{\rm 1 \ cent})$'

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
        if gal_type == 'LRG': counter += 1; continue
        print(gal_label)
        data = np.load(f"data/{gal_type:s}_{n_gal:s}_fp_{snapshot:d}.npz")
        prob_acent = data['prob_acent']
        prob_anysat = data['prob_anysat']
        prob_acent_given_anysat = data['prob_acent_given_anysat']
        prob_anysat_given_acent = data['prob_anysat_given_acent']
        mbinc = data['mbinc']

        ax_scatter.plot(mbinc, prob_acent, color=hexcolors_bright[counter], ls='--')
        ax_scatter.plot(mbinc, prob_acent_given_anysat, color=hexcolors_bright[counter], ls='-')

        ax_histx.plot(mbinc, prob_acent_given_anysat/prob_acent, label=rf"${z_label}, \ {gal_label}$", color=hexcolors_bright[counter], ls='-')

        counter += 1 


ax_scatter.plot([], [], label=label_prob_acent, ls='--', color='k')
ax_scatter.plot([], [], label=label_prob_acent_given_anysat, ls='-', color='k')
ax_scatter.set_xscale('log')
ax_scatter.set_ylabel(r'$P(X|M_{\rm halo})$')
xmin, xmax = ax_scatter.get_xlim()
ax_scatter.legend(fontsize=22)

ax_histx.set_xscale('log')
ax_histx.set_xlabel(r'$M_{\rm halo} \ [M_\odot/h]$')
#ax_histx.set_ylabel(r'${\rm Ratio}$')
ax_histx.set_ylabel(r'$\frac{P({\rm 1 \ cent}|{\rm >0 \ sat})}{P({\rm 1 \ cent})}$')#, fontsize=22)
ax_histx.legend(fontsize=18)
ax_histx.set_ylim([0, 2.5])
ax_histx.set_xlim([xmin, xmax])
#plt.savefig(f"prob_cent_{n_gal:s}_fp_{snapshot:d}.png")
plt.savefig(f"figs/prob_cent_{n_gal}.png", bbox_inches='tight', pad_inches=0.)
plt.show()


quit()
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

ax_scatter.plot(mbinc, prob_anysat, label=label_prob_anysat)
ax_scatter.plot(mbinc, prob_anysat_given_acent, label=label_prob_anysat_given_acent)
ax_scatter.set_xscale('log')
ax_scatter.set_ylabel(r'$P(X|M_{\rm halo})$')
xmin, xmax = ax_scatter.get_xlim()
ax_scatter.legend()

ax_histx.plot(mbinc, prob_anysat_given_acent/prob_anysat)
ax_histx.set_xscale('log')
ax_histx.set_xlabel(r'$M_{\rm halo} \ [M_\odot/h]$')
ax_histx.set_ylabel(r'${\rm Ratio}$')
ax_histx.set_ylim([0, 2.5])
ax_histx.set_xlim([xmin, xmax])
plt.savefig(f"prob_sats_{n_gal:s}_fp_{snapshot:d}.png")
plt.show()
quit()

plt.figure(1, figsize=(9, 7))
plt.plot(mbinc, prob_acent, label=label_prob_acent)
plt.plot(mbinc, prob_acent_given_anysat, label=label_prob_acent_given_anysat)
plt.xscale('log')
plt.legend()
plt.savefig(f"prob_cent_{n_gal:s}_fp_{snapshot:d}.png")
#plt.close()


plt.figure(2, figsize=(9, 7))
plt.plot(mbinc, prob_anysat, label=label_prob_anysat)
plt.plot(mbinc, prob_anysat_given_acent, label=label_prob_anysat_given_acent)
plt.legend()
plt.xscale('log')
plt.savefig(f"prob_sats_{n_gal:s}_fp_{snapshot:d}.png")
#plt.close()
plt.show()
