import numpy as np
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()
import distinct_colours

fontsize = 22

color_sam = 'dodgerblue'
color_tng = '#CC6677'
offsets = [-0.4,-0.2,0.2,0.4]

data_dir = "data_env/"
opt = "env"

bin_centers = np.load(data_dir+"bin_cents.npy")
cross_mean = np.load(data_dir+"cross_true_mean.npy")
cross_err = np.load(data_dir+"cross_true_error.npy")

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


cross_opt_mean = np.load(data_dir+"cross_"+opt+"_mean.npy")
cross_opt_err = np.load(data_dir+"cross_"+opt+"_error.npy")

rat_opt_mean = np.load(data_dir+"rat_cross_"+opt+"_mean.npy")
rat_opt_err = np.load(data_dir+"rat_cross_"+opt+"_error.npy")

#ax_histx.tick_params(direction='out')#(direction='in', labelbottom=False)

power = 0


ax_scatter.plot(bin_centers, cross_opt_mean*bin_centers**power, color=color_sam, label=r"$\rm SAM$")
ax_scatter.plot(bin_centers, cross_mean*bin_centers**power, color=color_tng, label=r"${\rm TNG}$")

ax_scatter.set_ylabel(r"$\xi_{gv} (r)$",fontsize=fontsize+6)
ax_scatter.set_xscale('log')
#ax_scatter.set_yscale('log')

# now determine nice limits by hand:
ax_scatter.set_xlim([0.8,20])
ax_scatter.set_ylim([-1.2,0.25])
ax_scatter.legend(frameon=False)

line = np.linspace(0,50,3)
ax_histx.plot(line,np.ones(len(line)),'k--')

ax_histx.errorbar(bin_centers, rat_opt_mean, yerr=rat_opt_err, color=color_sam, ls='-', linewidth=2.5, fmt='o', capsize=4)

ax_histx.set_xlim(ax_scatter.get_xlim())
ax_histx.set_xscale('log')
ax_histx.set_xlabel(r"$r \ [\mathrm{Mpc}/h]$",fontsize=fontsize+6)
ax_histx.set_ylabel(r'${\rm Ratio}$')
ax_histx.set_ylim([0.65,1.35])

plt.savefig("all_cross_voids.pdf", bbox_inches='tight', pad_inches=0.)
plt.show()
