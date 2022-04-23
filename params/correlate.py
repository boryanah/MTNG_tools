import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


fp_dm = 'fp'; snapshot = 179
#fp_dm = 'dm'; snapshot = 184
#tng_dir = f"/mnt/gosling1/boryanah/TNG300/"
tng_dir = f"/mnt/alan1/boryanah/MTNG/data_{fp_dm:s}/"
if snapshot == 99: snap_str = ''
else: snap_str = f'_{snapshot:d}'
R = 1

#prop_y = np.load(tng_dir+f"GroupAnnEnv_R{R:d}_{fp_dm:s}{snap_str:s}.npy")
#prop_x = np.load(tng_dir+f"GroupShear_R{R:d}_{fp_dm:s}{snap_str:s}.npy")
#prop_x = np.load(tng_dir+f"Group_M_Mean200_{fp_dm:s}{snap_str:s}.npy")*1.e10
prop_x = np.load(tng_dir+f"GroupVelAni_{fp_dm:s}{snap_str:s}.npy")
#prop_x = np.load(tng_dir+f"GroupMarkedEnv_R{R:d}_s0.25_p2_{fp_dm:s}{snap_str:s}.npy")
prop_x[prop_x == 0.] = 1.e-6
print(np.sum(np.isnan(prop_x)))
prop_x[np.isnan(prop_x)] = 1.e-6
prop_x[np.isinf(prop_x)] = 1.e-6

prop_y = np.load(tng_dir+f"GroupEnv_R{R:d}_{fp_dm:s}{snap_str:s}.npy")
prop_z = np.load(tng_dir+f"Group_M_Mean200_{fp_dm:s}{snap_str:s}.npy")*1.e10

inds = np.arange(len(prop_z))[prop_z > 1.e12]
prop_x, prop_y = prop_x[inds], prop_y[inds]

#bins = np.logspace(11, 15, 31)
#binc = (bins[1:] + bins[:-1])/2.
bins = np.linspace(prop_x.min(), prop_x.max(), 31)
#bins = np.logspace(np.log10(prop_x.min()), np.log10(prop_x.max()), 31)
binc = (bins[1:] + bins[:-1])/2.
median, _, _ = stats.binned_statistic(prop_x, prop_y, statistic='median', bins=bins)


plt.figure(figsize=(9, 7))
plt.scatter(prop_x, prop_y, s=1, alpha=0.3)
plt.plot(binc, median, 'k-', lw=3.)
#plt.xscale('log')
#plt.yscale('log')
#plt.ylim([-1, 1.5])
plt.show()
