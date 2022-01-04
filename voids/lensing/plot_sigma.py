import numpy as np
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()

#data_dir = "data_tng"
data_dir = "data_mtng"
binc = np.load(data_dir+"binc.npy")
Sigma = np.load(data_dir+"Sigma_norm.npy")

plt.figure(figsize=(9, 7))
plt.plot(binc, np.ones(len(binc)), 'k--')
plt.plot(binc, Sigma)
plt.xlabel(r"$r/R_{\rm eff}$")
plt.ylabel(r"$\Sigma (r)/\overline{\Sigma}$")
plt.savefig("Sigma.png")
plt.show()
