
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate

#sim_type = "MTNG"
sim_type = "TNG"
data_dir_dic = {'TNG': "data_tng/", 'MTNG': "data_mtng/"}
data_dir = data_dir_dic[sim_type]

# load rho and r
rs = np.load("../density/"+data_dir+"/r_bins.npy")
binc = np.load("../density/"+data_dir+"/r_cents.npy")
shells = np.load("../density/"+data_dir+"/shells_part.npy")
vdiffs = 4*np.pi*(rs[1:]**3-rs[:-1]**3)/3.
rho = interp1d(binc, shells/vdiffs, bounds_error=False, fill_value=1.)#"extrapolate")

L = 3.

Sigma = np.zeros(len(binc))
# int rho(sqrt(r2+l2)) dl from -L to L where L = 3 R_eff
for i in range(len(binc)):
    print("i = ", i, end='\r')
    r = binc[i]
    integrand_rho = lambda l: rho(np.sqrt(r**2+l**2))
    Sigma[i] = integrate.quad(integrand_rho, -L, L)[0]

    
print("Sigma = ", Sigma)
# sigma bar = 2*L cause rho_mean = 1 and dl is from -3 to 3 
np.save(data_dir+"Sigma_norm.npy", Sigma/(L-(-L)))
np.save(data_dir+"binc.npy", binc)

