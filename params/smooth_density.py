import os

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interpn
from numba import njit
import numpy.linalg as la

from tools import numba_tsc_3D

fp_dm = 'dm'; snapshot = '184'
#fp_dm = 'fp'; snapshot = '179'
sim_type = "MTNG"
#sim_type = "TNG"; snapshot = '99'
tng_dir_dic = {'TNG': "/mnt/gosling1/boryanah/TNG300/", 'MTNG': "/mnt/alan1/boryanah/MTNG/"}
Lbox_dic = {'TNG': 205., 'MTNG': 500.}
pos_part_dic = {'TNG': "parts_position_tng300-2_99.npy", 'MTNG': f"data_parts/pos_down_1000_snap_{snapshot:s}_{fp_dm:s}.npy"}
N_dim_dic = {'TNG': 512, 'MTNG': 512}# can use the really high-res file 1024
sim_name_dic = {'TNG': "tng300-2", 'MTNG': "mtng"}

tng_dir = tng_dir_dic[sim_type]
Lbox = Lbox_dic[sim_type]
N_dim = N_dim_dic[sim_type]
pos_parts = np.load(tng_dir+pos_part_dic[sim_type])
sim_name = sim_name_dic[sim_type]

if sim_type == "TNG":
    pos_parts /= 1000. # deps on file
    GroupPos = np.load(tng_dir+f'GroupPos_{fp_dm:s}.npy')/1.e3
else:
    GroupPos = np.load(tng_dir+f'data_{fp_dm:s}/GroupPos_{fp_dm:s}_'+str(snapshot)+'.npy')


def smooth_density(D, R, N_dim, Lbox):
    # cell size
    cell = Lbox/N_dim
    # smoothing scale
    R /= cell
    D_smooth = gaussian_filter(D, R)
    return D_smooth

# tophat
@njit(nopython=True)
def Wth(ksq, r):
    k = np.sqrt(ksq)
    w = 3*(np.sin(k*r)-k*r*np.cos(k*r))/(k*r)**3
    return w

# gaussian
@njit(nopython=True)
def Wg(k, r):
    return np.exp(-k*r*r/2.)
            

@njit(nopython=True)
def get_tidal(dfour, karr, N_dim, R):
    
    # initiate array
    tfour = np.zeros(shape=(N_dim, N_dim, N_dim, 3, 3),dtype=np.complex128)#complex)
    
    # computing tidal tensor
    for a in range(N_dim):
        for b in range(N_dim):
            for c in range(N_dim):
                if (a, b, c) == (0, 0, 0): continue
                
                ksq = karr[a]**2 + karr[b]**2 + karr[c]**2
                # smoothed density Gauss fourier
                #dksmo[a, b, c] = Wg(ksq)*dfour[a, b, c]
                # smoothed density TH fourier
                #dkth[a, b, c] = Wth(ksq)*dfour[a, b, c]
                # all 9 components
                tfour[a, b, c, 0, 0] = karr[a]*karr[a]*dfour[a, b, c]/ksq
                tfour[a, b, c, 1, 1] = karr[b]*karr[b]*dfour[a, b, c]/ksq
                tfour[a, b, c, 2, 2] = karr[c]*karr[c]*dfour[a, b, c]/ksq
                tfour[a, b, c, 1, 0] = karr[a]*karr[b]*dfour[a, b, c]/ksq
                tfour[a, b, c, 0, 1] = tfour[a, b, c, 1, 0]
                tfour[a, b, c, 2, 0] = karr[a]*karr[c]*dfour[a, b, c]/ksq
                tfour[a, b, c, 0, 2] = tfour[a, b, c, 2, 0]
                tfour[a, b, c, 1, 2] = karr[b]*karr[c]*dfour[a, b, c]/ksq
                tfour[a, b, c, 2, 1] = tfour[a, b, c, 1, 2]
                if R is not None:
                    tfour[a, b, c, :, :] *= Wth(ksq, R)
    return tfour

@njit(nopython=True)
def get_shear_nb(tidr, N_dim):
    shear = np.zeros(shape=(N_dim, N_dim, N_dim), dtype=np.float64)
    for a in range(N_dim):
        for b in range(N_dim):
            for c in range(N_dim):
                t = tidr[a, b, c]
                evals, evects = la.eig(t)
                # ascending
                idx = evals.argsort()
                evals = evals[idx]
                evects = evects[:, idx]
                l1 = evals[0]
                l2 = evals[1]
                l3 = evals[2]
                shear[a, b, c] = 0.5*((l2-l1)**2 + (l3-l1)**2 + (l3-l2)**2)
    return shear

def get_shear(dsmo, N_dim, Lbox, R=None):
    # fourier transform the density field
    dfour = np.fft.fftn(dsmo)
        
    # k values
    karr = np.fft.fftfreq(N_dim, d=Lbox/(2*np.pi*N_dim))
    
    # creating empty arrays for future use
    tfour = get_tidal(dfour, karr, N_dim, R)
    tidr = np.real(np.fft.ifftn(tfour, axes = (0, 1, 2)))

    # compute shear
    shear = np.sqrt(get_shear_nb(tidr, N_dim))
    
    return shear

    
cell = Lbox/N_dim
GroupPos = (GroupPos/cell).astype(int)%N_dim
Rs = [2, 1, 5] # Mpc/h
delta_s = 0.25
p = 2

# if you don't have an unsmoothed density map
dens = np.zeros((N_dim, N_dim, N_dim))
numba_tsc_3D(pos_parts, dens, Lbox)
np.save(tng_dir+'unsmoothed_density_'+sim_name+'_'+snapshot+'.npy', dens)

# alternative
#dens = np.load(tng_dir+f'density_ngrid_{N_dim:d}_snap_{snapshot:s}_{fp_dm:s}.npy')

for R in Rs:
    print("R = ", R)
    try:
        dens_smooth = np.load(tng_dir+f'smoothed_density_R{R:d}_'+sim_name+'_'+snapshot+'.npy')
    except:
        print("creating smoothed density map")
        dens_smooth = smooth_density(dens, R, N_dim, Lbox)
        np.save(tng_dir+f'smoothed_density_R{R:d}_'+sim_name+'_'+snapshot+'.npy', dens_smooth)


    print("interping")
    GroupEnv = interpn((np.arange(N_dim),np.arange(N_dim),np.arange(N_dim)), dens_smooth, GroupPos)
    print("interped")
    # what we do usually
    #GroupEnv = dens_smooth[GroupPos[:, 0], GroupPos[:, 1], GroupPos[:, 2]]
    np.save(tng_dir+f'GroupEnv_R{R:d}_{fp_dm:s}_{snapshot:s}.npy', GroupEnv)
        
    # fourier transform to get t_ij and then diagonalize 
    # with gaussian smoothing
    mark = get_shear(dens_smooth, N_dim, Lbox)
    # with tophat smoothing
    #mark = get_shear(dens, N_dim, Lbox, R)
    print("computed shear")
    
    #if os.path.exists(tng_dir+f'GroupShear_R{R:d}_{fp_dm:s}.npy'): continue
    GroupShear = interpn((np.arange(N_dim), np.arange(N_dim), np.arange(N_dim)), mark, GroupPos)
    np.save(tng_dir+f'GroupShear_R{R:d}_{fp_dm:s}_{snapshot:s}.npy', GroupShear)

    #if os.path.exists(tng_dir+f'GroupEnv_R{R:d}_{fp_dm:s}.npy'): continue

    # void mark
    mark = (1+delta_s/(1+delta_s+dens_smooth))**p
    GroupMarkedEnv = interpn((np.arange(N_dim),np.arange(N_dim),np.arange(N_dim)), mark, GroupPos)
    # what we do usually
    #GroupMarkedEnv = mark[GroupPos[:, 0], GroupPos[:, 1], GroupPos[:, 2]]
    np.save(tng_dir+f'GroupMarkedEnv_R{R:d}_s{delta_s:.2f}_p{p:d}_{fp_dm:s}_{snapshot:s}.npy', GroupMarkedEnv)


