import os

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interpn
from numba import njit
import numpy.linalg as la

def get_density(pos, N_dim, Lbox):
    # total number of objects
    N = pos.shape[0]
    # get a 3d histogram with number of objects in each cell
    D, edges = np.histogramdd(pos, bins=N_dim, range=[[0,Lbox],[0,Lbox],[0,Lbox]])
    # average number of particles per cell
    D_avg = N*1./N_dim**3
    D /= D_avg
    D -= 1.
    return D


def smooth_density(D, R, N_dim, Lbox):
    # cell size
    cell = Lbox/N_dim
    # smoothing scale
    R /= cell
    D_smooth = gaussian_filter(D, R)
    return D_smooth

#sim_type = "MTNG"; fp_dm = 'fp' # no choice
#sim_type = "MTNG_DM"; fp_dm = 'dm' # no choice
sim_type = "TNG"; fp_dm = 'dm' # actual choice
tng_dir_dic = {'TNG': "/mnt/gosling1/boryanah/TNG300/", 'MTNG': "/mnt/alan1/boryanah/MTNG/"}
Lbox_dic = {'TNG': 205., 'MTNG': 500.}
pos_part_dic = {'TNG': "parts_position_tng300-3_99.npy", 'MTNG': "data_parts/pos_down_1000_snap_179_fp.npy"}
void_dic = {'TNG': "../finders/tracers.SVF_recen_ovl0.5", 'MTNG': "../visualize/data/pos_down_10000_snap_179_fp.SVF_recen_ovl0.5"}
data_dir_dic = {'TNG': "data_tng/", 'MTNG': "data_mtng/"}
N_dim_dic = {'TNG': 512, 'MTNG':512}
sim_name_dic = {'TNG': "tng300-2", 'MTNG': "mtng"}

#pos_part = np.load(tng_dir+"parts_position_tng300-3_99.npy")/1000.
tng_dir = tng_dir_dic[sim_type]
Lbox = Lbox_dic[sim_type]
pos_part = np.load(tng_dir+pos_part_dic[sim_type])
data_dir = data_dir_dic[sim_type]
N_dim = N_dim_dic[sim_type]
sim_name = sim_name_dic[sim_type]

if sim_type == "TNG":
    snapshot = '99'
    pos_part /= 1000. # deps on file
    if fp_dm == 'fp':
        GroupPos = np.load(tng_dir+'GroupPos_fp.npy')/1.e3
    else:
        GroupPos = np.load(tng_dir+'GroupPos_dm.npy')/1.e3
else:
    snapshot = '179'
    if fp_dm == 'fp':
        GroupPos = np.load(tng_dir+'data_fp/GroupPos_fp_'+str(snapshot)+'.npy')
    else:
        GroupPos = np.load(tng_dir+'data_dm/GroupPos_dm_'+str(snapshot)+'.npy')
    print(GroupPos_dm.max())

cell = Lbox/N_dim
GroupPos = (GroupPos/cell).astype(int)%N_dim
#Rs = [1, 2, 5] # Mpc/h
Rs = [1.1, 1.5, 2., 2.5] # Mpc/h
delta_s = 0.25
p = 2

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
    shear = np.zeros(shape=(N_dim, N_dim, N_dim), dtype=np.float64)#float)
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
    
#try:
dens = np.load(tng_dir+'unsmoothed_density_'+sim_name+'_'+snapshot+'.npy')
#except:
#dens = get_density(pos_part, N_dim, Lbox)
#np.save(tng_dir+'unsmoothed_density_'+sim_name+'_'+snapshot+'.npy', dens)

for R in Rs:
    print("R = ", R)
    #try:
    dens_smooth = np.load(tng_dir+f'smoothed_density_R{R:.1f}_'+sim_name+'_'+snapshot+'.npy')
    #except:
    #dens_smooth = smooth_density(dens, R, N_dim, Lbox)
    #np.save(tng_dir+f'smoothed_density_R{R:.1f}_'+sim_name+'_'+snapshot+'.npy', dens_smooth)
    print("smoothed")
    
    GroupEnv = interpn((np.arange(N_dim),np.arange(N_dim),np.arange(N_dim)), dens_smooth, GroupPos)
    # what we do usually
    #GroupEnv = dens_smooth[GroupPos[:, 0], GroupPos[:, 1], GroupPos[:, 2]]
    np.save(tng_dir+f'GroupEnv_R{R:.1f}_{fp_dm:s}_{snapshot:s}.npy', GroupEnv)
# TESTING
if False:
    
    # fourier transform to get t_ij and then diagonalize 
    # with gaussian smoothing
    #mark = get_shear(dens_smooth, N_dim, Lbox)
    # with tophat smoothing
    mark = get_shear(dens, N_dim, Lbox, R)
    print("computed shear")

    #if os.path.exists(tng_dir+f'GroupShear_R{R:d}_dm.npy'): continue
    GroupShear = interpn((np.arange(N_dim), np.arange(N_dim), np.arange(N_dim)), mark, GroupPos_dm)
    np.save(tng_dir+f'GroupShear_R{R:d}_dm_{snapshot:d}.npy', GroupShear)
    
    #if os.path.exists(tng_dir+f'GroupShear_R{R:d}_{fp_dm:s}.npy'): continue
    GroupShear = interpn((np.arange(N_dim), np.arange(N_dim), np.arange(N_dim)), mark, GroupPos)
    np.save(tng_dir+f'GroupShear_R{R:d}_{fp_dm:s}_{snapshot:d}.npy', GroupShear)

    #if os.path.exists(tng_dir+f'GroupEnv_R{R:d}_{fp_dm:s}.npy'): continue

    # void mark
    mark = (1+delta_s/(1+delta_s+dens_smooth))**p
    GroupMarkedEnv = interpn((np.arange(N_dim),np.arange(N_dim),np.arange(N_dim)), mark, GroupPos)
    # what we do usually
    #GroupMarkedEnv = mark[GroupPos[:, 0], GroupPos[:, 1], GroupPos[:, 2]]
    np.save(tng_dir+f'GroupMarkedEnv_R{R:d}_s{delta_s:.2f}_p{p:d}_{fp_dm:s}_{snapshot:d}.npy', GroupMarkedEnv)

