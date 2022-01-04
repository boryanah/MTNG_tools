import numpy as np
import time

import Corrfunc
from Corrfunc.theory import DDrppi
from Corrfunc.theory import DDsmu
from Corrfunc.utils import convert_rp_pi_counts_to_wp
from Corrfunc.utils import convert_3d_counts_to_cf
from halotools.mock_observables import tpcf_multipole

def get_RR(N1, N2, Lbox, rbins):
    vol_all = 4./3*np.pi*rbins**3
    vol_bins = vol_all[1:]-vol_all[:-1]
    n2 = N2/Lbox**3
    n2_bin = vol_bins*n2
    pairs = N1*n2_bin
    return pairs

def get_RRsmu(N1, N2, Lbox, rbins, mubins):
    vol_all = 2./3*np.pi*rbins**3
    dvol = vol_all[1:]-vol_all[:-1]
    dmu = mubins[1:]-mubins[:-1]
    n2 = N2/Lbox**3
    n2_bin = dvol[:, None]*n2*dmu[None, :]
    pairs = N1*n2_bin
    pairs *= 2.
    return pairs

def get_RRrppi(N1, N2, Lbox, rpbins, pibins):
    vol_all = np.pi*rpbins**2
    dvol = vol_all[1:]-vol_all[:-1]
    dpi = pibins[1:]-pibins[:-1]
    n2 = N2/Lbox**3
    n2_bin = dvol[:, None]*n2*dpi[None, :]
    pairs = N1*n2_bin
    pairs *= 2.
    return RR

def get_xi_l0l2(pos, Lbox, bins, mu_max=1., nmu_bins=20, nthreads=8, periodic=True, rand_pos=np.array([0., 0., 0.])):
    N = pos.shape[0]
    RAND_N = rand_pos.shape[0]
    if periodic:
        assert RAND_N == 3
    
    # angular bins
    mu_bins = np.linspace(0., mu_max, nmu_bins+1)
    nbins = len(bins)-1
    
    # Specify that an autocorrelation is wanted
    autocorr = 1
    DD_counts = DDsmu(autocorr, nthreads, bins, mu_max, nmu_bins, pos[:, 0], pos[:, 1], pos[:, 2], periodic=periodic, boxsize=Lbox)
    DD_smu = DD_counts['npairs'].reshape(nbins, nmu_bins)

    if not periodic:
        # Cross pair counts in DR
        autocorr = 0
        DR_counts = DDsmu(autocorr, nthreads, bins, mu_max, nmu_bins, pos[:, 0], pos[:, 1], pos[:, 2], periodic=periodic, boxsize=Lbox,
                          X2=rand_pos[:, 0], Y2=rand_pos[:, 1], Z2=rand_pos[:, 2])

        # Auto pairs counts in RR
        autocorr = 1
        RR_counts = DDsmu(autocorr, nthreads, bins, mu_max, nmu_bins, rand_pos[:, 0], rand_pos[:, 1], rand_pos[:, 2], periodic=periodic, boxsize=Lbox)

        # All the pair counts are done, get the angular correlation function
        xismu = convert_3d_counts_to_cf(N, N, RAND_N, RAND_N, DD_counts, DR_counts, DR_counts, RR_counts)
        xismu = xismu.reshape(nbins, nmu_bins)
    else:
        # compute the randoms analytically
        RR_smu = get_RRsmu(N, N, Lbox, bins, mu_bins)
        xismu = DD_smu/RR_smu - 1.
        
    # transformation
    xil0 = tpcf_multipole(xismu, mu_bins, order=0)
    xil2 = tpcf_multipole(xismu, mu_bins, order=2)

    return xil0, xil2, bins

def get_jack_xi_l0l2(pos_tr, pos_sh, Lbox, N_dim, bins, mu_max=1., nmu_bins=20, nthreads=8, periodic=True):
    N = pos_tr.shape[0]
    assert N == pos_sh.shape[0]
    if not periodic:
        RAND_N = N*15
        rand_pos = np.vstack((np.random.random(RAND_N), np.random.random(RAND_N), np.random.random(RAND_N))).T*Lbox
        rand_pos = rand_pos.astype(np.float32)
    N_bin = len(bins)
    
    # empty arrays to record data
    Ratl0 = np.zeros((N_bin-1, N_dim**3))
    Xil0_sh = np.zeros((N_bin-1, N_dim**3))
    Xil0_tr = np.zeros((N_bin-1, N_dim**3))
    
    Ratl2 = np.zeros((N_bin-1, N_dim**3))
    Xil2_sh = np.zeros((N_bin-1, N_dim**3))
    Xil2_tr = np.zeros((N_bin-1, N_dim**3))
    for i_x in range(N_dim):
        for i_y in range(N_dim):
            for i_z in range(N_dim):
                print("i, j, k = ", i_x, i_y, i_z)
                pos_sh_jack = pos_sh.copy()
                pos_tr_jack = pos_tr.copy()
                if not periodic:
                    rand_pos_jack = rand_pos.copy()
                
                xyz = np.array([i_x,i_y,i_z], dtype=int)
                size = Lbox/N_dim
                
                bool_arr = np.prod((xyz == (pos_sh/size).astype(int)), axis=1).astype(bool)
                pos_sh_jack[bool_arr] = np.array([0., 0., 0.])
                pos_sh_jack = pos_sh_jack[np.sum(pos_sh_jack, axis=1) != 0.]
                
                bool_arr = np.prod((xyz == (pos_tr/size).astype(int)), axis=1).astype(bool)
                pos_tr_jack[bool_arr] = np.array([0., 0., 0.])
                pos_tr_jack = pos_tr_jack[np.sum(pos_tr_jack, axis=1) != 0.]

                if not periodic:
                    bool_arr = np.prod((xyz == (rand_pos/size).astype(int)), axis=1).astype(bool)
                    rand_pos_jack[bool_arr] = np.array([0., 0., 0.])
                    rand_pos_jack = rand_pos_jack[np.sum(rand_pos_jack, axis=1) != 0.]
                else:
                    rand_pos_jack = np.array([0., 0., 0.])
                xil0_tr, xil2_tr, _ = get_xi_l0l2(pos_tr_jack, Lbox, bins, mu_max, nmu_bins, nthreads, periodic, rand_pos_jack)
                xil0_sh, xil2_sh, _ = get_xi_l0l2(pos_sh_jack, Lbox, bins, mu_max, nmu_bins, nthreads, periodic, rand_pos_jack)
                
                ratl0 = xil0_sh/xil0_tr
                Ratl0[:, i_x+N_dim*i_y+N_dim**2*i_z] = ratl0
                Xil0_sh[:, i_x+N_dim*i_y+N_dim**2*i_z] = xil0_sh
                Xil0_tr[:, i_x+N_dim*i_y+N_dim**2*i_z] = xil0_tr
                
                ratl2 = xil2_sh/xil2_tr
                Ratl2[:, i_x+N_dim*i_y+N_dim**2*i_z] = ratl2
                Xil2_sh[:, i_x+N_dim*i_y+N_dim**2*i_z] = xil2_sh
                Xil2_tr[:, i_x+N_dim*i_y+N_dim**2*i_z] = xil2_tr
                
    # compute mean and error
    ratl0_mean = np.mean(Ratl0, axis=1)
    ratl0_err = np.sqrt(N_dim**3-1)*np.std(Ratl0, axis=1)
    xil0_sh_mean = np.mean(Xil0_sh, axis=1)
    xil0_sh_err = np.sqrt(N_dim**3-1)*np.std(Xil0_sh, axis=1)
    xil0_tr_mean = np.mean(Xil0_tr, axis=1)
    xil0_tr_err = np.sqrt(N_dim**3-1)*np.std(Xil0_tr, axis=1)
    
    ratl2_mean = np.mean(Ratl2, axis=1)
    ratl2_err = np.sqrt(N_dim**3-1)*np.std(Ratl2, axis=1)
    xil2_sh_mean = np.mean(Xil2_sh, axis=1)
    xil2_sh_err = np.sqrt(N_dim**3-1)*np.std(Xil2_sh, axis=1)
    xil2_tr_mean = np.mean(Xil2_tr, axis=1)
    xil2_tr_err = np.sqrt(N_dim**3-1)*np.std(Xil2_tr, axis=1)
    
    return ratl0_mean, ratl0_err, ratl2_mean, ratl2_err, xil0_sh_mean, xil0_sh_err, xil0_tr_mean, xil0_tr_err, xil2_sh_mean, xil2_sh_err, xil2_tr_mean, xil2_tr_err, bins
