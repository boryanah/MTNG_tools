"""
Tools for computing correlations
"""
import numpy as np
import time

from Corrfunc.theory import DDsmu, DDrppi, DD, xi
from Corrfunc.utils import convert_rp_pi_counts_to_wp, convert_3d_counts_to_cf
from halotools.mock_observables import tpcf_multipole
from numba_2pcf.cf import numba_pairwise_vel

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

def get_RRrppi(N1, N2, Lbox, rpbins, pimax):
    vol_all = np.pi*rpbins**2
    dvol = vol_all[1:]-vol_all[:-1]
    pibins = np.arange(pimax)
    dpi = pibins[1:]-pibins[:-1]
    n2 = N2/Lbox**3
    n2_bin = dvol[:, None]*n2*dpi[None, :]
    pairs = N1*n2_bin
    pairs *= 2.
    return pairs

"""
def get_RRrppi(N1, N2, Lbox, rpbins, pibins):
    vol_all = np.pi*rpbins**2
    dvol = vol_all[1:]-vol_all[:-1]
    dpi = pibins[1:]-pibins[:-1]
    n2 = N2/Lbox**3
    n2_bin = dvol[:, None]*n2*dpi[None, :]
    pairs = N1*n2_bin
    pairs *= 2.
    return pairs
"""
def get_xirppi(pos1, pos2, lbox, rpbins, pimax, pi_bin_size, Nthread=16, num_cells = 20, x2 = None, y2 = None, z2 = None):
    start = time.time()
    if not isinstance(pimax, int):
        raise ValueError("pimax needs to be an integer")
    if not isinstance(pi_bin_size, int):
        raise ValueError("pi_bin_size needs to be an integer")
    if not pimax % pi_bin_size == 0:
        raise ValueError("pi_bin_size needs to be an integer divisor of pimax, current values are ", pi_bin_size, pimax)

    x1, y1, z1 = pos1[:, 0], pos1[:, 1], pos1[:, 2]
    x2, y2, z2 = pos2[:, 0], pos2[:, 1], pos2[:, 2]
    
    ND1 = float(len(x1))
    if x2 is not None:
        ND2 = len(x2)
        autocorr = 0
    else:
        autocorr = 1
        ND2 = ND1
        
    # single precision mode
    # to do: make this native
    cf_start = time.time()
    rpbins = rpbins.astype(np.float32)
    pimax = np.float32(pimax)
    x1 = x1.astype(np.float32)
    y1 = y1.astype(np.float32)
    z1 = z1.astype(np.float32)
    lbox = np.float32(lbox)
    
    if autocorr == 1:
        results = DDrppi(autocorr, Nthread, pimax, rpbins, x1, y1, z1,
                         boxsize = lbox, periodic = True, max_cells_per_dim = num_cells)
        DD_counts = results['npairs']
    else:
        x2 = x2.astype(np.float32)
        y2 = y2.astype(np.float32)
        z2 = z2.astype(np.float32)
        results = DDrppi(autocorr, Nthread, pimax, rpbins, x1, y1, z1, X2 = x2, Y2 = y2, Z2 = z2,
                         boxsize = lbox, periodic = True, max_cells_per_dim = num_cells)
        DD_counts = results['npairs']
    print("corrfunc took time ", time.time() - cf_start)
        
    DD_counts_new = np.array([np.sum(DD_counts[i:i+pi_bin_size]) for i in range(0, len(DD_counts), pi_bin_size)])
    DD_counts_new = DD_counts_new.reshape((len(rpbins) - 1, int(pimax/pi_bin_size)))
        
    # RR_counts_new = np.zeros((len(rpbins) - 1, int(pimax/pi_bin_size)))
    RR_counts_new = np.pi*(rpbins[1:]**2 - rpbins[:-1]**2)*pi_bin_size / lbox**3 * ND1 * ND2 * 2
    xirppi = DD_counts_new / RR_counts_new[:, None] - 1
    print("corrfunc took ", time.time() - start, "ngal ", len(x1))
    return xirppi

    
def get_xil0l2(pos, Lbox, bins, mu_max=1., nmu_bins=20, nthreads=8, periodic=True, rand_pos=np.array([0., 0., 0.])):
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


def get_jack_xirppi(xyz_true, xyz_hod, Lbox, pimax, pi_bin_size, N_dim=3, nthreads=16, bins=np.logspace(-1, 1, 41)):
    
    # bins for the correlation function
    N_bin = len(bins)
    bin_centers = (bins[:-1] + bins[1:])/2.

    true_max = xyz_true.max()
    true_min = xyz_true.min()
    hod_max = xyz_hod.max()
    hod_min = xyz_hod.min()
    
    if true_max > Lbox or true_min < 0. or hod_max > Lbox or hod_min < 0.:
        print("NOTE: we are wrapping positions")
        xyz_true = xyz_true % Lbox
        xyz_hod = xyz_hod % Lbox

    # empty arrays to record data
    Rat_hodtrue = np.zeros((N_bin-1, int(pimax/pi_bin_size), N_dim**3))
    Xirppi_hod = np.zeros((N_bin-1, int(pimax/pi_bin_size), N_dim**3))
    Xirppi_true = np.zeros((N_bin-1, int(pimax/pi_bin_size), N_dim**3))
    for i_x in range(N_dim):
        for i_y in range(N_dim):
            for i_z in range(N_dim):
                print("i, j, k = ", i_x, i_y, i_z)
                xyz_hod_jack = xyz_hod.copy()
                xyz_true_jack = xyz_true.copy()
            
                xyz = np.array([i_x,i_y,i_z],dtype=int)
                size = Lbox/N_dim

                bool_arr = np.prod((xyz == (xyz_hod/size).astype(int)),axis=1).astype(bool)
                xyz_hod_jack[bool_arr] = np.array([0.,0.,0.])
                xyz_hod_jack = xyz_hod_jack[np.sum(xyz_hod_jack,axis=1)!=0.]

                bool_arr = np.prod((xyz == (xyz_true/size).astype(int)),axis=1).astype(bool)
                xyz_true_jack[bool_arr] = np.array([0.,0.,0.])
                xyz_true_jack = xyz_true_jack[np.sum(xyz_true_jack,axis=1)!=0.]


                # compute xirppiwise
                xirppi_hod = get_xirppi(xyz_hod_jack, xyz_hod_jack, Lbox, bins, pimax, pi_bin_size, Nthread=nthreads)
                xirppi_true = get_xirppi(xyz_true_jack, xyz_true_jack, Lbox, bins, pimax, pi_bin_size, Nthread=nthreads)
                
                rat_hodtrue = xirppi_hod/xirppi_true
                Rat_hodtrue[:, :, i_x+N_dim*i_y+N_dim**2*i_z] = rat_hodtrue
                Xirppi_hod[:, :, i_x+N_dim*i_y+N_dim**2*i_z] = xirppi_hod
                Xirppi_true[:, :, i_x+N_dim*i_y+N_dim**2*i_z] = xirppi_true

    # compute without jackknife cause biased
    xirppi_hod = get_xirppi(xyz_hod, xyz_hod, Lbox, bins, pimax, pi_bin_size, Nthread=nthreads)
    xirppi_true = get_xirppi(xyz_true, xyz_true, Lbox, bins, pimax, pi_bin_size, Nthread=nthreads)

    # compute mean and error
    axis = 2
    #Rat_hodtrue_mean = np.mean(Rat_hodtrue,axis=axis)
    Rat_hodtrue_mean = xirppi_hod/xirppi_true
    Rat_hodtrue_err = np.sqrt(N_dim**3-1)*np.std(Rat_hodtrue,axis=axis)
    assert rat_hodtrue.shape == Rat_hodtrue_mean.shape
    #Xirppi_mean_hod = np.mean(Xirppi_hod,axis=axis)
    Xirppi_mean_hod = xirppi_hod
    Xirppi_err_hod = np.sqrt(N_dim**3-1)*np.std(Xirppi_hod,axis=axis)
    #Xirppi_mean_true = np.mean(Xirppi_true,axis=axis)
    Xirppi_mean_true = xirppi_true
    Xirppi_err_true = np.sqrt(N_dim**3-1)*np.std(Xirppi_true,axis=axis)

    return Rat_hodtrue_mean, Rat_hodtrue_err, Xirppi_mean_hod, Xirppi_err_hod,  Xirppi_mean_true, Xirppi_err_true, bin_centers


def get_jack_xil0l2(pos_tr, pos_sh, Lbox, N_dim, bins, mu_max=1., nmu_bins=20, nthreads=8, periodic=True):
    N = pos_tr.shape[0]
    print(pos_sh.shape[0], pos_tr.shape[0])
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
                xil0_tr, xil2_tr, _ = get_xil0l2(pos_tr_jack, Lbox, bins, mu_max, nmu_bins, nthreads, periodic, rand_pos_jack)
                xil0_sh, xil2_sh, _ = get_xil0l2(pos_sh_jack, Lbox, bins, mu_max, nmu_bins, nthreads, periodic, rand_pos_jack)
                
                ratl0 = xil0_sh/xil0_tr
                Ratl0[:, i_x+N_dim*i_y+N_dim**2*i_z] = ratl0
                Xil0_sh[:, i_x+N_dim*i_y+N_dim**2*i_z] = xil0_sh
                Xil0_tr[:, i_x+N_dim*i_y+N_dim**2*i_z] = xil0_tr

                # OLD definition
                #ratl2 = xil2_sh/xil2_tr
                # NEW definition
                ratl2 = (xil2_sh-xil2_tr)/xil0_tr
                Ratl2[:, i_x+N_dim*i_y+N_dim**2*i_z] = ratl2
                Xil2_sh[:, i_x+N_dim*i_y+N_dim**2*i_z] = xil2_sh
                Xil2_tr[:, i_x+N_dim*i_y+N_dim**2*i_z] = xil2_tr
                
                
    if periodic:
        rand_pos = np.array([0., 0., 0.])
    xil0_tr, xil2_tr, _ = get_xil0l2(pos_tr, Lbox, bins, mu_max, nmu_bins, nthreads, periodic, rand_pos)
    xil0_sh, xil2_sh, _ = get_xil0l2(pos_sh, Lbox, bins, mu_max, nmu_bins, nthreads, periodic, rand_pos)

    # compute mean and error
    #ratl0_mean = np.mean(Ratl0, axis=1)
    ratl0_mean = xil0_sh/xil0_tr
    ratl0_err = np.sqrt(N_dim**3-1)*np.std(Ratl0, axis=1)
    #xil0_sh_mean = np.mean(Xil0_sh, axis=1)
    xil0_sh_mean = xil0_sh
    xil0_sh_err = np.sqrt(N_dim**3-1)*np.std(Xil0_sh, axis=1)
    #xil0_tr_mean = np.mean(Xil0_tr, axis=1)
    xil0_tr_mean = xil0_tr
    xil0_tr_err = np.sqrt(N_dim**3-1)*np.std(Xil0_tr, axis=1)
    
    #ratl2_mean = np.mean(Ratl2, axis=1)
    # OLD definition
    #ratl2_mean = xil2_sh/xil2_tr
    # NEW definition
    ratl2_mean = (xil2_sh-xil2_tr)/xil0_tr
    ratl2_err = np.sqrt(N_dim**3-1)*np.std(Ratl2, axis=1)
    #xil2_sh_mean = np.mean(Xil2_sh, axis=1)
    xil2_sh_mean = xil2_sh
    xil2_sh_err = np.sqrt(N_dim**3-1)*np.std(Xil2_sh, axis=1)
    #xil2_tr_mean = np.mean(Xil2_tr, axis=1)
    xil2_tr_mean = xil2_tr
    xil2_tr_err = np.sqrt(N_dim**3-1)*np.std(Xil2_tr, axis=1)
    
    return ratl0_mean, ratl0_err, ratl2_mean, ratl2_err, xil0_sh_mean, xil0_sh_err, xil0_tr_mean, xil0_tr_err, xil2_sh_mean, xil2_sh_err, xil2_tr_mean, xil2_tr_err, bins


def get_jack_corr(xyz_true, w_true, xyz_hod, w_hod, Lbox, N_dim=3, nthreads=16, bins=np.logspace(-1, 1, 21)):
    
    # bins for the correlation function
    N_bin = len(bins)
    bin_centers = (bins[:-1] + bins[1:])/2.

    true_max = xyz_true.max()
    true_min = xyz_true.min()
    hod_max = xyz_hod.max()
    hod_min = xyz_hod.min()
    
    if true_max > Lbox or true_min < 0. or hod_max > Lbox or hod_min < 0.:
        print("NOTE: we are wrapping positions")
        xyz_true = xyz_true % Lbox
        xyz_hod = xyz_hod % Lbox

    # empty arrays to record data
    Rat_hodtrue = np.zeros((N_bin-1,N_dim**3))
    Corr_hod = np.zeros((N_bin-1,N_dim**3))
    Corr_true = np.zeros((N_bin-1,N_dim**3))
    for i_x in range(N_dim):
        for i_y in range(N_dim):
            for i_z in range(N_dim):
                xyz_hod_jack = xyz_hod.copy()
                xyz_true_jack = xyz_true.copy()
                w_hod_jack = w_hod.copy()
                w_true_jack = w_true.copy()
            
                xyz = np.array([i_x,i_y,i_z],dtype=int)
                size = Lbox/N_dim

                bool_arr = np.prod((xyz == (xyz_hod/size).astype(int)),axis=1).astype(bool)
                xyz_hod_jack[bool_arr] = np.array([0.,0.,0.])
                xyz_hod_jack = xyz_hod_jack[np.sum(xyz_hod_jack,axis=1)!=0.]
                w_hod_jack[bool_arr] = -1
                w_hod_jack = w_hod_jack[np.abs(w_hod_jack+1) > 1.e-6]

                bool_arr = np.prod((xyz == (xyz_true/size).astype(int)),axis=1).astype(bool)
                xyz_true_jack[bool_arr] = np.array([0.,0.,0.])
                xyz_true_jack = xyz_true_jack[np.sum(xyz_true_jack,axis=1)!=0.]
                w_true_jack[bool_arr] = -1
                w_true_jack = w_true_jack[np.abs(w_true_jack+1) > 1.e-6]


                # in case we don't have weights
                if np.abs(np.sum(w_hod_jack)-len(w_hod_jack)) < 1.e-6:
                    res_hod = xi(X=xyz_hod_jack[:,0],Y=xyz_hod_jack[:,1],Z=xyz_hod_jack[:,2],boxsize=Lbox,nthreads=nthreads,binfile=bins)
                    xi_hod = res_hod['xi']
                else:
                    res_hod = xi(X=xyz_hod_jack[:,0],Y=xyz_hod_jack[:,1],Z=xyz_hod_jack[:,2],boxsize=Lbox,weights=w_hod_jack,weight_type='pair_product',nthreads=nthreads,binfile=bins)
                    xi_hod = res_hod['xi']
                res_true = xi(X=xyz_true_jack[:,0],Y=xyz_true_jack[:,1],Z=xyz_true_jack[:,2],boxsize=Lbox,weights=w_true_jack,weight_type='pair_product',nthreads=16,binfile=bins)
                xi_true = res_true['xi']
                
                rat_hodtrue = xi_hod/xi_true
                Rat_hodtrue[:,i_x+N_dim*i_y+N_dim**2*i_z] = rat_hodtrue
                Corr_hod[:,i_x+N_dim*i_y+N_dim**2*i_z] = xi_hod
                Corr_true[:,i_x+N_dim*i_y+N_dim**2*i_z] = xi_true

    xi_true = xi(boxsize=Lbox, nthreads=16, X=xyz_true[:, 0], Y=xyz_true[:, 1], Z=xyz_true[:, 2], weights=w_true, weight_type='pair_product', binfile=bins)['xi']
    xi_hod = xi(boxsize=Lbox, nthreads=16, X=xyz_hod[:, 0], Y=xyz_hod[:, 1], Z=xyz_hod[:, 2], weights=w_hod, weight_type='pair_product', binfile=bins)['xi']
                
    # compute mean and error
    #Rat_hodtrue_mean = np.mean(Rat_hodtrue,axis=1)
    Rat_hodtrue_mean = xi_hod/xi_true
    Rat_hodtrue_err = np.sqrt(N_dim**3-1)*np.std(Rat_hodtrue,axis=1)
    #Corr_mean_hod = np.mean(Corr_hod,axis=1)
    Corr_mean_hod = xi_hod
    Corr_err_hod = np.sqrt(N_dim**3-1)*np.std(Corr_hod,axis=1)
    #Corr_mean_true = np.mean(Corr_true,axis=1)
    Corr_mean_true = xi_true
    Corr_err_true = np.sqrt(N_dim**3-1)*np.std(Corr_true,axis=1)

    return Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers

def get_jack_cross_corr(xyz_true, xyz_hod, Lbox, N_dim=3, nthreads=16, bins=np.logspace(-1, 1, 21)):
    # create randoms
    N_rand = 15*xyz_true.shape[0]
    xyz_rand = np.random.rand(N_rand, 3)*Lbox

    # cross correlation
    xi_cross = get_xi_cross(xyz_true, xyz_hod, xyz_rand, xyz_rand, Lbox, bins)
    
    # bins for the correlation function
    N_bin = len(bins)
    bin_centers = (bins[:-1] + bins[1:])/2.

    # empty arrays to record data
    Corr_cross = np.zeros((N_bin-1, N_dim**3))
    for i_x in range(N_dim):
        for i_y in range(N_dim):
            for i_z in range(N_dim):
                print(i_x, i_y, i_z)
                xyz_hod_jack = xyz_hod.copy()
                xyz_true_jack = xyz_true.copy()
                xyz_rand_jack = xyz_rand.copy()
            
                xyz = np.array([i_x,i_y,i_z],dtype=int)
                size = Lbox/N_dim

                bool_arr = np.prod((xyz == (xyz_hod/size).astype(int)),axis=1).astype(bool)
                xyz_hod_jack[bool_arr] = np.array([0.,0.,0.])
                xyz_hod_jack = xyz_hod_jack[np.sum(xyz_hod_jack,axis=1)!=0.]

                bool_arr = np.prod((xyz == (xyz_true/size).astype(int)),axis=1).astype(bool)
                xyz_true_jack[bool_arr] = np.array([0.,0.,0.])
                xyz_true_jack = xyz_true_jack[np.sum(xyz_true_jack,axis=1)!=0.]

                bool_arr = np.prod((xyz == (xyz_rand/size).astype(int)),axis=1).astype(bool)
                xyz_rand_jack[bool_arr] = np.array([0.,0.,0.])
                xyz_rand_jack = xyz_rand_jack[np.sum(xyz_rand_jack,axis=1)!=0.]

                xi_cross_jack = get_xi_cross(xyz_true_jack, xyz_hod_jack, xyz_rand_jack, xyz_rand_jack, Lbox, bins)
                Corr_cross[:, i_x+N_dim*i_y+N_dim**2*i_z] = xi_cross_jack
                
    # compute mean and error
    Corr_mean_cross = xi_cross
    Corr_err_cross = np.sqrt(N_dim**3 - 1)*np.std(Corr_cross, axis=1)

    return Corr_mean_cross, Corr_err_cross, bin_centers


def get_jack_pair(xyz_true, v_true, xyz_hod, v_hod, Lbox, N_dim=3, nthreads=16, bins=np.linspace(0., 100., 51)):
    #assert np.isclose(np.sum(w_hod_jack), len(w_hod_jack)), "weights are not ones, pairwise not implemented"
    assert np.isclose(bins[0], 0)
    
    # bins for the correlation function
    N_bin = len(bins)
    bin_centers = (bins[:-1] + bins[1:])/2.

    true_max = xyz_true.max()
    true_min = xyz_true.min()
    hod_max = xyz_hod.max()
    hod_min = xyz_hod.min()
    
    if true_max > Lbox or true_min < 0. or hod_max > Lbox or hod_min < 0.:
        print("NOTE: we are wrapping positions")
        xyz_true = xyz_true % Lbox
        xyz_hod = xyz_hod % Lbox

    # empty arrays to record data
    Rat_hodtrue = np.zeros((N_bin-1,N_dim**3))
    Pair_hod = np.zeros((N_bin-1,N_dim**3))
    Pair_true = np.zeros((N_bin-1,N_dim**3))
    for i_x in range(N_dim):
        for i_y in range(N_dim):
            for i_z in range(N_dim):
                print("i, j, k = ", i_x, i_y, i_z)
                xyz_hod_jack = xyz_hod.copy()
                xyz_true_jack = xyz_true.copy()
                v_hod_jack = v_hod.copy()
                v_true_jack = v_true.copy()
            
                xyz = np.array([i_x,i_y,i_z],dtype=int)
                size = Lbox/N_dim

                bool_arr = np.prod((xyz == (xyz_hod/size).astype(int)),axis=1).astype(bool)
                xyz_hod_jack[bool_arr] = np.array([0.,0.,0.])
                xyz_hod_jack = xyz_hod_jack[np.sum(xyz_hod_jack,axis=1)!=0.]
                #v_hod_jack[bool_arr] = -1
                #v_hod_jack = v_hod_jack[np.abs(v_hod_jack+1) > 1.e-6]
                v_hod_jack[bool_arr] = np.array([0.,0.,0.])
                v_hod_jack = v_hod_jack[np.sum(v_hod_jack,axis=1)!=0.]

                bool_arr = np.prod((xyz == (xyz_true/size).astype(int)),axis=1).astype(bool)
                xyz_true_jack[bool_arr] = np.array([0.,0.,0.])
                xyz_true_jack = xyz_true_jack[np.sum(xyz_true_jack,axis=1)!=0.]
                #v_true_jack[bool_arr] = -1
                #v_true_jack = v_true_jack[np.abs(v_true_jack+1) > 1.e-6]
                v_true_jack[bool_arr] = np.array([0.,0.,0.])
                v_true_jack = v_true_jack[np.sum(v_true_jack,axis=1)!=0.]

                # compute pairwise
                pair_hod = numba_pairwise_vel(xyz_hod_jack, v_hod_jack, box=Lbox, Rmax=np.max(bins), nbin=len(bins)-1, corrfunc=False, nthread=nthreads, periodic=True)['pairwise']
                pair_true = numba_pairwise_vel(xyz_true_jack, v_true_jack, box=Lbox, Rmax=np.max(bins), nbin=len(bins)-1, corrfunc=False, nthread=nthreads, periodic=True)['pairwise']
                
                rat_hodtrue = pair_hod/pair_true
                Rat_hodtrue[:,i_x+N_dim*i_y+N_dim**2*i_z] = rat_hodtrue
                Pair_hod[:,i_x+N_dim*i_y+N_dim**2*i_z] = pair_hod
                Pair_true[:,i_x+N_dim*i_y+N_dim**2*i_z] = pair_true

    pair_hod = numba_pairwise_vel(xyz_hod, v_hod, box=Lbox, Rmax=np.max(bins), nbin=len(bins)-1, corrfunc=False, nthread=nthreads, periodic=True)['pairwise']
    pair_true = numba_pairwise_vel(xyz_true, v_true, box=Lbox, Rmax=np.max(bins), nbin=len(bins)-1, corrfunc=False, nthread=nthreads, periodic=True)['pairwise']

                
    # compute mean and error
    #Rat_hodtrue_mean = np.mean(Rat_hodtrue,axis=1)
    Rat_hodtrue_mean = pair_hod/pair_true
    Rat_hodtrue_err = np.sqrt(N_dim**3-1)*np.std(Rat_hodtrue,axis=1)
    #Pair_mean_hod = np.mean(Pair_hod,axis=1)
    Pair_mean_hod = pair_hod
    Pair_err_hod = np.sqrt(N_dim**3-1)*np.std(Pair_hod,axis=1)
    #Pair_mean_true = np.mean(Pair_true,axis=1)
    Pair_mean_true = pair_true
    Pair_err_true = np.sqrt(N_dim**3-1)*np.std(Pair_true,axis=1)

    return Rat_hodtrue_mean, Rat_hodtrue_err, Pair_mean_hod, Pair_err_hod,  Pair_mean_true, Pair_err_true, bin_centers


def get_xi_cross(D1, D2, R1, R2, Lbox, bins):
    N_D1 = D1.shape[0]
    N_D2 = D2.shape[0]
    N_R1 = R1.shape[0]
    N_R2 = R2.shape[0]
    D1 = D1.astype(np.float32)
    D2 = D2.astype(np.float32)
    R1 = R1.astype(np.float32)
    R2 = R2.astype(np.float32)
    
    autocorr = 0
    results = DD(autocorr, nthreads=16, binfile=bins,
                 X1=D1[:, 0], Y1=D1[:, 1], Z1=D1[:, 2],
                 X2=D2[:, 0], Y2=D2[:, 1], Z2=D2[:, 2],
                 boxsize=Lbox, periodic=True)
    D1D2 = results['npairs'].astype(float)
    D1D2 /= (N_D1*N_D2)

    autocorr = 0
    results = DD(autocorr, nthreads=16, binfile=bins,
                 X1=R1[:, 0], Y1=R1[:, 1], Z1=R1[:, 2],
                 X2=D2[:, 0], Y2=D2[:, 1], Z2=D2[:, 2],
                 boxsize=Lbox, periodic=True)
    R1D2 = results['npairs'].astype(float)
    R1D2 /= (N_R1*N_D2)

    autocorr = 0
    results = DD(autocorr, nthreads=16, binfile=bins,
                 X1=D1[:, 0], Y1=D1[:, 1], Z1=D1[:, 2],
                 X2=R2[:, 0], Y2=R2[:, 1], Z2=R2[:, 2],
                 boxsize=Lbox, periodic=True)
    D1R2 = results['npairs'].astype(float)
    D1R2 /= (N_D1*N_R2)

    """
    autocorr = 0
    results = DD(autocorr, nthreads=16, binfile=bins,
                 X1=R1[:, 0], Y1=R1[:, 1], Z1=R1[:, 2],
                 X2=R2[:, 0], Y2=R2[:, 1], Z2=R2[:, 2],
                 boxsize=Lbox, periodic=True)
    R1R2 = results['npairs'].astype(float)
    """
    R1R2 = get_RR(N_R1, N_R2, Lbox, bins)
    R1R2 /= (N_R1*N_R2)

    return (D1D2 - R1D2 - D1R2 + R1R2)/R1R2
