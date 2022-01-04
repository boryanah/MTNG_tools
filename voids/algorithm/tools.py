import numpy as np
import numba
from numba import njit
from nbodykit.lab import ArrayCatalog, FieldMesh
from nbodykit.base.mesh import MeshFilter


@njit
def dist(pos1, pos2, L=None):
    '''
    Calculate L2 norm distance between a set of points
    and either a reference point or another set of points.
    Optionally includes periodicity.
    
    Parameters
    ----------
    pos1: ndarray of shape (N,m)
        A set of points
    pos2: ndarray of shape (N,m) or (m,) or (1,m)
        A single point or set of points
    L: float, optional
        The box size. Will do a periodic wrap if given.
    
    Returns
    -------
    dist: ndarray of shape (N,)
        The distances between pos1 and pos2
    '''
    
    # read dimension of data
    N, nd = pos1.shape
    
    # allow pos2 to be a single point
    pos2 = np.atleast_2d(pos2)
    assert pos2.shape[-1] == nd
    broadcast = len(pos2) == 1
    
    dist = np.empty(N, dtype=pos1.dtype)
    
    i2 = 0
    for i in range(N):
        delta = 0.
        for j in range(nd):
            dx = pos1[i][j] - pos2[i2][j]
            if L is not None:
                if dx >= L/2:
                    dx -= L
                elif dx < -L/2:
                    dx += L
            delta += dx*dx
        dist[i] = np.sqrt(delta)
        if not broadcast:
            i2 += 1
    return dist

@numba.jit(nopython=True, nogil=True)
def numba_tsc_3D(positions, density, boxsize, weights=np.empty(0)):
    gx = np.uint32(density.shape[0])
    gy = np.uint32(density.shape[1])
    gz = np.uint32(density.shape[2])
    threeD = gz != 1
    W = 1.
    Nw = len(weights)
    for n in range(len(positions)):
        # broadcast scalar weights
        if Nw == 1:
            W = weights[0]
        elif Nw > 1:
            W = weights[n]
        
        # convert to a position in the grid
        px = (positions[n,0]/boxsize + .5)*gx
        py = (positions[n,1]/boxsize + .5)*gy
        if threeD:
            pz = (positions[n,2]/boxsize + .5)*gz
        
        # round to nearest cell center
        ix = np.int32(round(px))
        iy = np.int32(round(py))
        if threeD:
            iz = np.int32(round(pz))
        
        # calculate distance to cell center
        dx = ix - px
        dy = iy - py
        if threeD:
            dz = iz - pz
        
        # find the tsc weights for each dimension
        wx = .75 - dx**2
        wxm1 = .5*(.5 + dx)**2
        wxp1 = .5*(.5 - dx)**2
        wy = .75 - dy**2
        wym1 = .5*(.5 + dy)**2
        wyp1 = .5*(.5 - dy)**2
        if threeD:
            wz = .75 - dz**2
            wzm1 = .5*(.5 + dz)**2
            wzp1 = .5*(.5 - dz)**2
        else:
            wz = 1.
        
        # find the wrapped x,y,z grid locations of the points we need to change
        # negative indices will be automatically wrapped
        ixm1 = (ix - 1)
        ixw  = rightwrap(ix    , gx)
        ixp1 = rightwrap(ix + 1, gx)
        iym1 = (iy - 1)
        iyw  = rightwrap(iy    , gy)
        iyp1 = rightwrap(iy + 1, gy)
        if threeD:
            izm1 = (iz - 1)
            izw  = rightwrap(iz    , gz)
            izp1 = rightwrap(iz + 1, gz)
        else:
            izw = np.uint32(0)
        
        # change the 9 or 27 cells that the cloud touches
        density[ixm1, iym1, izw ] += wxm1*wym1*wz  *W
        density[ixm1, iyw , izw ] += wxm1*wy  *wz  *W
        density[ixm1, iyp1, izw ] += wxm1*wyp1*wz  *W
        density[ixw , iym1, izw ] += wx  *wym1*wz  *W
        density[ixw , iyw , izw ] += wx  *wy  *wz  *W
        density[ixw , iyp1, izw ] += wx  *wyp1*wz  *W
        density[ixp1, iym1, izw ] += wxp1*wym1*wz  *W
        density[ixp1, iyw , izw ] += wxp1*wy  *wz  *W
        density[ixp1, iyp1, izw ] += wxp1*wyp1*wz  *W
        
        if threeD:
            density[ixm1, iym1, izm1] += wxm1*wym1*wzm1*W
            density[ixm1, iym1, izp1] += wxm1*wym1*wzp1*W

            density[ixm1, iyw , izm1] += wxm1*wy  *wzm1*W
            density[ixm1, iyw , izp1] += wxm1*wy  *wzp1*W

            density[ixm1, iyp1, izm1] += wxm1*wyp1*wzm1*W
            density[ixm1, iyp1, izp1] += wxm1*wyp1*wzp1*W

            density[ixw , iym1, izm1] += wx  *wym1*wzm1*W
            density[ixw , iym1, izp1] += wx  *wym1*wzp1*W

            density[ixw , iyw , izm1] += wx  *wy  *wzm1*W
            density[ixw , iyw , izp1] += wx  *wy  *wzp1*W

            density[ixw , iyp1, izm1] += wx  *wyp1*wzm1*W
            density[ixw , iyp1, izp1] += wx  *wyp1*wzp1*W

            density[ixp1, iym1, izm1] += wxp1*wym1*wzm1*W
            density[ixp1, iym1, izp1] += wxp1*wym1*wzp1*W

            density[ixp1, iyw , izm1] += wxp1*wy  *wzm1*W
            density[ixp1, iyw , izp1] += wxp1*wy  *wzp1*W

            density[ixp1, iyp1, izm1] += wxp1*wyp1*wzm1*W
            density[ixp1, iyp1, izp1] += wxp1*wyp1*wzp1*W



@njit
def paint_tsc(pos, meshflat, mesh_strides, period):
    """
    Doesn't work -- I suspect issue with mesh_strides or meshflat[int(ind)] which used to be meshflat[int(ind)/mesh.itemsize]
    """
    Ndim = pos.shape[1]
    Np = pos.shape[0]
    Nmax = int(3 ** Ndim)
    ignore = False
    period = int(period)
    outbound = 0
    for i in range(Np):
        for n in range(Nmax):
            ignore = False
            kernel = 1.0
            ind = 0
            for d in range(Ndim):
                intpos = np.rint(pos[i, d]) # NGP index
                diff = intpos - pos[i, d]
                rel = (n // 3**d) % 3 - 1 # maps offset (rel. to NGP) to (-1, 0, 1)
                targetpos = intpos + rel - 1
                if rel == -1: # before NGP
                    kernel *= 0.5 * (0.5+diff)*(0.5+diff)
                elif rel == 0: # NGP
                    kernel *= 0.75 - diff*diff
                else: # after NGPs
                    kernel *= 0.5 * (0.5-diff)*(0.5-diff)
                
                # wrap by period
                if period > 0:
                    while targetpos >= period:
                        targetpos -= period
                    while targetpos < 0:
                        targetpos += period
                if targetpos < 0 or \
                        targetpos >= period:
                    ignore = True
                    break
                ind += mesh_strides[d]*targetpos

            if ignore:
                outbound += 1
                continue
            meshflat[int(ind)] += kernel

    return outbound


@njit
def W_th(ksq, r):
    """
    Tophat filter
    """
    k = np.sqrt(ksq)
    w = 3*(np.sin(k*r)-k*r*np.cos(k*r))/(k*r)**3
    return w

class TopHat(MeshFilter):
    """ A TopHat filter defined in Fourier space.
    
    Notes
    -----
    A fourier space filter is different from a configuration space
    filter. The TopHat in fourier space creates ringing effects
    due to the truncation / discretization of modes.
    
    """
    kind = 'wavenumber'
    mode = 'complex'
    
    def __init__(self, r):
        """
        Parameters
        ----------
        r : float
        radius of the TopHat filter
        """
        self.r = r
    
    
    def filter(self, k, v):
        r = self.r
        k = sum(ki ** 2 for ki in k) ** 0.5
        kr = k * r
        w = 3 * (np.sin(kr) / kr **3 - np.cos(kr) / kr ** 2)
        w[k == 0] = 1.0
        return w * v

def get_density(pos, N_dim, Lbox):
    """
    Slow function for getting the overdensity
    """
    # total number of objects
    N = pos.shape[0]
    # get a 3d histogram with number of objects in each cell
    D, edges = np.histogramdd(pos, bins=N_dim, range=[[0,Lbox],[0,Lbox],[0,Lbox]])
    # average number of particles per cell
    D_avg = N*1./N_dim**3
    #D /= D_avg
    #D -= 1.
    return D
    
@njit
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

    return tfour

@numba.vectorize
def rightwrap(x, L):
    if x >= L:
        return x - L
    return x
