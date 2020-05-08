import numpy as np
from numba import njit, jit

from scipy import stats          # For method_silverman
from scipy.special import gamma  # For rvkde_sigmas
from scipy.spatial import KDTree # For rvkde_sigmas

# TO EXPORT: method_silverman, method_abramson, method_RVKDE, method_ERVKDE, MSE

# Note: methods currently may not work for dimensions other than 2

# Versions used in development:
# Python 3.8.1
# numpy 1.18.1
# numba 0.49.0
# scipy 1.4.1


@njit
def kde(x, samples, sigmas, dim=2):
    """
    f_hat(x) -> density at x
    
    For: get_density, abramson_sigmas
    """
    dist_sq = np.sum((x - samples)**2, axis=1)
    kde_val = (1/((sigmas**2)*2*np.pi))**(0.5*dim)*np.exp(-dist_sq/(2*(sigmas**2)))
    return np.mean(kde_val)


def get_density(samples, xs, ys, sigmas, dim=2):
    """
    samples, xs, ys -> kde(x,y) for x in xs, y in ys for given samples

    xs, ys: np.linspace representing X, Y axes
    dim: dimension
    samples: np.array of samples
    
    Dependencies: kde
    For: abramson_sigmas, method_abramson, method_RVKDE, method_ERVKDE
    """
    return np.array([[kde(np.array([x,y]), samples, sigmas, dim) for x in xs] for y in ys])


@njit
def ISE_loop(samples, s, dim=2):
    """
    Optimized loop for abramson_sigmas
    
    samples: np.array of samples
    s: fixed sigma (one value)
    dim: dimension
    """
    N = len(samples)
    total = 0
    for i in range(N):
        for j in range(N):
            dist_sq = np.sum((samples[i]-samples[j])**2)
            total += (i != j)*(1/((s**2)*2*np.pi))**(0.5*dim)*np.exp(-dist_sq/(2*(s**2)))
    return -1*(2*total/N/(N-1))


@jit(looplift=True)
def abramson_sigmas(samples, xs, ys, dim=2): # , xy_range=[-4,4] (Removed parameter, due to memory issue)
    """
    samples, xs, ys -> sigmas
    
    dim: dimension
    xs, ys: np.linspace representing X, Y axes
    
    Assumption: xs and ys are in range [-4,4] 
    
    Dependencies: kde, ISE_loop
    For: method_abramson
    """
#     xy_range: np.linspace range for xs and ys (Removed from docstring, memory issue)

    xy_interval = abs(xs[1] - xs[0])*abs(ys[1] - ys[0])

#     sigma_range = [2**x for x in range(*xy_range)]
    sigma_range = [2**x for x in range(-4,4)]
    best_s, best_ISE_est = 0, np.inf
    N = len(samples)
    
    # Estimate the best sigma value
    for s in sigma_range:
        # Compute Fixed Gaussian KDE
        fixed_sigmas = np.array([s] * N)
        density_squared_sum = np.square(get_density(samples, xs, ys, fixed_sigmas)).sum()

        # Get ISE estimate
        ISE_estimate = ISE_loop(samples, s, dim) + density_squared_sum * xy_interval
        
        if ISE_estimate < best_ISE_est:
            best_s, best_ISE_est = s, ISE_estimate

    best_fixed_sigmas = np.array([best_s] * N)
    
    # Use estimate to compute sigmas
    sigmas = []
    for center in samples:
        sigmas.append(1 / np.sqrt(kde(center, samples, best_fixed_sigmas)))
        
    sigmas = np.array(sigmas)
    
    # Make similar to fixed gaussian
    sigmas = sigmas / sigmas.mean() * best_s
    
    return sigmas


@jit(looplift=True)
def rvkde_sigmas(samples, dim=2, K=None, beta=2, smoothing=True):
    """
    samples -> sigmas
    
    K: if None, use defaults
    
    Dependencies: scipy.special.gamma, scipy.spatial.KDTree
    For: method_RVKDE, method_ERVKDE
    """
    if K==None:
        K = int(len(samples)/10)
    
    tree = KDTree(samples, int(K*1.5))
    
    R_scale = np.sqrt(np.pi)/np.power((K+1)*gamma(dim/2+1), 1./dim)
    sigma_scale = R_scale*beta*(dim+1.)/dim/K
    
    # Calculate sigmas
    sigmas = []
    for x in samples:
        # Find K nearest neighbor
        knn = tree.query(x, K+1)
        sigma = np.sum(knn[0])*sigma_scale
        sigmas.append(sigma)
    
    # Sigma smoothing
    new_sigmas = []
    if smoothing:
        for x in samples:
            neighbor_indices = tree.query(x, K+1)[1][1:K+1] # indices of nearest-neighbors
            new_sigma = np.sum([sigmas[i] for i in neighbor_indices])/K
            new_sigmas.append(new_sigma)
    else:
        new_sigmas = sigmas
            
    new_sigmas = np.array(new_sigmas)

    return new_sigmas


def method_silverman(samples, tofit):
    """
    samples, tofit (X,Y axes) -> Silverman KDE density
    
    samples: 2D numpy array
    tofit: [x,y] pairs for X,Y np.linspace axes
    
    Example tofit:
       tofit = np.array([[x,y] for y in ys for x in xs])
          ( where xs = ys = np.linspace(-4,4,100) )
    
    Dependencies: scipy.stats
    """
    gaus_kde = stats.gaussian_kde(samples.T)
    return gaus_kde(tofit.T)


def method_abramson(samples, xs, ys, dim=2):  # xy_range=[-4,4] (Removed parameter, memory constraint)
    """
    samples, xs, ys -> Abramson KDE density
    
    samples: 2D numpy array
    xs, ys: np.linspace(start, end, num_points) for X, Y axes
       (represents X and Y axes)
       
    Assumption for abramson_sigmas: xs and ys are in range [-4,4] 
    
    Dependencies: abramson_sigmas, get_density
    """
#     xy_range: np.linspace range for xs and ys (Removed from docstring, memory issue)

    sigmas = abramson_sigmas(samples, xs, ys, dim) # , xy_range removed
    return get_density(samples, xs, ys, sigmas, dim)


def method_RVKDE(samples, xs, ys, dim=2, K=None, beta=2, smoothing=True):
    """
    samples, xs, ys -> RVKDE density
    
    samples: 2D numpy array
    xs, ys: np.linspace(start, end, num_points) for X, Y axes
       (represents X and Y axes)
       
    Dependencies: rvkde_sigmas, get_density
    """
    K = int(len(samples)/10) if (K is None) else K
    sigmas = rvkde_sigmas(samples, dim, K, beta, smoothing)
    return get_density(samples, xs, ys, sigmas, dim)


def method_ERVKDE(samples, xs, ys, dim=2, K=None, beta=2, smoothing=True):
    """
    samples, xs, ys -> ERVKDE density
    
    samples: 2D numpy array
    xs, ys: np.linspace(start, end, num_points) for X, Y axes
       (represents X and Y axes)
       
   Dependencies: rvkde_sigmas, get_density
    """
    K = int(len(samples)/10) if (K is None) else K
    sigmas = rvkde_sigmas(samples, dim, K, beta, smoothing)
    sig_avg = np.mean(np.std(samples))
    diff = ((4*sig_avg**5/(len(samples)*(dim+2)))**(1/(dim+4))) - np.median(sigmas)
    elevated_sigmas = np.array([s + diff for s in sigmas])
    return get_density(samples, xs, ys, elevated_sigmas, dim)


def MSE(predicted_density, true_density):
    """Mean Square Error"""
    predicted_density = np.array(predicted_density)
    true_density = np.array(true_density)
    return ((predicted_density-true_density)**2).mean()
