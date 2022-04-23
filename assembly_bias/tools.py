import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def vrange(starts, lengths):
    """ Create concatenated ranges of integers for multiple start/length

    Args:
        starts (numpy.array): starts for each range
        lengths (numpy.array): lengths for each range (same length as starts)

    Returns:
        numpy.array: concatenated ranges

    See the following illustrative example:

        starts = np.array([1, 3, 4, 6])
        lengths = np.array([0, 2, 3, 0])

        print vrange(starts, lengths)
        >>> [3 4 4 5 6]

    """

    # Repeat start position index length times and concatenate
    cat_start = np.repeat(starts, lengths)

    # Create group counter that resets for each start/length
    cat_counter = np.arange(lengths.sum()) - np.repeat(lengths.cumsum() - lengths, lengths)

    # Add group counter to group specific starts
    cat_range = cat_start + cat_counter

    return cat_range

def draw_pseudo(x, alpha, mu):
    """
    P(x)=(αμ)^αx/(αx)! e−μα
    https://math.stackexchange.com/questions/1245024/is-there-a-analytical-formula-for-super-and-sub-poissonian-distributions
    """
    prob = np.zeros(len(x))
    for i in range(len(x)):
        #alpha_x_int = int(np.round(alpha*x[i]))
        #prob[i] = (alpha*mu)**(alpha*x[i])/np.math.factorial(alpha_x_int)*np.exp(-mu*alpha) # chunky
        #prob[i] = (alpha*mu)**(alpha_x_int)/np.math.factorial(alpha_x_int)*np.exp(-mu*alpha) # og
        prob[i] = (alpha*mu)**(alpha*x[i])/gamma(alpha*x[i]+1.)*np.exp(-mu*alpha) # chunky # new
    return prob



def main():
    #alpha = 1.
    alpha = 0.8
    mu = 9.5
    
    n_max = 20
    ints = np.arange(0, n_max)
    ps = draw_pseudo(ints, alpha, mu)
    print(ps)
    print(np.sum(ps))
    ps /= np.sum(ps)

    n_tries = 100000
    mine = np.zeros(n_tries)
    nump = np.zeros(n_tries)
    for i in range(n_tries):
        mine[i] = (np.random.choice(ints, p=ps))
        nump[i] = (np.random.poisson(lam=mu, size=1))

    bins = np.linspace(0, n_max, n_max+1)
    hm, _ = np.histogram(mine, bins=bins)
    hn, edges = np.histogram(nump, bins=bins)
    binc = (edges[1:]+edges[:-1])*.5
    
    print("std/sqrt(mu), mean (nump) = ", np.std(nump)/np.sqrt(mu), np.mean(nump))
    print("std/sqrt(mu), mean (mine) = ", np.std(mine)/np.sqrt(mu), np.mean(mine))
    
    plt.plot(binc, hm, label="mine")
    plt.plot(binc, hn, label="nump")
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    main()
