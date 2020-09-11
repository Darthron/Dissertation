import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from graph import Graph
from functools import partial
from scipy.special import factorial
from sklearn.metrics.pairwise import euclidean_distances

# Assumes f(0) = 0 and f(inf) = 0 and has at most two modes, one negative one, optional
# and exctly one positive one
def find_function_endpoint(h):
    h_eps = 1e-12
    dx = 1e-5
    x_end = 1.1

    h_xend = h(x_end)
    while not (h_xend >= 0 and h_xend < h_eps and h(x_end + dx) <= h_xend):
        if np.isinf(x_end):
            exit(0)
        x_end *= 2
        h_xend = h(x_end)

    x_beg = dx
    x_L = (x_beg + x_end) / 2

    while x_end - x_beg > 1e-3:
        h_xL = h(x_L)
        h_xL_add_dx = h(x_L + dx)

        if h_xL > 0 and h_xL <= h_eps and h_xL_add_dx < h_xL:
            x_end = x_L
        else:
            x_beg = x_L

        x_L = (x_beg + x_end) / 2

    return x_L

def riemann_sum(h, x_beg = None, x_end = 1.0, dx = 1e-4):
    # Midpoint Riemann sum
    x_beg = 0 if x_beg is None else 10 ** (-15.95)
    
    xs = np.arange(x_beg, x_end, dx)
    hs = dx * h(xs)
    hs[[0, -1]] /= 2

    if False:
        plt.plot(xs, h(xs))
        plt.xlim(left = 0)
        plt.xlim(right = 5)
        plt.ylim(top = 4)
        plt.show()

    return (hs.sum(), hs, xs)

def generate_rgg(N, embeddings_prior, squared_dist = True):
    embeddings = embeddings_prior(N)
    eta = 2 if squared_dist else 1

    adjacency_matrix = scipy.stats.bernoulli.rvs(np.exp(-euclidean_distances(embeddings, embeddings) ** eta)) # not symmetric, we will only consider the upper triangle
    np.fill_diagonal(adjacency_matrix, 0)

    upper_triangle = np.triu(adjacency_matrix)
    adjacency_matrix = upper_triangle.T + upper_triangle

    edges = np.argwhere(adjacency_matrix)
    curr_index = 0
    edges_lists = []
    i = 0
    while i < edges.shape[0]:
        while curr_index <= edges[i][0]:
            edges_lists.append(set([]))
            curr_index += 1

        while i < edges.shape[0] and edges[i][0] == curr_index - 1:
            edges_lists[-1].add(edges[i][1])
            i += 1

    while curr_index < N:
        edges_lists.append(set([]))
        curr_index += 1

    return Graph(N, edges_lists)

#written by Enzo Michelangeli, style changes by josef-pktd
# Student's T random variable
def multivariate_t_rvs(m, S, df = np.inf, n = 1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n) / df
    z = np.random.multivariate_normal(np.zeros(d), S, (n,))

    return m + z / np.sqrt(x)[:, None]   # same output format as random.multivariate_normal

def gaussian_mixtures(ms, Ss, pis, N):
    Ns = np.random.multinomial(N, pis)

    embeddings = np.random.multivariate_normal(ms[0], Ss[0], Ns[0])
    for i in range(1, len(pis)):
        embeddings = np.concatenate((embeddings, np.random.multivariate_normal(ms[i], Ss[i], Ns[i])), axis = 0)

    return embeddings











