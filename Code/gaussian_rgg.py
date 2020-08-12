import typing
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from graph import Graph
from scipy.special import factorial

def generate_gaussian_rgg(N, sigma, m):
    n = 2 * m
    cov = (sigma ** 2) * np.identity(n)
    embeddings = np.random.multivariate_normal(np.zeros(n), cov, N)

    edges = []
    dists = []
    for i in range(N):
        for j in range(i + 1, N):
            dist_sq = np.linalg.norm(embeddings[i] - embeddings[j]) ** 2
            dists.append(dist_sq)
            p = np.exp(-dist_sq)

            if scipy.stats.bernoulli.rvs(p):
                edges.append((i + 1, j + 1))

    return Graph(N, edges)

