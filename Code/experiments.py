import numpy as np
import matplotlib.pyplot as plt
import utils
import pickle
from functools import partial
from graph import Graph
from utils import generate_rgg, multivariate_t_rvs, gaussian_mixtures
from scipy.special import factorial
from rgg import RGG

def total_variance(p1, p2):
    if isinstance(p1, dict):
        ret = 0.
        for k, v in p1.items():
            if k not in p2:
                ret = max(ret, v)
            else:
                ret = max(ret, abs(v - p2[k]))

        for k, v in p2.items():
            if k not in p1:
                ret = max(ret, v)
    else:
        ret = max(p1 - p2)

    return ret

# Gaussian spherical prior
def gaussian_spherical_prior():
    original_edges_counts = []
    found_edges_counts = []

    total_variance_graph_statistics = []

    N = 300
    m = 1

    i = -1
    original_sigmas = list(np.arange(0.1, 3.5, 0.3))
    for original_sigma in original_sigmas:
        original_edges_counts.append([])
        found_edges_counts.append([])
        total_variance_graph_statistics.append([[], [], []])
        i += 1

        for run in range(20):
            g_o = generate_rgg(N, partial(np.random.multivariate_normal, np.zeros(2 * m), original_sigma ** 2 * np.identity(2 * m)))
            o_dks, o_eps, o_geodesics = g_o.get_degree_stats()

            rgg = RGG(g_o, m = m)
            found_sigma = rgg.sigma_binary_search() / np.sqrt(2)

            g_f = generate_rgg(N, partial(np.random.multivariate_normal, np.zeros(2 * m), found_sigma ** 2 * np.identity(2 * m)))
            f_dks, f_eps, f_geodesics = g_f.get_degree_stats()

            original_edges_counts[i].append(g_o.edges_count)
            found_edges_counts[i].append(g_f.edges_count)

            total_variance_graph_statistics[i][0].append(total_variance(o_dks, f_dks))
            total_variance_graph_statistics[i][1].append(total_variance(o_eps, f_eps))
            total_variance_graph_statistics[i][2].append(total_variance(o_geodesics, f_geodesics))
            
        print(original_sigma)

    f = open('spherical_gaussian_result.txt', 'wb')
    pickle.dump(original_sigmas, f)
    pickle.dump(original_edges_counts, f)
    pickle.dump(found_edges_counts, f)
    pickle.dump(total_variance_graph_statistics, f)

# Gaussian spherical prior
def experiment(params, priors, name, m = 1):
    original_edges_counts = []
    found_edges_counts = []

    found_sigmas = []
    total_variance_graph_statistics = []

    N = 300

    i = -1
    for i in range(len(priors)):
        original_edges_counts.append([])
        found_edges_counts.append([])
        total_variance_graph_statistics.append([[], [], []])
        found_sigmas.append([])

        for run in range(20):
            print(run)
            g_o = generate_rgg(N, priors[i])
            o_dks, o_eps, o_geodesics = g_o.get_degree_stats()

            rgg = RGG(g_o, m = m)
            found_sigma = rgg.sigma_binary_search() / np.sqrt(2)

            g_f = generate_rgg(N, partial(np.random.multivariate_normal, np.zeros(2 * m), found_sigma ** 2 * np.identity(2 * m)))
            f_dks, f_eps, f_geodesics = g_f.get_degree_stats()

            found_sigmas[i].append(found_sigma)
            original_edges_counts[i].append(g_o.edges_count)
            found_edges_counts[i].append(g_f.edges_count)

            total_variance_graph_statistics[i][0].append(total_variance(o_dks, f_dks))
            total_variance_graph_statistics[i][1].append(total_variance(o_eps, f_eps))
            total_variance_graph_statistics[i][2].append(total_variance(o_geodesics, f_geodesics))
        print(found_sigmas[-1])

    f = open(name + '_m' + str(m) + '.txt', 'wb')
    pickle.dump(m, f)
    pickle.dump(params, f)
    pickle.dump(found_sigmas, f)
    pickle.dump(original_edges_counts, f)
    pickle.dump(found_edges_counts, f)
    pickle.dump(total_variance_graph_statistics, f)

params = np.arange(0.1, 3.5, 0.3)
priors = [partial(multivariate_t_rvs, np.zeros(2 * 1), s ** 2 * np.identity(2 * 1), 15) for s in params]
experiment(params, priors, 'student_t', m = 1)
