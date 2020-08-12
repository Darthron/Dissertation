import numpy as np
import matplotlib.pyplot as plt
import utils
from functools import partial
from graph import Graph
from utils import generate_rgg, multivariate_t_rvs, gaussian_mixtures
from scipy.special import factorial

MAX_EXPONENT = 709
MIN_INVERSE_EXPONENT = -745

class RGG:
    m = 2

    def __init__(self, graph : Graph = None, sigma = 1.0, m = 1):
        self.graph = graph
        self.q_sigma = sigma
        RGG.m = m

    def sigma_binary_search(self):
        s_l = 1e-6
        s_L = 1.01
        ds = 1e-3

        while RGG.free_energy(self.graph.vertices_count, self.graph.edges_count, s_L ** 2) < RGG.free_energy(self.graph.vertices_count, self.graph.edges_count, (s_L + ds) ** 2):
            s_L *= 2

        while s_L - s_l >= 1e-6:
            s = (s_l + s_L) / 2
            curr_F = RGG.free_energy(self.graph.vertices_count, self.graph.edges_count, s ** 2)

            if RGG.free_energy(self.graph.vertices_count, self.graph.edges_count, (s - ds) ** 2) < curr_F:
                s_l = s
            else:
                s_L = s

        return (s_l + s_L) / 2

    @staticmethod
    def free_energy(vertices_count, edges_count, s_sq):
        total_possible_edges_count = vertices_count * (vertices_count - 1) / 2
        n = 2 * RGG.m

        F = -total_possible_edges_count * n * s_sq 
        F += (total_possible_edges_count - edges_count) * RGG.calculate_non_edge_term(s_sq)
        F -= RGG.m * s_sq - RGG.m - RGG.m * np.log(s_sq)

        return F

    @staticmethod
    def calculate_non_edge_term(sigma_sq):
        h = partial(RGG.h, sigma_sq)
        x_end = utils.find_function_endpoint(h)

        return utils.riemann_sum(h, x_end = x_end)[0] / ((2 * sigma_sq) ** RGG.m * factorial(RGG.m - 1))

    @staticmethod
    def h(sigma_sq, x):
        def log_term(x):
            #x = np.sqrt(x)
            if x < 10 ** -15.5:
                return np.log(np.exp(10 ** -15.5) - 1) if RGG.m == 1 else 0.0
            elif x > MAX_EXPONENT:
                return x

            return np.log(np.exp(x) - 1)

        if not isinstance(x, np.ndarray):
            return log_term(x) * x ** (RGG.m - 1) * np.exp(-x / (2 * sigma_sq))

        log_terms = np.array(list(map(lambda u: log_term(u), x)))

        return log_terms * x ** (RGG.m - 1) * np.exp(-x / (2 * sigma_sq))

"""
gaussian_mixtures([np.zeros(2 * RGG.m), np.ones(2 * RGG.m)], [np.identity(2 * RGG.m), np.identity(2 * RGG.m)], [0.3, 0.7], 10)

orig_sigma = 0.4
g = generate_rgg(300, partial(np.random.multivariate_normal, np.zeros(2 * RGG.m), orig_sigma ** 2 * np.identity(2 * RGG.m)))
#dks, eps = g.get_degree_stats()

orig_sigma = 0.4
g = generate_rgg(300, partial(multivariate_t_rvs, np.zeros(2 * RGG.m), orig_sigma ** 2 * np.identity(2 * RGG.m), np.inf))

rgg = RGG(g)
found_sig = rgg.sigma_binary_search() / np.sqrt(2)#rgg._gradient_descent() / np.sqrt(2)
print(found_sig)

orig_sigma = 0.4
g = generate_gaussian_rgg(300, orig_sigma, 1)
rgg = RGG(g)
"""
