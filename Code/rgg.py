import numpy as np
import matplotlib.pyplot as plt
import utils
from functools import partial
from graph import Graph
from utils import generate_rgg, multivariate_t_rvs, gaussian_mixtures
from scipy.special import factorial, gamma

MAX_EXPONENT = 709
MIN_INVERSE_EXPONENT = -745

class RGG:
    m = 2

    def __init__(self, graph : Graph = None, sigma = 1.0, m = 1, squared_dist = True):
        self.graph = graph
        self.q_sigma = sigma
        self.squared_dist = squared_dist

        RGG.m = m

    def sigma_binary_search(self):
        s_l = 1e-6
        s_L = 1.01
        ds = 1e-3

        while RGG.free_energy(self.graph.vertices_count, self.graph.edges_count, s_L ** 2, self.squared_dist) < RGG.free_energy(self.graph.vertices_count, self.graph.edges_count, (s_L + ds) ** 2, self.squared_dist):
            s_L *= 2

        while s_L - s_l >= 1e-6:
            s = (s_l + s_L) / 2
            curr_F = RGG.free_energy(self.graph.vertices_count, self.graph.edges_count, s ** 2, self.squared_dist)

            if RGG.free_energy(self.graph.vertices_count, self.graph.edges_count, (s - ds) ** 2, self.squared_dist) < curr_F:
                s_l = s
            else:
                s_L = s


        return (s_l + s_L) / 2

    @staticmethod
    def free_energy(vertices_count, edges_count, s_sq, squared_dist):
        total_possible_edges_count = vertices_count * (vertices_count - 1) / 2
        n = 2 * RGG.m

        if squared_dist:
            F = -total_possible_edges_count * n * s_sq 
            F += (total_possible_edges_count - edges_count) * RGG.calculate_non_edge_term(s_sq, squared_dist)
            F -= vertices_count * (RGG.m * s_sq - RGG.m - RGG.m * np.log(s_sq))
        else:
            s = np.sqrt(s_sq)

            F = -total_possible_edges_count * np.sqrt(2) * s * gamma(RGG.m + 0.5) / gamma(RGG.m)
            a = F
            F += (total_possible_edges_count - edges_count) * RGG.calculate_non_edge_term(s_sq, squared_dist)
            F -= vertices_count * (RGG.m * s_sq - RGG.m - RGG.m * np.log(s_sq))

        return F

    @staticmethod
    def calculate_non_edge_term(sigma_sq, squared_dist):
        h = partial(RGG.h, sigma_sq, squared_dist)
        x_end = utils.find_function_endpoint(h)
        """
        plt.plot(np.arange(0, x_end, 1e-1), h(np.arange(0, x_end, 1e-1)))
        plt.show()
        """

        if squared_dist:
            return utils.riemann_sum(h, x_end = x_end)[0] / ((2 * sigma_sq) ** RGG.m * factorial(RGG.m - 1))
        else:
            s = np.sqrt(sigma_sq)
            return utils.riemann_sum(h, x_end = x_end)[0] / (s ** (2 * RGG.m - 1) * 2 ** (RGG.m - 1) * factorial(RGG.m - 1))

    @staticmethod
    def h(sigma_sq, squared_dist, x):
        def log_term(x):
            #x = np.sqrt(x)
            if x < 10 ** -15.5:
                return np.log(np.exp(10 ** -15.5) - 1) if RGG.m == 1 else 0.0
            elif x > MAX_EXPONENT:
                return x

            return np.log(np.exp(x) - 1)

        if not isinstance(x, np.ndarray):
            if squared_dist:
                return log_term(x) * x ** (RGG.m - 1) * np.exp(-x / (2 * sigma_sq))
            else:
                return log_term(x) * x ** (2 * RGG.m - 1) * np.exp(-x ** 2 / (2 * sigma_sq))

        log_terms = np.array(list(map(lambda u: log_term(u), x)))

        if squared_dist:
            return log_terms * x ** (RGG.m - 1) * np.exp(-x / (2 * sigma_sq))
        else:
            return log_terms * x ** (2 * RGG.m - 1) * np.exp(-x ** 2 / (2 * sigma_sq))

    @staticmethod
    def plot_free_energy(vertices_count, edges_count, s_beg = 0.1, s_end = 3.0, step = 1e-1):
        sigmas = list(np.arange(s_beg, s_end, step))
        Fs = np.array(list(map(lambda s: RGG.free_energy(vertices_count, edges_count, s ** 2, False), sigmas)))

        i = np.argmax(Fs)
        print(sigmas[i])
        print(sigmas[i] / np.sqrt(2), Fs[i])

        plt.plot(sigmas, Fs)
        plt.xlim(right = s_end + step)
        plt.xlim(left = 0)
        plt.show()

