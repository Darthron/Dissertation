import typing
import numpy as np
import matplotlib.pyplot as plt
import utils
from functools import partial
from graph import Graph
from gaussian_rgg import generate_gaussian_rgg
from scipy.special import factorial

MAX_EXPONENT = 709
MIN_INVERSE_EXPONENT = -745

class RGG:
    graph : Graph
    q_sigma : float

    def __init__(self, graph : Graph = None):
        self.graph = graph
        self.q_sigma = 1.5

    def _gradient_descent(self):
        curr_F = RGG.free_energy(self.graph.vertices_count, self.graph.edges_count, self.q_sigma ** 2)
        step_size = 1e-6

        print('GRAD')
        while True:
            #print(curr_F)
            print(self.q_sigma)
            old_sigma = self.q_sigma
            #print('Sigma = ', str(old_sigma))
            dsigma = self._differentiate_free_energy_sigma()
            self.q_sigma += step_size * dsigma
            #print('NSigma = ', str(self.q_sigma))
            new_F = RGG.free_energy(self.graph.vertices_count, self.graph.edges_count, self.q_sigma ** 2)

            print(curr_F, new_F)
            if new_F <= curr_F:
                break

            curr_F = new_F

            #print('------------------')

        return old_sigma

    def _differentiate_free_energy_sigma(self):
        dF_dsigma = -2 * self.graph.vertices_count * (self.graph.vertices_count - 1) * self.q_sigma
        dF_dsigma += (self.graph.vertices_count * (self.graph.vertices_count - 1) / 2 - self.graph.edges_count) * self._differentiate_sigma_term_non_edge()
        dF_dsigma -= self.q_sigma * 2 - 4 / self.q_sigma

        return dF_dsigma

    def _differentiate_sigma_term_non_edge(self):
        h = partial(RGG.h, self.q_sigma ** 2)
        x_end = utils.find_function_endpoint(h)
        S, hs, xs = utils.riemann_sum(h, x_end)

        res = -S * 2 / self.q_sigma
        res += hs @ xs * self.q_sigma ** (-3)

        return res / (2 * self.q_sigma ** 2)

    @staticmethod
    def free_energy(vertices_count, edges_count, s_sq):
        total_possible_edges_count = vertices_count * (vertices_count - 1) / 2

        F = -2 * total_possible_edges_count * s_sq 
        F += (total_possible_edges_count - edges_count) * RGG.calculate_non_edge_term(s_sq)
        F -= s_sq - 1 + np.log(2) / 2 - np.log(s_sq)

        return F

    @staticmethod
    def calculate_non_edge_term(sigma_sq):
        h = partial(RGG.h, sigma_sq)
        x_end = utils.find_function_endpoint(h)

        return utils.riemann_sum(h, x_end)[0] / (2 * sigma_sq)

    @staticmethod
    def h(sigma_sq, x):
        def log_term(x):
            if x < 10 ** -15.5:
                return np.log(np.exp(10 ** -15.5) - 1)
            elif x > MAX_EXPONENT:
                return x

            return np.log(np.exp(x) - 1)

        if not isinstance(x, np.ndarray):
            return np.exp(-x / (2 * sigma_sq)) * log_term(x)

        log_terms = np.array(list(map(lambda u: log_term(u), x)))

        return np.exp(-x / (2 * sigma_sq)) * log_terms

    @staticmethod
    def plot_free_energy(vertices_count, edges_count, s_beg = 0.3, s_end = 1.8, step = 1e-3):
        sigmas = list(np.arange(s_beg, s_end, step))
        Fs = np.array(list(map(lambda s: RGG.free_energy(vertices_count, edges_count, s ** 2), sigmas)))
        
        i = np.argmax(Fs)
        print(sigmas[i])
        print(sigmas[i] / np.log(2), Fs[i])

        plt.plot(sigmas, Fs)
        plt.xlim(right = 3)
        plt.show()

    @staticmethod
    def plot_non_edge_term(s_beg = 0.3, s_end = 2.5, step = 1e-3):
        sigmas = list(np.arange(s_beg, s_end, step))
        Ts = np.array(list(map(lambda s: RGG.calculate_non_edge_term(s ** 2), sigmas)))

        plt.plot(sigmas, Ts)
        plt.xlim(right = 3)
        plt.show()

orig_sigma = 0.4
g = generate_gaussian_rgg(300, orig_sigma, 1)
rgg = RGG(g)

if 1:
    found_sig = rgg._gradient_descent() / np.sqrt(2)

    print(150 * 299 / (1 + 4 * orig_sigma ** 2))
    print(150 * 299 / (1 + 4 * found_sig ** 2))
    print(g.edges_count)
    print(np.sqrt(75 * 299 / g.edges_count - 0.5) / np.sqrt(2))
    print(found_sig)
else:
    RGG.plot_free_energy(g.vertices_count, g.edges_count, 0.1, 3.0, 1e-1)
    #RGG.plot_non_edge_term(g.vertices_count, g.edges_count)
