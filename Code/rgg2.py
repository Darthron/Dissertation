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

    m = 2

    def __init__(self, graph : Graph = None, sigma = 1.0):
        self.graph = graph
        self.q_sigma = sigma

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

    # TODO: Replace gradient descent with binary search
    def _gradient_descent(self):
        curr_F = RGG.free_energy(self.graph.vertices_count, self.graph.edges_count, self.q_sigma ** 2)
        step_size = 1e-7

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
        dF_dsigma = -2 * RGG.m * self.graph.vertices_count * (self.graph.vertices_count - 1) * self.q_sigma
        dF_dsigma += (self.graph.vertices_count * (self.graph.vertices_count - 1) / 2 - self.graph.edges_count) * self._differentiate_sigma_term_non_edge()
        dF_dsigma -= self.q_sigma * 2 * RGG.m - 2 * RGG.m / self.q_sigma

        return dF_dsigma

    def _differentiate_sigma_term_non_edge(self):
        h = partial(RGG.h, self.q_sigma ** 2)
        x_end = utils.find_function_endpoint(h)
        S, hs, xs = utils.riemann_sum(h, x_end = x_end)

        res = -S * 2 * RGG.m / self.q_sigma
        res += hs @ xs * self.q_sigma ** (-3)

        return res / ((2 * self.q_sigma ** 2) ** RGG.m * factorial(RGG.m - 1))

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

    @staticmethod
    def plot_free_energy(vertices_count, edges_count, s_beg = 0.1, s_end = 3.0, step = 1e-1):
        sigmas = list(np.arange(s_beg, s_end, step))
        Fs = np.array(list(map(lambda s: RGG.free_energy(vertices_count, edges_count, s ** 2), sigmas)))

        i = np.argmax(Fs)
        print(sigmas[i])
        print(sigmas[i] / np.sqrt(2), Fs[i])

        plt.plot(sigmas, Fs)
        plt.xlim(right = s_end + step)
        plt.xlim(left = 0)
        plt.show()

    @staticmethod
    def plot_non_edge_term(s_beg = 0.3, s_end = 2.5, step = 1e-1):
        sigmas = list(np.arange(s_beg, s_end, step))
        Ts = np.array(list(map(lambda s: RGG.calculate_non_edge_term(s ** 2), sigmas)))

        plt.plot(sigmas, Ts)
        plt.xlim(right = 3)
        plt.xlim(left = 0)
        plt.show()

"""
xs = np.arange(0, 50, 1e-3)
hs = RGG.h(1.2, xs)
print(RGG.calculate_non_edge_term(1.2))
print(utils.find_function_endpoint(partial(RGG.h, 1.2)))
plt.plot(xs, hs)
plt.show()
exit(0)
orig_sigma = 1.2
g = generate_gaussian_rgg(300, orig_sigma, RGG.m)
rgg = RGG(g)

if True:
    found_sig = rgg.sigma_binary_search() / np.sqrt(2)#rgg._gradient_descent() / np.sqrt(2)

    print("==================================")
    print(150 * 299 / (1 + 4 * orig_sigma ** 2) ** RGG.m)
    print(150 * 299 / (1 + 4 * found_sig ** 2) ** RGG.m)
    print(g.edges_count)
    print(np.sqrt(((g.vertices_count * (g.vertices_count - 1) / (2 * g.edges_count)) ** (1 / RGG.m) - 1) / 2) / np.sqrt(2))
    print(found_sig)
    print("====================== ", str(np.log(orig_sigma + 1) - found_sig))
else:
    RGG.plot_free_energy(g.vertices_count, g.edges_count, 0.1, 0.8, 1e-2)
    #RGG.plot_non_edge_term(g.vertices_count, g.edges_count)

"""
origs = []
found = []
for orig_sigma in np.arange(0.4, 1.0, 0.1):
    #orig_sigma = 0.7
    print(orig_sigma)
    g = generate_gaussian_rgg(300, orig_sigma, RGG.m)
    rgg = RGG(g)

    if True:
        found_sig = rgg.sigma_binary_search() / np.sqrt(2)#rgg._gradient_descent() / np.sqrt(2)
        print(found_sig)
        continue

        print("==================================")
        print(150 * 299 / (1 + 4 * orig_sigma ** 2) ** RGG.m)
        print(150 * 299 / (1 + 4 * found_sig ** 2) ** RGG.m)
        print(g.edges_count)
        print(np.sqrt(((g.vertices_count * (g.vertices_count - 1) / (2 * g.edges_count)) ** (1 / RGG.m) - 1) / 2) / np.sqrt(2))
        print(found_sig)
        print("====================== ", str(np.log(orig_sigma + 1) - found_sig))
        origs.append(orig_sigma)
        found.append(found_sig)
    else:
        RGG.plot_free_energy(g.vertices_count, g.edges_count, 0.1, 0.8, 1e-2)
        #RGG.plot_non_edge_term(g.vertices_count, g.edges_count)
plt.plot([0, 0], [2, 2])
print(origs, found)
plt.scatter(np.exp(np.array(found)), origs)
plt.show()
