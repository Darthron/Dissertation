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
    embeddings_sigma : float
    distance_sigma : float

    m = 2

    def __init__(self, graph : Graph = None, sigma = 1.0):
        self.graph = graph
        self.embeddings_sigma = sigma
        self.distance_sigma = sigma * np.sqrt(2)
        self.h = np.vectorize(self._scalar_h, excluded = ['sigma_sq'])

    def sigma_binary_search(self):
        s_l = 1e-6
        s_L = 1.01
        ds = 1e-3

        while self.free_energy(s_L ** 2) < self.free_energy((s_L + ds) ** 2):
            s_L *= 2

        while s_L - s_l >= 1e-6:
            s = (s_l + s_L) / 2
            print(s)
            curr_F = self.free_energy(s ** 2)

            if self.free_energy((s - ds) ** 2) < curr_F:
                s_l = s
            else:
                s_L = s

        return (s_l + s_L) / 2

    def free_energy(self, s_sq):
        total_possible_edges_count = self.graph.vertices_count * (self.graph.vertices_count - 1) / 2
        n = 2 * RGG.m

        F = -self.graph.edges_count * n * s_sq 
        F += (total_possible_edges_count - self.graph.edges_count) * self.calculate_non_edge_integral(s_sq) / ((2 * s_sq) ** self.m) / factorial(self.m - 1)
        F -= self.m * s_sq - self.m - self.m * np.log(s_sq)

        return F

    def calculate_non_edge_integral(self, s_sq):
        dx = 2 * 1e-3
        xs = np.arange(dx / 2, 20.0, dx)
        hs = self.h(s_sq, xs)

        return hs.sum() * dx

    def _scalar_h(self, sigma_sq, x):
        if x <= 1e-12:
            return 0.0

        log_term = np.log(1 - np.exp(-x))

        return log_term * (x ** (RGG.m - 1)) * np.exp(-x / (2 * sigma_sq))

    """
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

"""
xs = np.arange(0, 50, 1e-3)
hs = RGG.h(1.2, xs)
print(RGG.calculate_non_edge_term(1.2))
print(utils.find_function_endpoint(partial(RGG.h, 1.2)))
plt.plot(xs, hs)
plt.show()
exit(0)
"""

"""
orig_sigmas = np.arange(0.2, 4.1, 0.1)
founds = []
for orig_sigma in orig_sigmas:
    print(orig_sigma)
    g = generate_gaussian_rgg(300, orig_sigma, RGG.m)
    g.edges_count = 150 * 299 / (1 + 4 * orig_sigma ** 2) ** RGG.m
    rgg = RGG(g)

    founds.append(rgg.sigma_binary_search() / np.sqrt(2))

print(founds)
plt.scatter(orig_sigmas, np.exp(np.array(founds)), label = 'exp(found_sigma)')
plt.scatter(orig_sigmas, founds, label = 'found_sigma')
plt.xlabel('orig_sigma')
plt.legend()
plt.show()

exit(0)
"""


orig_sigma = 2.7
g = generate_gaussian_rgg(300, orig_sigma, RGG.m)
rgg = RGG(g)
g.vertices_count = 1000
g.edges_count = g.vertices_count * (g.vertices_count - 1) / (2 * (1 + 4 * orig_sigma ** 2) ** RGG.m)

"""
fs = []
for s_sq in np.arange(0.5, 3, 0.01):
    fs.append(rgg.free_energy(s_sq))

print(np.argmax(np.array(fs)))
plt.plot(np.arange(0.5, 3, 0.01), fs)
plt.show()
"""
print(rgg.sigma_binary_search() ** 2)

exit(0)


origs = []
found = []
for orig_sigma in np.arange(0.2, 1.0, 0.1):
    #orig_sigma = 0.7
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
        origs.append(orig_sigma)
        found.append(found_sig)
    else:
        RGG.plot_free_energy(g.vertices_count, g.edges_count, 0.1, 0.8, 1e-2)
        #RGG.plot_non_edge_term(g.vertices_count, g.edges_count)
plt.plot([0, 0], [2, 2])
print(origs, found)
plt.scatter(np.exp(np.array(found)), origs)
plt.show()

