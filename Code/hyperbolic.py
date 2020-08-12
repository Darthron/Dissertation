import numpy as np
import matplotlib.pyplot as plt

def plot_distance_histogram(zeta, alpha, R, beta):
    max_dist = np.arccosh(np.cosh(zeta * R) ** 2 + np.sinh(zeta * R) ** 2) / zeta
    dd = 1e-2
    num_samples = int(1e6)
    num_bins = 1 + int(max_dist / dd)
    bins = np.zeros(num_bins)

    # Maximum distance between two points in the disk of radius R is [cosh(zeta * R)]^2
    C = np.cosh(zeta * R) - 1

    mean = 0
    for i in range(num_samples):
        # Sample two radii
        r1 = np.arccosh(1 + C * np.random.uniform()) / alpha;
        r2 = np.arccosh(1 + C * np.random.uniform()) / alpha;
        
        # Sample two angles
        t1 = 2 * np.pi * np.random.uniform()
        t2 = 2 * np.pi * np.random.uniform()

        dt = np.pi - np.abs(np.pi - np.abs(t1 - t2))
        dist = np.arccosh(np.cosh(zeta * r1) * np.cosh(zeta * r2) - np.sinh(zeta * r1) * np.sinh(zeta * r2) * np.cos(dt)) / zeta

        #bins[int(dist / dd)] += 1
        mean += dist / num_samples

    print(max_dist)
    print(mean)
    #print(bins[-10:])
    #plt.hist(np.linspace(0, max_dist, num_bins), np.linspace(0, max_dist, num_bins), weights = bins, range = (0, max_dist))
    #plt.show()

def plot_pdf_quasiuniform(alpha, R):
    dr = 1e-2
    rs = np.arange(0.0, R + dr, dr)

    ax = plt.gca()
    ax.cla()
    ax.set_xlim((-R - 2 * dr, R + 2 * dr))
    ax.set_ylim((-R - 2 * dr, R + 2 * dr))
    ax.set_aspect('equal')

    for r in rs:
        c = hex(int((np.cosh(alpha * r) - 1) / (np.cosh(alpha * R) - 1) * 255))
        if len(c) == 3:
            c = "#0" + c[-1] + "0000"
        else:
            c = "#" + c[-2 :] + "0000"

        ax.add_artist(plt.Circle((0.0, 0.0), r, color = c, fill = False))

    half_cdf_circle = plt.Circle((0.0, 0.0), np.arccosh(1 + (np.cosh(alpha * R) - 1) / 2) / alpha, color = '#ffffff', fill = False)
    ax.add_artist(half_cdf_circle)
    ax.legend([half_cdf_circle], ['r = ' + str(np.arccosh(1 + (np.cosh(alpha * R) - 1) / 2) / alpha)])
    plt.show()


#plot_distance_histogram(1.0, 1, 10.0, 1.9)
plot_pdf_quasiuniform(1, 10.0)
