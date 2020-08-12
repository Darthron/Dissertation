import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.special import factorial

class RGG:
    m = 2
    sigma = 1

def scalar_h(sigma_sq, x):
    if x <= 1e-12:
        return 0.0

    sigma_sq *= 2 
    #log_term = x if x > 700 else np.log(np.exp(x) - 1)
    log_term = np.log(1 - np.exp(-x))

    return log_term * (x ** (RGG.m - 1)) * np.exp(-x / (2 * sigma_sq))

def integral(sigma_sq):
    x = 1e-7
    x_end = 20.0
    dx = 1e-7

    res = 0
    while x < x_end:
        res += scalar_h(x) * 2 * dx
        x += 2 * dx

    return res

def integral2():
    xs = np.arange(1e-6, 20.0, 2 * 1e-6)
    #hs = np.array([scalar_h(x) for x in xs])
    hs = h(1, xs)

    return hs.sum() * 2 * 1e-6

"""
h = np.vectorize(scalar_h, excluded = ['sigma_sq'])
print(integral2())




exit(0)

#print(integral(1))
start = time.time()
print(integral(1))
end = time.time()
print(end - start)
exit(0)

xs = np.arange(0.0, 20.0, 0.01)
hs = h(xs)

plt.plot(xs, hs)
plt.show()
"""

fig, axs = plt.subplots(2, 2, sharex = 'all')

for j, (sigma_sq, m) in enumerate([(1.5, 2), (1.2, 1), (0.7, 3), (0.9, 2)]):

    cov = sigma_sq * np.identity(2 * m)
    bins = np.zeros(500)
    i = 0
    mean = 0
    sims = 1e7
    while i < sims:
        embeddings = np.random.multivariate_normal(np.zeros(2 * m), cov, 2)
        d = np.linalg.norm(embeddings[0] - embeddings[1]) ** 2
        mean += d / sims

        if d >= 50:
            continue

        bins[int(d / 1e-1)] += 1
        i += 1

    us = np.arange(0, 50, 1e-1)
    pdf = np.array([(u ** (m - 1)) * np.exp(-u / (4 * sigma_sq)) / (4 * sigma_sq) ** m / factorial(m - 1) for u in us])

    axs[int(j / 2), j % 2].set_title(r'$(\sigma^2, m) = $' + str((sigma_sq, m)))
    axs[int(j / 2), j % 2].plot(us, pdf)
    axs[int(j / 2), j % 2].hist(np.linspace(0, 50, 500), np.linspace(0, 50, 500), weights = pdf.max() / bins.max() * bins, range = (0, 50))

plt.show()
