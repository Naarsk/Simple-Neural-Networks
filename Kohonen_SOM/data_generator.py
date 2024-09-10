import numpy as np


def circular(M):
    x = np.empty(shape=(M, 2))
    for j in range(M):
        th = 2 * np.pi * j / M
        er = np.random.normal(scale=0.1)
        x[j, 0] = (er + 1) * np.cos(th)
        x[j, 1] = (er + 1) * np.sin(th)
    return x


def cup(M):
    x = np.empty(shape=(M, 3))
    for j in range(M):
        phi = 2 * np.pi * j / M
        th = np.random.normal(loc=-np.pi, scale=0.5)
        er = np.random.normal(scale=0.1)
        x[j, 0] = (er + 1) * np.sin(th) * np.cos(phi)
        x[j, 1] = (er + 1) * np.sin(th) * np.sin(phi)
        x[j, 2] = (er + 1) * np.cos(th)
    return x


def spiral(M):
    x = np.empty(shape=(M, 2))
    for j in range(M):
        phi = 4 * np.pi * j / M
        er = np.random.normal(scale=0.01)
        x[j, 0] = (0.5+er) * np.exp(0.1 * phi) * np.cos(phi)
        x[j, 1] = (0.5+er) * np.exp(0.1 * phi) * np.sin(phi)
    return x
