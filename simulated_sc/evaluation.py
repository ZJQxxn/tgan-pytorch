import numpy as np
import scipy.stats

def MSE(a, b):
    a = a.detach().numpy().reshape(-1)
    b = b.detach().numpy().reshape(-1)
    return np.linalg.norm(a - b) / np.prod(a.shape)

def correlation(a, b):
    a = a.detach().numpy().reshape(-1)
    b = b.detach().numpy().reshape(-1)
    return scipy.stats.pearsonr(a.reshape(-1), b.reshape(-1))