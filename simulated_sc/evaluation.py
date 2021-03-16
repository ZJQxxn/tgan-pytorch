import numpy as np
import scipy.stats

def MSE(a, b):
    a = a.detach().numpy()
    b = b.detach().numpy()
    return np.linalg.norm(a - b) / np.prod(a.shape)

def correlation(a, b):
    a = a.detach().numpy()
    b = b.detach().numpy()
    return scipy.stats.pearsonr(a.reshape(-1), b.reshape(-1))