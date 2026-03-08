import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.array(x, dtype=float)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))