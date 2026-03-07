import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float).flatten()
    
    m, n = X.shape                        # m samples, n features
    w = np.zeros(n, dtype=float)
    b = 0.0
    
    for _ in range(steps):
        z = X @ w + b                     # or np.dot(X, w)
        p = _sigmoid(z)
        
        # Analytical gradients
        dw = (1/m) * (X.T @ (p - y))      # shape (n,)
        db = (1/m) * np.sum(p - y)        # scalar
        
        w = w - lr * dw
        b = b - lr * db
    
    return (w, b)