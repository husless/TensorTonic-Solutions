import numpy as np


def power_iteration(A, max_iter=5000, tol=1e-8):
    """Finds the dominant eigenvalue and eigenvector of matrix A."""
    n = A.shape[0]
    # Initialize a random vector and normalize it
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    for _ in range(max_iter):
        v_next = A.T @ v
        v_next_norm = np.linalg.norm(v_next)
        
        if v_next_norm < tol:
            # Handle zero matrix or subspace collapse
            return 0.0, v
            
        v_next = v_next / v_next_norm
        
        # Check convergence (accounts for sign flipping)
        if np.allclose(v, v_next, atol=tol) or np.allclose(v, -v_next, atol=tol):
            v = v_next
            break
        v = v_next

    # Rayleigh Quotient
    eigenvalue = np.dot(v, (A.T @ v))
    return eigenvalue, v


def get_top_k_eigenpairs(A, k, max_iter=5000, tol=1e-8):
    """Computes top-k eigenvalues and eigenvectors using Hotelling Deflation."""
    A_working = A.copy().astype(float)
    eigenvalues = []
    eigenvectors = []
    
    for i in range(k):
        # 1. Compute dominant pair of the current deflated matrix
        lam, v = power_iteration(A_working, max_iter, tol)
        
        eigenvalues.append(lam)
        eigenvectors.append(v)
        
        # 2. Deflate the matrix for the next iteration step
        # A_{i+1} = A_i - lambda_i * (v_i * v_i^T)
        A_working -= lam * np.outer(v, v)
        
    return np.array(eigenvalues), np.array(eigenvectors).T


def pca_projection(X, k):
    """
    Project data onto the top-k principal components.

    X: (n_samples, d_features)
    k: int
    """
    X = np.asarray(X)
    n, d = X.shape

    Xc = X - np.mean(X, axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / (n-1)
    print(Xc)
    print(cov)

    # top-k eigenvectors
    # (d, k) matrix
    # eigv, W = get_top_k_eigenpairs(cov, k)
    eigv, W = np.linalg.eig(cov)
    print('eig:', eigv)
    w = W[:, np.argsort(eigv)[::-1]]
    X_proj = Xc @ w[:,:k]
    return X_proj