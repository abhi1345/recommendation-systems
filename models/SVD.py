import numpy as np
from scipy.sparse.linalg import svds

#Singular Value Decomposition

def svd_flip(u, v, u_based_decision=True):
    if u_based_decision:
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v

def svd_fit(X, n):
    U, Sigma, VT = svds(X, k=n)
    Sigma = Sigma[::-1]
    U, VT = svd_flip(U[:, ::-1], VT[::-1])
    X_transformed = U * Sigma
    return X_transformed
