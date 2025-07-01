import numpy as np
from scipy.io import savemat, loadmat
from picard._core_picard import core_picard

def amari_index(W_true, W_est):
    P = W_est @ np.linalg.inv(W_true)
    C = np.abs(P)
    row_sum = np.sum(C, axis=1, keepdims=True)
    col_sum = np.sum(C, axis=0, keepdims=True)
    r = np.sum((C / row_sum - 1/P.shape[1])**2)
    c = np.sum((C / col_sum - 1/P.shape[0])**2)
    return (r + c) / (2 * P.shape[0])

# fixed seed for reproducibility
np.random.seed(0)

# generate sources
N, T = 5, 10000
S = np.random.laplace(size=(N, T))

# random mixing
A = np.random.randn(N, N)
X = A @ S

# save data for MATLAB
savemat('picard_data.mat', {'X': X, 'A': A})

# run Python Picard
# from threadpoolctl import threadpool_limits
#with threadpool_limits(limits=1, user_api="blas"):
Y_py, W_py, info_py = core_picard(X.copy(), ortho=False, extended=False,
                                  max_iter=200, tol=1e-6, m=10,
                                  lambda_min=0.01, ls_tries=10, verbose=True)
print('Python converged in', info_py['n_iterations'], 'iterations')
