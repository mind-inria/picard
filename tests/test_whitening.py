from eegprep import pop_loadset
import numpy as np
from scipy import linalg

def whitening(X1, n_components):
    n, p = X1.shape
    u, d, _ = linalg.svd(X1, full_matrices=False)
    # build K = (u / d).T[:n_components] * sqrt(p)
    K = (u / d).T[:n_components]
    K *= np.sqrt(p)

    # enforce fixed‐sign: make the max‐abs entry in each row positive
    for i in range(K.shape[0]):
        j = np.argmax(np.abs(K[i]))
        if K[i, j] < 0:
            K[i] = -K[i]

    # whiten the data
    X1 = K.dot(X1)
    covariance = np.eye(n_components)  # for extended

    return X1, K, covariance

EEG = pop_loadset('tests/eeglab_data_with_ica_tmp.set');

X1, K, covariance = whitening(EEG['data'].astype(np.float64), 32)
print(K[0:6,0:6])
print(covariance[0:6,0:6])