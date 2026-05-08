"""Test that X.copy(order='K') preserves memory layout for MATLAB parity.

When input X is F-contiguous (column-major, as loaded from MATLAB .mat files),
X.copy() silently converts to C-contiguous, changing float accumulation order
in np.mean(axis=-1). This produces ~1e-13 centering differences that cascade
through SVD -> whitening -> L-BFGS, causing measurable ICA divergence on
real 64-channel EEG data (AMARI up to 0.004).

The fix: X.copy(order='K') preserves the input's memory layout, so
F-contiguous data stays F-contiguous and produces the same centering as MATLAB.
"""

import numpy as np
from numpy.testing import assert_allclose
from scipy import linalg

from picard import picard
from picard._core_picard import core_picard
from picard.densities import Tanh


def generate_ica_data(n_sources=10, n_samples=5000, seed=42):
    """Generate synthetic ICA data with a known mixing matrix."""
    rng = np.random.RandomState(seed)
    S = rng.laplace(size=(n_sources, n_samples))
    A = rng.randn(n_sources, n_sources)
    X = A @ S
    return X, A, S


def manual_picard_f_contiguous(X):
    """Reference implementation: manual centering/whitening on F-contiguous data.

    This replicates what MATLAB does: column-major storage, column-major
    accumulation in mean(), SVD whitening with K-row sign flip.
    """
    X = np.asfortranarray(X, dtype='float64')
    N, T = X.shape

    # Center (F-contiguous accumulation, matching MATLAB)
    X = X - X.mean(axis=1)[:, np.newaxis]

    # Whiten via SVD on data
    u, d, _ = linalg.svd(X, full_matrices=False)
    K = (u / d).T[:N] * np.sqrt(T)

    # K-row sign flip (matching picard v0.8.2 solver.py lines 178-182)
    for i in range(K.shape[0]):
        j = np.argmax(np.abs(K[i]))
        if K[i, j] < 0:
            K[i] = -K[i]

    X_white = K @ X

    # Run core picard with identity init
    w_init = np.eye(N)
    X_white = w_init @ X_white
    Y, W_algo, _infos = core_picard(
        X_white, density=Tanh(), ortho=False, extended=False,
        m=10, max_iter=512, tol=1e-7, lambda_min=0.01, ls_tries=10,
        verbose=False,
    )
    W = W_algo @ w_init
    return W @ K


def test_picard_matches_manual_f_contiguous():
    """picard() on F-contiguous data must match manual F-contiguous computation.

    This is the key parity test: picard()'s internal centering must use the
    same float accumulation order as an explicit F-contiguous computation
    (which matches MATLAB's column-major behavior).
    """
    X, _, _ = generate_ica_data()
    X_f = np.asfortranarray(X)
    N = X_f.shape[0]

    # picard() standard call
    K_p, W_p, _ = picard(X_f, ortho=False, extended=False,
                         whiten=True, max_iter=512, tol=1e-7,
                         m=10, lambda_min=0.01, ls_tries=10,
                         w_init=np.eye(N), verbose=False)
    icaweights_picard = W_p @ K_p

    # Manual reference (F-contiguous throughout)
    icaweights_manual = manual_picard_f_contiguous(X_f)

    # Compare: permutation matrix should be close to identity (up to signs)
    P = icaweights_picard @ np.linalg.pinv(icaweights_manual)
    P_abs = np.abs(P)
    row_max = P_abs.max(axis=1, keepdims=True)
    col_max = P_abs.max(axis=0, keepdims=True)
    n = N
    amari = (np.sum(P_abs / row_max) - n + np.sum(P_abs / col_max) - n) / (2 * n * (n - 1))

    print(f"  AMARI (picard vs manual F-contiguous): {amari:.2e}")
    assert amari < 1e-5, (
        f"picard() diverged from manual F-contiguous reference: AMARI = {amari:.6f}. "
        f"X.copy() may be converting F-contiguous input to C-contiguous."
    )


def test_centering_layout_preserved():
    """Verify that centering on F-contiguous data matches element-wise."""
    X, _, _ = generate_ica_data()
    X_f = np.asfortranarray(X)

    # Simulate what picard() does internally
    X_copy_K = X_f.copy(order='K')
    X_copy_C = X_f.copy()  # default = C-contiguous

    mean_K = X_copy_K.mean(axis=-1)
    mean_C = X_copy_C.mean(axis=-1)
    mean_F_direct = X_f.mean(axis=-1)

    # order='K' preserves F layout, so mean should match direct F computation
    diff_K = np.max(np.abs(mean_K - mean_F_direct))
    diff_C = np.max(np.abs(mean_C - mean_F_direct))

    print(f"  max|mean(copy_K) - mean(F_direct)| = {diff_K:.2e}")
    print(f"  max|mean(copy_C) - mean(F_direct)| = {diff_C:.2e}")

    # copy(order='K') should match F-direct exactly (same memory layout)
    assert_allclose(mean_K, mean_F_direct, atol=0,
                    err_msg="copy(order='K') changed centering result")


def test_input_not_modified():
    """Verify that picard() does not alter the caller's input array."""
    X, _, _ = generate_ica_data()
    X_f = np.asfortranarray(X)
    X_original = X_f.copy(order='K')
    N = X_f.shape[0]

    _ = picard(X_f, ortho=False, whiten=True, max_iter=10,
               w_init=np.eye(N), verbose=False)

    assert np.array_equal(X_f, X_original), "picard() modified the input array"


if __name__ == "__main__":
    print("Test 1: picard() matches manual F-contiguous reference")
    test_picard_matches_manual_f_contiguous()
    print("PASSED\n")

    print("Test 2: Centering layout preserved by copy(order='K')")
    test_centering_layout_preserved()
    print("PASSED\n")

    print("Test 3: Input array not modified")
    test_input_not_modified()
    print("PASSED\n")

    print("All tests passed.")
