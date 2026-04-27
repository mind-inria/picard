# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pytest

from picard import picard, permute, amari_distance
from picard.densities import Tanh, Exp, Cube, check_density


def test_dimension_reduction():
    N, T = 5, 1000
    n_components = 3
    rng = np.random.RandomState(42)
    S = rng.laplace(size=(N, T))
    A = rng.randn(N, N)
    X = np.dot(A, S)
    K, W, Y = picard(
        X, n_components=n_components, ortho=False, random_state=rng, max_iter=2
    )
    assert K.shape == (n_components, N)
    assert W.shape == (n_components, n_components)
    assert Y.shape, (n_components, T)
    with warnings.catch_warnings(record=True) as w:
        K, W, Y = picard(
            X, n_components=n_components, ortho=False, whiten=False, max_iter=1
        )
        assert len(w) == 2


@pytest.mark.parametrize("n_component", [5, 3])
@pytest.mark.parametrize("whiten", [False, True])
@pytest.mark.parametrize("ortho", [False, True])
@pytest.mark.parametrize("w_init", [None, "id"])
def test_dots(n_component, whiten, ortho, w_init):
    N, T = 5, 100
    rng = np.random.RandomState(42)
    S = rng.laplace(size=(N, T))
    A = rng.randn(N, N)
    X = np.dot(A, S)
    if w_init == "id":
        if whiten:
            w_init = np.eye(n_component)
        else:
            w_init = np.eye(N)
    with warnings.catch_warnings(record=True):
        K, W, Y, X_mean = picard(
            X,
            ortho=ortho,
            whiten=whiten,
            return_X_mean=True,
            w_init=w_init,
            n_components=n_component,
            random_state=rng,
            max_iter=2,
            verbose=False,
        )
    if not whiten:
        K = np.eye(N)
    if ortho and whiten:
        assert_allclose(Y.dot(Y.T) / T, np.eye(n_component), atol=1e-8)
    Y_prime = np.dot(W, K).dot(X - X_mean[:, None])
    assert_allclose(Y, Y_prime, atol=1e-7)


def test_pre_fastica():
    N, T = 3, 1000
    rng = np.random.RandomState(42)
    names = ["tanh", "cube"]
    for j, fun in enumerate([Tanh(params=dict(alpha=0.5)), "cube"]):
        if j == 0:
            S = rng.laplace(size=(N, T))
        else:
            S = rng.uniform(low=-1, high=1, size=(N, T))
        A = rng.randn(N, N)
        X = np.dot(A, S)
        K, W, Y = picard(
            X, fun=fun, ortho=False, random_state=0, fastica_it=10
        )
        if fun == "tanh":
            fun = Tanh()
        elif fun == "exp":
            fun = Exp()
        elif fun == "cube":
            fun = Cube()
        # Get the final gradient norm
        psiY = fun.score_and_der(Y)[0]
        G = np.inner(psiY, Y) / float(T) - np.eye(N)
        err_msg = "fun %s, gradient norm greater than tol" % names[j]
        assert_allclose(G, np.zeros((N, N)), atol=1e-7, err_msg=err_msg)
        assert Y.shape == X.shape
        assert W.shape == A.shape
        assert K.shape == A.shape
        WA = W.dot(K).dot(A)
        WA = permute(WA)  # Permute and scale
        err_msg = "fun %s, wrong unmixing matrix" % names[j]
        assert_allclose(WA, np.eye(N), rtol=0, atol=1e-1, err_msg=err_msg)


def test_picard():
    N, T = 3, 1000
    rng = np.random.RandomState(42)
    names = ["tanh", "cube"]
    for j, fun in enumerate([Tanh(params=dict(alpha=0.5)), "cube"]):
        if j == 0:
            S = rng.laplace(size=(N, T))
        else:
            S = rng.uniform(low=-1, high=1, size=(N, T))
        A = rng.randn(N, N)
        X = np.dot(A, S)
        K, W, Y = picard(X, fun=fun, ortho=False, random_state=0)
        if fun == "tanh":
            fun = Tanh()
        elif fun == "exp":
            fun = Exp()
        elif fun == "cube":
            fun = Cube()
        # Get the final gradient norm
        psiY = fun.score_and_der(Y)[0]
        G = np.inner(psiY, Y) / float(T) - np.eye(N)
        err_msg = "fun %s, gradient norm greater than tol" % names[j]
        assert_allclose(G, np.zeros((N, N)), atol=1e-7, err_msg=err_msg)
        assert Y.shape == X.shape
        assert W.shape == A.shape
        assert K.shape == A.shape
        WA = W.dot(K).dot(A)
        WA = permute(WA)  # Permute and scale
        err_msg = "fun %s, wrong unmixing matrix" % names[j]
        assert_allclose(WA, np.eye(N), rtol=0, atol=1e-1, err_msg=err_msg)


def test_extended():
    N, T = 4, 2000
    n = N // 2
    rng = np.random.RandomState(42)

    S = np.concatenate(
        (rng.laplace(size=(n, T)), rng.uniform(low=-1, high=1, size=(n, T))),
        axis=0,
    )
    print(S.shape)
    A = rng.randn(N, N)
    X = np.dot(A, S)
    K, W, Y = picard(X, ortho=False, random_state=0, extended=True)
    assert Y.shape == X.shape
    assert W.shape == A.shape
    assert K.shape == A.shape
    WA = W.dot(K).dot(A)
    WA = permute(WA)  # Permute and scale
    err_msg = "wrong unmixing matrix"
    assert_allclose(WA, np.eye(N), rtol=0, atol=1e-1, err_msg=err_msg)


def test_shift():
    N, T = 5, 1000
    rng = np.random.RandomState(42)
    S = rng.laplace(size=(N, T))
    A = rng.randn(N, N)
    offset = rng.randn(N)
    X = np.dot(A, S) + offset[:, None]
    _, W, Y, X_mean = picard(
        X, ortho=False, whiten=False, return_X_mean=True, random_state=rng
    )
    assert_allclose(offset, X_mean, rtol=0, atol=0.2)
    WA = W.dot(A)
    WA = permute(WA)
    assert_allclose(WA, np.eye(N), rtol=0, atol=0.2)
    _, W, Y, X_mean = picard(
        X,
        ortho=False,
        whiten=False,
        centering=False,
        return_X_mean=True,
        random_state=rng,
    )
    assert_allclose(X_mean, 0)


def test_picardo():
    N, T = 3, 2000
    rng = np.random.RandomState(4)
    S = rng.laplace(size=(N, T))
    A = rng.randn(N, N)
    X = np.dot(A, S)
    names = ["tanh", "exp", "cube"]
    for fastica_it in [None, 2]:
        for fun in names:
            print(fun)
            K, W, Y = picard(
                X,
                fun=fun,
                ortho=True,
                random_state=rng,
                fastica_it=fastica_it,
                verbose=True,
                extended=True,
            )
            if fun == "tanh":
                fun = Tanh()
            elif fun == "exp":
                fun = Exp(params={"alpha": 0.1})
            elif fun == "cube":
                fun = Cube()
            # Get the final gradient norm
            psiY = fun.score_and_der(Y)[0]
            G = np.inner(psiY, Y) / float(T) - np.eye(N)
            G = (G - G.T) / 2.0  # take skew-symmetric part
            err_msg = "fun %s, gradient norm greater than tol" % fun
            assert_allclose(G, np.zeros((N, N)), atol=1e-7, err_msg=err_msg)
            assert Y.shape == X.shape
            assert W.shape == A.shape
            assert K.shape == A.shape
            WA = W.dot(K).dot(A)
            WA = permute(WA)  # Permute and scale
            err_msg = "fun %s, wrong unmixing matrix" % fun
            assert_allclose(WA, np.eye(N), rtol=0, atol=0.1, err_msg=err_msg)


def test_bad_custom_density():
    class CustomDensity(object):
        def log_lik(self, Y):
            return Y**4 / 4

        def score_and_der(self, Y):
            return Y**3, 3 * Y**2 + 2.0

    fun = CustomDensity()
    X = np.random.randn(2, 10)
    try:
        picard(X, fun=fun, random_state=0)
    except AssertionError:
        pass
    else:
        raise (AssertionError, "Bad function undetected")


def test_fun():
    for fun in [Tanh(), Exp(), Cube()]:
        check_density(fun)


@pytest.mark.parametrize("mode", ["lap", "gauss"])
@pytest.mark.parametrize("ortho", [True, False])
def test_no_regression(mode, ortho):
    n_tests = 10
    baseline = {}
    baseline["lap", True] = 17.0
    baseline["lap", False] = 23.0
    baseline["gauss", True] = 58.0
    baseline["gauss", False] = 60.0
    N, T = 10, 1000
    n_iters = []
    for i in range(n_tests):
        rng = np.random.RandomState(i)
        if mode == "lap":
            S = rng.laplace(size=(N, T))
        else:
            S = rng.randn(N, T)
        A = rng.randn(N, N)
        X = np.dot(A, S)
        _, _, _, n_iter = picard(
            X, return_n_iter=True, ortho=ortho, random_state=rng
        )
        n_iters.append(n_iter)
    n_mean = np.mean(n_iters)
    nb_mean = baseline[mode, ortho]
    err_msg = "mode=%s, ortho=%s. %d iterations, expecting <%d."
    assert n_mean < nb_mean, err_msg % (mode, ortho, n_mean, nb_mean)


def test_line_search_overflow():
    """Test that _line_search handles expm overflow gracefully.

    Regression test for NumPy 2.x compatibility: when the L-BFGS direction
    has large entries, expm overflows internally. The line search should
    reject such steps instead of crashing.
    """
    from picard._core_picard import _line_search

    N, T = 5, 1000
    rng = np.random.RandomState(42)
    Y = rng.randn(N, T)
    W = np.eye(N)
    density = Tanh()
    signs = np.ones(N)

    # A direction with large diagonal causes expm to overflow internally
    direction = np.eye(N) * 800
    with np.errstate(over='raise', invalid='raise'):
        converged, _, _, _, _ = _line_search(
            Y, W, density, direction, signs, None, 10, False, True, False
        )
    assert not converged


def test_amari_distance():
    p = 3
    rng = np.random.RandomState(0)
    A = rng.randn(p, p)
    W = np.linalg.pinv(A)
    scale = rng.randn(p)
    perm = np.argsort(rng.randn(p))
    W = W[perm]
    W *= scale[:, None]
    assert amari_distance(W, A) < 1e-6
