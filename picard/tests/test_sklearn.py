# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)
import warnings
import pytest

import numpy as np

from sklearn.decomposition import FastICA
from sklearn.utils._testing import assert_array_almost_equal

from picard import Picard


def center_and_norm(x, axis=-1):
    """ Centers and norms x **in place**
    Parameters
    -----------
    x: ndarray
        Array with an axis of observations (statistical units) measured on
        random variables.
    axis: int, optional
        Axis along which the mean and variance are calculated.
    """
    x = np.rollaxis(x, axis)
    x -= x.mean(axis=0)
    x /= x.std(axis=0)


def test_picard_nowhiten():
    m = [[0, 1], [1, 0]]

    ica = Picard(n_components=1, whiten=False, random_state=0)
    with pytest.warns(UserWarning):
        ica.fit(m)
    assert hasattr(ica, 'mixing_')


def test_fit_transform():
    # Test Picard.fit_transform
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10))
    for whiten, n_components in [[True, 5], [False, None]]:
        n_components_ = (n_components if n_components is not None else
                         X.shape[1])

        ica = Picard(n_components=n_components, whiten=whiten, random_state=0)
        Xt = ica.fit_transform(X)
        assert ica.components_.shape == (n_components_, 10)
        assert Xt.shape == (100, n_components_)

        ica = Picard(n_components=n_components, whiten=whiten, random_state=0)
        ica.fit(X)
        assert ica.components_.shape == (n_components_, 10)
        Xt2 = ica.transform(X)

        assert_array_almost_equal(Xt, Xt2)


def test_inverse_transform():
    # Test Picard.inverse_transform
    n_features = 10
    n_samples = 100
    n1, n2 = 5, 10
    rng = np.random.RandomState(0)
    X = rng.random_sample((n_samples, n_features))
    expected = {(True, n1): (n_features, n1),
                (True, n2): (n_features, n2),
                (False, n1): (n_features, n2),
                (False, n2): (n_features, n2)}
    for whiten in [True, False]:
        for n_components in [n1, n2]:
            n_components_ = (n_components if n_components is not None else
                             X.shape[1])
            ica = Picard(n_components=n_components, random_state=rng,
                         whiten=whiten)
            with warnings.catch_warnings(record=True):
                # catch "n_components ignored" warning
                Xt = ica.fit_transform(X)
            expected_shape = expected[(whiten, n_components_)]
            assert ica.mixing_.shape == expected_shape
            X2 = ica.inverse_transform(Xt)
            assert X.shape == X2.shape

            # reversibility test in non-reduction case
            if n_components == X.shape[1]:
                assert_array_almost_equal(X, X2)


def test_picard_errors():
    with pytest.raises(ValueError, match='max_iter should be greater than 1'):
        Picard(max_iter=0)


def test_picard_fastica_fun_parameter_constraints():
    """Test that the 'fun' parameter constraints have been updated to include
    'tanh', 'exp', 'cube', and callable.
    This test ensures that the modifications made to the Picard class to
    update the parameter constraints for 'fun' do not inadvertently affect
    the FastICA class from scikit-learn.
    """
    picard = Picard()
    fast_ica = FastICA()
    assert picard._parameter_constraints != fast_ica._parameter_constraints
