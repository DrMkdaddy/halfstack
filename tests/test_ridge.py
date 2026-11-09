import numpy as np
import pytest

from halfstack.ridge import fit, row_cosine


def make_problem(seed=0, n=2000, d_in=32, d_out=16, noise=1e-3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d_in))
    A = rng.normal(size=(d_out, d_in)) / np.sqrt(d_in)
    b = rng.normal(size=d_out)
    Y = X @ A.T + b + rng.normal(scale=noise, size=(n, d_out))
    return X, Y


def test_recovers_a_linear_map():
    X, Y = make_problem()
    m = fit(X[:1500], Y[:1500], lam=1e-8)
    pred = m.predict(X[1500:])
    assert row_cosine(pred, Y[1500:]).mean() > 0.9999


def test_alpha_uses_the_trace():
    X, Y = make_problem()
    m = fit(X, Y, lam=0.03)
    d_in = X.shape[1]
    expected = 0.03 * np.trace((X - X.mean(0)).T @ (X - X.mean(0))) / d_in
    assert m.alpha == pytest.approx(expected)


def test_prediction_matches_explicit_form():
    X, Y = make_problem()
    m = fit(X, Y, lam=0.01)
    Xc = X - X.mean(0)
    explicit = Xc @ m.A.T + Y.mean(0)
    assert np.allclose(m.predict(X), explicit, atol=1e-9)


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        fit(np.zeros((4, 3)), np.zeros((5, 3)), 0.1)


def test_cosine_of_identical_rows_is_one():
    x = np.random.default_rng(1).normal(size=(10, 8))
    assert np.allclose(row_cosine(x, x), 1.0)
