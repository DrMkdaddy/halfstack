"""Closed-form ridge maps for RidgeFill, in NumPy.

The map for a late layer predicts that layer's input from the split state:

    x_hat_l = A_l @ x_S + b_l

`fit` solves the regularised least-squares problem with the same parameterisation
as arXiv:2611.09709:

    A = Yc^T Xc (Xc^T Xc + alpha I)^-1
    b = mean(Y) - A mean(X)
    alpha = lam * trace(Xc^T Xc) / d_in
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RidgeMap:
    A: np.ndarray  # (d_out, d_in)
    b: np.ndarray  # (d_out,)
    lam: float
    alpha: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.A.T + self.b


def fit(X: np.ndarray, Y: np.ndarray, lam: float) -> RidgeMap:
    """Fit one layer's map. X is (n, d_in), Y is (n, d_out)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError(f"bad shapes: X={X.shape} Y={Y.shape}")

    mean_x = X.mean(axis=0)
    mean_y = Y.mean(axis=0)
    Xc = X - mean_x
    Yc = Y - mean_y

    xtx = Xc.T @ Xc
    d_in = X.shape[1]
    alpha = lam * float(np.trace(xtx)) / d_in
    A = (Yc.T @ Xc) @ np.linalg.inv(xtx + alpha * np.eye(d_in, dtype=np.float64))
    b = mean_y - A @ mean_x

    return RidgeMap(A=A, b=b, lam=lam, alpha=alpha)


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity for each row, shape (n,)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.einsum("ij,ij->i", a, b)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.maximum(den, 1e-12)
