"""Cardinality-constrained mean-variance asset selection as a QUBO.

Problem
-------
Choose exactly K of n assets (x_i in {0, 1}) to minimise

    f(x) = q * x^T Sigma x - mu^T x      subject to   sum_i x_i = K,

with expected returns mu, covariance Sigma and risk aversion q.

The constraint is enforced with a penalty P * (sum_i x_i - K)^2, which gives an
unconstrained binary quadratic model (QUBO)

    E(x) = x^T Q x + P*K^2,   x in {0, 1}^n,

the input format used by quantum annealers and QAOA. Here it is solved classically,
either exhaustively (small n) or by simulated annealing, as a reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def build_qubo(
    mu: np.ndarray,
    cov: np.ndarray,
    k: int,
    risk_aversion: float = 1.0,
    penalty: float | None = None,
) -> tuple[np.ndarray, float]:
    """Return (Q, penalty) with Q symmetric so that E(x) = x^T Q x (+ P*K^2).

    The default penalty, sum|mu| + q*sum|Sigma| + 1, is a heuristic. It is large
    enough that the test suite finds feasible optima, but ``evaluate_portfolio``
    should be used to confirm feasibility of any returned solution.
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = mu.shape[0]
    if cov.shape != (n, n) or not np.allclose(cov, cov.T):
        raise ValueError("cov must be a symmetric (n, n) matrix matching mu")
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= n")
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be non-negative")

    if penalty is None:
        penalty = float(np.abs(mu).sum() + risk_aversion * np.abs(cov).sum() + 1.0)

    Q = risk_aversion * cov.copy()
    # Binary variables satisfy x_i^2 = x_i, so linear terms go on the diagonal.
    Q[np.diag_indices(n)] += -mu + penalty * (1 - 2 * k)
    # Pairwise terms from P*(sum x)^2: each off-diagonal entry appears twice in x^T Q x.
    Q += penalty * (np.ones((n, n)) - np.eye(n))
    return Q, float(penalty)


def energy(Q: np.ndarray, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x)


def solve_exact(Q: np.ndarray, max_n: int = 18) -> tuple[np.ndarray, float]:
    """Exhaustive minimisation over all 2**n bitstrings. Practical for n <= ~18."""
    n = Q.shape[0]
    if n > max_n:
        raise ValueError(f"exhaustive search limited to n <= {max_n}; use solve_annealing")
    idx = np.arange(2**n)
    X = ((idx[:, None] >> np.arange(n)) & 1).astype(float)
    E = np.einsum("ij,ij->i", X @ Q, X)
    best = int(np.argmin(E))
    return X[best].astype(int), float(E[best])


def solve_annealing(
    Q: np.ndarray,
    n_sweeps: int = 400,
    t_start: float | None = None,
    t_end: float = 1e-3,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, float]:
    """Single-bit-flip simulated annealing with a geometric cooling schedule."""
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    n = Q.shape[0]
    x = gen.integers(0, 2, size=n).astype(float)
    field = Q @ x                 # field[i] = (Q x)_i, kept up to date incrementally
    E = float(x @ field)
    best_x, best_E = x.copy(), E

    if t_start is None:
        t_start = max(float(np.abs(Q).max()), 1e-9)
    temps = np.geomspace(t_start, t_end, n_sweeps)

    for T in temps:
        for i in gen.permutation(n):
            d = 1.0 - 2.0 * x[i]                  # flipping bit i changes x_i by d
            dE = 2.0 * d * field[i] + Q[i, i]     # exact since d**2 == 1
            if dE <= 0 or gen.random() < math.exp(-dE / T):
                x[i] += d
                field += d * Q[:, i]
                E += dE
                if E < best_E - 1e-12:
                    best_E, best_x = E, x.copy()
    return best_x.astype(int), float(best_E)


@dataclass(frozen=True)
class PortfolioMetrics:
    selected: tuple[int, ...]
    expected_return: float
    variance: float
    feasible: bool
    objective: float   # q * variance - expected_return


def evaluate_portfolio(
    x: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    k: int,
    risk_aversion: float = 1.0,
) -> PortfolioMetrics:
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    ret = float(mu @ x)
    var = float(x @ cov @ x)
    return PortfolioMetrics(
        selected=tuple(int(i) for i in np.flatnonzero(x > 0.5)),
        expected_return=ret,
        variance=var,
        feasible=int(round(x.sum())) == k,
        objective=risk_aversion * var - ret,
    )
