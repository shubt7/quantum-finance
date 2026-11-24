import itertools

import numpy as np
import pytest

from qfinance import build_qubo, evaluate_portfolio, solve_annealing, solve_exact


def _instance(n, seed):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.02, 0.12, n)
    F = rng.normal(0, 0.1, (n, 2))
    cov = F @ F.T + np.diag(rng.uniform(0.005, 0.02, n))
    return mu, cov


def test_qubo_energy_matches_direct_objective_plus_penalty():
    mu, cov = _instance(6, 0)
    k, q = 3, 0.7
    Q, P = build_qubo(mu, cov, k, risk_aversion=q)
    for bits in itertools.product([0, 1], repeat=6):
        x = np.array(bits, dtype=float)
        direct = q * x @ cov @ x - mu @ x + P * (x.sum() - k) ** 2
        # build_qubo drops the constant P*K^2, so add it back before comparing.
        assert x @ Q @ x + P * k**2 == pytest.approx(direct, abs=1e-9)


def test_exact_solver_returns_best_feasible_subset():
    mu, cov = _instance(8, 3)
    k, q = 3, 1.0
    Q, _ = build_qubo(mu, cov, k, risk_aversion=q)
    x, _ = solve_exact(Q)
    assert x.sum() == k

    best = min(
        evaluate_portfolio(np.isin(np.arange(8), c).astype(float), mu, cov, k, q).objective
        for c in itertools.combinations(range(8), k)
    )
    assert evaluate_portfolio(x, mu, cov, k, q).objective == pytest.approx(best, abs=1e-9)


def test_annealing_matches_exact_on_small_instance():
    mu, cov = _instance(10, 5)
    k = 4
    Q, _ = build_qubo(mu, cov, k)
    x_exact, E_exact = solve_exact(Q)
    x_sa, E_sa = solve_annealing(Q, n_sweeps=800, rng=11)
    assert E_sa == pytest.approx(E_exact, abs=1e-6)
    assert x_sa.sum() == k


def test_rejects_bad_cardinality():
    mu, cov = _instance(4, 0)
    with pytest.raises(ValueError):
        build_qubo(mu, cov, k=5)
