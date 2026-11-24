"""Select 4 of 12 assets by mean-variance QUBO: exhaustive vs simulated annealing.

Run:  python examples/select_portfolio.py
"""

import numpy as np

from qfinance import build_qubo, evaluate_portfolio, solve_annealing, solve_exact

rng = np.random.default_rng(7)
n, k, risk = 12, 4, 1.0

mu = rng.uniform(0.02, 0.12, n)
factors = rng.normal(0, 0.1, (n, 2))
cov = factors @ factors.T + np.diag(rng.uniform(0.005, 0.02, n))

Q, penalty = build_qubo(mu, cov, k, risk_aversion=risk)
print(f"{n} assets, choose {k}, risk aversion {risk}, penalty {penalty:.3f}\n")

x_exact, _ = solve_exact(Q)
x_sa, _ = solve_annealing(Q, n_sweeps=600, rng=1)

for label, x in [("exhaustive", x_exact), ("annealing", x_sa)]:
    m = evaluate_portfolio(x, mu, cov, k, risk)
    print(
        f"{label:>10}: assets {list(m.selected)}  return {m.expected_return:.4f}  "
        f"variance {m.variance:.5f}  objective {m.objective:.5f}  feasible={m.feasible}"
    )
