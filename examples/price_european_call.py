"""Compare simulated QAE pricing with Monte Carlo at equal payoff-evaluation budgets.

Run:  python examples/price_european_call.py
"""

from qfinance import call_price, price_european_call_mc, price_european_call_qae

S0, K, r, sigma, T = 100.0, 105.0, 0.03, 0.25, 1.0
N_QUBITS = 10  # 1024-node price grid

bs = call_price(S0, K, r, sigma, T)
grid = price_european_call_qae(S0, K, r, sigma, T, n_qubits=N_QUBITS, m_qubits=1)
print(f"Black-Scholes reference price      : {bs:.4f}")
print(f"Exact expectation on {2**N_QUBITS}-node grid : {grid.discretized_price:.4f}")
print()
print(f"{'m':>2} {'oracle calls':>12} {'QAE price':>10} {'QAE |err|':>10} {'MC price':>9} {'MC s.e.':>8} {'MC |err|':>9}")

for m in range(3, 15):
    q = price_european_call_qae(S0, K, r, sigma, T, n_qubits=N_QUBITS, m_qubits=m)
    mc_price, mc_se = price_european_call_mc(S0, K, r, sigma, T, n_samples=q.oracle_calls, rng=m)
    print(
        f"{m:>2} {q.oracle_calls:>12} {q.price:>10.4f} {abs(q.price - bs):>10.4f} "
        f"{mc_price:>9.4f} {mc_se:>8.4f} {abs(mc_price - bs):>9.4f}"
    )
