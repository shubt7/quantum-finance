# quantum-finance

Quantum algorithms for finance, written as a small, tested Python library with classical reference implementations.

It covers two problems where quantum methods are most often proposed:

1. **European option pricing with quantum amplitude estimation (QAE).** A statevector-level simulation of QAE, compared against Black-Scholes and Monte Carlo at equal payoff-evaluation budgets.
2. **Cardinality-constrained portfolio selection as a QUBO.** The mean-variance problem encoded as a binary quadratic model, the input format used by quantum annealers and QAOA, solved classically as a reference.

## Dashboard

[`docs/dashboard.html`](docs/dashboard.html) is a self-contained page with the convergence chart, portfolio results, and tables. Regenerate it with:

```bash
PYTHONPATH=src python scripts/build_dashboard.py
```

## Quick start

```bash
pip install -e ".[dev]"
pytest -q                               # 47 tests
python examples/price_european_call.py  # QAE vs Monte Carlo table
python examples/select_portfolio.py     # QUBO: exhaustive vs annealing
```

Requires Python 3.10+, NumPy and SciPy.

## Scope and honesty notes

- **QAE is simulated exactly, not run on hardware.** The module computes the exact outcome distribution of quantum phase estimation applied to the Grover iterate, with no sampling noise and no approximation of the QPE step.
- **State preparation and the payoff comparator are assumed ideal.** Their circuit cost is not modelled. The amplitude `a` is computed classically and then handed to QAE. The project isolates the amplitude-estimation step.
- **The quadratic speedup is in oracle calls, not wall-clock time.** QAE uses `M - 1` applications of the Grover iterate for `m` ancilla qubits (`M = 2^m`), so error scales as `~1/M`. Monte Carlo error scales as `~1/sqrt(N)`. Reaching error ε therefore takes roughly `1/ε` oracle calls for QAE and `1/ε²` samples for Monte Carlo. The simulator itself runs on a classical computer and realises no speedup.
- **The portfolio penalty is a heuristic.** `build_qubo` picks a default penalty that worked on the test instances. Use `evaluate_portfolio` to confirm feasibility of any returned solution.

## Results

Call option, S₀ = 100, K = 105, r = 3%, σ = 25%, T = 1. The QAE grid uses 2¹⁰ = 1024 nodes. Monte Carlo is given the same number of payoff evaluations as QAE's oracle calls.

```
Black-Scholes reference price      : 9.1218
Exact expectation on 1024-node grid : 9.1165

 m oracle calls  QAE price  QAE |err|  MC price  MC s.e.  MC |err|
 3            7    17.4441     8.3223    9.3217   8.4092    0.1999
 4           15     4.5336     4.5882    7.4949   3.8821    1.6269
 5           31    10.0373     0.9155    5.4341   2.1359    3.6877
 6           63    10.0373     0.9155   10.3923   1.9494    1.2705
 7          127     8.4734     0.6484    5.2612   1.0292    3.8606
 8          255     9.2402     0.1184    9.9528   1.0658    0.8310
 9          511     9.2402     0.1184    9.4172   0.7184    0.2954
10         1023     9.0456     0.0762    8.0359   0.4657    1.0859
11         2047     9.1427     0.0209    9.0908   0.3582    0.0310
12         4095     9.0941     0.0277    9.2137   0.2539    0.0919
13         8191     9.1184     0.0034    9.2212   0.1833    0.0994
14        16383     9.1184     0.0034    9.2698   0.1281    0.1480
```

Single-seed run. The QAE error is measured against Black-Scholes, so it includes about 0.005 of grid bias. The QAE error is not monotone in `m` because the estimate is the most likely outcome and the outcome map is symmetric. The Brassard et al. guarantee applies to outcomes with probability at least 8/π², not to every `m`.

Portfolio selection (12 assets, choose 4, risk aversion 1): the annealer recovers the exhaustive optimum, assets `[0, 1, 5, 8]`, objective −0.33242.

## Method

### Quantum amplitude estimation

Let `A|0⟩ = √(1−a)|bad⟩ + √a|good⟩`, with `sin²θ = a`. The Grover iterate `Q` rotates by `2θ` in span{|bad⟩, |good⟩}. Its eigenvectors `(|bad⟩ ∓ i|good⟩)/√2` have eigenvalues `e^{±2iθ}`. Running QPE on `Q` with `m` ancillas gives the outcome distribution

```
P(y) = ½ |g(θ/π − y/M)|² + ½ |g(−θ/π − y/M)|²,   g(δ) = (1/M) Σ_k e^{2πikδ},   M = 2^m
```

The estimate is `â = sin²(πy/M)`. The closed form of `g` is a geometric series, so the simulation costs `O(M)`. `qae.py` also exposes the Brassard et al. error bound.

### Option pricing

1. Discretise `S_T` on `N = 2^n` equal-probability midpoint quantiles of the lognormal law.
2. Normalise the payoff to `f_i = max(S_i − K, 0) / c ∈ [0, 1]`, so `a = Σ p_i f_i` is an amplitude.
3. Estimate `a` with QAE.
4. Price `= e^{−rT} · c · â`.

### Portfolio QUBO

Minimise `q·xᵀΣx − μᵀx + P·(Σx_i − K)²` over `x ∈ {0,1}ⁿ`. Expanding the square with `x_i² = x_i` gives a symmetric matrix `Q`, and the energy is `xᵀQx + P·K²`. Two solvers are included: exhaustive search (n ≤ 18) and single-bit-flip simulated annealing.

## Layout

```
src/qfinance/
  blackscholes.py    closed-form call/put (reference)
  qae.py             exact QAE outcome distribution and error bound
  option_pricing.py  QAE pricer, Monte Carlo baseline
  portfolio.py       QUBO construction, exhaustive and annealing solvers
tests/               47 tests: identities, theorem checks, solver agreement
examples/            the two scripts above
.github/workflows/   CI on Python 3.10–3.12
```

## References

- Brassard, Høyer, Mosca, Tapp, *Quantum Amplitude Amplification and Estimation* (2002).
- Stamatopoulos et al., *Option Pricing using Quantum Computers* (Quantum, 2020).
- Glover, Kochenberger, Du, *A Tutorial on Formulating and Using QUBO Models* (2018).

## License

MIT
