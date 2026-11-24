"""European call pricing: QAE-based pricing versus classical Monte Carlo.

QAE pipeline (see ``qae.py`` for the amplitude-estimation step)
---------------------------------------------------------------
1. Discretise the terminal price S_T on 2**n_qubits equal-probability quantile
   nodes of the lognormal law (midpoint quantiles, so each node has weight 1/N).
2. Normalise the payoff max(S_i - K, 0) to f_i = payoff_i / c in [0, 1], where
   c = max payoff. Then a = sum_i p_i f_i is an expectation, i.e. an amplitude.
3. Run amplitude estimation on a, giving a_hat with error O(1/M).
4. Price = exp(-rT) * c * a_hat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .blackscholes import call_price
from .qae import qae_distribution


@dataclass(frozen=True)
class QuantumPricingResult:
    price: float                 # QAE-based price estimate
    bs_price: float              # Black-Scholes closed form
    discretized_price: float     # exact expectation on the n-qubit grid (no QAE error)
    amplitude_exact: float       # a = E[f] on the grid
    amplitude_estimate: float    # a_hat returned by QAE
    oracle_calls: int            # applications of the Grover iterate
    n_qubits: int
    m_qubits: int

    @property
    def abs_error(self) -> float:
        """|QAE price - Black-Scholes|, including grid discretisation error."""
        return abs(self.price - self.bs_price)


def lognormal_quantile_grid(
    S0: float, r: float, sigma: float, T: float, n_qubits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (prices, probabilities) for a 2**n_qubits quantile grid of S_T."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1")
    N = 2**n_qubits
    z = norm.ppf((np.arange(N) + 0.5) / N)
    prices = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)
    probs = np.full(N, 1.0 / N)
    return prices, probs


def price_european_call_qae(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_qubits: int = 10,
    m_qubits: int = 8,
) -> QuantumPricingResult:
    """Price a European call with simulated quantum amplitude estimation."""
    if S0 <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("S0, K, sigma and T must all be strictly positive")

    prices, probs = lognormal_quantile_grid(S0, r, sigma, T, n_qubits)
    payoff = np.maximum(prices - K, 0.0)
    scale = float(payoff.max())
    f = payoff / scale if scale > 0 else payoff
    a_exact = float(np.dot(probs, f))

    qae = qae_distribution(a_exact, m_qubits)
    most_likely = int(np.argmax(qae.probabilities))
    a_hat = float(qae.estimates[most_likely])

    discount = math.exp(-r * T)
    return QuantumPricingResult(
        price=discount * scale * a_hat,
        bs_price=call_price(S0, K, r, sigma, T),
        discretized_price=discount * scale * a_exact,
        amplitude_exact=a_exact,
        amplitude_estimate=a_hat,
        oracle_calls=qae.oracle_calls,
        n_qubits=n_qubits,
        m_qubits=m_qubits,
    )


def price_european_call_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_samples: int,
    rng: np.random.Generator | int | None = None,
) -> tuple[float, float]:
    """Classical Monte Carlo price and its standard error, using ``n_samples`` payoffs."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    z = gen.standard_normal(n_samples)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)
    discount = math.exp(-r * T)
    payoff = np.maximum(ST - K, 0.0)
    price = discount * payoff.mean()
    stderr = discount * payoff.std(ddof=1) / math.sqrt(n_samples)
    return float(price), float(stderr)
