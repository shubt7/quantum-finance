"""Closed-form Black-Scholes-Merton prices, used as the classical reference."""

from __future__ import annotations

import math

from scipy.stats import norm


def _d1_d2(S: float, K: float, r: float, sigma: float, T: float) -> tuple[float, float]:
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("S, K, sigma and T must all be strictly positive")
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return d1, d1 - sigma * sqrt_T


def call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Price of a European call under geometric Brownian motion."""
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def put_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Price of a European put under geometric Brownian motion."""
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
