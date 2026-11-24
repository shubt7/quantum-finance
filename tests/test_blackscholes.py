import math

import pytest

from qfinance import call_price, put_price


def test_known_atm_call_value():
    # Textbook reference: S=K=100, r=5%, sigma=20%, T=1 -> call ~ 10.4506
    assert call_price(100, 100, 0.05, 0.20, 1.0) == pytest.approx(10.4506, abs=1e-4)


def test_put_call_parity():
    S, K, r, sigma, T = 92.0, 100.0, 0.03, 0.35, 0.75
    lhs = call_price(S, K, r, sigma, T) - put_price(S, K, r, sigma, T)
    rhs = S - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        call_price(100, 100, 0.05, 0.0, 1.0)
