
import pytest

from qfinance import call_price, price_european_call_mc, price_european_call_qae
from qfinance.qae import qae_error_bound

PARAMS = dict(S0=100.0, K=105.0, r=0.03, sigma=0.25, T=1.0)


def _bs():
    return call_price(S=PARAMS["S0"], K=PARAMS["K"], r=PARAMS["r"], sigma=PARAMS["sigma"], T=PARAMS["T"])


def test_grid_discretisation_converges_to_black_scholes():
    bs = _bs()
    coarse = price_european_call_qae(**PARAMS, n_qubits=6, m_qubits=3).discretized_price
    fine = price_european_call_qae(**PARAMS, n_qubits=12, m_qubits=3).discretized_price
    assert abs(fine - bs) < abs(coarse - bs)
    assert abs(fine - bs) < 0.05


def test_qae_price_within_theoretical_error_of_discretised_expectation():
    res = price_european_call_qae(**PARAMS, n_qubits=10, m_qubits=8)
    # Price = discount * scale * a_hat, so the QAE price error is discount * scale * |a_hat - a|.
    discount_scale = res.discretized_price / res.amplitude_exact
    bound = discount_scale * qae_error_bound(res.amplitude_exact, res.m_qubits)
    assert abs(res.price - res.discretized_price) <= bound


def test_out_of_the_money_payoff_is_zero_when_deep_otm():
    res = price_european_call_qae(S0=10.0, K=1000.0, r=0.01, sigma=0.1, T=0.5, n_qubits=6, m_qubits=4)
    assert res.price == 0.0
    assert res.oracle_calls == 2**4 - 1


def test_monte_carlo_agrees_with_black_scholes():
    bs = _bs()
    price, stderr = price_european_call_mc(**PARAMS, n_samples=200_000, rng=1)
    assert abs(price - bs) < 4 * stderr


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        price_european_call_qae(S0=-1, K=100, r=0.0, sigma=0.2, T=1.0)
