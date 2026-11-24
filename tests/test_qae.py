import math

import numpy as np
import pytest

from qfinance import qae_distribution, qae_error_bound
from qfinance.qae import _qpe_amplitude


def test_closed_form_matches_brute_force_qpe_sum():
    M = 32
    y = np.arange(M)
    for phase in [0.0, 0.07, 0.31, -0.19, 0.5]:
        brute = np.exp(2j * np.pi * np.outer(phase - y / M, np.arange(M))).mean(axis=1)
        assert np.allclose(_qpe_amplitude(phase, M, y), brute, atol=1e-12)


@pytest.mark.parametrize("a", [0.0, 0.05, 0.3, 0.5, 0.87, 1.0])
@pytest.mark.parametrize("m", [1, 3, 6])
def test_distribution_is_normalised(a, m):
    res = qae_distribution(a, m)
    assert res.probabilities.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(res.probabilities >= -1e-12)


def test_zero_and_one_are_exact():
    assert qae_distribution(0.0, 5).probabilities[0] == pytest.approx(1.0)
    res = qae_distribution(1.0, 5)
    assert res.estimates[res.probabilities.argmax()] == pytest.approx(1.0)


@pytest.mark.parametrize("a", [0.1, 0.25, 0.6, 0.93])
@pytest.mark.parametrize("m", [3, 5, 7])
def test_brassard_guarantee_holds(a, m):
    # Theorem: the outcome mass within the error bound is at least 8/pi^2.
    res = qae_distribution(a, m)
    within = np.abs(res.estimates - a) <= qae_error_bound(a, m) + 1e-12
    assert res.probabilities[within].sum() >= 8 / math.pi**2 - 1e-9


def test_error_shrinks_with_more_qubits():
    a = 0.37

    def mle_error(m):
        res = qae_distribution(a, m)
        return abs(res.estimates[res.probabilities.argmax()] - a)

    assert mle_error(8) < mle_error(4) < mle_error(2) + 1e-12


def test_oracle_calls_scale_linearly_in_M():
    assert qae_distribution(0.5, 6).oracle_calls == 2**6 - 1


def test_rejects_out_of_range_amplitude():
    with pytest.raises(ValueError):
        qae_distribution(1.2, 4)
