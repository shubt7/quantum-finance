"""Exact statevector-level simulation of Quantum Amplitude Estimation (QAE).

Setting
-------
Let A be a state-preparation unitary with

    A|0> = sqrt(1 - a) |bad> + sqrt(a) |good>,     a in [0, 1],

where ``a`` is the unknown quantity we want to estimate (for option pricing it is
the normalised expected payoff). With sin^2(theta) = a, the Grover iterate Q acts
as a rotation by 2*theta inside span{|bad>, |good>}.

QAE runs quantum phase estimation (QPE) on Q with m ancilla qubits, M = 2**m.
Q has eigenvectors

    v_+ = (|bad> - i|good>)/sqrt(2)   with eigenvalue exp(+2i*theta)  -> phase  +theta/pi
    v_- = (|bad> + i|good>)/sqrt(2)   with eigenvalue exp(-2i*theta)  -> phase  -theta/pi

and A|0> = alpha_+ v_+ + alpha_- v_- with |alpha_+| = |alpha_-| = 1/sqrt(2).
Because v_+ and v_- are orthogonal, the measurement distribution of the ancilla
register is an incoherent sum of the two QPE outcome distributions. That is what
this module computes exactly (no sampling noise, no approximation of the QPE step).

What is *not* simulated here: the cost and circuit of loading the probability
distribution and the payoff comparator. Those are assumed ideal, and ``a`` is
supplied classically. This module isolates the amplitude-estimation step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QAEResult:
    """Exact output distribution of QAE with ``m_qubits`` ancillas.

    Attributes
    ----------
    probabilities : array of shape (M,)
        P(y) for each measured integer y in {0, ..., M-1}. Sums to 1.
    estimates : array of shape (M,)
        The amplitude estimate a_hat(y) = sin^2(pi * y / M) that outcome y maps to.
    oracle_calls : int
        Number of applications of the Grover iterate Q, which is M - 1
        (controlled-Q^(2^k) for k = 0..m-1).
    """

    probabilities: np.ndarray
    estimates: np.ndarray
    oracle_calls: int


def qae_distribution(a: float, m_qubits: int) -> QAEResult:
    """Exact QAE outcome distribution for amplitude ``a`` with ``m_qubits`` ancillas."""
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"amplitude must lie in [0, 1], got {a}")
    if m_qubits < 1:
        raise ValueError("m_qubits must be at least 1")

    M = 2**m_qubits
    theta = math.asin(math.sqrt(a))
    y = np.arange(M)

    # |alpha_+|^2 = |alpha_-|^2 = 1/2 for any theta, so each branch carries weight 1/2.
    probs = 0.5 * (
        np.abs(_qpe_amplitude(theta / np.pi, M, y)) ** 2
        + np.abs(_qpe_amplitude(-theta / np.pi, M, y)) ** 2
    )
    estimates = np.sin(np.pi * y / M) ** 2
    return QAEResult(probabilities=probs, estimates=estimates, oracle_calls=M - 1)


def _qpe_amplitude(phase: float, M: int, y: np.ndarray) -> np.ndarray:
    """Inverse-QFT amplitude (1/M) * sum_k exp(2*pi*i*k*delta), delta = phase - y/M.

    Evaluated in closed form as a geometric series, so the cost is O(M) not O(M^2).
    Since M*delta = M*phase - y with y an integer, exp(2*pi*i*M*delta) depends only
    on the phase, which keeps the expression well-conditioned.
    """
    delta = phase - y / M
    num = 1.0 - np.exp(2j * np.pi * M * phase)
    den = 1.0 - np.exp(2j * np.pi * delta)
    safe = np.abs(den) > 1e-12
    out = np.ones(M, dtype=complex)  # delta integer: every term equals 1
    out[safe] = num / (M * den[safe])
    return out


def qae_error_bound(a: float, m_qubits: int) -> float:
    """Brassard-Hoyer-Mosca-Tapp bound: |a_hat - a| <= 2*pi*sqrt(a(1-a))/M + (pi/M)^2.

    The theorem guarantees this holds for an outcome carrying probability at least
    8/pi^2 (about 0.81).
    """
    M = 2**m_qubits
    return 2 * math.pi * math.sqrt(a * (1 - a)) / M + (math.pi / M) ** 2
