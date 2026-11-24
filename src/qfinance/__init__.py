"""qfinance: quantum algorithms for finance, with classical reference implementations."""

from .blackscholes import call_price, put_price
from .option_pricing import (
    QuantumPricingResult,
    lognormal_quantile_grid,
    price_european_call_mc,
    price_european_call_qae,
)
from .portfolio import (
    PortfolioMetrics,
    build_qubo,
    evaluate_portfolio,
    solve_annealing,
    solve_exact,
)
from .qae import QAEResult, qae_distribution, qae_error_bound

__version__ = "0.1.0"

__all__ = [
    "call_price",
    "put_price",
    "QAEResult",
    "qae_distribution",
    "qae_error_bound",
    "QuantumPricingResult",
    "lognormal_quantile_grid",
    "price_european_call_qae",
    "price_european_call_mc",
    "PortfolioMetrics",
    "build_qubo",
    "evaluate_portfolio",
    "solve_annealing",
    "solve_exact",
]
