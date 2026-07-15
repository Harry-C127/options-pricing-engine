"""
Option Greeks: analytical (closed-form) and finite-difference implementations.

Analytical Greeks come directly from differentiating the Black-Scholes
formula. Finite-difference Greeks are computed by bumping an input and
re-pricing, which is model-agnostic and works for any pricer (Black-Scholes,
binomial tree, Monte Carlo) — useful for validating the analytical formulas
and for pricers that don't have closed-form Greeks.
"""

from dataclasses import dataclass, replace
from math import exp, sqrt, pi
from scipy.stats import norm

from black_scholes import OptionParams, _d1_d2, price


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _phi(x: float) -> float:
    """Standard normal PDF."""
    return exp(-0.5 * x**2) / sqrt(2 * pi)


def analytical_greeks(p: OptionParams, option_type: str = "call") -> Greeks:
    """Closed-form Greeks from the Black-Scholes formula.

    Vega and Gamma are the same for calls and puts. Delta, Theta and Rho
    differ by option type.
    """
    d1, d2 = _d1_d2(p)
    disc_q = exp(-p.q * p.T)
    disc_r = exp(-p.r * p.T)

    gamma = disc_q * _phi(d1) / (p.S * p.sigma * sqrt(p.T))
    vega = p.S * disc_q * _phi(d1) * sqrt(p.T)  # per unit change in sigma (not %)

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            -p.S * disc_q * _phi(d1) * p.sigma / (2 * sqrt(p.T))
            - p.r * p.K * disc_r * norm.cdf(d2)
            + p.q * p.S * disc_q * norm.cdf(d1)
        )
        rho = p.K * p.T * disc_r * norm.cdf(d2)
    elif option_type == "put":
        delta = disc_q * (norm.cdf(d1) - 1)
        theta = (
            -p.S * disc_q * _phi(d1) * p.sigma / (2 * sqrt(p.T))
            + p.r * p.K * disc_r * norm.cdf(-d2)
            - p.q * p.S * disc_q * norm.cdf(-d1)
        )
        rho = -p.K * p.T * disc_r * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def finite_difference_greeks(
    p: OptionParams, option_type: str = "call", bump: float = 1e-4
) -> Greeks:
    """Model-agnostic Greeks via bump-and-reprice (central differences).

    Works with any pricing function that takes an OptionParams instance,
    so this same approach generalises to the binomial tree and Monte Carlo
    pricers later — swap out `price` for those pricers to sanity-check them.
    """
    base_up_S = replace(p, S=p.S * (1 + bump))
    base_dn_S = replace(p, S=p.S * (1 - bump))
    delta = (price(base_up_S, option_type) - price(base_dn_S, option_type)) / (
        2 * p.S * bump
    )

    gamma = (
        price(base_up_S, option_type)
        - 2 * price(p, option_type)
        + price(base_dn_S, option_type)
    ) / (p.S * bump) ** 2

    vol_bump = max(p.sigma * bump, 1e-6)
    up_sigma = replace(p, sigma=p.sigma + vol_bump)
    dn_sigma = replace(p, sigma=p.sigma - vol_bump)
    vega = (price(up_sigma, option_type) - price(dn_sigma, option_type)) / (
        2 * vol_bump
    )

    # Theta: price loses time, so bump T downward (closer to expiry)
    time_bump = min(p.T * bump, p.T * 0.5)
    dn_T = replace(p, T=p.T - time_bump)
    theta = -(price(p, option_type) - price(dn_T, option_type)) / time_bump

    rate_bump = max(abs(p.r) * bump, 1e-6)
    up_r = replace(p, r=p.r + rate_bump)
    dn_r = replace(p, r=p.r - rate_bump)
    rho = (price(up_r, option_type) - price(dn_r, option_type)) / (2 * rate_bump)

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


if __name__ == "__main__":
    params = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)

    analytical = analytical_greeks(params, "call")
    numerical = finite_difference_greeks(params, "call")

    print("Greek       Analytical    Finite-diff   Abs diff")
    for name in ["delta", "gamma", "vega", "theta", "rho"]:
        a = getattr(analytical, name)
        n = getattr(numerical, name)
        print(f"{name:<10}  {a:>10.6f}   {n:>10.6f}   {abs(a - n):.2e}")
