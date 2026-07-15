"""
Monte Carlo option pricer.

Simulates terminal (and, for path-dependent options, intermediate) asset
prices under geometric Brownian motion, then averages discounted payoffs.

Includes two variance reduction techniques, since naive Monte Carlo
converges slowly (error ~ 1/sqrt(n_paths)):

- Antithetic variates: for every random path driven by Z, also simulate
  the path driven by -Z. This costs almost nothing extra and reduces
  variance because the two paths' errors partially cancel.
- Control variates: use the Black-Scholes price (a known, closely
  correlated quantity) to correct the Monte Carlo estimate.
"""

import numpy as np

from black_scholes import OptionParams, call_price, put_price


def _simulate_terminal_prices(
    p: OptionParams, n_paths: int, antithetic: bool, rng: np.random.Generator
) -> np.ndarray:
    """Simulate terminal stock prices S_T under risk-neutral GBM."""
    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)

    drift = (p.r - p.q - 0.5 * p.sigma**2) * p.T
    diffusion = p.sigma * np.sqrt(p.T) * z
    return p.S * np.exp(drift + diffusion)


def monte_carlo_price(
    p: OptionParams,
    option_type: str = "call",
    n_paths: int = 100_000,
    antithetic: bool = True,
    control_variate: bool = True,
    seed: int | None = 42,
) -> dict:
    """Price a European option via Monte Carlo.

    Returns a dict with the price estimate, standard error, and the raw
    (uncorrected) estimate so you can see the effect of control variates.
    """
    rng = np.random.default_rng(seed)
    s_t = _simulate_terminal_prices(p, n_paths, antithetic, rng)
    disc = np.exp(-p.r * p.T)

    if option_type == "call":
        payoffs = np.maximum(s_t - p.K, 0.0)
        bs_price = call_price(p)
    elif option_type == "put":
        payoffs = np.maximum(p.K - s_t, 0.0)
        bs_price = put_price(p)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = disc * payoffs
    raw_estimate = discounted.mean()
    raw_stderr = discounted.std(ddof=1) / np.sqrt(len(discounted))

    if control_variate:
        # Control variate: the underlying itself has a known expectation
        # under the risk-neutral measure, E[S_T] = S0 * e^((r-q)T).
        control = disc * s_t
        control_mean = p.S * np.exp(-p.q * p.T)  # true discounted E[S_T]

        cov = np.cov(discounted, control)[0, 1]
        var_control = np.var(control, ddof=1)
        beta = cov / var_control if var_control > 0 else 0.0

        adjusted = discounted - beta * (control - control_mean)
        price_estimate = adjusted.mean()
        stderr = adjusted.std(ddof=1) / np.sqrt(len(adjusted))
    else:
        price_estimate = raw_estimate
        stderr = raw_stderr

    return {
        "price": price_estimate,
        "stderr": stderr,
        "raw_price": raw_estimate,
        "raw_stderr": raw_stderr,
        "bs_reference": bs_price,
    }


def monte_carlo_asian_price(
    p: OptionParams,
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int = 100,
    antithetic: bool = True,
    seed: int | None = 42,
) -> dict:
    """Price an arithmetic-average Asian option via Monte Carlo.

    Asian options have no closed-form solution under standard Black-Scholes
    assumptions (the arithmetic average of lognormals isn't lognormal),
    which is exactly why Monte Carlo earns its keep here rather than just
    being a slower way to reproduce Black-Scholes.
    """
    rng = np.random.default_rng(seed)
    dt = p.T / n_steps

    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal((half, n_steps))
        z = np.concatenate([z, -z], axis=0)
    else:
        z = rng.standard_normal((n_paths, n_steps))

    drift = (p.r - p.q - 0.5 * p.sigma**2) * dt
    diffusion = p.sigma * np.sqrt(dt) * z
    log_returns = drift + diffusion
    log_paths = np.log(p.S) + np.cumsum(log_returns, axis=1)
    paths = np.exp(log_paths)

    average_price = paths.mean(axis=1)
    disc = np.exp(-p.r * p.T)

    if option_type == "call":
        payoffs = np.maximum(average_price - p.K, 0.0)
    elif option_type == "put":
        payoffs = np.maximum(p.K - average_price, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = disc * payoffs
    return {
        "price": discounted.mean(),
        "stderr": discounted.std(ddof=1) / np.sqrt(len(discounted)),
    }


if __name__ == "__main__":
    params = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)

    print("European call — effect of variance reduction (n_paths=50,000):")
    naive = monte_carlo_price(params, "call", n_paths=50_000, antithetic=False, control_variate=False)
    anti = monte_carlo_price(params, "call", n_paths=50_000, antithetic=True, control_variate=False)
    full = monte_carlo_price(params, "call", n_paths=50_000, antithetic=True, control_variate=True)

    print(f"  Naive MC:              price={naive['price']:.4f}  stderr={naive['stderr']:.5f}")
    print(f"  + Antithetic:          price={anti['price']:.4f}  stderr={anti['stderr']:.5f}")
    print(f"  + Antithetic+Control:  price={full['price']:.4f}  stderr={full['stderr']:.5f}")
    print(f"  Black-Scholes ref:     price={full['bs_reference']:.4f}")

    print("\nAsian call option (no closed-form comparison available):")
    asian = monte_carlo_asian_price(params, "call", n_paths=50_000, n_steps=100)
    print(f"  Price={asian['price']:.4f}  stderr={asian['stderr']:.5f}")
    print(f"  (Note: Asian call is cheaper than vanilla call — averaging reduces variance of the payoff)")
