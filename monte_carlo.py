"""
Cox-Ross-Rubinstein (CRR) binomial tree option pricer.

Unlike Black-Scholes, the binomial tree naturally handles American-style
early exercise, since it re-evaluates at every node whether exercising now
beats holding. As the number of steps N grows, European binomial prices
converge to the Black-Scholes closed form (with a well-known oscillation
around the true value for even vs odd N).
"""

from math import exp, sqrt

from black_scholes import OptionParams


def binomial_price(
    p: OptionParams,
    option_type: str = "call",
    n_steps: int = 200,
    american: bool = False,
) -> float:
    """Price an option using a CRR binomial tree.

    n_steps: number of time steps in the tree. Higher = more accurate,
    slower. A few hundred steps is typically enough for convergence to
    within a few cents of the Black-Scholes price for European options.
    american: if True, allows early exercise at every node.
    """
    dt = p.T / n_steps
    u = exp(p.sigma * sqrt(dt))
    d = 1 / u
    disc = exp(-p.r * dt)
    # Risk-neutral up-probability, adjusted for continuous dividend yield q
    prob_up = (exp((p.r - p.q) * dt) - d) / (u - d)

    if not (0 <= prob_up <= 1):
        raise ValueError(
            "Risk-neutral probability outside [0,1] — check inputs "
            "(often caused by too few steps or extreme sigma/r/q values)"
        )

    # Terminal stock prices at maturity, for each of n_steps+1 final nodes
    terminal_prices = [p.S * u**(n_steps - j) * d**j for j in range(n_steps + 1)]

    if option_type == "call":
        values = [max(s - p.K, 0.0) for s in terminal_prices]
    elif option_type == "put":
        values = [max(p.K - s, 0.0) for s in terminal_prices]
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Step backward through the tree
    for step in range(n_steps - 1, -1, -1):
        new_values = []
        for j in range(step + 1):
            continuation = disc * (
                prob_up * values[j] + (1 - prob_up) * values[j + 1]
            )
            if american:
                stock_price = p.S * u**(step - j) * d**j
                if option_type == "call":
                    exercise = max(stock_price - p.K, 0.0)
                else:
                    exercise = max(p.K - stock_price, 0.0)
                new_values.append(max(continuation, exercise))
            else:
                new_values.append(continuation)
        values = new_values

    return values[0]


if __name__ == "__main__":
    from black_scholes import call_price, put_price

    params = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)

    bs_call = call_price(params)
    bs_put = put_price(params)

    euro_call = binomial_price(params, "call", n_steps=500, american=False)
    euro_put = binomial_price(params, "put", n_steps=500, american=False)
    amer_call = binomial_price(params, "call", n_steps=500, american=True)
    amer_put = binomial_price(params, "put", n_steps=500, american=True)

    print("Convergence check vs Black-Scholes (European, N=500 steps):")
    print(f"  Call: BS={bs_call:.4f}  Binomial={euro_call:.4f}  diff={abs(bs_call-euro_call):.5f}")
    print(f"  Put:  BS={bs_put:.4f}  Binomial={euro_put:.4f}  diff={abs(bs_put-euro_put):.5f}")

    print("\nAmerican vs European (early exercise premium):")
    print(f"  Call: American={amer_call:.4f}  European={euro_call:.4f}  premium={amer_call-euro_call:.5f}")
    print(f"  Put:  American={amer_put:.4f}  European={euro_put:.4f}  premium={amer_put-euro_put:.5f}")
