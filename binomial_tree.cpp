"""
Test suite validating all three pricing methods against each other and
against known reference values / no-arbitrage relationships.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from black_scholes import OptionParams, call_price, put_price, put_call_parity_check
from binomial_tree import binomial_price
from monte_carlo import monte_carlo_price
from greeks import analytical_greeks, finite_difference_greeks


ATM_PARAMS = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)


class TestBlackScholes:
    def test_atm_call_matches_known_value(self):
        # Standard textbook reference value for these parameters
        assert call_price(ATM_PARAMS) == pytest.approx(10.4506, abs=1e-3)

    def test_atm_put_matches_known_value(self):
        assert put_price(ATM_PARAMS) == pytest.approx(5.5735, abs=1e-3)

    def test_put_call_parity(self):
        assert put_call_parity_check(ATM_PARAMS)

    def test_deep_itm_call_approaches_intrinsic(self):
        p = OptionParams(S=200, K=100, T=0.01, r=0.05, sigma=0.2)
        # Deep ITM, near expiry: price should approach intrinsic value
        assert call_price(p) == pytest.approx(p.S - p.K, abs=1.0)

    def test_price_increases_with_volatility(self):
        low_vol = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.1)
        high_vol = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.4)
        assert call_price(high_vol) > call_price(low_vol)


class TestBinomialTree:
    def test_converges_to_black_scholes_call(self):
        bt = binomial_price(ATM_PARAMS, "call", n_steps=1000, american=False)
        bs = call_price(ATM_PARAMS)
        assert bt == pytest.approx(bs, abs=0.01)

    def test_converges_to_black_scholes_put(self):
        bt = binomial_price(ATM_PARAMS, "put", n_steps=1000, american=False)
        bs = put_price(ATM_PARAMS)
        assert bt == pytest.approx(bs, abs=0.01)

    def test_american_call_no_dividend_equals_european(self):
        # Well-known result: never optimal to early-exercise an American
        # call on a non-dividend-paying stock.
        euro = binomial_price(ATM_PARAMS, "call", n_steps=500, american=False)
        amer = binomial_price(ATM_PARAMS, "call", n_steps=500, american=True)
        assert amer == pytest.approx(euro, abs=1e-6)

    def test_american_put_has_early_exercise_premium(self):
        euro = binomial_price(ATM_PARAMS, "put", n_steps=500, american=False)
        amer = binomial_price(ATM_PARAMS, "put", n_steps=500, american=True)
        assert amer > euro

    def test_rejects_invalid_probability(self):
        # Extreme rate/dt combination should raise, not silently misprice
        bad_params = OptionParams(S=100, K=100, T=1.0, r=5.0, sigma=0.01)
        with pytest.raises(ValueError):
            binomial_price(bad_params, "call", n_steps=2)


class TestMonteCarlo:
    def test_converges_to_black_scholes_within_stderr(self):
        result = monte_carlo_price(ATM_PARAMS, "call", n_paths=200_000, seed=1)
        bs = call_price(ATM_PARAMS)
        # Within 4 standard errors is a reasonable statistical bound
        assert abs(result["price"] - bs) < 4 * result["stderr"]

    def test_control_variate_reduces_stderr(self):
        no_cv = monte_carlo_price(
            ATM_PARAMS, "call", n_paths=50_000, control_variate=False, seed=1
        )
        with_cv = monte_carlo_price(
            ATM_PARAMS, "call", n_paths=50_000, control_variate=True, seed=1
        )
        assert with_cv["stderr"] < no_cv["stderr"]

    def test_antithetic_reduces_or_matches_stderr(self):
        no_anti = monte_carlo_price(
            ATM_PARAMS, "call", n_paths=50_000, antithetic=False,
            control_variate=False, seed=1
        )
        anti = monte_carlo_price(
            ATM_PARAMS, "call", n_paths=50_000, antithetic=True,
            control_variate=False, seed=1
        )
        assert anti["stderr"] <= no_anti["stderr"] * 1.05  # allow small noise margin


class TestGreeks:
    def test_analytical_matches_finite_difference(self):
        analytical = analytical_greeks(ATM_PARAMS, "call")
        numerical = finite_difference_greeks(ATM_PARAMS, "call")

        assert analytical.delta == pytest.approx(numerical.delta, abs=1e-3)
        assert analytical.gamma == pytest.approx(numerical.gamma, abs=1e-3)
        assert analytical.vega == pytest.approx(numerical.vega, abs=1e-2)
        assert analytical.rho == pytest.approx(numerical.rho, abs=1e-2)

    def test_call_delta_between_zero_and_one(self):
        g = analytical_greeks(ATM_PARAMS, "call")
        assert 0 <= g.delta <= 1

    def test_put_delta_between_minus_one_and_zero(self):
        g = analytical_greeks(ATM_PARAMS, "put")
        assert -1 <= g.delta <= 0

    def test_gamma_is_positive(self):
        # Gamma is always non-negative for vanilla options (long or short
        # doesn't matter here — this is the gamma of the option itself)
        g_call = analytical_greeks(ATM_PARAMS, "call")
        g_put = analytical_greeks(ATM_PARAMS, "put")
        assert g_call.gamma > 0
        assert g_put.gamma > 0
        assert g_call.gamma == pytest.approx(g_put.gamma)  # gamma same for call/put


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
