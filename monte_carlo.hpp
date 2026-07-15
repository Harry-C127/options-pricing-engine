#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "black_scholes.hpp"

// CRR binomial tree pricer. Mirrors python/binomial_tree.py.
// American early exercise is checked at every node when american=true.
inline double binomial_price(
    const OptionParams& p,
    bool is_call,
    int n_steps = 200,
    bool american = false
) {
    double dt = p.T / n_steps;
    double u = std::exp(p.sigma * std::sqrt(dt));
    double d = 1.0 / u;
    double disc = std::exp(-p.r * dt);
    double prob_up = (std::exp((p.r - p.q) * dt) - d) / (u - d);

    if (prob_up < 0.0 || prob_up > 1.0) {
        throw std::invalid_argument(
            "Risk-neutral probability outside [0,1] - check inputs");
    }

    std::vector<double> values(n_steps + 1);

    // Terminal payoffs
    for (int j = 0; j <= n_steps; ++j) {
        double s_t = p.S * std::pow(u, n_steps - j) * std::pow(d, j);
        values[j] = is_call ? std::max(s_t - p.K, 0.0)
                             : std::max(p.K - s_t, 0.0);
    }

    // Backward induction
    for (int step = n_steps - 1; step >= 0; --step) {
        for (int j = 0; j <= step; ++j) {
            double continuation = disc * (prob_up * values[j] + (1 - prob_up) * values[j + 1]);
            if (american) {
                double s_t = p.S * std::pow(u, step - j) * std::pow(d, j);
                double exercise = is_call ? std::max(s_t - p.K, 0.0)
                                           : std::max(p.K - s_t, 0.0);
                values[j] = std::max(continuation, exercise);
            } else {
                values[j] = continuation;
            }
        }
    }

    return values[0];
}
