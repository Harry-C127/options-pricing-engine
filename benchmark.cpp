#include <iostream>
#include "black_scholes.hpp"
#include "monte_carlo.hpp"

int main() {
    OptionParams p{100, 100, 1.0, 0.05, 0.2};

    double bs_ref = bs_call_price(p);

    auto naive = monte_carlo_price(p, true, 50000, false);
    auto anti = monte_carlo_price(p, true, 50000, true);

    std::cout << "European call (n_paths=50,000):\n";
    std::cout << "  Naive MC:     price=" << naive.price << "  stderr=" << naive.stderr_ << "\n";
    std::cout << "  Antithetic:   price=" << anti.price << "  stderr=" << anti.stderr_ << "\n";
    std::cout << "  Black-Scholes ref: " << bs_ref << "\n";

    return 0;
}
