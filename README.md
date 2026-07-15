.PHONY: test benchmark-python benchmark-cpp clean install

install:
	pip install -r requirements.txt

test:
	cd tests && python3 -m pytest test_pricers.py -v

benchmark-python:
	cd python && python3 benchmarks.py

benchmark-cpp:
	cd cpp && g++ -O2 -std=c++17 benchmark.cpp -o benchmark && ./benchmark

build-cpp:
	cd cpp && \
	g++ -O2 -std=c++17 black_scholes.cpp -o black_scholes_test && \
	g++ -O2 -std=c++17 binomial_tree.cpp -o binomial_tree_test && \
	g++ -O2 -std=c++17 monte_carlo.cpp -o monte_carlo_test && \
	g++ -O2 -std=c++17 benchmark.cpp -o benchmark

clean:
	rm -f cpp/black_scholes_test cpp/binomial_tree_test cpp/monte_carlo_test cpp/benchmark
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
