CXX ?= clang++
CXXFLAGS ?= -O3 -mcpu=native -std=c++17 -Wall -Wextra -DNDEBUG

all: lattice_pt

lattice_pt: main.cpp lattice.hpp
	$(CXX) $(CXXFLAGS) main.cpp -o $@

test: tests/test_all.cpp lattice.hpp
	$(CXX) $(CXXFLAGS) tests/test_all.cpp -o tests/test_all
	./tests/test_all

bench: bench_moves.cpp
	$(CXX) $(CXXFLAGS) bench_moves.cpp -o bench_moves
	./bench_moves 500 2.5

clean:
	rm -f lattice_pt tests/test_all bench_moves

.PHONY: all test bench clean
