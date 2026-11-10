CXX ?= clang++
BOOST_INC ?= /opt/homebrew/include
CXXFLAGS ?= -O3 -std=c++17 -I../third_party/metal-cpp -I. -I$(BOOST_INC)
LDFLAGS = -framework Metal -framework Foundation -framework CoreGraphics
MSL_PATH = $(CURDIR)/kernels.metal

all: driver

driver: driver.cpp lattice.hpp sweeps_cpu.hpp kernels.metal
	$(CXX) $(CXXFLAGS) -DSOLID_MSL_PATH='"$(MSL_PATH)"' driver.cpp -o $@ $(LDFLAGS)

test: driver
	python3 check_correctness.py

test-long: driver
	python3 check_correctness.py --long

clean:
	rm -f driver run.npz

.PHONY: all test test-long clean
