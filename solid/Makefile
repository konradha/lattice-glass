CXX ?= clang++
CXXFLAGS ?= -O3 -std=c++17 -I../third_party/metal-cpp -I.
LDFLAGS = -framework Metal -framework Foundation -framework CoreGraphics
MSL_PATH = $(CURDIR)/kernels.metal

all: driver

driver: driver.cpp lattice.hpp sweeps_cpu.hpp kernels.metal
	$(CXX) $(CXXFLAGS) -DSOLID_MSL_PATH='"$(MSL_PATH)"' driver.cpp -o $@ $(LDFLAGS)

test: driver
	./driver 4 2.0 0.75 0.3 0.45 8 2 CPU
	mv run.npz /tmp/solid_cpu.npz
	./driver 4 2.0 0.75 0.3 0.45 8 2 GPU
	mv run.npz /tmp/solid_gpu.npz
	python3 -c "import numpy as np; c=np.load('/tmp/solid_cpu.npz'); g=np.load('/tmp/solid_gpu.npz'); [ (_ for _ in ()).throw(AssertionError(k)) for k in ['initial','final','snapshots','energy','accept_local','accept_nonlocal'] if not np.array_equal(c[k],g[k]) ]; print('CPU/GPU byte-identical')"

clean:
	rm -f driver run.npz

.PHONY: all test clean
