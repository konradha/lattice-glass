CXX ?= c++
BOOST_INC ?= /opt/homebrew/include
CXXFLAGS ?= -O2 -march=native -mtune=native -funroll-loops -fomit-frame-pointer -fno-math-errno -fno-trapping-math -fno-signed-zeros -DNDEBUG -std=c++17 -I../cpu -I../../third_party/metal-cpp -I$(BOOST_INC) -DSOLID_HAVE_METAL=1 -DSOLID_MSL_PATH='"$(CURDIR)/kernels.metal"'
LDFLAGS = -framework Metal -framework Foundation -framework CoreGraphics
all: driver
driver: driver.cpp ../cpu/lattice.hpp ../cpu/efma.hpp ../cpu/sweeps_cpu.hpp kernels.metal
	$(CXX) $(CXXFLAGS) driver.cpp -o $@ $(LDFLAGS)
clean:
	rm -f driver run.npz
.PHONY: all clean
