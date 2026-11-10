CXX ?= c++
BOOST_INC ?= /opt/homebrew/include
CXXFLAGS ?= -O2 -march=native -mtune=native -funroll-loops -fomit-frame-pointer -fno-math-errno -fno-trapping-math -fno-signed-zeros -DNDEBUG -std=c++17 -I. -I$(BOOST_INC) -DSOLID_HAVE_METAL=0
all: driver
driver: driver.cpp lattice.hpp efma.hpp sweeps_cpu.hpp
	$(CXX) $(CXXFLAGS) driver.cpp -o $@
clean:
	rm -f driver run.npz
.PHONY: all clean
