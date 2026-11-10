CXX ?= clang++
BOOST_INC ?= /opt/homebrew/include
UNAME_S := $(shell uname -s)
HAVE_METAL ?= $(if $(filter Darwin,$(UNAME_S)),1,0)

ifeq ($(HAVE_METAL),1)
  ifeq ($(wildcard /System/Library/Frameworks/Metal.framework),)
    HAVE_METAL := 0
  endif
endif

ifeq ($(HAVE_METAL),1)
  METAL_FLAGS = -I../third_party/metal-cpp -DSOLID_HAVE_METAL=1 \
                -DSOLID_MSL_PATH='"$(CURDIR)/kernels.metal"'
  METAL_LIBS = -framework Metal -framework Foundation -framework CoreGraphics
else
  $(warning Metal backend unavailable; building CPU-only)
  METAL_FLAGS = -DSOLID_HAVE_METAL=0
  METAL_LIBS =
endif

CXXFLAGS ?= -O3 -std=c++17 -I. -I$(BOOST_INC) $(METAL_FLAGS)
LDFLAGS = $(METAL_LIBS)

all: driver

driver: driver.cpp lattice.hpp sweeps_cpu.hpp efma.hpp kernels.metal
	$(CXX) $(CXXFLAGS) driver.cpp -o $@ $(LDFLAGS)

test: driver test-production
	python3 check_correctness.py

test-production: driver
	python3 check_production.py

test-long: driver
	python3 check_correctness.py --long

clean:
	rm -f driver run.npz

.PHONY: all test test-production test-long clean
