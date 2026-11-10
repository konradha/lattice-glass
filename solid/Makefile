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

HAVE_CUDA ?= 0
NVCC ?= nvcc
CUDA_FLAGS ?= -O3 -std=c++17 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90
ifeq ($(HAVE_CUDA),1)
  CUDA_OBJECT = cuda_backend.o
  CUDA_TARGET = cuda_check
  LINK = $(NVCC)
else
  LINK = $(CXX)
endif

CXXFLAGS ?= -O3 -std=c++17 -I. -I$(BOOST_INC) $(METAL_FLAGS)
LDFLAGS = $(METAL_LIBS)
ifeq ($(HAVE_CUDA),1)
  CXXFLAGS += -DSOLID_HAVE_CUDA=1
endif

all: driver $(CUDA_TARGET)

driver: driver.cpp lattice.hpp sweeps_cpu.hpp efma.hpp kernels.metal $(CUDA_OBJECT)
	$(LINK) $(CXXFLAGS) driver.cpp $(CUDA_OBJECT) -o $@ $(LDFLAGS)

cuda_backend.o: cuda_backend.cu cuda_backend.hpp
	$(NVCC) $(CUDA_FLAGS) -I. -I$(BOOST_INC) -c cuda_backend.cu -o $@

cuda_check: cuda_check.cpp cuda_backend.hpp cuda_backend.o lattice.hpp sweeps_cpu.hpp efma.hpp
	$(NVCC) $(CXXFLAGS) -DSOLID_HAVE_CUDA=1 cuda_check.cpp cuda_backend.o -o $@

cuda_check_prepare: cuda_check.cpp cuda_backend.hpp lattice.hpp sweeps_cpu.hpp efma.hpp
	$(CXX) $(CXXFLAGS) -USOLID_HAVE_CUDA -DSOLID_HAVE_CUDA=0 cuda_check.cpp -o $@

test: driver test-production
	python3 check_correctness.py

test-production: driver
	python3 check_production.py

test-long: driver
	python3 check_correctness.py --long

clean:
	rm -f driver cuda_backend.o cuda_check cuda_check_prepare run.npz

.PHONY: all test test-production test-long clean
