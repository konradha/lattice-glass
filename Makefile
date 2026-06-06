CXX ?= clang++-17
CXXFLAGS ?= -I. -ftree-vectorize -pedantic -ffast-math -march=native -O3 -Wall -Wunknown-pragmas -fopenmp -std=c++17
LDFLAGS ?=
LDLIBS ?= -lm -lstdc++

ifneq ($(CONDA_PREFIX),)
LDFLAGS += -Wl,-rpath,$(CONDA_PREFIX)/lib
endif

TARGET = to_omp
SRC = sim_omp.cpp
EXCHANGE_DELTA_TEST = tests/test_exchange_delta
SPECIES_REDUCTION_TEST = tests/test_species_reduction
BALANCED_CLUSTER_TEST = tests/test_balanced_cluster

.PHONY: all check clean

all: $(TARGET)

$(TARGET): $(SRC) maps_omp.h npy.hpp tsc.h
	$(CXX) $(CXXFLAGS) $(SRC) $(LDFLAGS) $(LDLIBS) -o $(TARGET)

measure_rng: measure_rng.cpp tsc.h
	$(CXX) $(CXXFLAGS) measure_rng.cpp $(LDFLAGS) $(LDLIBS) -o measure_rng

$(EXCHANGE_DELTA_TEST): tests/test_exchange_delta.cpp $(SRC) maps_omp.h npy.hpp
	$(CXX) $(CXXFLAGS) tests/test_exchange_delta.cpp $(LDFLAGS) $(LDLIBS) -o $(EXCHANGE_DELTA_TEST)

$(SPECIES_REDUCTION_TEST): tests/test_species_reduction.cpp species_reduction.h
	$(CXX) $(CXXFLAGS) tests/test_species_reduction.cpp $(LDFLAGS) $(LDLIBS) -o $(SPECIES_REDUCTION_TEST)

$(BALANCED_CLUSTER_TEST): tests/test_balanced_cluster.cpp balanced_cluster.h
	$(CXX) $(CXXFLAGS) tests/test_balanced_cluster.cpp $(LDFLAGS) $(LDLIBS) -o $(BALANCED_CLUSTER_TEST)

check: $(EXCHANGE_DELTA_TEST) $(SPECIES_REDUCTION_TEST) $(BALANCED_CLUSTER_TEST)
	./$(EXCHANGE_DELTA_TEST)
	./$(SPECIES_REDUCTION_TEST)
	./$(BALANCED_CLUSTER_TEST)
	python -m unittest discover -s tests

clean:
	rm -f $(TARGET) measure_rng $(EXCHANGE_DELTA_TEST) $(SPECIES_REDUCTION_TEST) $(BALANCED_CLUSTER_TEST)
