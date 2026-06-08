CXX ?= clang++-17
CPPFLAGS ?=
CXXFLAGS ?= -I. -ftree-vectorize -pedantic -ffast-math -march=native -O3 -Wall -Wunknown-pragmas -fopenmp -std=c++17
# Tests rely on assert(); keep them enabled even when the environment injects -DNDEBUG.
TESTFLAGS = -UNDEBUG
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
BALANCED_CLUSTER_PILOT = experiments/pilot_balanced_cluster
CLUSTER_DIAGNOSTICS = experiments/cluster_diagnostics
REFERENCE_CANCELLATION_TEST = tests/test_reference_cancellation
FP_EFFICIENCY = experiments/fp_efficiency
COLLECTIVE_MOVES_TEST = tests/test_collective_moves
OCCUPANCY_EFFICIENCY = experiments/occupancy_efficiency
LIFTED_VACANCY_TEST = tests/test_lifted_vacancy
LIFTED_SWAP_TEST = tests/test_lifted_swap

.PHONY: all check clean pilot diag fpbench

all: $(TARGET)

$(TARGET): $(SRC) maps_omp.h npy.hpp tsc.h compare_mode.h balanced_cluster.h species_reduction.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SRC) $(LDFLAGS) $(LDLIBS) -o $(TARGET)

measure_rng: measure_rng.cpp tsc.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) measure_rng.cpp $(LDFLAGS) $(LDLIBS) -o measure_rng

$(EXCHANGE_DELTA_TEST): tests/test_exchange_delta.cpp $(SRC) maps_omp.h npy.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TESTFLAGS) tests/test_exchange_delta.cpp $(LDFLAGS) $(LDLIBS) -o $(EXCHANGE_DELTA_TEST)

$(SPECIES_REDUCTION_TEST): tests/test_species_reduction.cpp species_reduction.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TESTFLAGS) tests/test_species_reduction.cpp $(LDFLAGS) $(LDLIBS) -o $(SPECIES_REDUCTION_TEST)

$(BALANCED_CLUSTER_TEST): tests/test_balanced_cluster.cpp balanced_cluster.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TESTFLAGS) tests/test_balanced_cluster.cpp $(LDFLAGS) $(LDLIBS) -o $(BALANCED_CLUSTER_TEST)

$(BALANCED_CLUSTER_PILOT): experiments/pilot_balanced_cluster.cpp balanced_cluster.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TESTFLAGS) experiments/pilot_balanced_cluster.cpp $(LDFLAGS) $(LDLIBS) -o $(BALANCED_CLUSTER_PILOT)

$(CLUSTER_DIAGNOSTICS): experiments/cluster_diagnostics.cpp balanced_cluster.h species_reduction.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TESTFLAGS) experiments/cluster_diagnostics.cpp $(LDFLAGS) $(LDLIBS) -o $(CLUSTER_DIAGNOSTICS)

$(REFERENCE_CANCELLATION_TEST): tests/test_reference_cancellation.cpp fp_sampler.h balanced_cluster.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TESTFLAGS) tests/test_reference_cancellation.cpp $(LDFLAGS) $(LDLIBS) -o $(REFERENCE_CANCELLATION_TEST)

$(FP_EFFICIENCY): experiments/fp_efficiency.cpp fp_sampler.h balanced_cluster.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TESTFLAGS) experiments/fp_efficiency.cpp $(LDFLAGS) $(LDLIBS) -o $(FP_EFFICIENCY)

$(COLLECTIVE_MOVES_TEST): tests/test_collective_moves.cpp collective_moves.h fp_sampler.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TESTFLAGS) tests/test_collective_moves.cpp $(LDFLAGS) $(LDLIBS) -o $(COLLECTIVE_MOVES_TEST)

$(OCCUPANCY_EFFICIENCY): experiments/occupancy_efficiency.cpp fp_sampler.h collective_moves.h lifted_vacancy.h species_reduction.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TESTFLAGS) experiments/occupancy_efficiency.cpp $(LDFLAGS) $(LDLIBS) -o $(OCCUPANCY_EFFICIENCY)

$(LIFTED_VACANCY_TEST): tests/test_lifted_vacancy.cpp lifted_vacancy.h fp_sampler.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TESTFLAGS) tests/test_lifted_vacancy.cpp $(LDFLAGS) $(LDLIBS) -o $(LIFTED_VACANCY_TEST)

$(LIFTED_SWAP_TEST): tests/test_lifted_swap.cpp lifted_swap.h fp_sampler.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(TESTFLAGS) tests/test_lifted_swap.cpp $(LDFLAGS) $(LDLIBS) -o $(LIFTED_SWAP_TEST)

diag: $(CLUSTER_DIAGNOSTICS)
	./$(CLUSTER_DIAGNOSTICS)

fpbench: $(FP_EFFICIENCY)
	./$(FP_EFFICIENCY)

pilot: $(BALANCED_CLUSTER_PILOT)
	./$(BALANCED_CLUSTER_PILOT)

check: $(EXCHANGE_DELTA_TEST) $(SPECIES_REDUCTION_TEST) $(BALANCED_CLUSTER_TEST) $(REFERENCE_CANCELLATION_TEST) $(COLLECTIVE_MOVES_TEST) $(LIFTED_VACANCY_TEST) $(LIFTED_SWAP_TEST)
	./$(EXCHANGE_DELTA_TEST)
	./$(SPECIES_REDUCTION_TEST)
	./$(BALANCED_CLUSTER_TEST)
	./$(REFERENCE_CANCELLATION_TEST)
	./$(COLLECTIVE_MOVES_TEST)
	./$(LIFTED_VACANCY_TEST)
	./$(LIFTED_SWAP_TEST)
	python -m unittest discover -s tests

clean:
	rm -f $(TARGET) measure_rng $(EXCHANGE_DELTA_TEST) $(SPECIES_REDUCTION_TEST) $(BALANCED_CLUSTER_TEST) $(BALANCED_CLUSTER_PILOT) $(CLUSTER_DIAGNOSTICS) $(REFERENCE_CANCELLATION_TEST) $(FP_EFFICIENCY) $(COLLECTIVE_MOVES_TEST) $(OCCUPANCY_EFFICIENCY) $(LIFTED_VACANCY_TEST) $(LIFTED_SWAP_TEST)
