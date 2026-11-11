CXX ?= g++
NVCC ?= nvcc
BOOST_INC ?= /usr/include
BUILD_DIR ?= build
CXXFLAGS ?= -O3 -std=c++17 -pthread
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90
EXTRA_NVCCFLAGS ?=
CPPFLAGS += -I../common -I. -I$(BOOST_INC)
HEADERS = backend.hpp ../common/lattice.hpp ../common/sweeps_cpu.hpp ../common/efma.hpp
NATIVE_INPUTS = driver.cpp check.cpp fast_empty_window.cu packed_type_commit.cu packed_fast_empty.cu $(HEADERS) Makefile
# Bind native inputs and compiler settings, not Python/docs. No fast-math.
SOURCE_HASH := $(shell { cat $(NATIVE_INPUTS); printf '%s\n' '$(CXX) $(CXXFLAGS) $(CPPFLAGS)' '$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS)'; } | sha256sum | cut -d' ' -f1)
BUILD_DEFINES = -DSOLID_SOURCE_HASH=\"$(SOURCE_HASH)\" '-DSOLID_BUILD_FLAGS="$(CXXFLAGS) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS)"'
INPUT_STAMP = $(BUILD_DIR)/inputs-$(SOURCE_HASH)

.PHONY: all checkers clean
.SECONDARY:
all: driver-depth driver-throughput driver-ampere
checkers: check-depth check-throughput check-ampere

$(BUILD_DIR):
	mkdir -p $@
$(INPUT_STAMP): | $(BUILD_DIR)
	touch $@

$(BUILD_DIR)/driver.o: driver.cpp $(HEADERS) Makefile $(INPUT_STAMP) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(BUILD_DEFINES) -DSOLID_HAVE_CUDA=1 -c $< -o $@
$(BUILD_DIR)/check.o: check.cpp $(HEADERS) Makefile $(INPUT_STAMP) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(BUILD_DEFINES) -DSOLID_HAVE_CUDA=1 -c $< -o $@
$(BUILD_DIR)/depth.o: fast_empty_window.cu $(HEADERS) Makefile $(INPUT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(BUILD_DEFINES) -c $< -o $@
$(BUILD_DIR)/throughput.o: packed_type_commit.cu $(HEADERS) Makefile $(INPUT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(BUILD_DEFINES) -c $< -o $@
$(BUILD_DIR)/ampere.o: packed_fast_empty.cu $(HEADERS) Makefile $(INPUT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(BUILD_DEFINES) -c $< -o $@

driver-%: $(BUILD_DIR)/driver.o $(BUILD_DIR)/%.o
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $^ -Xcompiler -pthread -o $@
check-%: $(BUILD_DIR)/check.o $(BUILD_DIR)/%.o
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $^ -Xcompiler -pthread -o $@
# Fixture preparation remains available without linking a GPU backend.
check-cpu: check.cpp $(HEADERS) Makefile $(INPUT_STAMP)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(BUILD_DEFINES) -DSOLID_HAVE_CUDA=0 $< -o $@

clean:
	rm -rf $(BUILD_DIR) driver-depth driver-throughput driver-ampere check-depth check-throughput check-ampere check-cpu
