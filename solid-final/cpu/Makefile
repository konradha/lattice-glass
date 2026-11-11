CXX ?= c++
OS := $(shell uname -s)

# Header-only Boost.Multiprecision is required; no Boost libraries are linked.
# macOS: brew install boost; Debian/Ubuntu: apt install libboost-dev.
# Autodetect common prefixes, or use BOOST_INC=/path/to/include (or CPPFLAGS).
BOOST_INC ?= $(firstword $(patsubst %/boost/multiprecision/cpp_int.hpp,%,$(wildcard /opt/homebrew/include/boost/multiprecision/cpp_int.hpp /usr/local/include/boost/multiprecision/cpp_int.hpp /usr/include/boost/multiprecision/cpp_int.hpp)))
CPPFLAGS += -I../common $(if $(BOOST_INC),-I$(BOOST_INC))

ifeq ($(OS),Darwin)
# Preserve the measured opt2 Apple production algorithm and O2/native flags.
CPU_CACHED := 0
CXXFLAGS ?= -O2 -march=native -mtune=native -funroll-loops -fomit-frame-pointer -fno-math-errno -fno-trapping-math -fno-signed-zeros -DNDEBUG -std=c++17
else
# Validated opt3 Fast<true> K/type-fast CPU algorithm, not its benchmark main.
CPU_CACHED := 1
CXXFLAGS ?= -O3 -march=native -mtune=native -std=c++17
endif

INPUTS = driver.cpp fast.hpp ../common/lattice.hpp ../common/sweeps_cpu.hpp ../common/efma.hpp Makefile
SHA256 := $(shell command -v sha256sum 2>/dev/null || command -v shasum 2>/dev/null)
ifeq ($(notdir $(SHA256)),shasum)
SHA256 += -a 256
endif
SOURCE_HASH := $(shell cat $(INPUTS) | $(SHA256) | cut -d' ' -f1)
DEFINES = -DSOLID_CPU_CACHED=$(CPU_CACHED) -DSOLID_SOURCE_HASH=\"$(SOURCE_HASH)\"

.PHONY: all clean
all: driver

# Build only. No PGO, training, tests, simulations, or GPU invocation.
driver: $(INPUTS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(DEFINES) -pthread driver.cpp $(LDFLAGS) $(LDLIBS) -o $@

clean:
	rm -f driver
