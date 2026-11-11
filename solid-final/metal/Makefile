CXX ?= c++
BOOST_INC ?= /opt/homebrew/include
CXXFLAGS ?= -O2 -march=native -mtune=native -funroll-loops -fomit-frame-pointer -fno-math-errno -fno-trapping-math -fno-signed-zeros -DNDEBUG -std=c++17
CPPFLAGS += -I../common -I../third_party/metal-cpp -I$(BOOST_INC) -DSOLID_HAVE_METAL=1
LDLIBS += -framework Metal -framework Foundation -framework CoreGraphics
COMMON_HEADERS := ../common/lattice.hpp ../common/efma.hpp ../common/sweeps_cpu.hpp
APPLE_HEADERS := $(wildcard ../third_party/metal-cpp/Metal/*.hpp ../third_party/metal-cpp/Foundation/*.hpp)

all: driver_persistent driver_physical

driver_persistent: driver_persistent.cpp kernels_persistent.metal $(COMMON_HEADERS) $(APPLE_HEADERS) | require-darwin
	$(CXX) $(CPPFLAGS) -DSOLID_MSL_PATH='"$(CURDIR)/kernels_persistent.metal"' $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

driver_physical: driver_physical.cpp kernels_physical.metal $(COMMON_HEADERS) $(APPLE_HEADERS) | require-darwin
	$(CXX) $(CPPFLAGS) -DSOLID_MSL_PATH='"$(CURDIR)/kernels_physical.metal"' $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

require-darwin:
	@test "$$(uname -s)" = Darwin || { echo 'Metal samplers require macOS and Apple Metal; no CPU fallback' >&2; exit 2; }

clean:
	rm -f driver_persistent driver_physical

.PHONY: all require-darwin clean
