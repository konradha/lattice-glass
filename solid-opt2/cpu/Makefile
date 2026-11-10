CXX ?= c++
BOOST_INC ?= /opt/homebrew/include
CXXFLAGS ?= -O2 -march=native -mtune=native -funroll-loops -fomit-frame-pointer -fno-math-errno -fno-trapping-math -fno-signed-zeros -DNDEBUG -std=c++17 -I. -I$(BOOST_INC) -DSOLID_HAVE_METAL=0
all: driver
driver: driver.cpp lattice.hpp efma.hpp sweeps_cpu.hpp
	$(CXX) $(CXXFLAGS) driver.cpp -o $@
# --- Optional profile-guided optimization ---------------------------------
#
# The default build above stays plain -O2. PGO is a separate, explicit build:
#
#   make profile-generate   instrumented binary: driver.profile-generate
#   make profile-use        train it, then build driver.profile-use
#   make pgo                alias for profile-use
#
# profile-use runs the instrumented binary once on PGO_TRAIN_ARGS (a short L21
# warm-up that exercises both sweep kinds) inside $(PGO_DIR)/train, finalizes
# the profile, and rebuilds with the profile data. Profiles are tied to the
# exact compiler, flags, and source: regenerate after any source change.
# The PGO binary is left as driver.profile-use so the default `driver` target
# keeps its plain -O2 code. Everything below is overridable, e.g.
#   make profile-use CXX=g++ PGO_TRAIN_ARGS="21 2.0 0.75 0.30 0.45 1000 4 CPU both 100"
#
# PGO_DIR          profile data directory        (default: $(CURDIR)/pgo)
# PGO_OBJ          shared object name for both PGO compiles; GCC keys the
#                  .gcda file on the object base name, so it must not vary
#                  between the instrumented and the profile-use build
# PGO_TRAIN_ARGS   driver arguments for training (default: short L21 run)
# PGO_GEN_FLAGS    instrumented compile flags
# PGO_USE_FLAGS    profile-consuming compile flags
# PGO_MERGE        command that finalizes the profile before profile-use
PGO_DIR ?= $(CURDIR)/pgo
PGO_OBJ ?= driver.pgo.o
PGO_TRAIN_ARGS ?= 21 2.0 0.75 0.30 0.45 200 1 CPU both 20
PGO_GEN_FLAGS ?= -fprofile-generate=$(PGO_DIR) -fprofile-update=atomic
PGO_USE_FLAGS ?= -fprofile-use=$(PGO_DIR)

PGO_CLANG := $(findstring clang,$(shell $(CXX) --version 2>/dev/null | head -n 1))
ifeq ($(PGO_CLANG),)
# GCC reads the .gcda files written under $(PGO_DIR) directly.
PGO_MERGE ?= true
else
# Clang writes raw .profraw files; -fprofile-use wants default.profdata.
PGO_MERGE ?= xcrun llvm-profdata merge -o $(PGO_DIR)/default.profdata $(PGO_DIR)/*.profraw
endif

profile-generate: driver.profile-generate
driver.profile-generate: driver.cpp lattice.hpp efma.hpp sweeps_cpu.hpp
	$(CXX) $(CXXFLAGS) $(PGO_GEN_FLAGS) -c driver.cpp -o $(PGO_OBJ)
	$(CXX) $(CXXFLAGS) $(PGO_GEN_FLAGS) $(PGO_OBJ) -o $@

profile-use: driver.profile-generate
	rm -rf $(PGO_DIR)
	mkdir -p $(PGO_DIR)/train
	cd $(PGO_DIR)/train && ../../driver.profile-generate $(PGO_TRAIN_ARGS)
	$(PGO_MERGE)
	$(CXX) $(CXXFLAGS) $(PGO_USE_FLAGS) -c driver.cpp -o $(PGO_OBJ)
	$(CXX) $(CXXFLAGS) $(PGO_USE_FLAGS) $(PGO_OBJ) -o driver.profile-use

pgo: profile-use

clean:
	rm -f driver run.npz driver.profile-generate driver.profile-use $(PGO_OBJ)
	rm -rf $(PGO_DIR)
.PHONY: all clean profile-generate profile-use pgo
