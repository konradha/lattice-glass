#pragma once

#include "../common/efma.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace solid {
struct CudaTrace {
  uint64_t proposal;
  uint32_t a, b;
  int32_t delta;
  uint32_t flags; // valid=1, no-op=2, accepted=4
};
static_assert(sizeof(CudaTrace) == 24, "stable trace ABI");
struct CudaLaunch {
  int window = 256;
  int threads = 256;
  int proposals_per_thread = 1;
  bool profile = false;
};
enum class CudaStage { Rng, Address, LoadDelta, Conflict, Commit, Round, Count };
struct CudaStats {
  bool measured = false;
  uint64_t rounds = 0, evaluated = 0, finalized = 0, replay_rounds = 0;
  uint64_t cycles[6] = {};
  uint32_t smid = 0;
};
struct CudaResult {
  std::vector<uint8_t> state;
  std::vector<uint32_t> energy, acceptLocal, acceptNonlocal;
  std::vector<CudaTrace> trace;
  std::vector<CudaStats> stats;
  float milliseconds = 0;
};
struct CudaDeviceInfo { std::string name; int sms, clockKhz; };
CudaDeviceInfo cudaDeviceInfo();
class CudaBackend {
public:
  CudaBackend(int L, int chains, uint64_t seed,
              const std::vector<uint8_t>& initial,
              const std::vector<Threshold>& thresholds, CudaLaunch launch = {},
              uint64_t chain_offset = 0);
  ~CudaBackend();
  CudaBackend(const CudaBackend&) = delete;
  CudaBackend& operator=(const CudaBackend&) = delete;
  void reserve(uint32_t max_sweeps, uint32_t max_trace_per_chain = 0);
  // 0: nonlocal then physical-local; 1: nonlocal; 2: physical-local.
  // Synchronous only to this instance's nonblocking stream.
  void run(uint64_t start, uint32_t sweeps, int mode, CudaResult& result,
           uint32_t trace_per_chain = 0);
private:
  struct Impl;
  Impl* impl_;
};
} // namespace solid
