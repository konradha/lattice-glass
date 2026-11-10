#pragma once

#include "efma.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace solid {

struct CudaTrace {
  uint64_t proposal;
  uint32_t a, b;
  int32_t delta;
  uint32_t flags; // bit 0: valid; bit 1: no-op; bit 2: accepted
};

struct CudaResult {
  std::vector<uint8_t> state;
  std::vector<uint32_t> energy, acceptLocal, acceptNonlocal;
  std::vector<CudaTrace> trace;
  float milliseconds = 0;
};

struct CudaDeviceInfo {
  std::string name;
  int sms;
  int clockKhz;
};

CudaDeviceInfo cudaDeviceInfo();

class CudaBackend {
public:
  CudaBackend(int L, int chains, uint64_t seed,
              const std::vector<uint8_t>& initial,
              const std::vector<Threshold>& thresholds, int window = 128,
              uint64_t chain_offset = 0);
  ~CudaBackend();
  CudaBackend(const CudaBackend&) = delete;
  CudaBackend& operator=(const CudaBackend&) = delete;

  // mode: 0 = nonlocal then local, 1 = nonlocal, 2 = local.
  // Trace proposal IDs are the full keyed RNG indices, not call-relative IDs.
  void run(uint64_t start, uint32_t sweeps, int mode, CudaResult& result,
           uint32_t trace_per_chain = 0);

private:
  struct Impl;
  Impl* impl_;
};

} // namespace solid
