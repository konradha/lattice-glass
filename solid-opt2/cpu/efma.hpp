#pragma once
#include "lattice.hpp"
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace solid {

// EFMA threshold: T = floor(2^64 * exp(-beta*d)), split into two u32 words.
struct Threshold {
  uint32_t hi = 0;
  uint32_t lo = 0;
};

constexpr uint64_t EFMA_GOLDEN = 0x9e3779b97f4a7c15ULL;

// Counter-based word stream. A word depends only on (key, word index).
struct WordStream {
  uint64_t key = 0;
  uint32_t cursor = 0;
  uint64_t cache = 0;
  bool have_cache = false;
  inline uint32_t next() {
    const uint32_t pair = cursor >> 1;
    ++cursor;
    if (have_cache) {
      have_cache = false;
      return uint32_t(cache);
    }
    cache = mix(key + uint64_t(pair) * EFMA_GOLDEN);
    have_cache = true;
    return uint32_t(cache >> 32);
  }
};

// Lemire rejection: exact uniform integer in [0, n). t = 2^32 mod n.
inline uint32_t uniformIndex(WordStream& rng, uint32_t n, uint32_t t) {
  for (;;) {
    uint32_t r = rng.next();
    uint64_t m = uint64_t(r) * uint64_t(n);
    uint32_t lo = uint32_t(m);
    if (lo < t) continue;
    return uint32_t(m >> 32);
  }
}

inline uint64_t proposalKey(uint64_t seed, uint64_t chain, uint64_t index,
                            bool local) {
  return seed ^ (chain * EFMA_GOLDEN) ^
         (index * 0xd1342543de82ef95ULL) ^
         (local ? 0x1234ULL : 0x5678ULL);
}

inline bool acceptEFMA(uint32_t u_hi, WordStream& rng,
                       const std::vector<Threshold>& th, int dE) {
  if (dE <= 0) return true;
  if (dE >= int(th.size())) return false;
  const Threshold& t = th[dE];
  if (u_hi < t.hi) return true;
  if (u_hi > t.hi) return false;
  uint32_t u_lo = rng.next();
  return u_lo < t.lo;
}

// beta_text is the decimal CLI value. Do not round beta through double first.
inline std::vector<Threshold> makeThresholds(const std::string& beta_text) {
  using boost::multiprecision::cpp_dec_float_100;
  using boost::multiprecision::cpp_int;
  const cpp_dec_float_100 beta(beta_text);
  const cpp_dec_float_100 two64 = pow(cpp_dec_float_100(2), 64);
  std::vector<Threshold> th(1024);
  for (int d = 1; d < int(th.size()); ++d) {
    const cpp_dec_float_100 p = exp(-beta * cpp_dec_float_100(d));
    cpp_int t = static_cast<cpp_int>(p * two64);  // positive: trunc = floor
    const cpp_int mask = (cpp_int(1) << 32) - 1;
    th[d].lo = uint32_t((t & mask).convert_to<uint32_t>());
    th[d].hi = uint32_t(((t >> 32) & mask).convert_to<uint32_t>());
  }
  return th;
}

} // namespace solid
