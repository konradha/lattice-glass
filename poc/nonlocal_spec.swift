// ---------------------------------------------------------------------------
// nonlocal_spec.swift
//
// Speculative prefix kernel for nonlocal Metropolis swaps on the L=8 / L=21
// lattice glass.  One SIMD group (32 lanes) owns one chain; one threadgroup
// owns three chains and holds all 3 * L^3 packed state bytes in threadgroup
// memory (27,783 B at L=21).  This file is a self-contained spec: it does not
// call into warp_nonlocal.swift and shares no mutable state with it.
//
// Difference from warp_nonlocal.swift (the frozen harness)
// ---------------------------------------------------------
// The frozen kernel computes the conflict set first and only then evaluates
// the energy of the kept lanes; a conflicted lane keeps its proposal and does
// not consume an index.  This spec inverts that order and changes the RNG
// contract:
//
//   * every chain owns one monotone proposal cursor k.  Philox4x32-10 is
//     keyed by (chain, k) with counter (0, 0) and yields (a, b, u) with
//     a = w.x % N, b = w.y % N, u = w.z;
//   * in a round, lane l consumes the proposal at index k + l.  A draw with
//     a == b or torus-L1 distance < 3 is an invalid pair: it is a rejection
//     that consumes its index and is never re-drawn with a bumped attempt;
//   * all 32 lanes read the SAME round-start state, compute the integer dE
//     and the Metropolis accept bit without seeing each other's writes;
//   * accepted lanes (energy-accepted, valid pairs) publish their endpoints
//     and are OR-ed into a per-chain accepted mask with a threadgroup atomic;
//   * the cut j is the first lane whose two sites are within torus L1 <= 2
//     of either endpoint of any earlier ACCEPTED lane (four cross-endpoint
//     distances).  Rejected lanes - invalid pairs and Metropolis rejections -
//     never block;
//   * lanes < j are committed: accepted lanes among them write their two
//     endpoints and the neighbour counts (d = occ(tb) - occ(ta)), then
//     k += j.  Lane j's proposal is retried next round (it is now lane 0),
//     and every lane >= j is re-proposed from the same index stream next
//     round, shifted down by j.  Proposal indices behind the cursor are spent;
//     indices at or ahead of it are re-examined.  This reading of
//     "invalid pair consumes k" is the one implied by "next round starts at
//     k + j; proposal j is retried next round";
//   * the shuffled suffix means a lane's energy decision is always against
//     round-start state, so a committed prefix is a parallel prefix, not 32
//     chained sequential moves.
//
// Accepted lanes inside a prefix are pairwise non-conflicting, hence their
// endpoint and radius-1 neighbourhood writes are disjoint: the 12 neighbour
// sites of a and b do not overlap and no two committed lanes touch a shared
// site.  The commit is therefore race-free and equals a lane-order serial
// commit.
//
// No-op swaps.  Unlike the frozen harness this spec does NOT retire equal-type
// proposals before evaluation: a swap of two sites with the same type has
// dE == 0, is accepted, and writes nothing observable.  Those commits are
// reported separately as the no-op fraction.
//
// Energy / state model (identical integer model to golden/lattice_model.hpp):
//   state byte, one per site, bits 0..1 type, bits 2..4 occupied-neighbour
//   count; f(t, c) = 0 if t == 0, (c-3)^2 if t == 1, (c-5)^2 if t == 2.
//   Site indexing s = x + L*y + L^2*z; directions 0=+x 1=-x 2=+y 3=-y
//   4=+z 5=-z.  Acceptance is integer Metropolis with the shared threshold
//   table T[dE] = floor(2^32 * exp(-beta * dE)): accept dE <= 0, otherwise
//   accept iff u < T[dE].
//
// Metrics used throughout
//   committed proposals  sum of prefix lengths j over rounds
//   applied exchanges    accepted lanes with lane < j that were written
//   acceptance           applied / committed
//   mean prefix          committed / rounds (per chain)
//   no-op fraction       applied swaps with type(a) == type(b) / applied
//
// CUDA mapping
// -------------
//   Metal                         CUDA
//   simdgroup (32 lanes)          warp (32 lanes)
//   threadgroup (96 threads)      cooperative thread array / __shared__ block
//   threadgroup uchar st[3*N]     __shared__ uchar st[3*N]
//   simdgroup_barrier(...)        __syncwarp()
//   threadgroup_barrier(...)      __syncthreads()
//   simd_min(v)                   warp min reduction (__reduce_min_sync on sm_80+)
//   simd_sum(v)                   __reduce_add_sync / warp-shuffle reduction
//   simd_shuffle(x, i)            __shfl_sync(0xffffffff, x, i)
//   simd_ballot(p)                __ballot_sync(0xffffffff, p)
//   threadgroup atomic_uint       __shared__ int + atomicOr(&s, v)
//   atomic_fetch_or_explicit      atomicOr(&sharedMask, 1u << lane)
//   atomic_store/load_explicit    plain shared store / load after __syncwarp()
//   device uchar *gState          global uchar *gState
//   threadgroup_position_in_grid  blockIdx.x
//
// Gates (benchmark only after both gates pass)
//   G1  final lattice byte-identical to the exact sequential CPU reference
//       after 1e5 committed proposals, at L=8 and L=21, beta 2.8 and 3.2.
//       The kernel stops its per-chain cursor at exactly 1e5 proposal indices,
//       capping the final round's prefix at the budget boundary.
//   G2  256 chains, L=21, beta 2.8 and 3.0: GPU vs CPU means of final energy
//       and acceptance agree within 3 sigma.
//   Benchmark: L=21, 256 and 1024 chains, beta 2.8 and 3.2, after 50,000
//       burn-in rounds that are excluded from the timed window.  Timed state
//       is never the seeded initial state.
// ---------------------------------------------------------------------------

import Foundation
import Metal
import Darwin

// ---------------------------------------------------------------------------
// Fixed design constants
// ---------------------------------------------------------------------------

let kThresholdOffset = 512
let kThresholdCount = 1024
let kSimdWidth = 32
let kChainsPerThreadgroup = 3
let kThreadsPerThreadgroup = 96

// ---------------------------------------------------------------------------
// Packed-state helpers (CPU mirror of the MSL helpers)
// ---------------------------------------------------------------------------

@inline(__always)
func typeAt(_ st: [UInt8], _ i: Int) -> Int {
    Int(st[i] & 3)
}

@inline(__always)
func countAt(_ st: [UInt8], _ i: Int) -> Int {
    Int((st[i] >> 2) & 7)
}

@inline(__always)
func packSite(_ t: Int, _ c: Int) -> UInt8 {
    UInt8((c << 2) | (t & 3))
}

@inline(__always)
func energyOf(_ t: Int, _ c: Int) -> Int {
    if t == 0 { return 0 }
    let conn = t == 1 ? 3 : 5
    let d = c - conn
    return d * d
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

@inline(__always)
func neighbor(_ s: Int, _ d: Int, _ L: Int) -> Int {
    var x = s % L
    var y = (s / L) % L
    var z = s / (L * L)
    switch d {
    case 0: x = (x + 1) % L
    case 1: x = (x + L - 1) % L
    case 2: y = (y + 1) % L
    case 3: y = (y + L - 1) % L
    case 4: z = (z + 1) % L
    default: z = (z + L - 1) % L
    }
    return x + L * y + L * L * z
}

@inline(__always)
func torusL1(_ a: UInt32, _ b: UInt32, _ L: Int) -> Int {
    let lu = UInt32(L)
    let ax = Int(a % lu)
    let ay = Int((a / lu) % lu)
    let az = Int(a / (lu * lu))
    let bx = Int(b % lu)
    let by = Int((b / lu) % lu)
    let bz = Int(b / (lu * lu))
    var d = 0
    var t = abs(ax - bx); if t > L - t { t = L - t }; d += t
    t = abs(ay - by); if t > L - t { t = L - t }; d += t
    t = abs(az - bz); if t > L - t { t = L - t }; d += t
    return d
}

// ---------------------------------------------------------------------------
// Philox4x32-10.  Same routine and constants as the frozen harness; the only
// change is the key schedule: key = (chain, proposal index), counter = (0, 0).
// ---------------------------------------------------------------------------

@inline(__always)
func mulHi(_ a: UInt32, _ b: UInt32) -> UInt32 {
    UInt32((UInt64(a) &* UInt64(b)) >> 32)
}

@inline(__always)
func philox4x32_10(_ k0: UInt32, _ k1: UInt32,
                   _ c0in: UInt32, _ c1in: UInt32)
    -> (UInt32, UInt32, UInt32, UInt32)
{
    var c0 = c0in
    var c1 = c1in
    var c2: UInt32 = 0
    var c3: UInt32 = 0
    var k0 = k0
    var k1 = k1
    for _ in 0..<10 {
        let hi0 = mulHi(0xD2511F53, c0)
        let lo0 = 0xD2511F53 &* c0
        let hi1 = mulHi(0xCD9E8D57, c2)
        let lo1 = 0xCD9E8D57 &* c2
        let n0 = hi1 ^ c1 ^ k0
        let n1 = lo1
        let n2 = hi0 ^ c3 ^ k1
        let n3 = lo0
        c0 = n0; c1 = n1; c2 = n2; c3 = n3
        k0 &+= 0x9E3779B9
        k1 &+= 0xBB67AE85
    }
    return (c0, c1, c2, c3)
}

// Proposal at stream index k for a chain.  Same mapping as the MSL kernel.
@inline(__always)
func specProposal(chain: Int, index: UInt32, L: Int, N: Int)
    -> (a: UInt32, b: UInt32, u: UInt32, valid: Bool)
{
    let w = philox4x32_10(UInt32(truncatingIfNeeded: chain), index, 0, 0)
    let a = w.0 % UInt32(N)
    let b = w.1 % UInt32(N)
    let valid = a != b && torusL1(a, b, L) >= 3
    return (a: a, b: b, u: w.2, valid: valid)
}

// ---------------------------------------------------------------------------
// Deterministic start states and energy accounting
// ---------------------------------------------------------------------------

@inline(__always)
func splitmix64(_ x: inout UInt64) -> UInt64 {
    x &+= 0x9e3779b97f4a7c15
    var z = x
    z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
    z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
    return z ^ (z >> 31)
}

func makeInitialState(L: Int, chains: Int, seed: UInt64) -> [UInt8] {
    let N = L * L * L
    var out = [UInt8](repeating: 0, count: chains * N)
    let nocc = (3 * N + 2) / 4
    let n1 = (2 * nocc + 2) / 5
    let n2 = nocc - n1
    for c in 0..<chains {
        var sites = Array(0..<N)
        var x = seed &+ UInt64(c) &* 0x9e3779b97f4a7c15
        x ^= 0x9e3779b97f4a7c15
        if N > 1 {
            for i in stride(from: N, through: 2, by: -1) {
                let j = Int(splitmix64(&x) % UInt64(i))
                sites.swapAt(i - 1, j)
            }
        }
        let base = c * N
        for i in 0..<n1 { out[base + sites[i]] = 1 }
        for i in 0..<n2 { out[base + sites[n1 + i]] = 2 }
        for s in 0..<N {
            var cnt = 0
            for d in 0..<6 {
                if (out[base + neighbor(s, d, L)] & 3) != 0 { cnt += 1 }
            }
            out[base + s] = packSite(Int(out[base + s] & 3), cnt)
        }
    }
    return out
}

func fullEnergy(_ st: [UInt8], _ L: Int) -> Int {
    var e = 0
    for i in 0..<st.count {
        e += energyOf(typeAt(st, i), countAt(st, i))
    }
    return e
}

func makeThresholds(beta: Double) -> [UInt32] {
    var t = [UInt32](repeating: 0, count: kThresholdCount)
    for i in 0..<kThresholdCount {
        let dE = i - kThresholdOffset
        if dE <= 0 {
            t[i] = UInt32.max
        } else {
            let p = exp(-beta * Double(dE)) * 4294967296.0
            t[i] = UInt32(min(max(p, 0.0), 4294967295.0))
        }
    }
    return t
}

// ---------------------------------------------------------------------------
// CPU conflict grid
//
// Exact equivalent of the GPU's lane-ordered mask scan: for each lane in lane
// order, the lane is cut iff either of its endpoints already lies in the
// radius-2 ball of an earlier ACCEPTED lane.  Stamping only accepted lanes is
// what makes rejected lanes non-blocking.  O(32 + 25 * accepted) per round.
// ---------------------------------------------------------------------------

final class ConflictGrid {
    let L: Int
    let N: Int
    private var stamp: [UInt32]
    private var current: UInt32 = 0
    private let ball: [Int32]

    init(L: Int) {
        self.L = L
        self.N = L * L * L
        self.stamp = [UInt32](repeating: 0, count: N)
        var b = [Int32](repeating: 0, count: N * 25)
        for s in 0..<N {
            let x0 = s % L
            let y0 = (s / L) % L
            let z0 = s / (L * L)
            var k = 0
            for dx in -2...2 {
                for dy in -2...2 {
                    for dz in -2...2 {
                        if abs(dx) + abs(dy) + abs(dz) <= 2 {
                            let x = (x0 + dx + L) % L
                            let y = (y0 + dy + L) % L
                            let z = (z0 + dz + L) % L
                            b[s * 25 + k] = Int32(x + L * y + L * L * z)
                            k += 1
                        }
                    }
                }
            }
            precondition(k == 25)
        }
        self.ball = b
    }

    func beginRound() {
        current &+= 1
        if current == 0 {
            for i in 0..<stamp.count { stamp[i] = 0 }
            current = 1
        }
    }

    @inline(__always)
    func isBlocked(_ s: Int) -> Bool {
        stamp[s] == current
    }

    @inline(__always)
    func mark(_ s: Int) {
        let off = s * 25
        for k in 0..<25 {
            stamp[Int(ball[off + k])] = current
        }
    }
}

// ---------------------------------------------------------------------------
// Move evaluation and commit
// ---------------------------------------------------------------------------

func delta(_ st: [UInt8], _ a: Int, _ b: Int, _ L: Int) -> Int {
    let ta = typeAt(st, a)
    let tb = typeAt(st, b)
    let ca = countAt(st, a)
    let cb = countAt(st, b)
    let d = (tb != 0 ? 1 : 0) - (ta != 0 ? 1 : 0)
    var de = energyOf(tb, ca) - energyOf(ta, ca)
    de += energyOf(ta, cb) - energyOf(tb, cb)
    for k in 0..<6 {
        let q = neighbor(a, k, L)
        let tq = typeAt(st, q)
        let cq = countAt(st, q)
        de += energyOf(tq, cq + d) - energyOf(tq, cq)

        let r = neighbor(b, k, L)
        let tr = typeAt(st, r)
        let cr = countAt(st, r)
        de += energyOf(tr, cr - d) - energyOf(tr, cr)
    }
    return de
}

func commit(_ st: inout [UInt8], _ a: Int, _ b: Int, _ L: Int) {
    let ta = typeAt(st, a)
    let tb = typeAt(st, b)
    let ca = countAt(st, a)
    let cb = countAt(st, b)
    let d = (tb != 0 ? 1 : 0) - (ta != 0 ? 1 : 0)
    st[a] = packSite(tb, ca)
    st[b] = packSite(ta, cb)
    for k in 0..<6 {
        let q = neighbor(a, k, L)
        st[q] = packSite(typeAt(st, q), countAt(st, q) + d)
        let r = neighbor(b, k, L)
        st[r] = packSite(typeAt(st, r), countAt(st, r) - d)
    }
}

// ---------------------------------------------------------------------------
// Metal source.  L is a compile-time macro so the threadgroup state array has
// a static size.  The kernel is persistent: load state and cursor, run either
// a fixed number of rounds or until the chain's cursor reaches `limit`, store
// state, cursor, energy and counters.
//
// CUDA notes are inline; the mapping table is in the file header.
// ---------------------------------------------------------------------------

func metalSource(L: Int) -> String {
    return """
    #include <metal_stdlib>
    #include <metal_atomic>
    using namespace metal;

    #define L \(L)
    #define N (L * L * L)
    #define WARPS_PER_TG 3
    #define TG_THREADS 96
    #define THRESH_OFFSET 512
    #define THRESH_COUNT 1024

    inline uint philoxMulHi(uint a, uint b) {
        return uint((ulong(a) * ulong(b)) >> 32);
    }

    // Philox4x32-10, key = (chain, proposal index), counter = (0, 0).
    inline uint4 philox4x32_10(uint2 key, uint2 ctr) {
        uint c0 = ctr.x;
        uint c1 = ctr.y;
        uint c2 = 0u;
        uint c3 = 0u;
        uint k0 = key.x;
        uint k1 = key.y;
        for (int i = 0; i < 10; ++i) {
            uint hi0 = philoxMulHi(0xD2511F53u, c0);
            uint lo0 = 0xD2511F53u * c0;
            uint hi1 = philoxMulHi(0xCD9E8D57u, c2);
            uint lo1 = 0xCD9E8D57u * c2;
            uint n0 = hi1 ^ c1 ^ k0;
            uint n1 = lo1;
            uint n2 = hi0 ^ c3 ^ k1;
            uint n3 = lo0;
            c0 = n0; c1 = n1; c2 = n2; c3 = n3;
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
        return uint4(c0, c1, c2, c3);
    }

    inline uint coordX(uint s) { return s % L; }
    inline uint coordY(uint s) { return (s / L) % L; }
    inline uint coordZ(uint s) { return s / (L * L); }

    inline uint neighbor(uint s, uint d) {
        uint x = coordX(s);
        uint y = coordY(s);
        uint z = coordZ(s);
        if (d == 0u) x = (x + 1u) % L;
        else if (d == 1u) x = (x + L - 1u) % L;
        else if (d == 2u) y = (y + 1u) % L;
        else if (d == 3u) y = (y + L - 1u) % L;
        else if (d == 4u) z = (z + 1u) % L;
        else z = (z + L - 1u) % L;
        return x + L * y + L * L * z;
    }

    inline int torusL1(uint a, uint b) {
        int dx = abs(int(coordX(a)) - int(coordX(b)));
        int dy = abs(int(coordY(a)) - int(coordY(b)));
        int dz = abs(int(coordZ(a)) - int(coordZ(b)));
        dx = min(dx, L - dx);
        dy = min(dy, L - dy);
        dz = min(dz, L - dz);
        return dx + dy + dz;
    }

    // L <= 32 assumed: three 5-bit coordinate fields plus 5-bit z field.
    inline uint packCoord(uint s) {
        return (s % L) | (((s / L) % L) << 5) | ((s / (L * L)) << 10);
    }

    inline int torusL1p(uint a, uint b) {
        int dx = abs(int(a & 31u) - int(b & 31u));
        int dy = abs(int((a >> 5) & 31u) - int((b >> 5) & 31u));
        int dz = abs(int((a >> 10) & 31u) - int((b >> 10) & 31u));
        dx = min(dx, L - dx);
        dy = min(dy, L - dy);
        dz = min(dz, L - dz);
        return dx + dy + dz;
    }

    inline int energyOf(uint t, int c) {
        if (t == 0u) return 0;
        int conn = (t == 1u) ? 3 : 5;
        int d = c - conn;
        return d * d;
    }

    inline uint typeOf(threadgroup uchar *st, uint i) {
        return uint(st[i]) & 3u;
    }

    inline uint countOf(threadgroup uchar *st, uint i) {
        return (uint(st[i]) >> 2) & 7u;
    }

    inline void setSite(threadgroup uchar *st, uint i, uint t, uint c) {
        st[i] = uchar(t | (c << 2));
    }

    kernel void nonlocal_spec(device uchar *gState      [[buffer(0)]],
                              device uint *gK           [[buffer(1)]],
                              device uint *outEnergy    [[buffer(2)]],
                              device uint *outCommitted [[buffer(3)]],
                              device uint *outAccepted  [[buffer(4)]],
                              device uint *outNoop      [[buffer(5)]],
                              device uint *outValid     [[buffer(6)]],
                              device uint *outRounds    [[buffer(7)]],
                              constant uint &C          [[buffer(8)]],
                              constant uint &rounds     [[buffer(9)]],
                              constant uint &limit      [[buffer(10)]],
                              constant uint *thresh     [[buffer(11)]],
                              uint tgid [[threadgroup_position_in_grid]],
                              uint tid  [[thread_position_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg   [[simdgroup_index_in_threadgroup]])
    {
        // CUDA: __shared__ uchar st[3*N]; warp sg owns st[sg*N].
        threadgroup uchar st[WARPS_PER_TG * N];
        // Packed endpoints published for the accepted-mask scan.  CUDA: __shared__.
        threadgroup uint shPA[WARPS_PER_TG * 32];
        threadgroup uint shPB[WARPS_PER_TG * 32];
        // CUDA: __shared__ int shMask[3]; built with atomicOr.
        threadgroup atomic_uint shMask[WARPS_PER_TG];

        const uint baseChain = tgid * WARPS_PER_TG;
        for (uint i = tid; i < WARPS_PER_TG * N; i += TG_THREADS) {
            uint lc = i / N;
            st[i] = gState[(baseChain + lc) * N + (i - lc * N)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // __syncthreads()

        uint chain = baseChain + sg;
        uint slot = sg * 32u + lane;
        uint myCommitted = 0u;
        uint myAccepted = 0u;
        uint myNoop = 0u;
        uint myValid = 0u;
        uint myRounds = 0u;
        uint k = 0u;

        if (chain < C) {
            threadgroup uchar *mys = &st[sg * N];
            k = gK[chain];
            uint rIdx = 0u;

            while (true) {
                if (limit != 0u) {
                    if (k >= limit) break;
                } else if (rIdx >= rounds) {
                    break;
                }
                myRounds += 1u;

                // Lane l owns stream index k + l.  The limit boundary caps
                // the prefix: lanes at or beyond it are inactive.
                uint idx = k + lane;
                bool active = (limit == 0u) || (idx < limit);

                uint a = 0u, b = 0u, uBits = 0u;
                bool valid = false, acc = false, noop = false;
                uint ta = 0u, tb = 0u, ca = 0u, cb = 0u;
                int d = 0, dE = 0;
                uint na[6];
                uint nb[6];

                if (active) {
                    uint4 w = philox4x32_10(uint2(chain, idx), uint2(0u, 0u));
                    a = w.x % N;
                    b = w.y % N;
                    uBits = w.z;
                    // Invalid pair: structural rejection, consumes its index.
                    valid = (a != b) && (torusL1(a, b) >= 3);
                }

                // Every lane reads round-start state; no lane sees another
                // lane's commit in this block.
                if (valid) {
                    ta = typeOf(mys, a);
                    tb = typeOf(mys, b);
                    noop = (ta == tb);
                    if (!noop) {
                        ca = countOf(mys, a);
                        cb = countOf(mys, b);
                        d = int(tb != 0u) - int(ta != 0u);

                        dE += energyOf(tb, int(ca)) - energyOf(ta, int(ca));
                        dE += energyOf(ta, int(cb)) - energyOf(tb, int(cb));
                        #pragma unroll
                        for (uint q = 0u; q < 6u; ++q) {
                            na[q] = neighbor(a, q);
                            uint tq = typeOf(mys, na[q]);
                            int cq = int(countOf(mys, na[q]));
                            dE += energyOf(tq, cq + d) - energyOf(tq, cq);

                            nb[q] = neighbor(b, q);
                            uint tr = typeOf(mys, nb[q]);
                            int cr = int(countOf(mys, nb[q]));
                            dE += energyOf(tr, cr - d) - energyOf(tr, cr);
                        }

                        acc = (dE <= 0) ||
                              ((dE + THRESH_OFFSET < THRESH_COUNT) &&
                               (uBits < thresh[dE + THRESH_OFFSET]));
                    }
                }

                shPA[slot] = packCoord(a);
                shPB[slot] = packCoord(b);

                // Accepted-mask build.  CUDA: lane 0 stores 0 to __shared__,
                // atomicOr(&mask, 1u << lane) per accepted lane, __syncwarp()
                // before and after.  Bit i names lane i of this chain.
                if (lane == 0u) {
                    atomic_store_explicit(&shMask[sg], 0u, memory_order_relaxed);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);  // __syncwarp()
                if (acc) {
                    atomic_fetch_or_explicit(&shMask[sg], (1u << lane),
                                            memory_order_relaxed);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);  // __syncwarp()

                // Conflict scan over set bits strictly below this lane only.
                uint mask = atomic_load_explicit(&shMask[sg],
                                                 memory_order_relaxed);
                uint bits = mask & ((1u << lane) - 1u);
                uint pa = shPA[slot];
                uint pb = shPB[slot];
                bool conflict = false;
                if (acc) {
                    while (bits != 0u) {
                        uint i = uint(ctz(bits));
                        bits = bits & (bits - 1u);
                        uint qa = shPA[sg * 32u + i];
                        uint qb = shPB[sg * 32u + i];
                        conflict = conflict || (torusL1p(pa, qa) <= 2) ||
                                   (torusL1p(pa, qb) <= 2) ||
                                   (torusL1p(pb, qa) <= 2) ||
                                   (torusL1p(pb, qb) <= 2);
                    }
                }

                // Prefix cut: first ACCEPTED lane conflicting with an earlier
                // accepted lane, capped by the first inactive lane (limit
                // boundary).  A rejected lane writes nothing, so it neither
                // blocks nor is cut ("rejected lanes never block").
                uint cand = (acc && conflict) ? lane : 32u;
                if (!active) cand = min(cand, lane);
                uint j = simd_min(cand);  // warp min reduction

                // Commit accepted lanes < j.  Their neighbourhoods are
                // pairwise disjoint because all earlier accepted lanes are
                // conflict-free, so these read-modify-writes are race-free.
                if (acc && lane < j) {
                    setSite(mys, a, tb, ca);
                    setSite(mys, b, ta, cb);
                    #pragma unroll
                    for (uint q = 0u; q < 6u; ++q) {
                        uint qa = na[q];
                        setSite(mys, qa, typeOf(mys, qa),
                                uint(int(countOf(mys, qa)) + d));
                        uint qb = nb[q];
                        setSite(mys, qb, typeOf(mys, qb),
                                uint(int(countOf(mys, qb)) - d));
                    }
                }

                uint accSum = simd_sum((acc && lane < j) ? 1u : 0u);
                uint noopSum = simd_sum((noop && lane < j) ? 1u : 0u);
                uint validSum = simd_sum((valid && lane < j) ? 1u : 0u);
                if (lane == 0u) {
                    myCommitted += j;
                    myAccepted += accSum;
                    myNoop += noopSum;
                    myValid += validSum;
                }
                k += j;
                rIdx += 1u;
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        // All three warps must finish before the strided store-back.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < WARPS_PER_TG * N; i += TG_THREADS) {
            uint lc = i / N;
            gState[(baseChain + lc) * N + (i - lc * N)] = st[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0u && chain < C) {
            threadgroup uchar *mys = &st[sg * N];
            uint e = 0u;
            for (uint i = 0u; i < N; ++i) {
                e += uint(energyOf(typeOf(mys, i), int(countOf(mys, i))));
            }
            outEnergy[chain] = e;
            outCommitted[chain] = myCommitted;
            outAccepted[chain] = myAccepted;
            outNoop[chain] = myNoop;
            outValid[chain] = myValid;
            outRounds[chain] = myRounds;
            gK[chain] = k;
        }
    }
    """
}

// ---------------------------------------------------------------------------
// Host-side statistics helpers
// ---------------------------------------------------------------------------

struct Stats {
    var mean: Double
    var sigma: Double
    var se: Double
    var lo3: Double
    var hi3: Double
}

func stats(_ xs: [Double]) -> Stats {
    let n = Double(xs.count)
    let mean = xs.reduce(0, +) / n
    if xs.count < 2 {
        return Stats(mean: mean, sigma: 0, se: 0, lo3: mean, hi3: mean)
    }
    var v = 0.0
    for x in xs { v += (x - mean) * (x - mean) }
    v /= (n - 1)
    let sigma = sqrt(v)
    let se = sigma / sqrt(n)
    return Stats(mean: mean, sigma: sigma, se: se,
                 lo3: mean - 3 * se, hi3: mean + 3 * se)
}

// ---------------------------------------------------------------------------
// Metal runner and persistent session
// ---------------------------------------------------------------------------

func makeSharedBuffer<T>(_ device: MTLDevice, _ values: [T]) -> MTLBuffer {
    precondition(!values.isEmpty)
    return values.withUnsafeBytes { raw in
        device.makeBuffer(bytes: raw.baseAddress!, length: raw.count,
                          options: .storageModeShared)!
    }
}

func readArray<T>(_ buffer: MTLBuffer, _ type: T.Type, count: Int) -> [T] {
    let p = buffer.contents().bindMemory(to: T.self, capacity: count)
    return Array(UnsafeBufferPointer(start: p, count: count))
}

func readBytes(_ buffer: MTLBuffer, count: Int) -> [UInt8] {
    return readArray(buffer, UInt8.self, count: count)
}

final class MetalRunner {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipeline: MTLComputePipelineState
    let L: Int
    let N: Int

    init(L: Int) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal unavailable")
        }
        guard let queue = device.makeCommandQueue() else {
            fatalError("cannot create Metal command queue")
        }
        let library = try! device.makeLibrary(source: metalSource(L: L), options: nil)
        guard let fn = library.makeFunction(name: "nonlocal_spec") else {
            fatalError("nonlocal_spec function missing")
        }
        let pipeline = try! device.makeComputePipelineState(function: fn)
        let N = L * L * L
        precondition(pipeline.threadExecutionWidth == kSimdWidth,
                     "Metal SIMD width is \(pipeline.threadExecutionWidth), expected 32")
        precondition(pipeline.maxTotalThreadsPerThreadgroup >= kThreadsPerThreadgroup,
                     "threadgroup size \(kThreadsPerThreadgroup) exceeds pipeline limit")
        precondition(device.maxThreadgroupMemoryLength >= kChainsPerThreadgroup * N,
                     "threadgroup state \(kChainsPerThreadgroup * N) B exceeds device limit")
        self.device = device
        self.queue = queue
        self.pipeline = pipeline
        self.L = L
        self.N = N
    }
}

struct SpecRunOutput {
    var state: [UInt8]
    var k: [UInt32]
    var energy: [UInt32]
    var committed: [UInt32]
    var accepted: [UInt32]
    var noop: [UInt32]
    var valid: [UInt32]
    var rounds: [UInt32]
}

// One session owns the persistent device state for `chains` chains: state,
// cursor, thresholds and output counters.  Dispatches share the state and the
// cursor, which is what lets a benchmark burn in and then time a later window.
final class SpecSession {
    let runner: MetalRunner
    let chains: Int
    let paddedChains: Int
    let groups: Int
    let N: Int
    let stateBuf: MTLBuffer
    let kBuf: MTLBuffer
    let threshBuf: MTLBuffer
    let energyBuf: MTLBuffer
    let committedBuf: MTLBuffer
    let acceptedBuf: MTLBuffer
    let noopBuf: MTLBuffer
    let validBuf: MTLBuffer
    let roundsBuf: MTLBuffer

    init(runner: MetalRunner, state: [UInt8], chains: Int,
         thresholds: [UInt32]) {
        precondition(state.count == chains * runner.N)
        precondition(thresholds.count == kThresholdCount)
        self.runner = runner
        self.chains = chains
        self.N = runner.N
        self.groups = (chains + kChainsPerThreadgroup - 1) / kChainsPerThreadgroup
        self.paddedChains = groups * kChainsPerThreadgroup

        var padded = state
        padded.append(contentsOf: [UInt8](
            repeating: 0, count: (paddedChains - chains) * N))

        self.stateBuf = makeSharedBuffer(runner.device, padded)
        self.kBuf = makeSharedBuffer(
            runner.device, [UInt32](repeating: 0, count: paddedChains))
        self.threshBuf = makeSharedBuffer(runner.device, thresholds)
        self.energyBuf = makeSharedBuffer(
            runner.device, [UInt32](repeating: 0, count: paddedChains))
        self.committedBuf = makeSharedBuffer(
            runner.device, [UInt32](repeating: 0, count: paddedChains))
        self.acceptedBuf = makeSharedBuffer(
            runner.device, [UInt32](repeating: 0, count: paddedChains))
        self.noopBuf = makeSharedBuffer(
            runner.device, [UInt32](repeating: 0, count: paddedChains))
        self.validBuf = makeSharedBuffer(
            runner.device, [UInt32](repeating: 0, count: paddedChains))
        self.roundsBuf = makeSharedBuffer(
            runner.device, [UInt32](repeating: 0, count: paddedChains))
    }

    // limit != 0: run until each chain's cursor reaches the proposal budget.
    // limit == 0: run exactly `rounds` rounds.  Returns the host-measured
    // wall time of this dispatch only (commit + waitUntilCompleted).
    @discardableResult
    func dispatch(rounds: UInt32, limit: UInt32) -> Double {
        let cmd = runner.queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(runner.pipeline)
        enc.setBuffer(stateBuf, offset: 0, index: 0)
        enc.setBuffer(kBuf, offset: 0, index: 1)
        enc.setBuffer(energyBuf, offset: 0, index: 2)
        enc.setBuffer(committedBuf, offset: 0, index: 3)
        enc.setBuffer(acceptedBuf, offset: 0, index: 4)
        enc.setBuffer(noopBuf, offset: 0, index: 5)
        enc.setBuffer(validBuf, offset: 0, index: 6)
        enc.setBuffer(roundsBuf, offset: 0, index: 7)
        var cChains = UInt32(chains)
        var cRounds = rounds
        var cLimit = limit
        enc.setBytes(&cChains, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&cRounds, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&cLimit, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBuffer(threshBuf, offset: 0, index: 11)
        enc.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: kThreadsPerThreadgroup, height: 1, depth: 1))
        enc.endEncoding()

        let t0 = CFAbsoluteTimeGetCurrent()
        cmd.commit()
        cmd.waitUntilCompleted()
        let seconds = CFAbsoluteTimeGetCurrent() - t0
        precondition(cmd.error == nil,
                     "command buffer error: \(String(describing: cmd.error))")
        return seconds
    }

    func readBack(readState: Bool) -> SpecRunOutput {
        var state: [UInt8] = []
        if readState {
            let all = readBytes(stateBuf, count: paddedChains * N)
            state = Array(all[0..<(chains * N)])
        }
        return SpecRunOutput(
            state: state,
            k: readArray(kBuf, UInt32.self, count: chains),
            energy: readArray(energyBuf, UInt32.self, count: chains),
            committed: readArray(committedBuf, UInt32.self, count: chains),
            accepted: readArray(acceptedBuf, UInt32.self, count: chains),
            noop: readArray(noopBuf, UInt32.self, count: chains),
            valid: readArray(validBuf, UInt32.self, count: chains),
            rounds: readArray(roundsBuf, UInt32.self, count: chains))
    }
}

// ---------------------------------------------------------------------------
// Exact sequential CPU reference
//
// Same proposal stream (lane l <-> index k + l, Philox keyed (chain, k)), same
// round-start energy evaluation, same accepted-mask cut over earlier accepted
// lanes, same prefix commit and same cursor advance k += j.  Proposals at or
// beyond the cut are re-proposed next round from the same index stream; the
// ConflictGrid is the exact equivalent of the GPU's set-bit scan.
// ---------------------------------------------------------------------------

struct SpecChainStats {
    var energy = 0
    var committed = 0
    var accepted = 0
    var noop = 0
    var validRetired = 0
    var rounds = 0
    var k: UInt32 = 0
}

func simulateSpecChain(state: inout [UInt8], chain: Int, L: Int,
                       thresholds: [UInt32], kStart: UInt32,
                       rounds: Int, limit: UInt32,
                       grid: ConflictGrid) -> SpecChainStats
{
    let N = L * L * L
    var k = kStart
    var r = 0
    var stats = SpecChainStats()

    var a = [Int](repeating: 0, count: kSimdWidth)
    var b = [Int](repeating: 0, count: kSimdWidth)
    var u = [UInt32](repeating: 0, count: kSimdWidth)
    var acc = [Bool](repeating: false, count: kSimdWidth)
    var valid = [Bool](repeating: false, count: kSimdWidth)
    var active = [Bool](repeating: false, count: kSimdWidth)
    var noop = [Bool](repeating: false, count: kSimdWidth)

    while true {
        if limit != 0 {
            if k >= limit { break }
        } else if r >= rounds {
            break
        }

        for l in 0..<kSimdWidth {
            let idx = k &+ UInt32(l)
            let isActive = (limit == 0) || (idx < limit)
            active[l] = isActive
            valid[l] = false
            acc[l] = false
            noop[l] = false
            if !isActive { continue }

            let p = specProposal(chain: chain, index: idx, L: L, N: N)
            if !p.valid { continue }

            valid[l] = true
            a[l] = Int(p.a)
            b[l] = Int(p.b)
            u[l] = p.u

            let ta = typeAt(state, a[l])
            let tb = typeAt(state, b[l])
            noop[l] = (ta == tb)
            if !noop[l] {
                let de = delta(state, a[l], b[l], L)
                if de <= 0 {
                    acc[l] = true
                } else if de + kThresholdOffset < kThresholdCount {
                    acc[l] = u[l] < thresholds[de + kThresholdOffset]
                }
            }
        }

        // Lane-ordered cut: first lane blocked by any earlier accepted lane,
        // or the first lane outside the proposal budget.
        grid.beginRound()
        var j = kSimdWidth
        for l in 0..<kSimdWidth {
            if !active[l] { j = l; break }
            if acc[l] {
                if grid.isBlocked(a[l]) || grid.isBlocked(b[l]) {
                    j = l
                    break
                }
                grid.mark(a[l])
                grid.mark(b[l])
            }
        }

        // Commit the accepted prefix in lane order.  Accepted lanes < j are
        // pairwise non-conflicting, so the writes are disjoint by construction.
        for l in 0..<j where acc[l] {
            commit(&state, a[l], b[l], L)
            stats.accepted += 1
        }
        for l in 0..<j where noop[l] {
            stats.noop += 1
        }
        for l in 0..<j where valid[l] {
            stats.validRetired += 1
        }
        stats.committed += j
        k &+= UInt32(j)
        r += 1
    }

    stats.rounds = r
    stats.k = k
    stats.energy = fullEnergy(state, L)
    return stats
}

// ---------------------------------------------------------------------------
// G1: final lattice byte-identical to the CPU replay after 1e5 proposals
// ---------------------------------------------------------------------------

func gate1(_ r8: MetalRunner, _ r21: MetalRunner) -> Bool {
    print("=== G1: exact CPU replay, 1e5 committed proposals, L=8/L=21, beta 2.8/3.2 ===")
    let target: UInt32 = 100_000
    let cases: [(Int, MetalRunner)] = [(8, r8), (21, r21)]
    var allPass = true

    for (L, runner) in cases {
        let N = L * L * L
        for beta in [2.8, 3.2] {
            let th = makeThresholds(beta: beta)
            let initial = makeInitialState(L: L, chains: 1, seed: 1)
            let session = SpecSession(runner: runner, state: initial,
                                      chains: 1, thresholds: th)
            session.dispatch(rounds: 0, limit: target)
            let gpu = session.readBack(readState: true)

            var st = initial
            let grid = ConflictGrid(L: L)
            let cpu = simulateSpecChain(state: &st, chain: 0, L: L,
                                        thresholds: th, kStart: 0,
                                        rounds: 0, limit: target, grid: grid)

            let gpuState = Array(gpu.state[0..<N])
            let bytesEqual = (gpuState == st)
            let kEqual = (gpu.k[0] == cpu.k)
            let energyEqual = (Int(gpu.energy[0]) == cpu.energy)
            let committedEqual = (Int(gpu.committed[0]) == cpu.committed)
            let acceptedEqual = (Int(gpu.accepted[0]) == cpu.accepted)
            let noopEqual = (Int(gpu.noop[0]) == cpu.noop)
            let validEqual = (Int(gpu.valid[0]) == cpu.validRetired)
            let pass = bytesEqual && kEqual && energyEqual &&
                       committedEqual && acceptedEqual &&
                       noopEqual && validEqual
            allPass = allPass && pass

            let meanPrefix = cpu.rounds > 0
                ? Double(cpu.committed) / Double(cpu.rounds) : 0.0
            let noopFraction = cpu.committed > 0
                ? Double(cpu.noop) / Double(cpu.committed) : 0.0
            print(String(format:
                "L=%d beta=%.1f proposals=%u rounds=%d " +
                "E cpu=%d gpu=%d",
                L, beta, cpu.k, cpu.rounds,
                cpu.energy, Int(gpu.energy[0]))
                + " bytes_equal=\(bytesEqual ? "YES" : "NO")"
                + " k_equal=\(kEqual ? "YES" : "NO")"
                + String(format:
                " committed=%d accepted=%d noop=%d valid=%d " +
                "mean_prefix=%.3f noop_frac=%.4f",
                cpu.committed, cpu.accepted, cpu.noop, cpu.validRetired,
                meanPrefix, noopFraction)
                + " counters_equal=\(committedEqual && acceptedEqual && noopEqual && validEqual ? "YES" : "NO")"
                + "  " + (pass ? "PASS" : "FAIL"))
        }
    }
    print("G1 \(allPass ? "PASS" : "FAIL")")
    return allPass
}

// ---------------------------------------------------------------------------
// G2: 256 chains, beta 2.8 and 3.0, GPU vs CPU means within 3 sigma
// ---------------------------------------------------------------------------

func gate2(_ runner: MetalRunner, chains: Int = 256, rounds: Int = 2048) -> Bool {
    print("=== G2: 256 chains, L=21, CPU reference vs GPU (3 sigma) ===")
    let L = 21
    let N = L * L * L
    let initial = makeInitialState(L: L, chains: chains, seed: 1)
    let grid = ConflictGrid(L: L)
    var allPass = true

    for beta in [2.8, 3.0] {
        let th = makeThresholds(beta: beta)
        let session = SpecSession(runner: runner, state: initial,
                                  chains: chains, thresholds: th)
        session.dispatch(rounds: UInt32(rounds), limit: 0)
        let gpu = session.readBack(readState: true)

        var cpuEnergy = [Double](repeating: 0, count: chains)
        var cpuAccept = [Double](repeating: 0, count: chains)
        var cpuCommitted = 0
        var cpuAccepted = 0
        var cpuNoop = 0
        var cpuValid = 0
        var exactChains = 0

        for c in 0..<chains {
            var st = Array(initial[c * N..<(c + 1) * N])
            let res = simulateSpecChain(state: &st, chain: c, L: L,
                                        thresholds: th, kStart: 0,
                                        rounds: rounds, limit: 0, grid: grid)
            cpuEnergy[c] = Double(res.energy)
            cpuAccept[c] = res.committed > 0
                ? Double(res.accepted) / Double(res.committed) : 0.0
            cpuCommitted += res.committed
            cpuAccepted += res.accepted
            cpuNoop += res.noop
            cpuValid += res.validRetired

            let gpuSlice = Array(gpu.state[(c * N)..<((c + 1) * N)])
            if gpuSlice == st { exactChains += 1 }
        }

        var gpuEnergy = [Double](repeating: 0, count: chains)
        var gpuAccept = [Double](repeating: 0, count: chains)
        var gpuCommitted = 0
        var gpuAccepted = 0
        var gpuNoop = 0
        var gpuValid = 0
        var gpuRounds = 0
        for c in 0..<chains {
            gpuEnergy[c] = Double(gpu.energy[c])
            let cms = Int(gpu.committed[c])
            gpuCommitted += cms
            gpuAccepted += Int(gpu.accepted[c])
            gpuNoop += Int(gpu.noop[c])
            gpuValid += Int(gpu.valid[c])
            gpuRounds += Int(gpu.rounds[c])
            gpuAccept[c] = cms > 0 ? Double(gpu.accepted[c]) / Double(cms) : 0.0
        }

        let cE = stats(cpuEnergy)
        let gE = stats(gpuEnergy)
        let cA = stats(cpuAccept)
        let gA = stats(gpuAccept)
        let seE = sqrt(cE.se * cE.se + gE.se * gE.se)
        let seA = sqrt(cA.se * cA.se + gA.se * gA.se)
        let e3 = abs(cE.mean - gE.mean) <= 3 * seE
        let a3 = abs(cA.mean - gA.mean) <= 3 * seA
        let pass = e3 && a3
        allPass = allPass && pass

        let meanPrefix = Double(gpuCommitted) / Double(gpuRounds)
        let acceptance = gpuCommitted > 0
            ? Double(gpuAccepted) / Double(gpuCommitted) : 0.0
        let noopFraction = gpuCommitted > 0
            ? Double(gpuNoop) / Double(gpuCommitted) : 0.0

        print(String(format:
            "beta=%.1f  E: cpu %.3f [%.3f, %.3f]  gpu %.3f [%.3f, %.3f]",
            beta, cE.mean, cE.lo3, cE.hi3, gE.mean, gE.lo3, gE.hi3)
            + "  within3sigma=\(e3 ? "YES" : "NO")")
        print(String(format:
            "           acc: cpu %.6f [%.6f, %.6f]  gpu %.6f [%.6f, %.6f]",
            cA.mean, cA.lo3, cA.hi3, gA.mean, gA.lo3, gA.hi3)
            + "  within3sigma=\(a3 ? "YES" : "NO")")
        let g2line = String(format:
            "           committed=%d accepted=%d noop=%d valid=%d " +
            "acceptance=%.6f mean_prefix=%.3f noop_frac=%.4f exact_states=%d/%d",
            gpuCommitted, gpuAccepted, gpuNoop, gpuValid,
            acceptance, meanPrefix, noopFraction, exactChains, chains)
        let g2tag = pass ? "PASS" : "FAIL"
        print(g2line + "  cpu_committed=\(cpuCommitted) cpu_accepted=\(cpuAccepted)"
            + " cpu_noop=\(cpuNoop) cpu_valid=\(cpuValid)"
            + "  G2 beta=\(beta) \(g2tag)")
    }
    print("G2 \(allPass ? "PASS" : "FAIL")")
    return allPass
}

// ---------------------------------------------------------------------------
// Benchmark: L=21, 256/1024 chains, beta 2.8/3.2, after a 50,000-round burn-in
// that is excluded from the timed window.  Timed state is never the seeded
// initial state.
// ---------------------------------------------------------------------------

func benchmark(_ runner: MetalRunner) {
    print("=== Benchmark: L=21, burn-in 50,000 rounds (excluded from timing), " +
          "timed window 8192 rounds ===")
    print("(timed state is the post-burn-in state, never the seeded initial state)")
    let L = 21
    let N = L * L * L
    let burnRounds: UInt32 = 50_000
    let timedRounds: UInt32 = 8_192

    for beta in [2.8, 3.2] {
        let th = makeThresholds(beta: beta)
        for chains in [256, 1024] {
            let initial = makeInitialState(L: L, chains: chains, seed: 1)
            let session = SpecSession(runner: runner, state: initial,
                                      chains: chains, thresholds: th)

            // Burn-in: excluded from the timer.
            session.dispatch(rounds: burnRounds, limit: 0)

            // Timed window: continues from the burn-in state and cursor.
            let seconds = session.dispatch(rounds: timedRounds, limit: 0)
            let out = session.readBack(readState: false)

            var committed = 0
            var accepted = 0
            var noop = 0
            var validRetired = 0
            var energySum = 0
            for c in 0..<chains {
                committed += Int(out.committed[c])
                accepted += Int(out.accepted[c])
                noop += Int(out.noop[c])
                validRetired += Int(out.valid[c])
                energySum += Int(out.energy[c])
            }
            let laneProposals = Double(chains) * Double(timedRounds) * 32.0
            let committedPerSec = Double(committed) / seconds
            let appliedPerSec = Double(accepted) / seconds
            let lanePerSec = laneProposals / seconds
            let acceptance = committed > 0
                ? Double(accepted) / Double(committed) : 0.0
            let meanPrefix = Double(committed) / (Double(chains) * Double(timedRounds))
            let noopFraction = committed > 0
                ? Double(noop) / Double(committed) : 0.0
            let meanEnergy = Double(energySum) / Double(chains)

            let line = String(format:
                "beta=%.1f chains=%4d burn=%u timed=%u time=%.6f s  " +
                "committed=%.4f G/s  applied=%.4f G/s  lanes=%.4f G/s",
                beta, chains, burnRounds, timedRounds, seconds,
                committedPerSec / 1e9, appliedPerSec / 1e9, lanePerSec / 1e9)
            let tail = String(format:
                "  acceptance=%.6f mean_prefix=%.3f noop_frac=%.4f " +
                "valid_retired=%d mean_E=%.3f",
                acceptance, meanPrefix, noopFraction, validRetired, meanEnergy)
            print(line + tail)
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

var sharedRunner21: MetalRunner? = nil

func runner21() -> MetalRunner {
    if sharedRunner21 == nil {
        sharedRunner21 = MetalRunner(L: 21)
    }
    return sharedRunner21!
}

let mode = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1].lowercased() : "all"

let needG1 = (mode == "all" || mode == "g1")
let needG2 = (mode == "all" || mode == "g2")
let needBench = (mode == "all" || mode == "bench")

var g1Pass = true
var g2Pass = true

if needG1 {
    let r8 = MetalRunner(L: 8)
    let r21 = runner21()
    g1Pass = gate1(r8, r21)
}

if needG2 {
    g2Pass = gate2(runner21())
}

if needBench {
    if mode == "all" && !(g1Pass && g2Pass) {
        print("=== Benchmark skipped: gates did not pass ===")
    } else {
        benchmark(runner21())
    }
}

if mode == "all" {
    let pass = g1Pass && g2Pass
    print("=== SUMMARY: G1 \(g1Pass ? "PASS" : "FAIL"), " +
          "G2 \(g2Pass ? "PASS" : "FAIL") ===")
    if !pass { exit(1) }
}
