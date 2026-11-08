// ---------------------------------------------------------------------------
// warp_nonlocal.swift
//
// Warp-batched nonlocal Metropolis swaps for the L=8 / L=21 lattice glass.
// One SIMD group (32 lanes) owns one chain; one threadgroup owns three chains.
// The whole threadgroup state is 3 * L^3 packed bytes (27,783 B at L=21),
// under the 32 KiB threadgroup-memory limit.
//
// State byte, one per site:
//   bits 0..1  type: 0 vacancy, 1 and 2 occupied species
//   bits 2..4  cached occupied-neighbour count (0..6)
// Site indexing is s = x + L*y + L^2*z, matching the other PoCs.  Directions
// are 0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z.
//
// Energy (same integer model as golden/local_model.hpp and
// golden/lattice_model.hpp):
//   f(t,c) = 0 if t == 0, (c-3)^2 if t == 1, (c-5)^2 if t == 2
//
// Proposal / accept contract implemented here
//   * every round every lane holds one (a,b) with b != a and torus-L1
//     distance >= 3;
//   * a lane that was deferred (conflicted) last round carries the same
//     (a,b,u) and does not consume RNG; every other lane draws fresh from
//     Philox4x32-10 keyed by (chain, lane) with counter (round, attempt);
//     attempt advances on b == a or distance < 3;
//   * conflict is any of the four cross-endpoint torus-L1 distances <= 2;
//     greedy lane order: lane j is kept iff no earlier kept lane conflicts;
//   * kept lanes evaluate and commit before the next round, in lane order.
//
// Nonlocal endpoint counts.  Because the proposal enforces distance >= 3,
// b is outside a's closed neighbourhood and vice versa, so c[a] and c[b] do
// NOT move when the types are exchanged.  The twelve neighbours of a and b
// are distinct and take c += d (a's neighbours) and c -= d (b's neighbours),
// d = occ(tb) - occ(ta).  This is the update used by
// golden/lattice_model.hpp::apply_accept and SINGLE_CHAIN_ARCH.md section 1
// ("sites 2..7 get c += ob-oa, sites 8..13 get c += oa-ob").  A local bond
// swap is different: there a and b are neighbours, hence the c[a] -= d,
// c[b] += d rule from the local kernels.
//
// Null proposals (type(a) == type(b)) are retired without a commit and are
// not counted as accepted, matching golden/lattice_model.hpp::step.  They
// still occupy their lane for conflict purposes and are consumed; the lane
// draws a fresh proposal next round.  Retiring them avoids a deferred null
// pair being re-proposed forever.
//
// Acceptance uses the integer form of Metropolis: with u = Philox word w.z,
// accept dE <= 0, otherwise accept iff u < T[dE], where
// T[dE] = floor(2^32 * exp(-beta * dE)).  The host builds T once and both
// the CPU replay and the Metal kernel use exactly that table, so decisions
// are bit-identical without relying on device libm.
//
// Gates (run in this order; benchmark only after them):
//   G1  one chain, 10,000 rounds, L=8 and L=21.  GPU dumps all lane records
//       (round, lane, a, b, u, dE, kept, accepted) in lane/commit order.
//       The CPU replay below regenerates the same proposals, retry state,
//       greedy kept set, dE and acceptance and applies accepted commits in
//       that order; final types/counts must be byte-identical.
//   G2  256 chains at beta 2.8 and 3.0.  CPU reference (same Philox, retry,
//       conflict, greedy and commit logic) and GPU means for final energy and
//       acceptance with 3-sigma intervals.
//   G3  mean kept lanes per round at L=21 (expected about 25 of 32).
//   Benchmark kept nonlocal proposals/s for 1, 8, 64 and 256 chains,
//       compared with the 2.8 G/s local-bond kernel baseline.
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
let kThreadsPerThreadgroup = kSimdWidth * kChainsPerThreadgroup
let kLocalBaselinePerSecond = 2.8e9

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
// Philox4x32-10
//
// Spec constants: M0=0xD2511F53, M1=0xCD9E8D57, W0=0x9E3779B9,
// W1=0xBB67AE85, ten rounds.  The Metal side is the same routine in MSL.
// key = (chain, lane); counter = (round, attempt).
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

// ---------------------------------------------------------------------------
// Deterministic start states
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

// ---------------------------------------------------------------------------
// CPU conflict grid
//
// Equivalent, exact form of the GPU's greedy lane scan.  For each kept lane
// in lane order, every endpoint's radius-2 torus-L1 ball is stamped.  A lane
// is kept iff neither endpoint lies in the union of the earlier kept balls.
// This is the same predicate as the four simd_shuffle cross-endpoint checks
// on the GPU, with O(50) work per kept lane instead of a serial 31-lane scan.
// ---------------------------------------------------------------------------

final class ConflictGrid {
    let L: Int
    let N: Int
    let ballSize = 25
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
// CPU proposal draw and move evaluation
// ---------------------------------------------------------------------------

func drawProposal(chain: Int, lane: Int, round: Int, L: Int, N: Int)
    -> (a: UInt32, b: UInt32, u: UInt32)
{
    let key0 = UInt32(truncatingIfNeeded: chain)
    let key1 = UInt32(truncatingIfNeeded: lane)
    let ctr0 = UInt32(truncatingIfNeeded: round)
    var attempt: UInt32 = 0
    while true {
        let w = philox4x32_10(key0, key1, ctr0, attempt)
        let a = w.0 % UInt32(N)
        let b = w.1 % UInt32(N)
        if a != b && torusL1(a, b, L) >= 3 {
            return (a: a, b: b, u: w.2)
        }
        attempt &+= 1
    }
}

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
// Trace record: identical layout to the MSL struct.
// ---------------------------------------------------------------------------

struct TraceRec {
    var round: UInt32
    var lane: UInt32
    var a: UInt32
    var b: UInt32
    var u: UInt32
    var dE: UInt32
    var kept: UInt32
    var accepted: UInt32
}

// ---------------------------------------------------------------------------
// Metal source.  L is a compile-time macro so the threadgroup array has a
// static size.  The kernel is persistent: load state, run all rounds, store.
// ---------------------------------------------------------------------------

func metalSource(L: Int) -> String {
    return """
    #include <metal_stdlib>
    using namespace metal;

    #define L \(L)
    #define N (L * L * L)
    #define TG_THREADS 96
    #define THRESH_OFFSET 512
    #define THRESH_COUNT 1024
    #define TABSTOP 32

    struct TraceRec {
        uint round;
        uint lane;
        uint a;
        uint b;
        uint u;
        uint dE;
        uint kept;
        uint accepted;
    };

    inline uint philoxMulHi(uint a, uint b) {
        return uint((ulong(a) * ulong(b)) >> 32);
    }

    // Philox4x32-10.  key = (chain, lane), counter = (round, attempt).
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
        int ax = int(coordX(a)), ay = int(coordY(a)), az = int(coordZ(a));
        int bx = int(coordX(b)), by = int(coordY(b)), bz = int(coordZ(b));
        int dx = abs(ax - bx);
        int dy = abs(ay - by);
        int dz = abs(az - bz);
        dx = min(dx, L - dx);
        dy = min(dy, L - dy);
        dz = min(dz, L - dz);
        return dx + dy + dz;
    }
    inline uint packCoord(uint s) {
        return (s % L) | (((s / L) % L) << 5) | ((s / (L * L)) << 10);
    }
    inline int torusL1p(uint a, uint b) {
        int dx = abs(int(a & 31u) - int(b & 31u));
        int dy = abs(int((a >> 5) & 31u) - int((b >> 5) & 31u));
        int dz = abs(int((a >> 10) & 31u) - int((b >> 10) & 31u));
        dx = min(dx, L - dx); dy = min(dy, L - dy); dz = min(dz, L - dz);
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

    kernel void warp_nonlocal(device uchar *gState [[buffer(0)]],
                              device TraceRec *trace [[buffer(1)]],
                              device uint *outEnergy [[buffer(2)]],
                              device uint *outKept [[buffer(3)]],
                              device uint *outAccepted [[buffer(4)]],
                              device uint *outNonNullKept [[buffer(5)]],
                              constant uint &C [[buffer(6)]],
                              constant uint &rounds [[buffer(7)]],
                              constant uint &traceOn [[buffer(8)]],
                              constant uint *thresh [[buffer(9)]],
                              uint tgid [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
        threadgroup uchar st[3 * N];
        threadgroup uint shA[96];
        threadgroup uint shB[96];
        threadgroup uint shKept[3];

        const uint baseChain = tgid * 3u;
        for (uint i = tid; i < 3u * N; i += TG_THREADS) {
            uint lc = i / N;
            uint off = i - lc * N;
            st[i] = gState[(baseChain + lc) * N + off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint chain = baseChain + sg;
        threadgroup uchar *mys = &st[sg * N];

        uint a = 0u;
        uint b = 0u;
        uint uBits = 0u;
        bool deferred = false;
        uint myKept = 0u;
        uint myAccepted = 0u;
        uint myNonNullKept = 0u;

        if (chain < C) {
            for (uint r = 0u; r < rounds; ++r) {
                if (!deferred) {
                    uint attempt = 0u;
                    while (true) {
                        uint4 w = philox4x32_10(uint2(chain, lane), uint2(r, attempt));
                        uint na = w.x % N;
                        uint nb = w.y % N;
                        if (na != nb && torusL1(na, nb) >= 3) {
                            a = na;
                            b = nb;
                            uBits = w.z;
                            break;
                        }
                        attempt += 1u;
                    }
                }
                // Reconverge the divergent retry loop before simd_shuffle.
                simdgroup_barrier(mem_flags::mem_threadgroup);

                // Greedy lane order.  For lane j we compare against every
                // earlier lane i (31 possible earlier lanes) on all four
                // endpoint pairs.  At iteration i, lane i's kept bit is final,
                // so a uniform broadcast reproduces the serial greedy scan.
                // (31 x 4 cross-endpoint conflict detection.)
                uint ca = packCoord(a);
                uint cb = packCoord(b);
                bool kept = true;
                for (uint i = 0u; i < 32u; ++i) {
                    uint cai = simd_shuffle(ca, ushort(i));
                    uint cbi = simd_shuffle(cb, ushort(i));
                    uint ki = simd_shuffle(kept ? 1u : 0u, ushort(i));
                    if (i < lane && ki != 0u) {
                        if (torusL1p(ca, cai) <= 2 || torusL1p(ca, cbi) <= 2 ||
                            torusL1p(cb, cai) <= 2 || torusL1p(cb, cbi) <= 2) {
                            kept = false;
                        }
                    }
                }

                int dE = 0;
                bool acc = false;
                if (kept) {
                    myKept += 1u;
                    uint ta = typeOf(mys, a);
                    uint tb = typeOf(mys, b);
                    if (ta != tb) {
                        myNonNullKept += 1u;
                        uint ca = countOf(mys, a);
                        uint cb = countOf(mys, b);
                        int d = int(tb != 0u ? 1 : 0) - int(ta != 0u ? 1 : 0);

                        dE += energyOf(tb, int(ca)) - energyOf(ta, int(ca));
                        dE += energyOf(ta, int(cb)) - energyOf(tb, int(cb));

                        uint na[6];
                        uint nb[6];
                        #pragma unroll
                        for (uint k = 0u; k < 6u; ++k) {
                            na[k] = neighbor(a, k);
                            uint tq = typeOf(mys, na[k]);
                            int cq = int(countOf(mys, na[k]));
                            dE += energyOf(tq, cq + d) - energyOf(tq, cq);

                            nb[k] = neighbor(b, k);
                            uint tr = typeOf(mys, nb[k]);
                            int cr = int(countOf(mys, nb[k]));
                            dE += energyOf(tr, cr - d) - energyOf(tr, cr);
                        }

                        acc = (dE <= 0) ||
                              (dE + THRESH_OFFSET < THRESH_COUNT &&
                               uBits < thresh[dE + THRESH_OFFSET]);
                        if (acc) {
                            // Kept lanes have pairwise disjoint closed
                            // neighbourhoods, so these stores are race-free
                            // and equal a lane-order serial commit.
                            // Endpoints exchange types; their counts stay.
                            setSite(mys, a, tb, ca);
                            setSite(mys, b, ta, cb);
                            #pragma unroll
                            for (uint k = 0u; k < 6u; ++k) {
                                uint q = na[k];
                                setSite(mys, q, typeOf(mys, q),
                                        uint(int(countOf(mys, q)) + d));
                                uint p = nb[k];
                                setSite(mys, p, typeOf(mys, p),
                                        uint(int(countOf(mys, p)) - d));
                            }
                            myAccepted += 1u;
                        }
                    }
                }

                if (traceOn != 0u && chain == 0u) {
                    uint idx = r * TABSTOP + lane;
                    trace[idx].round = r;
                    trace[idx].lane = lane;
                    trace[idx].a = a;
                    trace[idx].b = b;
                    trace[idx].u = uBits;
                    trace[idx].dE = as_type<uint>(dE);
                    trace[idx].kept = kept ? 1u : 0u;
                    trace[idx].accepted = acc ? 1u : 0u;
                }

                deferred = !kept;
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        // All three SIMD groups must finish before the strided store-back.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < 3u * N; i += TG_THREADS) {
            uint lc = i / N;
            uint off = i - lc * N;
            gState[(baseChain + lc) * N + off] = st[i];
        }
        // Reconverge the strided store loop before the SIMD reductions.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint keptSum = simd_sum(myKept);
        uint accSum = simd_sum(myAccepted);
        uint nonNullSum = simd_sum(myNonNullKept);

        if (lane == 0u && chain < C) {
            uint e = 0u;
            for (uint i = 0u; i < N; ++i) {
                e += uint(energyOf(typeOf(mys, i), int(countOf(mys, i))));
            }
            outEnergy[chain] = e;
            outKept[chain] = keptSum;
            outAccepted[chain] = accSum;
            outNonNullKept[chain] = nonNullSum;
        }
    }
    """
}

// ---------------------------------------------------------------------------
// Host-side result and statistics helpers
// ---------------------------------------------------------------------------

struct MismatchReport {
    var count = 0
    var messages: [String] = []
}

func noteMismatch(_ report: inout MismatchReport, _ message: @autoclosure () -> String) {
    report.count += 1
    if report.messages.count < 20 {
        report.messages.append(message())
    }
}

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

struct ChainResult {
    var energy: Int
    var kept: Int
    var accepted: Int
    var nonNullKept: Int
}

struct RunOutput {
    var state: [UInt8]
    var trace: [TraceRec]
    var energy: [UInt32]
    var kept: [UInt32]
    var accepted: [UInt32]
    var nonNullKept: [UInt32]
    var seconds: Double
}

// ---------------------------------------------------------------------------
// Metal runner
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
        guard let fn = library.makeFunction(name: "warp_nonlocal") else {
            fatalError("warp_nonlocal function missing")
        }
        let pipeline = try! device.makeComputePipelineState(function: fn)
        precondition(pipeline.threadExecutionWidth == kSimdWidth,
                     "Metal SIMD width is \(pipeline.threadExecutionWidth), expected 32")
        precondition(pipeline.maxTotalThreadsPerThreadgroup >= kThreadsPerThreadgroup,
                     "threadgroup size \(kThreadsPerThreadgroup) exceeds pipeline limit")
        precondition(MemoryLayout<TraceRec>.stride == 32,
                     "TraceRec stride \(MemoryLayout<TraceRec>.stride), expected 32")
        let N = L * L * L
        precondition(device.maxThreadgroupMemoryLength >= kChainsPerThreadgroup * N,
                     "threadgroup state \(kChainsPerThreadgroup * N) B exceeds device limit")
        self.device = device
        self.queue = queue
        self.pipeline = pipeline
        self.L = L
        self.N = N
    }

    func run(state: [UInt8], chains: Int, rounds: Int, thresholds: [UInt32],
             traceOn: Bool, readState: Bool) -> RunOutput
    {
        precondition(state.count == chains * N)
        precondition(thresholds.count == kThresholdCount)

        let groups = (chains + kChainsPerThreadgroup - 1) / kChainsPerThreadgroup
        let paddedChains = groups * kChainsPerThreadgroup
        var padded = state
        padded.append(contentsOf: [UInt8](
            repeating: 0, count: (paddedChains - chains) * N))

        let stateBuf = makeSharedBuffer(device, padded)

        let traceCount = traceOn ? rounds * kSimdWidth : 1
        let emptyRec = TraceRec(round: 0, lane: 0, a: 0, b: 0,
                                u: 0, dE: 0, kept: 0, accepted: 0)
        let traceBuf = makeSharedBuffer(
            device, [TraceRec](repeating: emptyRec, count: traceCount))

        let energyBuf = makeSharedBuffer(
            device, [UInt32](repeating: 0, count: chains))
        let keptBuf = makeSharedBuffer(
            device, [UInt32](repeating: 0, count: chains))
        let accBuf = makeSharedBuffer(
            device, [UInt32](repeating: 0, count: chains))
        let nonNullBuf = makeSharedBuffer(
            device, [UInt32](repeating: 0, count: chains))
        let threshBuf = makeSharedBuffer(device, thresholds)

        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(stateBuf, offset: 0, index: 0)
        enc.setBuffer(traceBuf, offset: 0, index: 1)
        enc.setBuffer(energyBuf, offset: 0, index: 2)
        enc.setBuffer(keptBuf, offset: 0, index: 3)
        enc.setBuffer(accBuf, offset: 0, index: 4)
        enc.setBuffer(nonNullBuf, offset: 0, index: 5)
        var cChains = UInt32(chains)
        var cRounds = UInt32(rounds)
        var cTrace = UInt32(traceOn ? 1 : 0)
        enc.setBytes(&cChains, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&cRounds, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&cTrace, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBuffer(threshBuf, offset: 0, index: 9)
        enc.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: kThreadsPerThreadgroup, height: 1, depth: 1))
        enc.endEncoding()

        let t0 = CFAbsoluteTimeGetCurrent()
        cmd.commit()
        cmd.waitUntilCompleted()
        let seconds = CFAbsoluteTimeGetCurrent() - t0
        precondition(cmd.error == nil, "command buffer error: \(String(describing: cmd.error))")

        let outState = readState ? readBytes(stateBuf, count: paddedChains * N) : []
        let outTrace = traceOn
            ? readArray(traceBuf, TraceRec.self, count: traceCount) : []
        let outEnergy = readArray(energyBuf, UInt32.self, count: chains)
        let outKept = readArray(keptBuf, UInt32.self, count: chains)
        let outAcc = readArray(accBuf, UInt32.self, count: chains)
        let outNonNull = readArray(nonNullBuf, UInt32.self, count: chains)
        return RunOutput(state: outState, trace: outTrace,
                         energy: outEnergy, kept: outKept,
                         accepted: outAcc, nonNullKept: outNonNull,
                         seconds: seconds)
    }
}

// ---------------------------------------------------------------------------
// CPU reference chain simulation
//
// One function drives both the G1 trace replay and the G2/G3 CPU reference.
// Pass trace == nil for the statistical reference.  With a trace, every lane
// record is compared against the CPU's own proposal, kept, dE and accepted
// values, and accepted commits are applied in lane order.
// ---------------------------------------------------------------------------

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

func simulateChainCPU(
    state: inout [UInt8],
    chain: Int,
    rounds: Int,
    L: Int,
    thresholds: [UInt32],
    grid: ConflictGrid,
    trace: [TraceRec]?,
    report: inout MismatchReport
) -> ChainResult {
    let N = L * L * L
    var pa = [UInt32](repeating: 0, count: 32)
    var pb = [UInt32](repeating: 0, count: 32)
    var pu = [UInt32](repeating: 0, count: 32)
    var deferred = [Bool](repeating: false, count: 32)
    var keptTotal = 0
    var acceptedTotal = 0
    var nonNullKept = 0

    if let trace = trace {
        precondition(trace.count >= rounds * 32,
                     "trace too short for \(rounds) rounds")
    }

    for r in 0..<rounds {
        for j in 0..<32 {
            if !deferred[j] {
                let p = drawProposal(chain: chain, lane: j, round: r,
                                     L: L, N: N)
                pa[j] = p.a
                pb[j] = p.b
                pu[j] = p.u
            }
        }

        grid.beginRound()
        var keptMask: UInt32 = 0
        for j in 0..<32 {
            let a = Int(pa[j])
            let b = Int(pb[j])
            if !grid.isBlocked(a) && !grid.isBlocked(b) {
                keptMask |= (UInt32(1) << UInt32(j))
                grid.mark(a)
                grid.mark(b)
            }
        }

        for j in 0..<32 {
            let isKept = ((keptMask >> UInt32(j)) & 1) == 1
            var dE = 0
            var acc = false
            if isKept {
                let a = Int(pa[j])
                let b = Int(pb[j])
                let ta = typeAt(state, a)
                let tb = typeAt(state, b)
                if ta != tb {
                    dE = delta(state, a, b, L)
                    if dE <= 0 {
                        acc = true
                    } else if dE + kThresholdOffset < thresholds.count {
                        acc = pu[j] < thresholds[dE + kThresholdOffset]
                    }
                }
                keptTotal += 1
                if ta != tb { nonNullKept += 1 }
                if acc {
                    acceptedTotal += 1
                    commit(&state, a, b, L)
                }
            }

            if let trace = trace {
                let rec = trace[r * 32 + j]
                if rec.round != UInt32(r) {
                    noteMismatch(&report,
                        "round \(r) lane \(j): rec.round=\(rec.round)")
                }
                if rec.lane != UInt32(j) {
                    noteMismatch(&report,
                        "round \(r) lane \(j): rec.lane=\(rec.lane)")
                }
                if rec.a != pa[j] || rec.b != pb[j] || rec.u != pu[j] {
                    noteMismatch(&report,
                        "round \(r) lane \(j): proposal cpu=(\(pa[j]),\(pb[j]),\(pu[j])) gpu=(\(rec.a),\(rec.b),\(rec.u))")
                }
                if (rec.kept != 0) != isKept {
                    noteMismatch(&report,
                        "round \(r) lane \(j): kept cpu=\(isKept) gpu=\(rec.kept)")
                }
                let recDE = UInt32(bitPattern: Int32(dE))
                if rec.dE != recDE {
                    noteMismatch(&report,
                        "round \(r) lane \(j): dE cpu=\(dE) gpu=\(Int32(bitPattern: rec.dE))")
                }
                if (rec.accepted != 0) != acc {
                    noteMismatch(&report,
                        "round \(r) lane \(j): accepted cpu=\(acc) gpu=\(rec.accepted)")
                }
            }
        }

        for j in 0..<32 {
            deferred[j] = ((keptMask >> UInt32(j)) & 1) == 0
        }
    }

    return ChainResult(energy: fullEnergy(state, L), kept: keptTotal,
                       accepted: acceptedTotal, nonNullKept: nonNullKept)
}

// ---------------------------------------------------------------------------
// G1: GPU trace vs CPU replay at L=8 and L=21
// ---------------------------------------------------------------------------

func gate1(_ r8: MetalRunner, _ r21: MetalRunner) -> Bool {
    print("=== G1: warp-batched nonlocal trace replay (1 chain, 10,000 rounds) ===")
    var allPass = true
    let rounds = 10000
    let beta = 2.8
    let th = makeThresholds(beta: beta)
    let cases: [(Int, MetalRunner)] = [(8, r8), (21, r21)]

    for (L, runner) in cases {
        let N = L * L * L
        let initial = makeInitialState(L: L, chains: 1, seed: 1)
        let gpu = runner.run(state: initial, chains: 1, rounds: rounds,
                             thresholds: th, traceOn: true, readState: true)
        var st = initial
        let grid = ConflictGrid(L: L)
        var report = MismatchReport()
        let cpu = simulateChainCPU(state: &st, chain: 0, rounds: rounds,
                                   L: L, thresholds: th, grid: grid,
                                   trace: gpu.trace, report: &report)
        let gpuState = Array(gpu.state[0..<N])
        let bytesEqual = (gpuState == st)
        let gpuEnergy = Int(gpu.energy[0])
        let energyEqual = (cpu.energy == gpuEnergy)
        let traceCountOK = (gpu.trace.count == rounds * 32)
        let pass = report.count == 0 && bytesEqual && energyEqual && traceCountOK

        print("L=\(L) rounds=\(rounds) records=\(gpu.trace.count) " +
              "mismatches=\(report.count) bytes_equal=\(bytesEqual ? "YES" : "NO") " +
              "energy cpu=\(cpu.energy) gpu=\(gpuEnergy) " +
              "cpu_kept=\(cpu.kept) cpu_accepted=\(cpu.accepted)")
        for m in report.messages { print("  " + m) }
        if !pass {
            print("  G1 L=\(L) FAIL")
        }
        allPass = allPass && pass
    }
    print("G1 \(allPass ? "PASS" : "FAIL")")
    return allPass
}

// ---------------------------------------------------------------------------
// G2/G3: 256 chains, beta 2.8 and 3.0, CPU reference vs GPU
// ---------------------------------------------------------------------------

func gate2And3(_ runner: MetalRunner, chains: Int = 256, rounds: Int = 2048) -> Bool {
    print("=== G2: 256 chains CPU reference vs GPU (L=21) ===")
    print("=== G3: mean kept lanes per round (L=21) ===")
    let L = 21
    let N = L * L * L
    let initial = makeInitialState(L: L, chains: chains, seed: 1)
    let grid = ConflictGrid(L: L)
    var allPass = true

    for beta in [2.8, 3.0] {
        let th = makeThresholds(beta: beta)
        let gpu = runner.run(state: initial, chains: chains, rounds: rounds,
                             thresholds: th, traceOn: false, readState: true)

        var cpuStates = [UInt8](repeating: 0, count: chains * N)
        var cpuEnergy = [Int](repeating: 0, count: chains)
        var cpuKept = [Int](repeating: 0, count: chains)
        var cpuAccepted = [Int](repeating: 0, count: chains)
        var cpuNonNull = [Int](repeating: 0, count: chains)
        var dummy = MismatchReport()

        for c in 0..<chains {
            var st = Array(initial[c * N..<(c + 1) * N])
            let res = simulateChainCPU(state: &st, chain: c, rounds: rounds,
                                       L: L, thresholds: th, grid: grid,
                                       trace: nil, report: &dummy)
            cpuStates.replaceSubrange(c * N..<(c + 1) * N, with: st)
            cpuEnergy[c] = res.energy
            cpuKept[c] = res.kept
            cpuAccepted[c] = res.accepted
            cpuNonNull[c] = res.nonNullKept
        }

        let gpuState = Array(gpu.state[0..<(chains * N)])
        let stateExact = (gpuState == cpuStates)
        var energyMismatch = 0
        var keptMismatch = 0
        var acceptedMismatch = 0
        var nonNullMismatch = 0
        for c in 0..<chains {
            if Int(gpu.energy[c]) != cpuEnergy[c] { energyMismatch += 1 }
            if Int(gpu.kept[c]) != cpuKept[c] { keptMismatch += 1 }
            if Int(gpu.accepted[c]) != cpuAccepted[c] { acceptedMismatch += 1 }
            if Int(gpu.nonNullKept[c]) != cpuNonNull[c] { nonNullMismatch += 1 }
        }

        let cpuE = cpuEnergy.map { Double($0) }
        let gpuE = gpu.energy.map { Double($0) }
        let cpuAccRetired = (0..<chains).map {
            cpuKept[$0] > 0 ? Double(cpuAccepted[$0]) / Double(cpuKept[$0]) : 0.0
        }
        let gpuAccRetired = (0..<chains).map {
            gpu.kept[$0] > 0
                ? Double(gpu.accepted[$0]) / Double(gpu.kept[$0]) : 0.0
        }
        let cpuAccPhysical = (0..<chains).map {
            cpuNonNull[$0] > 0
                ? Double(cpuAccepted[$0]) / Double(cpuNonNull[$0]) : 0.0
        }
        let gpuAccPhysical = (0..<chains).map {
            gpu.nonNullKept[$0] > 0
                ? Double(gpu.accepted[$0]) / Double(gpu.nonNullKept[$0]) : 0.0
        }

        let cE = stats(cpuE)
        let gE = stats(gpuE)
        let cA = stats(cpuAccRetired)
        let gA = stats(gpuAccRetired)
        let cP = stats(cpuAccPhysical)
        let gP = stats(gpuAccPhysical)

        let seE = sqrt(cE.se * cE.se + gE.se * gE.se)
        let seA = sqrt(cA.se * cA.se + gA.se * gA.se)
        let seP = sqrt(cP.se * cP.se + gP.se * gP.se)
        let e3 = abs(cE.mean - gE.mean) <= 3 * seE || energyMismatch == 0
        let a3 = abs(cA.mean - gA.mean) <= 3 * seA || acceptedMismatch == 0
        let p3 = abs(cP.mean - gP.mean) <= 3 * seP || nonNullMismatch == 0

        let statePass = stateExact && energyMismatch == 0 &&
                        keptMismatch == 0 && acceptedMismatch == 0 &&
                        nonNullMismatch == 0
        let pass = statePass && e3 && a3 && p3
        allPass = allPass && pass

        let eFlag = (abs(cE.mean - gE.mean) <= 3 * seE) ? "yes" : "no"
        let aFlag = (abs(cA.mean - gA.mean) <= 3 * seA) ? "yes" : "no"
        let pFlag = (abs(cP.mean - gP.mean) <= 3 * seP) ? "yes" : "no"
        print(String(format:
            "beta=%.1f  E: cpu %.3f [%.3f, %.3f]  gpu %.3f [%.3f, %.3f]",
            beta, cE.mean, cE.lo3, cE.hi3, gE.mean, gE.lo3, gE.hi3)
            + "  within3sigma=\(eFlag)")
        print(String(format:
            "           acc(retired): cpu %.6f [%.6f, %.6f]  gpu %.6f [%.6f, %.6f]",
            cA.mean, cA.lo3, cA.hi3, gA.mean, gA.lo3, gA.hi3)
            + "  within3sigma=\(aFlag)")
        print(String(format:
            "           acc(non-null): cpu %.6f [%.6f, %.6f]  gpu %.6f [%.6f, %.6f]",
            cP.mean, cP.lo3, cP.hi3, gP.mean, gP.lo3, gP.hi3)
            + "  within3sigma=\(pFlag)")
        print("           per-chain mismatches: E=\(energyMismatch) kept=\(keptMismatch) " +
              "accepted=\(acceptedMismatch) nonnull=\(nonNullMismatch) " +
              "state_exact=\(stateExact ? "YES" : "NO")")
        print("           G2 beta=\(beta) \(pass ? "PASS" : "FAIL")")

        let keptPerRound = Double(gpu.kept.reduce(0) { $0 + Int($1) })
            / Double(chains * rounds)
        let inExpected = keptPerRound >= 22.0 && keptPerRound <= 29.0
        print(String(format:
            "G3 beta=%.1f  kept/round=%.4f (%.2f%% of 32) expected ~25/32",
            beta, keptPerRound, 100.0 * keptPerRound / 32.0)
            + "  " + (inExpected ? "within [22,29]" : "outside [22,29]"))
        allPass = allPass && inExpected
    }
    print("G2 \(allPass ? "PASS" : "FAIL")")
    return allPass
}

// ---------------------------------------------------------------------------
// Benchmark: kept nonlocal proposals/s, 1/8/64/256 chains, L=21
// ---------------------------------------------------------------------------

func benchmark(_ runner: MetalRunner) {
    print("=== Benchmark: kept nonlocal proposals/s (L=21, beta 2.8) ===")
    let L = 21
    let N = L * L * L
    let rounds = 8192
    let th = makeThresholds(beta: 2.8)
    let initial = makeInitialState(L: L, chains: 1024, seed: 1)

    for chains in [1, 8, 64, 256, 1024] {
        let state = Array(initial[0..<(chains * N)])
        // Warm the pipeline / allocator; the measured run starts from the
        // same input state, so its proposal stream is deterministic.
        _ = runner.run(state: state, chains: chains,
                       rounds: max(1, rounds / 4), thresholds: th,
                       traceOn: false, readState: false)
        let r = runner.run(state: state, chains: chains, rounds: rounds,
                           thresholds: th, traceOn: false, readState: false)
        let kept = r.kept.reduce(0) { $0 + Int($1) }
        let accepted = r.accepted.reduce(0) { $0 + Int($1) }
        let proposed = chains * rounds * 32
        let keptPerSec = Double(kept) / r.seconds
        let propPerSec = Double(proposed) / r.seconds
        let accPerSec = Double(accepted) / r.seconds
        print("chains=\(chains) rounds=\(rounds)  " + String(format:
            "time=%.6f s  kept=%.3f G/s  proposed=%.3f G/s  accepted=%.3f G/s  kept/local=%.4fx",
            r.seconds, keptPerSec / 1e9, propPerSec / 1e9,
            accPerSec / 1e9, keptPerSec / kLocalBaselinePerSecond))
    }
    print("(local baseline: \(kLocalBaselinePerSecond / 1e9) G/s local-bond proposals)")
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

let mode = CommandLine.arguments.count > 1 ? CommandLine.arguments[1].lowercased() : "all"

let needG1 = (mode == "all" || mode == "g1")
let needG2 = (mode == "all" || mode == "g2" || mode == "g3")
// Benchmark is only part of the "all" flow and only runs after both gates.
let needBench = (mode == "all")

var g1Pass = true
var g2Pass = true

if needG1 {
    let r8 = MetalRunner(L: 8)
    let r21 = runner21()
    g1Pass = gate1(r8, r21)
}

if needG2 {
    g2Pass = gate2And3(runner21())
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
    print("=== SUMMARY: G1 \(g1Pass ? "PASS" : "FAIL"), G2/G3 \(g2Pass ? "PASS" : "FAIL") ===")
    if !pass { exit(1) }
}




