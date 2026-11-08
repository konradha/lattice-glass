import Foundation
import Metal

// ---------------------------------------------------------------------------
// Optimized persistent single-CTA chain for the L=21 local lattice glass.
//
// Same model as poc/local_chain.swift (21 colours x 3 axes, 441 edges/colour,
// fp32 Metropolis on the exact integer dE, same splitmix acceptance hash, same
// threadgroup-resident state, same CPU coloured validation). What changed:
//
//   * one byte per site packs type (bits 0..1) and neighbour count (bits 2..4),
//     so the edge path loads one byte per site instead of two;
//   * the whole (axis, colour) plan is unrolled with axis and colour as
//     compile-time constants, so the anchor site arithmetic, the partner
//     direction and every torus wrap becomes a constant-folded add/select;
//   * the 12-site affinity list is held in named scalars (no local array, no
//     dynamic index), neighbours are derived from the anchor coordinates
//     instead of calling a div/mod based neighbour routine;
//   * the ten non-endpoint energy deltas use the exact expansion
//     (c+d-conn)^2 - (c-conn)^2 of the same integer energy.
//
// Coloured model note: the two endpoints of a bond swap occupancy, so a's own
// count (which tracks b) changes by -d and b's by +d, while their ten other
// affected neighbours take += d near a and -= d near b, exactly as in
// poc/local_beta.swift. That keeps c[s] equal to the six-neighbour occupancy
// popcount, so every dE is the true local energy.
// ---------------------------------------------------------------------------

let L = 21
let N = 9261
let TG = 512

func site(_ x: Int, _ y: Int, _ z: Int) -> Int { x + L*y + L*L*z }
func neigh(_ s: Int, _ d: Int) -> Int {
    var x = s % L, y = (s / L) % L, z = s / (L*L)
    switch d {
    case 0: x = (x + 1) % L
    case 1: x = (x + L - 1) % L
    case 2: y = (y + 1) % L
    case 3: y = (y + L - 1) % L
    case 4: z = (z + 1) % L
    default: z = (z + L - 1) % L
    }
    return site(x, y, z)
}
func smix(_ x0: UInt32) -> UInt32 {
    var x = x0
    x ^= x >> 16; x &*= 0x7feb352d
    x ^= x >> 15; x &*= 0x846ca68b
    x ^= x >> 16
    return x
}
@inline(__always) func hash32(_ c: Int, _ sw: Int, _ ax: Int, _ co: Int, _ t: Int) -> UInt32 {
    var h = UInt32(truncatingIfNeeded: c) &* 0x9e3779b9
    h ^= UInt32(truncatingIfNeeded: sw) &* 0x85ebca6b
    h ^= UInt32(truncatingIfNeeded: ax) &* 0xc2b2ae35
    h ^= UInt32(truncatingIfNeeded: co) &* 0x27d4eb2f
    h ^= UInt32(truncatingIfNeeded: t) &* 0x165667b1
    return smix(h)
}
@inline(__always) func energy(_ t: Int, _ c: Int) -> Int {
    if t == 0 { return 0 }
    let d = t == 1 ? c - 3 : c - 5
    return d*d
}
func initState() -> ([UInt8], [UInt8]) {
    var t = [UInt8](repeating: 0, count: N)
    var c = [UInt8](repeating: 0, count: N)
    var p = Array(0..<N)
    var x: UInt64 = 1000
    for i in stride(from: N-1, through: 1, by: -1) {
        x &+= 0x9e3779b97f4a7c15
        var z = x
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        z ^= z >> 31
        let j = Int(z % UInt64(i+1))
        p.swapAt(i, j)
    }
    for i in 0..<2315 { t[p[i]] = 0 }
    for i in 2315..<5093 { t[p[i]] = 1 }
    for i in 5093..<N { t[p[i]] = 2 }
    for s in 0..<N {
        var k = 0
        for d in 0..<6 { if t[neigh(s, d)] != 0 { k += 1 } }
        c[s] = UInt8(k)
    }
    return (t, c)
}
func packState(_ t: [UInt8], _ c: [UInt8]) -> [UInt8] {
    var p = [UInt8](repeating: 0, count: N)
    for i in 0..<N { p[i] = UInt8((t[i] & 3) | (c[i] << 2)) }
    return p
}
func digest(_ p: [UInt8]) -> UInt64 {
    var h: UInt64 = 0xcbf29ce484222325
    for b in p { h = (h ^ UInt64(b)) &* 0x100000001b3 }
    return h
}

// ---------------------------------------------------------------------------
// Metal kernel
// ---------------------------------------------------------------------------

let metalSrc = #"""
#include <metal_stdlib>
using namespace metal;

// Packed site byte: bits 0..1 type (0 vacancy, 1 and 2 species),
//                    bits 2..4 occupied-neighbour count (0..6).

inline uint smix(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16; return x;
}
inline uint h32(uint c, uint sw, uint ax, uint co, uint t) {
    uint h = c * 0x9e3779b9u;
    h ^= sw * 0x85ebca6bu;
    h ^= ax * 0xc2b2ae35u;
    h ^= co * 0x27d4eb2fu;
    h ^= t  * 0x165667b1u;
    return smix(h);
}

#define EN_E(T, C) (((T) == 0) ? 0 : (((T) == 1) ? (((C) - 3) * ((C) - 3)) : (((C) - 5) * ((C) - 5))))

// One edge colour: 441 independent bonds, one per thread.
template <uint AX, uint CO>
inline void edgeStep(threadgroup uchar *st, uint t, float beta, uint sw) {
    const uint p1 = t / 21u;
    const uint p2 = t - 21u * p1;
    int p0i = int(CO) - 3 * int(p1) - 5 * int(p2);
    p0i %= 21;
    p0i += (p0i < 0) ? 21 : 0;
    const uint p0 = uint(p0i);

    // Anchor coordinates, rotated by the (constant) axis.
    const uint x = (AX == 0u) ? p0 : ((AX == 1u) ? p2 : p1);
    const uint y = (AX == 0u) ? p1 : ((AX == 1u) ? p0 : p2);
    const uint z = (AX == 0u) ? p2 : ((AX == 1u) ? p1 : p0);
    const uint a = x + 21u * y + 441u * z;

    // Partner b = nbr(a, AX): +x on axis 0, -x on axis 1, +y on axis 2.
    uint bx = x, by = y;
    if (AX == 0u)      bx = (x == 20u) ? 0u : x + 1u;
    else if (AX == 1u) bx = (x == 0u) ? 20u : x - 1u;
    else               by = (y == 20u) ? 0u : y + 1u;
    const uint b = bx + 21u * by + 441u * z;

    const uint pa = uint(st[a]);
    const uint pb = uint(st[b]);
    const uint ta = pa & 3u;
    const uint tb = pb & 3u;
    if (ta == tb) return;
    const int dl = int(tb != 0u) - int(ta != 0u);

    // Neighbours of a and of b, as constant-folded torus wraps.
    const uint n0 = (x == 20u) ? a - 20u : a + 1u;
    const uint n1 = (x == 0u)  ? a + 20u : a - 1u;
    const uint n2 = (y == 20u) ? a - 420u : a + 21u;
    const uint n3 = (y == 0u)  ? a + 420u : a - 21u;
    const uint n4 = (z == 20u) ? a - 8820u : a + 441u;
    const uint n5 = (z == 0u)  ? a + 8820u : a - 441u;
    const uint m0 = (bx == 20u) ? b - 20u : b + 1u;
    const uint m1 = (bx == 0u)  ? b + 20u : b - 1u;
    const uint m2 = (by == 20u) ? b - 420u : b + 21u;
    const uint m3 = (by == 0u)  ? b + 420u : b - 21u;
    const uint m4 = (z == 20u)  ? b - 8820u : b + 441u;
    const uint m5 = (z == 0u)   ? b + 8820u : b - 441u;

    // Slots 2..6: five neighbours of a in direction order, a's partner dropped.
    const uint s2 = (AX == 0u) ? n1 : n0;
    const uint s3 = (AX == 2u) ? n1 : n2;
    const uint s4 = n3;
    const uint s5 = n4;
    const uint s6 = n5;
    // Slots 7..11: five neighbours of b, b's partner dropped.
    const uint s7  = (AX == 1u) ? m1 : m0;
    const uint s8  = (AX == 2u) ? m1 : m2;
    const uint s9  = (AX == 2u) ? m2 : m3;
    const uint s10 = m4;
    const uint s11 = m5;

    int de = 0;
    uchar nv0, nv1, nv2, nv3, nv4, nv5, nv6, nv7, nv8, nv9, nv10, nv11;

#define ACC_END(K, PK, OT, NEWT, DLT) {                              \
        const int oc = int(((PK) >> 2u) & 7u);                       \
        const int cn = oc + (DLT);                                   \
        de += EN_E((NEWT), cn) - EN_E((OT), oc);                     \
        nv##K = uchar(uint(NEWT) | (uint(cn) << 2));                 \
    }
#define ACC_NB(K, SITE, DLT) {                                       \
        const uint z_ = (SITE);                                      \
        const uint pz = uint(st[z_]);                                \
        const uint ot = pz & 3u;                                     \
        const int oc = int((pz >> 2u) & 7u);                         \
        const int cn = oc + (DLT);                                   \
        de += EN_E(int(ot), cn) - EN_E(int(ot), oc);                 \
        nv##K = uchar(ot | (uint(cn) << 2));                         \
    }

    // a and b swap occupancy, so c[a] = oc - dl and c[b] = oc + dl: each
    // endpoint's count tracks the other's new occupancy (poc/local_beta.swift).
    ACC_END(0, pa, int(ta), int(tb), -dl);
    ACC_END(1, pb, int(tb), int(ta),  dl);
    ACC_NB(2, s2,  dl);
    ACC_NB(3, s3,  dl);
    ACC_NB(4, s4,  dl);
    ACC_NB(5, s5,  dl);
    ACC_NB(6, s6,  dl);
    ACC_NB(7, s7, -dl);
    ACC_NB(8, s8, -dl);
    ACC_NB(9, s9, -dl);
    ACC_NB(10, s10, -dl);
    ACC_NB(11, s11, -dl);

#undef ACC_END
#undef ACC_NB

    const uint hh = h32(0u, sw, AX, CO, t);
    const float u = float(hh) * (1.0f / 4294967296.0f);
    if (de <= 0 || u < exp(-beta * float(de))) {
        st[a] = nv0;
        st[b] = nv1;
        st[s2] = nv2;
        st[s3] = nv3;
        st[s4] = nv4;
        st[s5] = nv5;
        st[s6] = nv6;
        st[s7] = nv7;
        st[s8] = nv8;
        st[s9] = nv9;
        st[s10] = nv10;
        st[s11] = nv11;
    }
}

#define RUN_COLOR(A, C) {                                            \
        if (tid < 441u) edgeStep<A, C>(st, tid, beta, sw);           \
        threadgroup_barrier(mem_flags::mem_threadgroup);             \
    }

template <uint AX>
inline void runAxis(threadgroup uchar *st, uint tid, float beta, uint sw) {
    RUN_COLOR(AX, 0);
    RUN_COLOR(AX, 1);
    RUN_COLOR(AX, 2);
    RUN_COLOR(AX, 3);
    RUN_COLOR(AX, 4);
    RUN_COLOR(AX, 5);
    RUN_COLOR(AX, 6);
    RUN_COLOR(AX, 7);
    RUN_COLOR(AX, 8);
    RUN_COLOR(AX, 9);
    RUN_COLOR(AX, 10);
    RUN_COLOR(AX, 11);
    RUN_COLOR(AX, 12);
    RUN_COLOR(AX, 13);
    RUN_COLOR(AX, 14);
    RUN_COLOR(AX, 15);
    RUN_COLOR(AX, 16);
    RUN_COLOR(AX, 17);
    RUN_COLOR(AX, 18);
    RUN_COLOR(AX, 19);
    RUN_COLOR(AX, 20);
}

#undef RUN_COLOR

kernel void chain(device uchar *g [[buffer(0)]],
                  constant uint &sweeps [[buffer(1)]],
                  constant float &beta [[buffer(2)]],
                  uint tid [[thread_position_in_threadgroup]]) {
    threadgroup uchar st[9261];
    for (uint i = tid; i < 9261u; i += 512u) st[i] = g[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint sw = 0; sw < sweeps; ++sw) {
        runAxis<0>(st, tid, beta, sw);
        runAxis<1>(st, tid, beta, sw);
        runAxis<2>(st, tid, beta, sw);
    }
    for (uint i = tid; i < 9261u; i += 512u) g[i] = st[i];
}
"""#

guard let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else { fatalError("no metal") }
let lib = try dev.makeLibrary(source: metalSrc, options: nil)
let pso = try dev.makeComputePipelineState(function: lib.makeFunction(name: "chain")!)
precondition(pso.maxTotalThreadsPerThreadgroup >= TG)
print("device=\(dev.name) maxTG=\(pso.maxTotalThreadsPerThreadgroup)")

let betaVal: Float = CommandLine.arguments.count > 1 ? (Float(CommandLine.arguments[1]) ?? 2.8) : 2.8

func run(_ buf: MTLBuffer, _ sweeps: UInt32) {
    let cmd = q.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(buf, offset: 0, index: 0)
    var s = sweeps
    var b = betaVal
    enc.setBytes(&s, length: 4, index: 1)
    enc.setBytes(&b, length: 4, index: 2)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: TG, height: 1, depth: 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
}

// ---------------------------------------------------------------------------
// Benchmark: one dispatch of 1/10/100/1000 sweeps, each from the same start
// state, reporting proposals/s (3 axes * 21 colours * 441 edges = 3N per sweep).
// The 1-sweep figure is dominated by encode/dispatch latency.
// ---------------------------------------------------------------------------

let (t0, c0) = initState()
let initPacked = packState(t0, c0)

run(dev.makeBuffer(bytes: initPacked, length: N, options: .storageModeShared)!, 1) // warm up

var benchBuf: MTLBuffer? = nil
for s in [1, 10, 100, 1000] {
    let buf = dev.makeBuffer(bytes: initPacked, length: N, options: .storageModeShared)!
    let t = CFAbsoluteTimeGetCurrent()
    run(buf, UInt32(s))
    let dt = CFAbsoluteTimeGetCurrent() - t
    let prop = Double(s) * 3.0 * Double(N)
    print(String(format: "sweeps=%4d time=%.5fs proposals=%.3e proposals/s=%.3f M/s",
                 s, dt, prop, prop / dt / 1e6))
    benchBuf = buf
}

// ---------------------------------------------------------------------------
// CPU coloured reference (packed state, identical update rule and hash).
// ---------------------------------------------------------------------------

func anchor(_ axis: Int, _ color: Int, _ t: Int) -> (Int, Int) {
    let p1 = t / 21, p2 = t % 21
    var p0 = color - 3*p1 - 5*p2
    p0 %= 21
    if p0 < 0 { p0 += 21 }
    var q = [0, 0, 0]
    q[axis] = p0
    q[(axis+1)%3] = p1
    q[(axis+2)%3] = p2
    let a = site(q[0], q[1], q[2])
    return (a, neigh(a, axis))
}
func applyCPU(_ st: inout [UInt8], _ ax: Int, _ co: Int, _ sw: Int, _ beta: Float) {
    for t in 0..<441 {
        let (a, b) = anchor(ax, co, t)
        let ta = Int(st[a] & 3), tb = Int(st[b] & 3)
        if ta == tb { continue }
        let d = (tb != 0 ? 1 : 0) - (ta != 0 ? 1 : 0)
        var aff = [a, b]
        for j in 0..<6 { let z = neigh(a, j); if z != b { aff.append(z) } }
        for j in 0..<6 { let z = neigh(b, j); if z != a { aff.append(z) } }
        var nt = [Int](repeating: 0, count: 12), nc = [Int](repeating: 0, count: 12), de = 0
        for k in 0..<12 {
            let z = aff[k]
            let pz = st[z]
            let ot = Int(pz & 3), oc = Int((pz >> 2) & 7)
            let tn = (z == a) ? tb : ((z == b) ? ta : ot)
            var cn = oc
            if k == 0 { cn -= d } else if k == 1 { cn += d } else if k < 7 { cn += d } else { cn -= d }
            de += energy(tn, cn) - energy(ot, oc)
            nt[k] = tn; nc[k] = cn
        }
        let hh = hash32(0, sw, ax, co, t)
        let u = Float(hh) * (1.0/4294967296.0)
        if de <= 0 || u < exp(-beta*Float(de)) {
            for k in 0..<12 { st[aff[k]] = UInt8(nt[k] | (nc[k] << 2)) }
        }
    }
}
func validateChain() {
    let (it, ic) = initState()
    var cpu = packState(it, ic)
    let buf = dev.makeBuffer(bytes: cpu, length: N, options: .storageModeShared)!
    run(buf, 2)
    for s in 0..<2 { for ax in 0..<3 { for co in 0..<21 { applyCPU(&cpu, ax, co, s, betaVal) } } }
    let gp = buf.contents().bindMemory(to: UInt8.self, capacity: N)
    var bad = 0
    for i in 0..<N where gp[i] != cpu[i] { bad += 1 }
    print("optimized persistent validate mismatches=\(bad) digest=\(String(format: "%016llx", digest(cpu)))")
}

validateChain()

let bp = benchBuf!.contents().bindMemory(to: UInt8.self, capacity: N)
var finalE = 0
for i in 0..<N { finalE += energy(Int(bp[i] & 3), Int((bp[i] >> 2) & 7)) }
print("bench sweeps=1000 energy=\(finalE)")
