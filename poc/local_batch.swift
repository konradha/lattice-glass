import Foundation
import Metal

// ---------------------------------------------------------------------------
// local_batch.swift — launch-strategy benchmark for the per-color local kernel.
//
// The MSL below is local_metal.swift's kernel (both endpoints of a swap hold
// each other's type, so their own counts move too), so every number here is
// directly comparable to that file's.
//
// Schedule: 21 colors per axis x 3 axes, 441 edges per color -> 63 dispatches
// per 3-axis sweep. Colors of one axis partition that axis' edges, but their
// 12-site affected sets overlap, so the colored schedule is strictly
// sequential: color k must observe every accepted update of color k-1, and
// sweep s+1 must observe sweep s.
//
// Launch modes
//   cb-per-color    one command buffer + host wait per dispatch (baseline from
//                   local_metal.swift): 63 command buffers and 63 host waits
//                   per sweep.
//   batch-encoders  one command buffer per full 63-dispatch sweep, each
//                   dispatch in its own compute encoder (the encoder boundary
//                   carries the dependency), no host wait between colors.
//   batch-barrier   one command buffer per sweep, all 63 dispatches in a single
//                   compute encoder separated by memoryBarrier(scope: .buffers).
//
// Memory ordering: buffers are MTLResource default (.tracked) shared-storage
// buffers; a single compute encoder does not order consecutive dispatches that
// touch the same buffer, hence the explicit memoryBarrier in batch-barrier.
// Across encoders and across command buffers on one queue the tracker orders
// access, which is what batch-encoders relies on.
//
// Exactness: the batched modes must reproduce the per-color mode bit for bit.
// C=1 is checked site by site against the CPU colored reference (identical
// (sweep, axis, color) order) after each of 2 sweeps; every (mode, C) run is
// also cross-checked by FNV-1a state checksum and total energy.
// ---------------------------------------------------------------------------

let N = 9261
let L = 21

func site(_ x: Int, _ y: Int, _ z: Int) -> Int { x + L * y + L * L * z }

func neigh(_ s: Int, _ d: Int) -> Int {
    var x = s % L, y = (s / L) % L, z = s / (L * L)
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

func hash32(_ c: Int, _ sw: Int, _ ax: Int, _ co: Int, _ t: Int) -> UInt32 {
    var h = UInt32(truncatingIfNeeded: c) &* 0x9e3779b9
    h ^= UInt32(truncatingIfNeeded: sw) &* 0x85ebca6b
    h ^= UInt32(truncatingIfNeeded: ax) &* 0xc2b2ae35
    h ^= UInt32(truncatingIfNeeded: co) &* 0x27d4eb2f
    h ^= UInt32(truncatingIfNeeded: t) &* 0x165667b1
    return smix(h)
}

func energy(_ t: Int, _ c: Int) -> Int {
    if t == 0 { return 0 }
    let d = t == 1 ? c - 3 : c - 5
    return d * d
}

func initState(_ configs: Int, _ seed: UInt64) -> ([UInt8], [UInt8]) {
    var types = [UInt8](repeating: 0, count: configs * N)
    var counts = [UInt8](repeating: 0, count: configs * N)
    for c in 0..<configs {
        var perm = Array(0..<N)
        var x = seed &+ UInt64(c) &* 0x9e3779b97f4a7c15
        for i in stride(from: N - 1, through: 1, by: -1) {
            x &+= 0x9e3779b97f4a7c15
            var z = x
            z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
            z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
            z ^= z >> 31
            let j = Int(z % UInt64(i + 1))
            perm.swapAt(i, j)
        }
        let n0 = 2315, n1 = 2778
        for i in 0..<n0 { types[c * N + perm[i]] = 0 }
        for i in n0..<(n0 + n1) { types[c * N + perm[i]] = 1 }
        for i in (n0 + n1)..<N { types[c * N + perm[i]] = 2 }
        for s in 0..<N {
            var cnt = 0
            for d in 0..<6 { if types[c * N + neigh(s, d)] != 0 { cnt += 1 } }
            counts[c * N + s] = UInt8(cnt)
        }
    }
    return (types, counts)
}

let metalSrc = #"""
#include <metal_stdlib>
using namespace metal;
#define L 21
#define N 9261
inline uint smix(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16; return x;
}
inline uint hash32(uint c, uint sw, uint ax, uint co, uint t) {
    uint h = c * 0x9e3779b9u;
    h ^= sw * 0x85ebca6bu;
    h ^= ax * 0xc2b2ae35u;
    h ^= co * 0x27d4eb2fu;
    h ^= t * 0x165667b1u;
    return smix(h);
}
inline int energy(int t, int c) {
    if (t == 0) return 0;
    int d = (t == 1) ? (c - 3) : (c - 5);
    return d*d;
}
inline uint siteOf(uint x, uint y, uint z) { return x + 21u*y + 441u*z; }
inline uint neigh(uint s, uint d) {
    int x = int(s % 21u), y = int((s / 21u) % 21u), z = int(s / 441u);
    if (d == 0) x = (x + 1) % 21;
    else if (d == 1) x = (x + 20) % 21;
    else if (d == 2) y = (y + 1) % 21;
    else if (d == 3) y = (y + 20) % 21;
    else if (d == 4) z = (z + 1) % 21;
    else z = (z + 20) % 21;
    return siteOf(uint(x),uint(y),uint(z));
}
kernel void local_color(device uchar *typ [[buffer(0)]],
                        device uchar *cnt [[buffer(1)]],
                        constant uint &C [[buffer(2)]],
                        constant uint &axis [[buffer(3)]],
                        constant uint &color [[buffer(4)]],
                        constant uint &sweep [[buffer(5)]],
                        constant float &beta [[buffer(6)]],
                        uint gid [[thread_position_in_grid]]) {
    uint t = gid % 441u;
    uint c = gid / 441u;
    if (c >= C) return;
    uint p1 = t / 21u, p2 = t % 21u;
    int p0 = int(color) - 3*int(p1) - 5*int(p2);
    p0 %= 21; if (p0 < 0) p0 += 21;
    uint co[3];
    co[axis] = uint(p0);
    co[(axis+1u)%3u] = p1;
    co[(axis+2u)%3u] = p2;
    uint a = siteOf(co[0],co[1],co[2]);
    uint b = neigh(a, axis);
    uint base = c*N;
    uchar ta = typ[base+a], tb = typ[base+b];
    if (ta == tb) return;
    int d = (tb != 0) - (ta != 0);
    uint aff[12]; int na = 0;
    aff[na++] = a; aff[na++] = b;
    for (uint j=0;j<6u;j++) { uint q=neigh(a,j); if (q!=b) aff[na++]=q; }
    for (uint j=0;j<6u;j++) { uint q=neigh(b,j); if (q!=a) aff[na++]=q; }
    int de = 0; int nt[12]; int nc[12];
    for (int k=0;k<12;k++) {
        uint q = aff[k]; int ot=typ[base+q]; int oc=cnt[base+q];
        int tnew = (q==a) ? tb : ((q==b) ? ta : ot);
        int cnew = oc;
        // Endpoints are each other's neighbours, so a's count sees b take ta
        // and b's count sees a take tb.
        if (k==0) cnew -= d;
        else if (k==1) cnew += d;
        else if (k<7) cnew += d;
        else cnew -= d;
        de += energy(tnew,cnew) - energy(ot,oc);
        nt[k]=tnew; nc[k]=cnew;
    }
    uint h = hash32(c, sweep, axis, color, t);
    float u = float(h) * (1.0f/4294967296.0f);
    bool acc = de <= 0 || u < exp(-beta * float(de));
    if (acc) for (int k=0;k<12;k++) { uint q=aff[k]; typ[base+q]=uchar(nt[k]); cnt[base+q]=uchar(nc[k]); }
}
"""#

guard let device = MTLCreateSystemDefaultDevice(),
      let queue = device.makeCommandQueue() else { fatalError("Metal unavailable") }
let library = try device.makeLibrary(source: metalSrc, options: nil)
guard let kernelFn = library.makeFunction(name: "local_color") else {
    fatalError("kernel local_color missing")
}
let pso = try device.makeComputePipelineState(function: kernelFn)

let beta: Float = CommandLine.arguments.count > 1 ? (Float(CommandLine.arguments[1]) ?? 2.8) : 2.8
let sweeps = 3
let configs = [1, 10, 100, 1000]
let threadsPerGroup = min(256, pso.maxTotalThreadsPerThreadgroup)

enum LaunchMode: CaseIterable, Equatable {
    case cbPerColor
    case batchEncoders
    case batchBarrier

    var batched: Bool { self != .cbPerColor }

    var label: String {
        switch self {
        case .cbPerColor: return "cb-per-color"
        case .batchEncoders: return "batch-encoders"
        case .batchBarrier: return "batch-barrier"
        }
    }
}

func padRight(_ s: String, _ width: Int) -> String {
    if s.count >= width { return s }
    return s + String(repeating: " ", count: width - s.count)
}

// Encode one (axis, color) dispatch into an existing compute encoder.
func encodeColor(_ enc: MTLComputeCommandEncoder, _ tb: MTLBuffer, _ cb: MTLBuffer,
                 _ C: Int, _ axis: Int, _ color: Int, _ sweep: Int, _ beta: Float) {
    enc.setComputePipelineState(pso)
    enc.setBuffer(tb, offset: 0, index: 0)
    enc.setBuffer(cb, offset: 0, index: 1)
    var vc = UInt32(C), va = UInt32(axis), vco = UInt32(color), vs = UInt32(sweep)
    var vb = beta
    enc.setBytes(&vc, length: 4, index: 2)
    enc.setBytes(&va, length: 4, index: 3)
    enc.setBytes(&vco, length: 4, index: 4)
    enc.setBytes(&vs, length: 4, index: 5)
    enc.setBytes(&vb, length: 4, index: 6)
    let total = C * 441
    enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
}

// Encode all 63 dispatches of one 3-axis sweep into a single command buffer.
// barrier == true: one encoder, memoryBarrier(scope: .buffers) between
// dispatches; barrier == false: one encoder per dispatch.
func encodeSweep(_ cmd: MTLCommandBuffer, _ tb: MTLBuffer, _ cb: MTLBuffer,
                 _ C: Int, _ sweep: Int, _ beta: Float, barrier: Bool) {
    if barrier {
        let enc = cmd.makeComputeCommandEncoder()!
        var first = true
        for ax in 0..<3 {
            for co in 0..<21 {
                if !first { enc.memoryBarrier(scope: .buffers) }
                first = false
                encodeColor(enc, tb, cb, C, ax, co, sweep, beta)
            }
        }
        enc.endEncoding()
    } else {
        for ax in 0..<3 {
            for co in 0..<21 {
                let enc = cmd.makeComputeCommandEncoder()!
                encodeColor(enc, tb, cb, C, ax, co, sweep, beta)
                enc.endEncoding()
            }
        }
    }
}

func checksum(_ tb: MTLBuffer, _ cb: MTLBuffer, _ C: Int) -> UInt64 {
    var h: UInt64 = 0xcbf29ce484222325
    let tp = tb.contents().bindMemory(to: UInt8.self, capacity: C * N)
    let cp = cb.contents().bindMemory(to: UInt8.self, capacity: C * N)
    for i in 0..<(C * N) { h = (h ^ UInt64(tp[i])) &* 0x100000001b3 }
    for i in 0..<(C * N) { h = (h ^ UInt64(cp[i])) &* 0x100000001b3 }
    return h
}

struct RunResult {
    let C: Int
    let mode: LaunchMode
    let seconds: Double
    let gpuSeconds: Double
    let energy: Int64
    let state: UInt64
}

// Run `sweeps` full 3-axis sweeps over C independent configs in `mode`.
// Buffer allocation and start-state construction are outside the timing.
func run(_ mode: LaunchMode, _ C: Int, _ sweeps: Int, _ beta: Float) -> RunResult {
    let (types, counts) = initState(C, 1000)
    let tb = device.makeBuffer(bytes: types, length: C * N, options: .storageModeShared)!
    let cb = device.makeBuffer(bytes: counts, length: C * N, options: .storageModeShared)!
    var gpuTotal = 0.0
    func noteGPU(_ cmd: MTLCommandBuffer) {
        if cmd.gpuStartTime > 0 && cmd.gpuEndTime > 0 { gpuTotal += cmd.gpuEndTime - cmd.gpuStartTime }
    }

    let t0 = CFAbsoluteTimeGetCurrent()
    if mode.batched {
        var pending: [MTLCommandBuffer] = []
        pending.reserveCapacity(sweeps)
        for s in 0..<sweeps {
            let cmd = queue.makeCommandBuffer()!
            encodeSweep(cmd, tb, cb, C, s, beta, barrier: mode == .batchBarrier)
            cmd.commit()
            pending.append(cmd)
        }
        for cmd in pending {
            cmd.waitUntilCompleted()
            noteGPU(cmd)
        }
    } else {
        for s in 0..<sweeps {
            for ax in 0..<3 {
                for co in 0..<21 {
                    let cmd = queue.makeCommandBuffer()!
                    let enc = cmd.makeComputeCommandEncoder()!
                    encodeColor(enc, tb, cb, C, ax, co, s, beta)
                    enc.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                    noteGPU(cmd)
                }
            }
        }
    }
    let seconds = CFAbsoluteTimeGetCurrent() - t0

    let tptr = tb.contents().bindMemory(to: UInt8.self, capacity: C * N)
    let cptr = cb.contents().bindMemory(to: UInt8.self, capacity: C * N)
    var e: Int64 = 0
    for i in 0..<(C * N) { e += Int64(energy(Int(tptr[i]), Int(cptr[i]))) }
    return RunResult(C: C, mode: mode, seconds: seconds, gpuSeconds: gpuTotal,
                     energy: e, state: checksum(tb, cb, C))
}

// ---- CPU colored reference (same arithmetic as local_metal.swift) ----

func anchor(_ axis: Int, _ color: Int, _ t: Int) -> (Int, Int) {
    let p1 = t / 21, p2 = t % 21
    var p0 = color - 3 * p1 - 5 * p2
    p0 %= 21; if p0 < 0 { p0 += 21 }
    var co = [0, 0, 0]
    co[axis] = p0; co[(axis + 1) % 3] = p1; co[(axis + 2) % 3] = p2
    let a = site(co[0], co[1], co[2])
    return (a, neigh(a, axis))
}

func applyCPU(_ typ: inout [UInt8], _ cnt: inout [UInt8],
              _ axis: Int, _ color: Int, _ sweep: Int, _ beta: Float) {
    for t in 0..<441 {
        let (a, b) = anchor(axis, color, t)
        let ta = typ[a], tb = typ[b]
        if ta == tb { continue }
        let d = (tb != 0 ? 1 : 0) - (ta != 0 ? 1 : 0)
        var aff = [a, b]
        for j in 0..<6 { let q = neigh(a, j); if q != b { aff.append(q) } }
        for j in 0..<6 { let q = neigh(b, j); if q != a { aff.append(q) } }
        var nt = [Int](repeating: 0, count: 12)
        var nc = [Int](repeating: 0, count: 12)
        var de = 0
        for k in 0..<12 {
            let q = aff[k]
            let ot = Int(typ[q]), oc = Int(cnt[q])
            let tn = (q == a) ? Int(tb) : ((q == b) ? Int(ta) : ot)
            var cn = oc
            // Endpoints are each other's neighbours, so a's count sees b take ta
            // and b's count sees a take tb.
            if k == 0 { cn -= d } else if k == 1 { cn += d }
            else if k < 7 { cn += d } else { cn -= d }
            de += energy(tn, cn) - energy(ot, oc)
            nt[k] = tn; nc[k] = cn
        }
        let h = hash32(0, sweep, axis, color, t)
        let u = Float(h) * (1.0 / 4294967296.0)
        if de <= 0 || u < exp(-beta * Float(de)) {
            for k in 0..<12 { typ[aff[k]] = UInt8(nt[k]); cnt[aff[k]] = UInt8(nc[k]) }
        }
    }
}

// C=1, 2 sweeps, checked against the CPU colored reference after each sweep.
// cb-per-color is additionally checked after every single color; the batch
// modes can only be checked at sweep boundaries (the host never sees the
// intermediate states), which is also the strongest ordering test available.
func validate(_ mode: LaunchMode, _ beta: Float) -> (checks: Int, badChecks: Int, badSites: Int) {
    let (initT, initC) = initState(1, 1000)
    let tb = device.makeBuffer(bytes: initT, length: N, options: .storageModeShared)!
    let cb = device.makeBuffer(bytes: initC, length: N, options: .storageModeShared)!
    var typ = initT
    var cnt = initC
    var checks = 0, badChecks = 0, badSites = 0
    let tptr = tb.contents().bindMemory(to: UInt8.self, capacity: N)
    let cptr = cb.contents().bindMemory(to: UInt8.self, capacity: N)

    func compare() {
        var bad = 0
        for i in 0..<N where tptr[i] != typ[i] || cptr[i] != cnt[i] { bad += 1 }
        checks += 1
        badSites += bad
        if bad > 0 { badChecks += 1 }
    }

    for sweep in 0..<2 {
        if mode.batched {
            let cmd = queue.makeCommandBuffer()!
            encodeSweep(cmd, tb, cb, 1, sweep, beta, barrier: mode == .batchBarrier)
            cmd.commit()
            cmd.waitUntilCompleted()
            for ax in 0..<3 {
                for co in 0..<21 { applyCPU(&typ, &cnt, ax, co, sweep, beta) }
            }
            compare()
        } else {
            for ax in 0..<3 {
                for co in 0..<21 {
                    let cmd = queue.makeCommandBuffer()!
                    let enc = cmd.makeComputeCommandEncoder()!
                    encodeColor(enc, tb, cb, 1, ax, co, sweep, beta)
                    enc.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                    applyCPU(&typ, &cnt, ax, co, sweep, beta)
                    compare()
                }
            }
        }
    }
    return (checks, badChecks, badSites)
}

// ---- main ----

print("device=\(device.name) kernel=local_color maxTG=\(pso.maxTotalThreadsPerThreadgroup) tg=\(threadsPerGroup)")
print("colors/axis=21 axes=3 dispatches/sweep=63 proposals/sweep/config=\(441 * 63) beta=\(beta) sweeps=\(sweeps)")
print("modes: \(LaunchMode.cbPerColor.label)=63 cmd buffers + 63 host waits per sweep; "
    + "\(LaunchMode.batchEncoders.label)=1 cmd buffer per sweep, 63 encoders, 0 host waits between colors; "
    + "\(LaunchMode.batchBarrier.label)=1 cmd buffer per sweep, 1 encoder + 62 buffer barriers, 0 host waits between colors")

// Warm-up: pipeline creation / first-dispatch costs stay out of every timing.
_ = run(.batchBarrier, 1, 1, beta)

for mode in LaunchMode.allCases {
    let v = validate(mode, beta)
    let verdict = v.badChecks == 0 ? "PASS" : "FAIL"
    print("validate " + padRight(mode.label, 15)
        + "C=1 sweeps=2 checks=\(v.checks) mismatchedChecks=\(v.badChecks) mismatchedSites=\(v.badSites) \(verdict)")
}

var results: [RunResult] = []
for mode in LaunchMode.allCases {
    for C in configs {
        let r = run(mode, C, sweeps, beta)
        results.append(r)
        let proposals = Double(C) * 441.0 * 63.0 * Double(sweeps)
        let rate = proposals / r.seconds
        let gpuRate = r.gpuSeconds > 0 ? proposals / r.gpuSeconds : 0
        print(padRight(mode.label, 15) + String(format:
            "C=%4d sweeps=%d time=%8.4fs gpu=%8.4fs proposals=%.3e rate=%9.3f M/s gpu_rate=%9.3f M/s energy=%lld state=%016llx",
            C, sweeps, r.seconds, r.gpuSeconds, proposals, rate / 1e6, gpuRate / 1e6, r.energy, r.state))
    }
}

// Every mode runs the identical deterministic schedule, so final state and
// energy must agree exactly; anything else means a missed dependency.
var crossBad = 0
for C in configs {
    let group = results.filter { $0.C == C }
    guard let ref = group.first else { continue }
    for r in group.dropFirst() where r.state != ref.state || r.energy != ref.energy {
        crossBad += 1
        print("MISMATCH C=\(C) mode=\(r.mode.label) energy=\(r.energy) state=\(String(format: "%016llx", r.state))"
            + " reference=\(ref.mode.label) energy=\(ref.energy) state=\(String(format: "%016llx", ref.state))")
    }
}
print("cross-mode state/energy agreement: " + (crossBad == 0 ? "PASS" : "FAIL (\(crossBad) mismatching runs)"))

print("summary: rate in M/s with speedup vs \(LaunchMode.cbPerColor.label) (wall time)")
for C in configs {
    let group = results.filter { $0.C == C }
    guard let base = group.first(where: { $0.mode == .cbPerColor }) else { continue }
    let proposals = Double(C) * 441.0 * 63.0 * Double(sweeps)
    let baseRate = proposals / base.seconds
    var line = "  C=\(C):"
    for r in group {
        let rate = proposals / r.seconds
        line += " " + r.mode.label + String(format: "=%.1f (x%.2f)", rate / 1e6, rate / baseRate)
    }
    print(line)
}
