// poc/m3_roofline.swift - Metal roofline microbenchmarks for the Apple M3 GPU.
//
// Run:  swift poc/m3_roofline.swift
//
// Four on-chip microbenchmarks, each folded into a checksum that is written to
// device memory and summed on the host, so none of the work can be deleted by
// the compiler:
//
//   1. random single-byte loads from a threadgroup-resident byte array
//      (8 / 16 / 32 KiB working set, 1 and 8 threadgroups)
//   2. random 32-bit loads from word arrays of the same sizes
//   3. integer ALU throughput: one dependent chain vs eight independent chains
//      for add / xor / multiply
//   4. atomic_fetch_min on one threadgroup atomic_uint shared by all 256
//      threads of a threadgroup (worst-case contention)
//
// Threadgroup memory is per-core on-chip SRAM, so the 1-threadgroup rows
// measure one GPU core and the 8-threadgroup rows measure all cores of the M3.
// Timing is wall clock around commit + waitUntilCompleted; one warmup dispatch
// is discarded and the best of `repeats` dispatches is reported.

import Foundation
import Metal
import Darwin
import Dispatch

// ---------------------------------------------------------------------------
// Metal source
// ---------------------------------------------------------------------------

let mslSource = #"""
#include <metal_stdlib>
using namespace metal;

// 32-bit avalanche mix (murmur3-style finalizer); seeds RNG streams and fills
// the threadgroup arrays deterministically.
static inline uint smix32(uint x) {
    x ^= x >> 16; x *= 0x7FEB352Du;
    x ^= x >> 15; x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

// xorshift32: full 2^32-1 period; the state must stay non-zero.
struct Rng32 {
    uint s;
    void seedWith(uint x) { s = (x == 0u) ? 0x9E3779B9u : x; }
    uint next() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s;
    }
};

// ---------------------------------------------------------------------------
// 1. random single-byte reads from threadgroup memory
// ---------------------------------------------------------------------------
// Each threadgroup fills its own byte array, then every thread walks four
// independent xorshift streams and accumulates four random byte loads per
// iteration. Within a warp the 32 addresses land in random banks, the worst
// realistic pattern for threadgroup SRAM (bank-conflict bound).
kernel void tg_rand_byte_read(threadgroup uchar *state   [[threadgroup(0)]],
                              constant uint &stateBytes  [[buffer(0)]],
                              constant uint &iters       [[buffer(1)]],
                              constant uint &seed        [[buffer(2)]],
                              device uint *checksums     [[buffer(3)]],
                              uint3 tgid3                 [[threadgroup_position_in_grid]],
                              uint3 lid3                  [[thread_position_in_threadgroup]],
                              uint3 tpg                  [[threads_per_threadgroup]])
{
    const uint tgid = tgid3.x;
    const uint lid = lid3.x;
    const uint tgSize = tpg.x;
    for (uint i = lid; i < stateBytes; i += tgSize) {
        state[i] = uchar(smix32(i * 0x9E3779B1u + seed) & 0xFFu);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint mask = stateBytes - 1u;      // working sets are powers of two

    uint h = smix32(seed ^ (tgid * 0x9E3779B1u) ^ (lid * 0x85EBCA6Bu));
    Rng32 r0; r0.seedWith(h);
    Rng32 r1; r1.seedWith(smix32(h ^ 0x27D4EB2Fu));
    Rng32 r2; r2.seedWith(smix32(h ^ 0xC2B2AE35u));
    Rng32 r3; r3.seedWith(smix32(h ^ 0x165667B1u));

    uint acc = 0u;
    for (uint it = 0u; it < iters; ++it) {
        uint a = r0.next() & mask;
        uint b = r1.next() & mask;
        uint c = r2.next() & mask;
        uint d = r3.next() & mask;
        acc += uint(state[a]) + uint(state[b]) + uint(state[c]) + uint(state[d]);
    }
    checksums[tgid * tgSize + lid] = acc;
}

// ---------------------------------------------------------------------------
// 2. random 32-bit reads from threadgroup memory
// ---------------------------------------------------------------------------
// Same shape as (1), one naturally aligned word per load instead of one byte.
kernel void tg_rand_word_read(threadgroup uint *state    [[threadgroup(0)]],
                              constant uint &stateWords  [[buffer(0)]],
                              constant uint &iters       [[buffer(1)]],
                              constant uint &seed        [[buffer(2)]],
                              device uint *checksums     [[buffer(3)]],
                              uint3 tgid3                 [[threadgroup_position_in_grid]],
                              uint3 lid3                  [[thread_position_in_threadgroup]],
                              uint3 tpg                  [[threads_per_threadgroup]])
{
    const uint tgid = tgid3.x;
    const uint lid = lid3.x;
    const uint tgSize = tpg.x;
    for (uint i = lid; i < stateWords; i += tgSize) {
        state[i] = smix32(i * 0x85EBCA6Bu + seed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint mask = stateWords - 1u;

    uint h = smix32(seed ^ (tgid * 0x27D4EB2Fu) ^ (lid * 0xC2B2AE35u));
    Rng32 r0; r0.seedWith(h);
    Rng32 r1; r1.seedWith(smix32(h ^ 0x9E3779B1u));
    Rng32 r2; r2.seedWith(smix32(h ^ 0x165667B1u));
    Rng32 r3; r3.seedWith(smix32(h ^ 0xD3A2646Cu));

    uint acc = 0u;
    for (uint it = 0u; it < iters; ++it) {
        uint a = r0.next() & mask;
        uint b = r1.next() & mask;
        uint c = r2.next() & mask;
        uint d = r3.next() & mask;
        acc += state[a] + state[b] + state[c] + state[d];
    }
    checksums[tgid * tgSize + lid] = acc;
}

// ---------------------------------------------------------------------------
// 3. integer ALU throughput
// ---------------------------------------------------------------------------
// op: 0 = add, 1 = xor, 2 = mul.  dependent: 1 = one loop-carried chain,
// 0 = eight independent chains in the same loop. The operand (i | 1) is data
// dependent on the loop counter, so the chain cannot be folded into a single
// strength-reduced operation. Only the chain operations are counted by the
// host; the index operand and loop bookkeeping are excluded.
kernel void int_alu(device uint *out          [[buffer(0)]],
                    constant uint &iters      [[buffer(1)]],
                    constant uint &op         [[buffer(2)]],
                    constant uint &dependent  [[buffer(3)]],
                    constant uint &seed       [[buffer(4)]],
                    uint gid                  [[thread_position_in_grid]])
{
    uint h = smix32(seed ^ (gid * 0x9E3779B1u)) | 1u;
    uint acc = 0u;

    if (dependent != 0u) {
        acc = h;
        if (op == 0u) {
            for (uint i = 0u; i < iters; ++i) acc = acc + (i | 1u);
        } else if (op == 1u) {
            for (uint i = 0u; i < iters; ++i) acc = acc ^ (i | 1u);
        } else {
            for (uint i = 0u; i < iters; ++i) acc = acc * (i | 1u);
        }
    } else {
        // Odd seeds keep every multiply chain non-zero for the whole run.
        uint a0 = h;
        uint a1 = (h + 0x9E3779B1u) | 1u;
        uint a2 = (h ^ 0x85EBCA6Bu) | 1u;
        uint a3 = (h + 0xC2B2AE35u) | 1u;
        uint a4 = (h ^ 0x27D4EB2Fu) | 1u;
        uint a5 = (h + 0x165667B1u) | 1u;
        uint a6 = (h ^ 0xD3A2646Cu) | 1u;
        uint a7 = (h + 0xFD7046C5u) | 1u;
        if (op == 0u) {
            for (uint i = 0u; i < iters; ++i) {
                uint v = i | 1u;
                a0 = a0 + v; a1 = a1 + v; a2 = a2 + v; a3 = a3 + v;
                a4 = a4 + v; a5 = a5 + v; a6 = a6 + v; a7 = a7 + v;
            }
        } else if (op == 1u) {
            for (uint i = 0u; i < iters; ++i) {
                uint v = i | 1u;
                a0 = a0 ^ v; a1 = a1 ^ v; a2 = a2 ^ v; a3 = a3 ^ v;
                a4 = a4 ^ v; a5 = a5 ^ v; a6 = a6 ^ v; a7 = a7 ^ v;
            }
        } else {
            for (uint i = 0u; i < iters; ++i) {
                uint v = i | 1u;
                a0 = a0 * v; a1 = a1 * v; a2 = a2 * v; a3 = a3 * v;
                a4 = a4 * v; a5 = a5 * v; a6 = a6 * v; a7 = a7 * v;
            }
        }
        acc = (a0 ^ a1) ^ (a2 + a3) ^ (a4 ^ a5) ^ (a6 + a7);
    }
    out[gid] = acc;
}

// ---------------------------------------------------------------------------
// 4. threadgroup atomic_fetch_min on a single contended cell
// ---------------------------------------------------------------------------
// Every thread hammers the same threadgroup atomic_uint; the cell is reset by
// lane 0, so the hardware cannot elide the read-modify-write, and the result is
// read back after the loop to prove the work happened.
kernel void tg_atomic_min(threadgroup atomic_uint *cell  [[threadgroup(0)]],
                          constant uint &iters           [[buffer(0)]],
                          constant uint &seed            [[buffer(1)]],
                          device uint *results           [[buffer(2)]],
                          uint tgid                      [[threadgroup_position_in_grid]],
                          uint lid                       [[thread_position_in_threadgroup]])
{
    if (lid == 0u) {
        atomic_store_explicit(cell, 0xFFFFFFFFu, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    Rng32 r;
    r.seedWith(smix32(seed ^ (tgid * 0xA5A5A5A5u) ^ (lid * 0x3C6EF372u)));
    for (uint it = 0u; it < iters; ++it) {
        atomic_fetch_min_explicit(cell, r.next(), memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        results[tgid] = atomic_load_explicit(cell, memory_order_relaxed);
    }
}
"""#

// ---------------------------------------------------------------------------
// sysctl helpers (CPU name, performance / efficiency core split)
// ---------------------------------------------------------------------------

func sysctlString(_ key: String) -> String? {
    var size = 0
    guard sysctlbyname(key, nil, &size, nil, 0) == 0, size > 0 else { return nil }
    var buf = [CChar](repeating: 0, count: size)
    guard sysctlbyname(key, &buf, &size, nil, 0) == 0 else { return nil }
    return String(cString: buf)
}

func sysctlInt(_ key: String) -> Int? {
    var value: Int64 = 0
    var size = MemoryLayout<Int64>.size
    guard sysctlbyname(key, &value, &size, nil, 0) == 0 else { return nil }
    return Int(value)
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

let maxThreadgroups = 8

struct Row {
    let name: String
    let tgs: Int
    let tgmem: String
    let iters: Int
    let opsPerThread: Int
    let bytesPerOp: Int
    let seconds: Double
    let checksum: UInt64
}

final class Harness {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let threads: Int
    let repeats: Int
    let out: MTLBuffer
    let byteRead: MTLComputePipelineState
    let wordRead: MTLComputePipelineState
    let intALU: MTLComputePipelineState
    let atomicMin: MTLComputePipelineState
    let maxThreadsPerThreadgroup: Int

    init(device: MTLDevice, threads: Int, repeats: Int) {
        self.device = device
        self.threads = threads
        self.repeats = repeats

        guard let queue = device.makeCommandQueue() else { fatalError("no command queue") }
        self.queue = queue

        let options = MTLCompileOptions()
        options.languageVersion = .version3_0
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: mslSource, options: options)
        } catch {
            fatalError("Metal compile failed: \(error)")
        }

        func makePSO(_ name: String) -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                fatalError("kernel \(name) not found")
            }
            do {
                return try device.makeComputePipelineState(function: fn)
            } catch {
                fatalError("pipeline \(name) failed: \(error)")
            }
        }

        let byteReadPSO = makePSO("tg_rand_byte_read")
        let wordReadPSO = makePSO("tg_rand_word_read")
        let aluPSO = makePSO("int_alu")
        let atomicPSO = makePSO("tg_atomic_min")
        self.byteRead = byteReadPSO
        self.wordRead = wordReadPSO
        self.intALU = aluPSO
        self.atomicMin = atomicPSO
        self.maxThreadsPerThreadgroup = byteReadPSO.maxTotalThreadsPerThreadgroup

        guard let out = device.makeBuffer(length: maxThreadgroups * threads * 4,
                                          options: .storageModeShared) else {
            fatalError("checksum buffer allocation failed")
        }
        self.out = out
    }

    /// One discarded warmup dispatch, then `repeats` timed dispatches; returns
    /// the best wall-clock seconds around commit + waitUntilCompleted.
    func time(_ pso: MTLComputePipelineState,
              threadgroups: Int,
              threadgroupBytes: Int,
              _ bind: (MTLComputeCommandEncoder) -> Void) -> Double {
        var best = Double.greatestFiniteMagnitude
        for rep in 0...repeats {
            guard let cb = queue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else {
                fatalError("cannot create command buffer / encoder")
            }
            enc.setComputePipelineState(pso)
            if threadgroupBytes > 0 {
                enc.setThreadgroupMemoryLength(threadgroupBytes, index: 0)
            }
            bind(enc)
            enc.dispatchThreadgroups(
                MTLSize(width: threadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
            enc.endEncoding()

            let t0 = DispatchTime.now().uptimeNanoseconds
            cb.commit()
            cb.waitUntilCompleted()
            let t1 = DispatchTime.now().uptimeNanoseconds

            if let error = cb.error { fatalError("GPU error: \(error)") }
            if rep > 0 {
                best = Swift.min(best, Double(t1 - t0) * 1e-9)
            }
        }
        return best
    }

    /// Sums the first `count` uint32 slots of the checksum buffer.
    func checksum(_ count: Int) -> UInt64 {
        let p = out.contents().bindMemory(to: UInt32.self, capacity: count)
        var s: UInt64 = 0
        for i in 0..<count { s &+= UInt64(p[i]) }
        return s
    }
}

@inline(__always) func setU32(_ enc: MTLComputeCommandEncoder, _ value: UInt32, _ index: Int) {
    var v = value
    enc.setBytes(&v, length: MemoryLayout<UInt32>.size, index: index)
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

func padR(_ s: String, _ w: Int) -> String {
    s.count >= w ? s : s + String(repeating: " ", count: w - s.count)
}

func padL(_ s: String, _ w: Int) -> String {
    s.count >= w ? s : String(repeating: " ", count: w - s.count) + s
}

func rate(_ r: Row, _ threads: Int) -> Double {
    Double(r.tgs * threads * r.opsPerThread) / r.seconds
}

func printTable(_ rows: [Row], threads: Int) {
    var header = padR("case", 24)
    header += padL("tgs", 4)
    header += padL("tgmem", 8)
    header += padL("iter/thr", 10)
    header += padL("ops/thr", 10)
    header += padL("total ops", 12)
    header += padL("ms", 9)
    header += padL("Mops/s", 11)
    header += padL("GB/s", 8)
    header += padL("checksum", 14)
    print(header)
    print(String(repeating: "-", count: header.count))

    for r in rows {
        let totalOps = Double(r.tgs * threads * r.opsPerThread)
        let opsPerSec = totalOps / r.seconds
        let bytesPerSec = Double(r.tgs * threads * r.opsPerThread * r.bytesPerOp) / r.seconds

        var line = padR(r.name, 24)
        line += padL("\(r.tgs)", 4)
        line += padL(r.tgmem, 8)
        line += padL("\(r.iters)", 10)
        line += padL("\(r.opsPerThread)", 10)
        line += padL(String(format: "%.3e", totalOps), 12)
        line += padL(String(format: "%.2f", r.seconds * 1e3), 9)
        line += padL(String(format: "%.1f", opsPerSec / 1e6), 11)
        line += r.bytesPerOp > 0
            ? padL(String(format: "%.1f", bytesPerSec / 1e9), 8)
            : padL("-", 8)
        line += padL("0x" + String(format: "%08x", UInt32(r.checksum & 0xFFFF_FFFF)), 14)
        print(line)
    }
}

// ---------------------------------------------------------------------------
// Sweep parameters (the exact loop counts are printed in the header)
// ---------------------------------------------------------------------------

let threadsPerThreadgroup = 256
let threadgroupSweep = [1, 8]
let stateByteSweep = [8 << 10, 16 << 10, 32 << 10]
let readIters = 1 << 17          // per thread; 4 random loads per iteration
let readsPerIteration = 4
let aluIters = 1 << 21           // per thread; 1 op (dep) or 8 ops (indep) per iter
let independentChains = 8
let atomicIters = 1 << 15        // per thread, all 256 threads on one cell
let repeats = 3
let seed: UInt32 = 0x5EED_1234

precondition(threadgroupSweep.allSatisfy { $0 <= maxThreadgroups },
             "threadgroupSweep exceeds maxThreadgroups (checksum buffer size)")

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal is not available")
}
let harness = Harness(device: device, threads: threadsPerThreadgroup, repeats: repeats)

var rows: [Row] = []

// --- 1. random byte reads ---------------------------------------------------
let maxTGMemory = device.maxThreadgroupMemoryLength
for tgs in threadgroupSweep {
    for bytes in stateByteSweep where bytes <= maxTGMemory {
        let secs = harness.time(harness.byteRead, threadgroups: tgs, threadgroupBytes: bytes) { enc in
            setU32(enc, UInt32(bytes), 0)
            setU32(enc, UInt32(readIters), 1)
            setU32(enc, seed, 2)
            enc.setBuffer(harness.out, offset: 0, index: 3)
        }
        rows.append(Row(name: "byte_rand_read",
                        tgs: tgs,
                        tgmem: "\(bytes >> 10) KiB",
                        iters: readIters,
                        opsPerThread: readIters * readsPerIteration,
                        bytesPerOp: 1,
                        seconds: secs,
                        checksum: harness.checksum(tgs * threadsPerThreadgroup)))
    }
}

// --- 2. random 32-bit reads -------------------------------------------------
for tgs in threadgroupSweep {
    for bytes in stateByteSweep where bytes <= maxTGMemory {
        let words = bytes / 4
        let secs = harness.time(harness.wordRead, threadgroups: tgs, threadgroupBytes: bytes) { enc in
            setU32(enc, UInt32(words), 0)
            setU32(enc, UInt32(readIters), 1)
            setU32(enc, seed, 2)
            enc.setBuffer(harness.out, offset: 0, index: 3)
        }
        rows.append(Row(name: "u32_rand_read",
                        tgs: tgs,
                        tgmem: "\(bytes >> 10) KiB",
                        iters: readIters,
                        opsPerThread: readIters * readsPerIteration,
                        bytesPerOp: 4,
                        seconds: secs,
                        checksum: harness.checksum(tgs * threadsPerThreadgroup)))
    }
}

// --- 3. integer ALU ---------------------------------------------------------
let opNames = ["add", "xor", "mul"]
for tgs in threadgroupSweep {
    for dependent in [1, 0] {
        for op in 0..<opNames.count {
            let opsPerThread = dependent == 1 ? aluIters : aluIters * independentChains
            let secs = harness.time(harness.intALU, threadgroups: tgs, threadgroupBytes: 0) { enc in
                enc.setBuffer(harness.out, offset: 0, index: 0)
                setU32(enc, UInt32(aluIters), 1)
                setU32(enc, UInt32(op), 2)
                setU32(enc, UInt32(dependent), 3)
                setU32(enc, seed, 4)
            }
            let mode = dependent == 1 ? "dep" : "indep8"
            rows.append(Row(name: "alu_\(opNames[op])_\(mode)",
                            tgs: tgs,
                            tgmem: "-",
                            iters: aluIters,
                            opsPerThread: opsPerThread,
                            bytesPerOp: 0,
                            seconds: secs,
                            checksum: harness.checksum(tgs * threadsPerThreadgroup)))
        }
    }
}

// --- 4. threadgroup atomic fetch-min ---------------------------------------
for tgs in threadgroupSweep {
    let secs = harness.time(harness.atomicMin, threadgroups: tgs, threadgroupBytes: 16) { enc in
        setU32(enc, UInt32(atomicIters), 0)
        setU32(enc, seed, 1)
        enc.setBuffer(harness.out, offset: 0, index: 2)
    }
    rows.append(Row(name: "atomic_fetch_min_u32",
                    tgs: tgs,
                    tgmem: "4 B",
                    iters: atomicIters,
                    opsPerThread: atomicIters,
                    bytesPerOp: 0,
                    seconds: secs,
                    checksum: harness.checksum(tgs)))
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

let cpuName = sysctlString("machdep.cpu.brand_string") ?? "unknown CPU"
let model = sysctlString("hw.model") ?? "unknown model"
let logicalCores = ProcessInfo.processInfo.processorCount
let activeCores = ProcessInfo.processInfo.activeProcessorCount
let perfCores = sysctlInt("hw.perflevel0.logicalcpu")
let effCores = sysctlInt("hw.perflevel1.logicalcpu")

print("Metal M3 roofline microbenchmarks")
if let p = perfCores, let e = effCores {
    print("CPU  : \(cpuName) [\(model)]  logical=\(logicalCores) active=\(activeCores)  P=\(p) E=\(e)")
} else {
    print("CPU  : \(cpuName) [\(model)]  logical=\(logicalCores) active=\(activeCores)")
}
print("GPU  : \(device.name)  maxThreadsPerThreadgroup=\(harness.maxThreadsPerThreadgroup)"
      + "  maxThreadgroupMemory=\(maxTGMemory) B")
print("")
print("exact loop counts (per thread):")
print("  byte/u32 reads : \(readIters) iterations x \(readsPerIteration) random loads"
      + " = \(readIters * readsPerIteration) loads/thread")
print("  integer ALU    : \(aluIters) iterations -> \(aluIters) counted ops (dependent)"
      + " or \(aluIters * independentChains) counted ops (\(independentChains) independent chains)")
print("  atomic min     : \(atomicIters) conflicting atomic_fetch_min calls per thread")
let tgList = threadgroupSweep.map { String($0) }.joined(separator: "/")
print("  dispatch       : \(threadsPerThreadgroup) threads/threadgroup, \(tgList) threadgroups,"
      + " best of \(repeats) reps after 1 warmup, seed=0x\(String(seed, radix: 16))")
if let skipped = stateByteSweep.first(where: { $0 > maxTGMemory }) {
    print("  skipped state  : \(skipped >> 10) KiB (above device max threadgroup memory)")
}
print("")

printTable(rows, threads: threadsPerThreadgroup)
print("")
print("legend: byte/u32 rows: ops = random loads, GB/s = logical bytes fetched (1 B or 4 B per load)")
print("        alu rows: ops = add/xor/mul issued on the accumulator chain(s); the index")
print("        operand (i|1) and loop bookkeeping are excluded from the counted ops")
print("        atomic rows: one threadgroup atomic_uint, all 256 threads contend on it")
print("        tgmem = per-threadgroup working set (each threadgroup owns a private copy)")
print("")

func peak(_ name: String) -> Row? {
    rows.filter { $0.name == name }
        .max { rate($0, threadsPerThreadgroup) < rate($1, threadsPerThreadgroup) }
}

if let r = peak("byte_rand_read") {
    let s = rate(r, threadsPerThreadgroup)
    print("peak threadgroup byte reads : \(String(format: "%.2f", s / 1e9)) Greads/s"
          + " (\(String(format: "%.2f", s / 1e9)) GB/s logical) @ \(r.tgs) tgs / \(r.tgmem)")
}
if let r = peak("u32_rand_read") {
    let s = rate(r, threadsPerThreadgroup)
    print("peak threadgroup u32 reads  : \(String(format: "%.2f", s / 1e9)) Greads/s"
          + " (\(String(format: "%.2f", s * 4 / 1e9)) GB/s logical) @ \(r.tgs) tgs / \(r.tgmem)")
}
for mode in ["dep", "indep8"] {
    var best: Row? = nil
    for op in opNames {
        guard let r = peak("alu_\(op)_\(mode)") else { continue }
        if let b = best {
            if rate(r, threadsPerThreadgroup) > rate(b, threadsPerThreadgroup) { best = r }
        } else {
            best = r
        }
    }
    guard let r = best else { continue }
    print("peak integer ALU \(padR(mode, 7)): \(String(format: "%.2f", rate(r, threadsPerThreadgroup) / 1e9))"
          + " Gops/s @ \(r.tgs) tgs (\(r.name))")
}
if let r = peak("atomic_fetch_min_u32") {
    print("peak atomic_fetch_min u32   : \(String(format: "%.2f", rate(r, threadsPerThreadgroup) / 1e9))"
          + " Gops/s @ \(r.tgs) tgs (single contended cell per threadgroup)")
}
