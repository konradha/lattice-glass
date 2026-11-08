import Foundation
import Metal

let N = 9261
let L = 21
let CMAX = 1000

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
    return site(x,y,z)
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
func initState(_ configs: Int, _ seed: UInt64) -> ([UInt8],[UInt8]) {
    var types = [UInt8](repeating: 0, count: configs*N)
    var counts = [UInt8](repeating: 0, count: configs*N)
    for c in 0..<configs {
        var perm = Array(0..<N)
        var x = seed &+ UInt64(c) &* 0x9e3779b97f4a7c15
        for i in stride(from: N-1, through: 1, by: -1) {
            x &+= 0x9e3779b97f4a7c15
            var z = x
            z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
            z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
            z ^= z >> 31
            let j = Int(z % UInt64(i+1))
            perm.swapAt(i,j)
        }
        let n0=2315, n1=2778
        for i in 0..<n0 { types[c*N+perm[i]] = 0 }
        for i in n0..<(n0+n1) { types[c*N+perm[i]] = 1 }
        for i in (n0+n1)..<N { types[c*N+perm[i]] = 2 }
        for s in 0..<N {
            var cnt = 0
            for d in 0..<6 { if types[c*N+neigh(s,d)] != 0 { cnt += 1 } }
            counts[c*N+s] = UInt8(cnt)
        }
    }
    return (types,counts)
}
func fullEnergy(_ types: [UInt8], _ counts: [UInt8], _ c: Int) -> Int {
    var e = 0
    for s in 0..<N { e += energy(Int(types[c*N+s]), Int(counts[c*N+s])) }
    return e
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
        if (k>=2 && k<7) cnew += d;
        else if (k>=7) cnew -= d;
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
let lib = try device.makeLibrary(source: metalSrc, options: nil)
let fn = lib.makeFunction(name: "local_color")!
let pso = try device.makeComputePipelineState(function: fn)
print("device=\(device.name) maxThreads=\(pso.maxTotalThreadsPerThreadgroup)")

func dispatchColor(_ tb: MTLBuffer, _ cb: MTLBuffer, _ C: Int, _ axis: Int, _ color: Int, _ sweep: Int, _ beta: Float) {
    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
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
    let total = C*441
    let tg = min(256, pso.maxTotalThreadsPerThreadgroup)
    enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
}

func runSweeps(_ C: Int, _ sweeps: Int, _ beta: Float) -> (Double, MTLBuffer, MTLBuffer) {
    let (types, counts) = initState(C, 1000)
    let tb = device.makeBuffer(bytes: types, length: C*N, options: .storageModeShared)!
    let cb = device.makeBuffer(bytes: counts, length: C*N, options: .storageModeShared)!
    let t0 = CFAbsoluteTimeGetCurrent()
    for s in 0..<sweeps {
        for ax in 0..<3 {
            for co in 0..<21 { dispatchColor(tb, cb, C, ax, co, s, beta) }
        }
    }
    let dt = CFAbsoluteTimeGetCurrent() - t0
    return (dt,tb,cb)
}

// correctness: C=1, one axis, 2 sweeps, compare to CPU colored reference later.
let beta: Float = 2.8
for C in [1,10,100,1000] {
    let (dt,tb,cb) = runSweeps(C, 3, beta)
    let tptr = tb.contents().bindMemory(to: UInt8.self, capacity: C*N)
    let cptr = cb.contents().bindMemory(to: UInt8.self, capacity: C*N)
    var e: Int64 = 0
    for c in 0..<C { var ec=0; for s in 0..<N { ec += energy(Int(tptr[c*N+s]), Int(cptr[c*N+s])) }; e += Int64(ec) }
    let prop = Double(C)*9261.0*3.0*Double(3) // 3 axes * 3 sweeps
    let rate = prop/dt
    print(String(format:"C=%4d time=%.4fs proposals=%.3e rate=%.3f M/s totalE=%lld", C, dt, prop, rate/1e6, e))
}

// ---- CPU colored reference for C=1 ----
func anchor(_ axis: Int, _ color: Int, _ t: Int) -> (Int,Int) {
    let p1 = t / 21, p2 = t % 21
    var p0 = color - 3*p1 - 5*p2
    p0 %= 21; if p0 < 0 { p0 += 21 }
    var co = [0,0,0]
    co[axis] = p0; co[(axis+1)%3] = p1; co[(axis+2)%3] = p2
    let a = site(co[0],co[1],co[2])
    return (a, neigh(a,axis))
}
func applyCPU(_ typ: inout [UInt8], _ cnt: inout [UInt8], _ axis: Int, _ color: Int, _ sweep: Int, _ beta: Float) {
    for t in 0..<441 {
        let (a,b) = anchor(axis,color,t)
        let ta = typ[a], tb = typ[b]
        if ta == tb { continue }
        let d = (tb != 0 ? 1 : 0) - (ta != 0 ? 1 : 0)
        var aff=[Int](); aff.append(a); aff.append(b)
        for j in 0..<6 { let q=neigh(a,j); if q != b { aff.append(q) } }
        for j in 0..<6 { let q=neigh(b,j); if q != a { aff.append(q) } }
        var nt=[Int](repeating:0,count:12), nc=[Int](repeating:0,count:12), de=0
        for k in 0..<12 {
            let q=aff[k]; let ot=Int(typ[q]), oc=Int(cnt[q])
            let tn=(q==a) ? Int(tb) : ((q==b) ? Int(ta) : ot)
            var cn=oc
            if k>=2 && k<7 { cn += d } else if k>=7 { cn -= d }
            de += energy(tn,cn) - energy(ot,oc)
            nt[k]=tn; nc[k]=cn
        }
        let h=hash32(0,sweep,axis,color,t)
        let u=Float(h)*(1.0/4294967296.0)
        if de <= 0 || u < exp(-beta*Float(de)) {
            for k in 0..<12 { typ[aff[k]]=UInt8(nt[k]); cnt[aff[k]]=UInt8(nc[k]) }
        }
    }
}
func validateGPU(beta: Float) {
    let (initT,initC)=initState(1,1000)
    let tb=device.makeBuffer(bytes:initT,length:N,options:.storageModeShared)!
    let cb=device.makeBuffer(bytes:initC,length:N,options:.storageModeShared)!
    var typ=initT, cnt=initC
    var bad=0
    for sw in 0..<2 {
        for ax in 0..<3 {
            for co in 0..<21 {
                dispatchColor(tb,cb,1,ax,co,sw,beta)
                applyCPU(&typ,&cnt,ax,co,sw,beta)
                let gp=tb.contents().bindMemory(to:UInt8.self,capacity:N)
                let gc=cb.contents().bindMemory(to:UInt8.self,capacity:N)
                for i in 0..<N { if gp[i] != typ[i] || gc[i] != cnt[i] { bad += 1; break } }
            }
        }
    }
    print("validate C=1 mismatches=\(bad)")
}
validateGPU(beta:2.8)
