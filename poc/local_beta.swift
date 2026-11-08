import Foundation
import Metal

// ---------------------------------------------------------------------------
// Beta sweep for the persistent one-CTA colored local kernel.
//
// Same geometry, state, 21-colour x 3-axis schedule and single-threadgroup
// persistent loop as poc/local_chain.swift, with three differences:
//
//   * beta is a runtime kernel argument (buffer 3) so the sweep reuses one
//     pipeline;
//   * the kernel counts accepted swaps. Each of the 441 participating threads
//     owns a threadgroup counter slot; thread 0 reduces them once at the end
//     and writes the total to a device buffer (buffer 4);
//   * the neighbour count c is maintained as the exact popcount of the
//     six-neighbour occupancy mask. Since a and b are adjacent and change
//     occupancy by +d and -d, c[a] changes by -d and c[b] by +d (the ten other
//     affected sites change by +d near a and -d near b). This keeps
//     c[s] == |{d : type(neighbour(s,d)) != 0}| after every color, so the
//     energy is the true local energy and the invariant check is meaningful.
//
// For each beta the state is reset to the same fixed-composition start, then
// 1000 full sweeps (3 axes x 21 colors x 441 proposals) are timed. Output is
// proposals/s, final energy, accepted moves and invariant checks.
//
// C=1 exactness: after 2 sweeps at beta 3.2 the whole device state (types and
// counts) and the accepted-move total are compared against a sequential CPU
// colored reference that runs the identical schedule, hash and Metropolis rule.
// ---------------------------------------------------------------------------

let L=21, N=9261, TG=512
let proposalsPerSweep = 3*N

func site(_ x:Int,_ y:Int,_ z:Int)->Int { x+L*y+L*L*z }
func neigh(_ s:Int,_ d:Int)->Int { var x=s%L,y=(s/L)%L,z=s/(L*L); if d==0{x=(x+1)%L}else if d==1{x=(x+L-1)%L}else if d==2{y=(y+1)%L}else if d==3{y=(y+L-1)%L}else if d==4{z=(z+1)%L}else{z=(z+L-1)%L}; return site(x,y,z) }
func initState()->([UInt8],[UInt8]) { var t=[UInt8](repeating:0,count:N); var c=[UInt8](repeating:0,count:N); var p=Array(0..<N); var x:UInt64=1000; for i in stride(from:N-1,through:1,by:-1){ x &+= 0x9e3779b97f4a7c15; var z=x; z=(z^(z>>30))&*0xbf58476d1ce4e5b9; z=(z^(z>>27))&*0x94d049bb133111eb; z^=z>>31; let j=Int(z%UInt64(i+1)); p.swapAt(i,j) }; for i in 0..<2315{t[p[i]]=0}; for i in 2315..<5093{t[p[i]]=1}; for i in 5093..<N{t[p[i]]=2}; for s in 0..<N{var k=0; for d in 0..<6{if t[neigh(s,d)] != 0{k+=1}}; c[s]=UInt8(k)}; return(t,c) }
func energy(_ t:Int,_ c:Int)->Int { if t==0{return 0}; let d=t==1 ? c-3 : c-5; return d*d }
func energyAll(_ t:[UInt8],_ c:[UInt8])->Int { var e=0; for s in 0..<N{e+=energy(Int(t[s]),Int(c[s]))}; return e }

// Fixed-composition totals: N0=2315 vacancies, N1=2778 type-1, N2=4168 type-2.
// Swaps exchange types, so the totals and sum(type) are invariant.
let N0=2315, N1=2778, N2=4168
let SUM_T=N1+2*N2

func verifyInvariants(_ t:[UInt8],_ c:[UInt8])->(ok:Bool,text:String){
  var badType=0,badCount=0,sum=0,cnt=[0,0,0]
  for s in 0..<N {
    let ts=Int(t[s])
    if ts>2 { badType+=1 } else { cnt[ts]+=1; sum+=ts }
    var k=0
    for d in 0..<6 { if t[neigh(s,d)] != 0 { k+=1 } }
    if Int(c[s]) != k { badCount+=1 }
  }
  let ok = badType==0 && badCount==0 && sum==SUM_T && cnt[0]==N0 && cnt[1]==N1 && cnt[2]==N2
  if ok {
    return (true,"inv OK (types valid, sum=\(sum), comp=\(cnt[0])/\(cnt[1])/\(cnt[2]), counts match masks)")
  }
  return (false,"inv FAIL (badTypes=\(badType), staleCounts=\(badCount), sum=\(sum), comp=\(cnt[0])/\(cnt[1])/\(cnt[2]))")
}

let src = #"""
#include <metal_stdlib>
using namespace metal;
#define NN 9261
#define TG 512
inline uint smix(uint x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x;}
inline uint h32(uint c,uint sw,uint ax,uint co,uint t){uint h=c*0x9e3779b9u;h^=sw*0x85ebca6bu;h^=ax*0xc2b2ae35u;h^=co*0x27d4eb2fu;h^=t*0x165667b1u;return smix(h);}
inline int en(int t,int c){if(t==0)return 0;int d=(t==1)?c-3:c-5;return d*d;}
inline uint siteOf(uint x,uint y,uint z){return x+21u*y+441u*z;}
inline uint nbr(uint s,uint d){int x=int(s%21u),y=int((s/21u)%21u),z=int(s/441u);if(d==0)x=(x+1)%21;else if(d==1)x=(x+20)%21;else if(d==2)y=(y+1)%21;else if(d==3)y=(y+20)%21;else if(d==4)z=(z+1)%21;else z=(z+20)%21;return siteOf(uint(x),uint(y),uint(z));}
inline void edge(threadgroup uchar *st, threadgroup uchar *ct, uint ax,uint co,uint t,float beta,uint sw,threadgroup uint *acc){
    uint p1=t/21u,p2=t%21u;int p0=int(co)-3*int(p1)-5*int(p2);p0%=21;if(p0<0)p0+=21;
    uint q[3];q[ax]=uint(p0);q[(ax+1u)%3u]=p1;q[(ax+2u)%3u]=p2;
    uint a=siteOf(q[0],q[1],q[2]), b=nbr(a,ax);
    uchar ta=st[a],tb=st[b]; if(ta==tb)return;
    int d=(tb!=0)-(ta!=0);
    uint aff[12];int na=0;aff[na++]=a;aff[na++]=b;
    for(uint j=0;j<6u;j++){uint z=nbr(a,j);if(z!=b)aff[na++]=z;}
    for(uint j=0;j<6u;j++){uint z=nbr(b,j);if(z!=a)aff[na++]=z;}
    int nt[12];int nc[12];int de=0;
    for(int k=0;k<12;k++){
        uint z=aff[k];int ot=st[z],oc=ct[z];
        int tn=(z==a)?tb:((z==b)?ta:ot);
        int cn=oc;
        if(k==0)cn-=d;else if(k==1)cn+=d;else if(k<7)cn+=d;else cn-=d;
        de+=en(tn,cn)-en(ot,oc);nt[k]=tn;nc[k]=cn;
    }
    uint hh=h32(0u,sw,ax,co,t);float u=float(hh)*(1.0f/4294967296.0f);
    if(de<=0||u<exp(-beta*float(de))){
        for(int k=0;k<12;k++){uint z=aff[k];st[z]=uchar(nt[k]);ct[z]=uchar(nc[k]);}
        *acc+=1u;
    }
}
kernel void chain(device uchar *gt [[buffer(0)]],device uchar *gc [[buffer(1)]],constant uint &sweeps [[buffer(2)]],constant float &beta [[buffer(3)]],device uint *accOut [[buffer(4)]],uint tid [[thread_position_in_threadgroup]]){
    threadgroup uchar st[NN];threadgroup uchar ct[NN];threadgroup uint acct[TG];
    acct[tid]=0u;
    for(uint i=tid;i<NN;i+=TG){st[i]=gt[i];ct[i]=gc[i];}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=0;s<sweeps;s++){
      for(uint ax=0;ax<3u;ax++){
        for(uint co=0;co<21u;co++){
          if(tid<441u) edge(st,ct,ax,co,tid,beta,s,&acct[tid]);
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0u){uint tot=0u;for(uint i=0;i<TG;i++)tot+=acct[i];accOut[0]=tot;}
    for(uint i=tid;i<NN;i+=TG){gt[i]=st[i];gc[i]=ct[i];}
}
"""#
guard let dev=MTLCreateSystemDefaultDevice(), let q=dev.makeCommandQueue() else{fatalError("no metal")}
let lib=try dev.makeLibrary(source:src,options:nil)
let fn=lib.makeFunction(name:"chain")!
let pso=try dev.makeComputePipelineState(function:fn)

struct RunResult { let t:[UInt8]; let c:[UInt8]; let accepted:Int; let seconds:Double }

func buf(_ a:[UInt8])->MTLBuffer { dev.makeBuffer(bytes:a,length:N,options:.storageModeShared)! }

func run(sweeps:UInt32,beta:Float,from state:([UInt8],[UInt8]))->RunResult{
  let tb=buf(state.0), cb=buf(state.1)
  let ab=dev.makeBuffer(length:MemoryLayout<UInt32>.size,options:.storageModeShared)!
  ab.contents().storeBytes(of:UInt32(0),as:UInt32.self)
  let cmd=q.makeCommandBuffer()!
  let enc=cmd.makeComputeCommandEncoder()!
  enc.setComputePipelineState(pso)
  enc.setBuffer(tb,offset:0,index:0)
  enc.setBuffer(cb,offset:0,index:1)
  var sw=sweeps, b=beta
  enc.setBytes(&sw,length:4,index:2)
  enc.setBytes(&b,length:4,index:3)
  enc.setBuffer(ab,offset:0,index:4)
  enc.dispatchThreadgroups(MTLSize(width:1,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:TG,height:1,depth:1))
  enc.endEncoding()
  let start=CFAbsoluteTimeGetCurrent()
  cmd.commit()
  cmd.waitUntilCompleted()
  let seconds=CFAbsoluteTimeGetCurrent()-start
  let accepted=Int(ab.contents().load(as:UInt32.self))
  let gp=tb.contents().bindMemory(to:UInt8.self,capacity:N)
  let gc=cb.contents().bindMemory(to:UInt8.self,capacity:N)
  return RunResult(t:Array(UnsafeBufferPointer(start:UnsafePointer(gp),count:N)),
                   c:Array(UnsafeBufferPointer(start:UnsafePointer(gc),count:N)),
                   accepted:accepted,seconds:seconds)
}

// ---------------------------------------------------------------------------
// CPU colored reference for one (sweep, axis, color). Same order, same
// affected set, same counter update, same hash and Metropolis rule as the
// kernel. Beta is Float to match the GPU's fp32 exp path. Returns accepts.
// ---------------------------------------------------------------------------
func smix(_ x0:UInt32)->UInt32{var x=x0;x^=x>>16;x&*=0x7feb352d;x^=x>>15;x&*=0x846ca68b;x^=x>>16;return x}
func hash32(_ c:Int,_ sw:Int,_ ax:Int,_ co:Int,_ t:Int)->UInt32{var h=UInt32(truncatingIfNeeded:c)&*0x9e3779b9;h^=UInt32(truncatingIfNeeded:sw)&*0x85ebca6b;h^=UInt32(truncatingIfNeeded:ax)&*0xc2b2ae35;h^=UInt32(truncatingIfNeeded:co)&*0x27d4eb2f;h^=UInt32(truncatingIfNeeded:t)&*0x165667b1;return smix(h)}
func anchor(_ axis:Int,_ color:Int,_ t:Int)->(Int,Int){let p1=t/21,p2=t%21;var p0=color-3*p1-5*p2;p0%=21;if p0<0{p0+=21};var q=[0,0,0];q[axis]=p0;q[(axis+1)%3]=p1;q[(axis+2)%3]=p2;let a=site(q[0],q[1],q[2]);return(a,neigh(a,axis))}
func applyCPU(_ st:inout[UInt8],_ ct:inout[UInt8],_ ax:Int,_ co:Int,_ sw:Int,_ beta:Float)->Int{
  var accepted=0
  for t in 0..<441 {
    let (a,b)=anchor(ax,co,t)
    let ta=st[a],tb=st[b]
    if ta==tb { continue }
    let d=(tb != 0 ? 1:0)-(ta != 0 ? 1:0)
    var aff=[a,b]
    for j in 0..<6 { let z=neigh(a,j); if z != b { aff.append(z) } }
    for j in 0..<6 { let z=neigh(b,j); if z != a { aff.append(z) } }
    var nt=[Int](repeating:0,count:12), nc=[Int](repeating:0,count:12), de=0
    for k in 0..<12 {
      let z=aff[k]
      let ot=Int(st[z]), oc=Int(ct[z])
      let tn=(z==a) ? Int(tb):((z==b) ? Int(ta):ot)
      var cn=oc
      if k==0 { cn-=d } else if k==1 { cn+=d } else if k<7 { cn+=d } else { cn-=d }
      de+=energy(tn,cn)-energy(ot,oc)
      nt[k]=tn; nc[k]=cn
    }
    let hh=hash32(0,sw,ax,co,t)
    let u=Float(hh)*(1.0/4294967296.0)
    if de<=0 || u<exp(-beta*Float(de)) {
      for k in 0..<12 { st[aff[k]]=UInt8(nt[k]); ct[aff[k]]=UInt8(nc[k]) }
      accepted+=1
    }
  }
  return accepted
}

// ---------------------------------------------------------------------------
// C=1 exactness: 2 sweeps at beta 3.2, device vs sequential CPU colored ref.
// ---------------------------------------------------------------------------
func validateC1(){
  let beta:Float=3.2, sweeps:UInt32=2
  let (t0,c0)=initState()
  let r=run(sweeps:sweeps,beta:beta,from:(t0,c0))
  var st=t0, ct=c0, cpuAcc=0
  for s in 0..<Int(sweeps){ for ax in 0..<3 { for co in 0..<21 { cpuAcc += applyCPU(&st,&ct,ax,co,s,beta) } } }
  var badT=0,badC=0
  for i in 0..<N { if r.t[i] != st[i] { badT+=1 }; if r.c[i] != ct[i] { badC+=1 } }
  let eG=energyAll(r.t,r.c), eC=energyAll(st,ct)
  let invG=verifyInvariants(r.t,r.c), invC=verifyInvariants(st,ct)
  let pass = badT==0 && badC==0 && r.accepted==cpuAcc && eG==eC && invG.ok && invC.ok
  print("")
  print(String(format:"validate C=1 beta=%.1f sweeps=%d",Double(beta),Int(sweeps)))
  print("  state mismatches: types=\(badT) counts=\(badC)")
  print("  accepted: GPU=\(r.accepted) CPU=\(cpuAcc)")
  print("  energy:   GPU=\(eG) CPU=\(eC)")
  print("  GPU \(invG.text)")
  print("  CPU \(invC.text)")
  print("  exactness: \(pass ? "PASS" : "FAIL")")
}

// ---------------------------------------------------------------------------
// Benchmark: same fixed-composition start for every beta, 1000 full sweeps.
// ---------------------------------------------------------------------------
let (t0,c0)=initState()
let initInv=verifyInvariants(t0,c0)
print("device=\(dev.name) maxTG=\(pso.maxTotalThreadsPerThreadgroup) threads=\(TG)")
print("initial energy=\(energyAll(t0,c0)); \(initInv.text)")

_ = run(sweeps:1,beta:2.8,from:(t0,c0)) // warmup (pipeline/state already built)

let betas:[Float]=[0.0,0.5,2.0,2.8,3.2,3.5]
let sweeps:UInt32=1000
let proposals=Double(sweeps)*Double(proposalsPerSweep)
print("")
print("beta   time(s)    proposals/s        M/s    final E   accepted  accepted%  invariants")
var allOK=initInv.ok
for beta in betas {
  let r=run(sweeps:sweeps,beta:beta,from:(t0,c0))
  let rate=proposals/r.seconds
  let e=energyAll(r.t,r.c)
  let inv=verifyInvariants(r.t,r.c)
  if !inv.ok { allOK=false }
  print(String(format:"%4.1f  %8.4f  %13.4e  %8.2f  %8d  %9d  %8.2f%%  %@",
               Double(beta),r.seconds,rate,rate/1e6,e,r.accepted,
               100.0*Double(r.accepted)/proposals,inv.text))
}
print("")
print(allOK ? "all invariant checks passed" : "INVARIANT FAILURES DETECTED")

validateC1()
