// DEPRECATED: this version leaves swapped-endpoint counts stale. Use poc/local_chain_opt.swift.
import Foundation
import Metal

let L=21, N=9261
func site(_ x:Int,_ y:Int,_ z:Int)->Int { x+L*y+L*L*z }
func neigh(_ s:Int,_ d:Int)->Int { var x=s%L,y=(s/L)%L,z=s/(L*L); if d==0{x=(x+1)%L}else if d==1{x=(x+L-1)%L}else if d==2{y=(y+1)%L}else if d==3{y=(y+L-1)%L}else if d==4{z=(z+1)%L}else{z=(z+L-1)%L}; return site(x,y,z) }
func initState()->([UInt8],[UInt8]) { var t=[UInt8](repeating:0,count:N); var c=[UInt8](repeating:0,count:N); var p=Array(0..<N); var x:UInt64=1000; for i in stride(from:N-1,through:1,by:-1){ x &+= 0x9e3779b97f4a7c15; var z=x; z=(z^(z>>30))&*0xbf58476d1ce4e5b9; z=(z^(z>>27))&*0x94d049bb133111eb; z^=z>>31; let j=Int(z%UInt64(i+1)); p.swapAt(i,j) }; for i in 0..<2315{t[p[i]]=0}; for i in 2315..<5093{t[p[i]]=1}; for i in 5093..<N{t[p[i]]=2}; for s in 0..<N{var k=0; for d in 0..<6{if t[neigh(s,d)] != 0{k+=1}}; c[s]=UInt8(k)}; return(t,c) }
func energy(_ t:Int,_ c:Int)->Int { if t==0{return 0}; let d=t==1 ? c-3 : c-5; return d*d }

let src = #"""
#include <metal_stdlib>
using namespace metal;
#define NN 9261
inline uint smix(uint x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x;}
inline uint h32(uint c,uint sw,uint ax,uint co,uint t){uint h=c*0x9e3779b9u;h^=sw*0x85ebca6bu;h^=ax*0xc2b2ae35u;h^=co*0x27d4eb2fu;h^=t*0x165667b1u;return smix(h);}
inline int en(int t,int c){if(t==0)return 0;int d=(t==1)?c-3:c-5;return d*d;}
inline uint siteOf(uint x,uint y,uint z){return x+21u*y+441u*z;}
inline uint nbr(uint s,uint d){int x=int(s%21u),y=int((s/21u)%21u),z=int(s/441u);if(d==0)x=(x+1)%21;else if(d==1)x=(x+20)%21;else if(d==2)y=(y+1)%21;else if(d==3)y=(y+20)%21;else if(d==4)z=(z+1)%21;else z=(z+20)%21;return siteOf(uint(x),uint(y),uint(z));}
inline void edge(threadgroup uchar *st, threadgroup uchar *ct, uint ax,uint co,uint t,float beta,uint sw){
    uint p1=t/21u,p2=t%21u;int p0=int(co)-3*int(p1)-5*int(p2);p0%=21;if(p0<0)p0+=21;
    uint q[3];q[ax]=uint(p0);q[(ax+1u)%3u]=p1;q[(ax+2u)%3u]=p2;
    uint a=siteOf(q[0],q[1],q[2]), b=nbr(a,ax);
    uchar ta=st[a],tb=st[b]; if(ta==tb)return;
    int d=(tb!=0)-(ta!=0);
    uint aff[12];int na=0;aff[na++]=a;aff[na++]=b;
    for(uint j=0;j<6u;j++){uint z=nbr(a,j);if(z!=b)aff[na++]=z;}
    for(uint j=0;j<6u;j++){uint z=nbr(b,j);if(z!=a)aff[na++]=z;}
    int nt[12];int nc[12];int de=0;
    for(int k=0;k<12;k++){uint z=aff[k];int ot=st[z],oc=ct[z];int tn=(z==a)?tb:((z==b)?ta:ot);int cn=oc;if(k>=2&&k<7)cn+=d;else if(k>=7)cn-=d;de+=en(tn,cn)-en(ot,oc);nt[k]=tn;nc[k]=cn;}
    uint hh=h32(0,sw,ax,co,t);float u=float(hh)*(1.0f/4294967296.0f);
    if(de<=0||u<exp(-beta*float(de))){for(int k=0;k<12;k++){uint z=aff[k];st[z]=uchar(nt[k]);ct[z]=uchar(nc[k]);}}
}
kernel void chain(device uchar *gt [[buffer(0)]],device uchar *gc [[buffer(1)]],constant uint &sweeps [[buffer(2)]],constant float &beta [[buffer(3)]],uint tid [[thread_position_in_threadgroup]]){
    threadgroup uchar st[NN];threadgroup uchar ct[NN];
    for(uint i=tid;i<NN;i+=512u){st[i]=gt[i];ct[i]=gc[i];}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=0;s<sweeps;s++){
      for(uint ax=0;ax<3u;ax++){
        for(uint co=0;co<21u;co++){
          if(tid<441u) edge(st,ct,ax,co,tid,beta,s);
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
    }
    for(uint i=tid;i<NN;i+=512u){gt[i]=st[i];gc[i]=ct[i];}
}
"""#
guard let dev=MTLCreateSystemDefaultDevice(), let q=dev.makeCommandQueue() else{fatalError("no metal")}
let lib=try dev.makeLibrary(source:src,options:nil)
let fn=lib.makeFunction(name:"chain")!
let pso=try dev.makeComputePipelineState(function:fn)
print("device=\(dev.name) maxTG=\(pso.maxTotalThreadsPerThreadgroup)")
let (t0,c0)=initState()
let tb=dev.makeBuffer(bytes:t0,length:N,options:.storageModeShared)!
let cb=dev.makeBuffer(bytes:c0,length:N,options:.storageModeShared)!
func run(_ sweeps:UInt32,_ beta:Float)->Double{
 let cmd=q.makeCommandBuffer()!;let enc=cmd.makeComputeCommandEncoder()!
 enc.setComputePipelineState(pso);enc.setBuffer(tb,offset:0,index:0);enc.setBuffer(cb,offset:0,index:1)
 var sw=sweeps,b=beta;enc.setBytes(&sw,length:4,index:2);enc.setBytes(&b,length:4,index:3)
 enc.dispatchThreadgroups(MTLSize(width:1,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:512,height:1,depth:1));enc.endEncoding();cmd.commit();cmd.waitUntilCompleted();return 0
}
for s in [1,10,100,1000] { let t=CFAbsoluteTimeGetCurrent(); _=run(UInt32(s),2.8); let dt=CFAbsoluteTimeGetCurrent()-t; let prop=Double(s)*3.0*9261.0; print(String(format:"sweeps=%4d time=%.4fs proposals=%.3e rate=%.3f M/s",s,dt,prop,prop/dt/1e6)) }
let gp=tb.contents().bindMemory(to:UInt8.self,capacity:N);let gcp=cb.contents().bindMemory(to:UInt8.self,capacity:N);var e=0;for i in 0..<N{e+=energy(Int(gp[i]),Int(gcp[i]))};print("energy=\(e)")

// CPU colored reference for the persistent kernel.
func smix(_ x0:UInt32)->UInt32{var x=x0;x^=x>>16;x&*=0x7feb352d;x^=x>>15;x&*=0x846ca68b;x^=x>>16;return x}
func hash32(_ c:Int,_ sw:Int,_ ax:Int,_ co:Int,_ t:Int)->UInt32{var h=UInt32(truncatingIfNeeded:c)&*0x9e3779b9;h^=UInt32(truncatingIfNeeded:sw)&*0x85ebca6b;h^=UInt32(truncatingIfNeeded:ax)&*0xc2b2ae35;h^=UInt32(truncatingIfNeeded:co)&*0x27d4eb2f;h^=UInt32(truncatingIfNeeded:t)&*0x165667b1;return smix(h)}
func anchor(_ axis:Int,_ color:Int,_ t:Int)->(Int,Int){let p1=t/21,p2=t%21;var p0=color-3*p1-5*p2;p0%=21;if p0<0{p0+=21};var q=[0,0,0];q[axis]=p0;q[(axis+1)%3]=p1;q[(axis+2)%3]=p2;let a=site(q[0],q[1],q[2]);return(a,neigh(a,axis))}
func applyCPU(_ st:inout[UInt8],_ ct:inout[UInt8],_ ax:Int,_ co:Int,_ sw:Int,_ beta:Float){for t in 0..<441{let(a,b)=anchor(ax,co,t);let ta=st[a],tb=st[b];if ta==tb{continue};let d=(tb != 0 ? 1:0)-(ta != 0 ? 1:0);var aff=[a,b];for j in 0..<6{let z=neigh(a,j);if z != b{aff.append(z)}};for j in 0..<6{let z=neigh(b,j);if z != a{aff.append(z)}};var nt=[Int](repeating:0,count:12),nc=[Int](repeating:0,count:12),de=0;for k in 0..<12{let z=aff[k];let ot=Int(st[z]),oc=Int(ct[z]);let tn=(z==a) ? Int(tb):((z==b) ? Int(ta):ot);var cn=oc;if k>=2&&k<7{cn+=d}else if k>=7{cn-=d};de+=energy(tn,cn)-energy(ot,oc);nt[k]=tn;nc[k]=cn};let hh=hash32(0,sw,ax,co,t);let u=Float(hh)*(1.0/4294967296.0);if de<=0||u<exp(-beta*Float(de)){for k in 0..<12{st[aff[k]]=UInt8(nt[k]);ct[aff[k]]=UInt8(nc[k])}}}}
func validateChain(){let(it,ic)=initState();let tb=dev.makeBuffer(bytes:it,length:N,options:.storageModeShared)!;let cb=dev.makeBuffer(bytes:ic,length:N,options:.storageModeShared)!;var st=it,ct=ic;let cmd=q.makeCommandBuffer()!;let enc=cmd.makeComputeCommandEncoder()!;enc.setComputePipelineState(pso);enc.setBuffer(tb,offset:0,index:0);enc.setBuffer(cb,offset:0,index:1);var sw:UInt32=2,beta:Float=2.8;enc.setBytes(&sw,length:4,index:2);enc.setBytes(&beta,length:4,index:3);enc.dispatchThreadgroups(MTLSize(width:1,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:512,height:1,depth:1));enc.endEncoding();cmd.commit();cmd.waitUntilCompleted();for s in 0..<2{for ax in 0..<3{for co in 0..<21{applyCPU(&st,&ct,ax,co,s,beta)}}};let gp=tb.contents().bindMemory(to:UInt8.self,capacity:N);let gc=cb.contents().bindMemory(to:UInt8.self,capacity:N);var bad=0;for i in 0..<N{if gp[i] != st[i] || gc[i] != ct[i]{bad+=1}};print("persistent validate mismatches=\(bad)")}
validateChain()
