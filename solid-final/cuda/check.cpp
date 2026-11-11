#include "../common/sweeps_cpu.hpp"
#ifndef SOLID_HAVE_CUDA
#define SOLID_HAVE_CUDA 0
#endif
#if SOLID_HAVE_CUDA
#include "backend.hpp"
#else
// The on-disk ABI is shared with the CUDA build without requiring CUDA headers.
namespace solid {
struct CudaTrace { uint64_t proposal; uint32_t a,b; int32_t delta; uint32_t flags; };
}
#endif
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

using namespace solid;
namespace {
static_assert(sizeof(CudaTrace)==24,"fixture trace ABI must remain 24 bytes");
constexpr uint64_t kMagic=0x5343554441303031ULL;
struct Fixture {
  uint32_t kind=0,L=21,chains=1,mode=1,beta=3200,sweeps=0,traceLimit=0;
  uint64_t seed=1,offset=0,start=0;
  std::vector<Threshold> thresholds;
  std::vector<uint8_t> initial;
  std::vector<uint32_t> checkpoints,energy,local,nonlocal;
  std::vector<std::vector<uint8_t>> states;
  std::vector<CudaTrace> trace;
};
template<class T> void put(std::ostream& s,const T& x) { s.write(reinterpret_cast<const char*>(&x),sizeof(x)); }
template<class T> void get(std::istream& s,T& x) {
  s.read(reinterpret_cast<char*>(&x),sizeof(x));
  if(!s) throw std::runtime_error("truncated fixture");
}
template<class T> void putv(std::ostream& s,const std::vector<T>& v) {
  put(s,uint64_t(v.size()));
  if(!v.empty()) s.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));
}
template<class T> void getv(std::istream& s,std::vector<T>& v) {
  uint64_t n;get(s,n);
  if(n>(1ULL<<31)/sizeof(T)) throw std::runtime_error("invalid fixture vector length");
  v.resize(size_t(n));
  if(n) s.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));
  if(!s) throw std::runtime_error("truncated fixture vector");
}
void save(const std::string& path,const std::vector<Fixture>& fs) {
  std::ofstream s(path,std::ios::binary);
  put(s,kMagic);put(s,uint32_t(fs.size()));
  for(const auto& f:fs) {
    put(s,f.kind);put(s,f.L);put(s,f.chains);put(s,f.mode);put(s,f.beta);put(s,f.sweeps);put(s,f.traceLimit);
    put(s,f.seed);put(s,f.offset);put(s,f.start);putv(s,f.thresholds);putv(s,f.initial);putv(s,f.checkpoints);
    for(const auto& v:f.states) putv(s,v);
    putv(s,f.energy);putv(s,f.local);putv(s,f.nonlocal);putv(s,f.trace);
  }
  if(!s) throw std::runtime_error("cannot write fixture: "+path);
}
std::vector<Fixture> load(const std::string& path) {
  std::ifstream s(path,std::ios::binary);uint64_t magic;get(s,magic);
  if(magic!=kMagic) throw std::runtime_error("invalid fixture magic");
  uint32_t n;get(s,n);
  if(n>1000) throw std::runtime_error("invalid fixture count");
  std::vector<Fixture> fs(n);
  for(auto& f:fs) {
    get(s,f.kind);get(s,f.L);get(s,f.chains);get(s,f.mode);get(s,f.beta);get(s,f.sweeps);get(s,f.traceLimit);
    get(s,f.seed);get(s,f.offset);get(s,f.start);getv(s,f.thresholds);getv(s,f.initial);getv(s,f.checkpoints);
    f.states.resize(f.checkpoints.size());for(auto& v:f.states)getv(s,v);
    getv(s,f.energy);getv(s,f.local);getv(s,f.nonlocal);getv(s,f.trace);
    if((f.L!=4&&f.L!=8&&f.L!=21)||!f.chains||f.mode>2||f.checkpoints.empty()||f.checkpoints.back()!=f.sweeps)
      throw std::runtime_error("invalid fixture dimensions");
    const size_t bytes=size_t(f.chains)*f.L*f.L*f.L,hist=size_t(f.chains)*f.sweeps;
    if(f.initial.size()!=bytes||f.energy.size()!=hist||f.local.size()!=hist||f.nonlocal.size()!=hist||f.trace.size()!=size_t(f.chains)*f.traceLimit)
      throw std::runtime_error("invalid fixture payload dimensions");
    uint32_t previous=0;
    for(size_t i=0;i<f.checkpoints.size();++i) {
      if(f.checkpoints[i]<=previous||f.states[i].size()!=bytes)throw std::runtime_error("invalid fixture checkpoint");
      previous=f.checkpoints[i];
    }
  }
  return fs;
}
// Independent geometry, masks and H: none of these use Lattice::nn, energy(),
// packedEnergy(), addAff(), nearPair(), or the incremental update machinery.
int neighbor(int s,int d,int L) {
  int x=s%L,y=(s/L)%L,z=s/(L*L);
  int* c=d<2?&x:(d<4?&y:&z);
  *c=(*c+((d&1)?L-1:1))%L;
  return x+L*y+L*L*z;
}
int independentSiteEnergy(int type,int mask) {
  if(!type)return 0;
  int count=0;for(int d=0;d<6;++d)count+=(mask>>d)&1;
  int q=count-(type==1?3:5);return q*q;
}
int rebuild(const std::vector<uint8_t>& input,int L,std::vector<uint8_t>* packed=nullptr) {
  int total=0;
  if(packed)packed->resize(input.size());
  for(size_t i=0;i<input.size();++i) {
    int t=input[i]&3,mask=0;
    if(t==3)throw std::runtime_error("invalid species in independent H");
    for(int d=0;d<6;++d)if(input[neighbor(int(i),d,L)]&3)mask|=1<<d;
    total+=independentSiteEnergy(t,mask);
    if(packed)(*packed)[i]=uint8_t(t|(mask<<2));
  }
  return total;
}
// Trace deltas use an independently constructed, deduplicated stencil. The CPU
// audit below compares this evaluator and the production delta against FULL H.
int stencilDelta(const Lattice& lat,const Proposal& p) {
  if(!p.valid||((lat.st[p.a]&3)==(lat.st[p.b]&3)))return 0;
  int sites[14],count=0;
  auto add=[&](int s){for(int i=0;i<count;++i)if(sites[i]==s)return;sites[count++]=s;};
  add(p.a);add(p.b);
  for(int d=0;d<6;++d){add(neighbor(p.a,d,lat.L));add(neighbor(p.b,d,lat.L));}
  auto type=[&](int s,bool after){return int(lat.st[after?(s==p.a?p.b:(s==p.b?p.a:s)):s]&3);};
  int delta=0;
  for(int i=0;i<count;++i)for(int after=0;after<2;++after) {
    int mask=0;for(int d=0;d<6;++d)if(type(neighbor(sites[i],d,lat.L),after))mask|=1<<d;
    delta+=(after?1:-1)*independentSiteEnergy(type(sites[i],after),mask);
  }
  return delta;
}
// Frozen CPU's public drawProposal takes int chain. This adapter is identical
// for representable positive IDs; larger IDs are explicitly the natural
// full-width counter-key extension, NOT a capability of the old CPU driver.
Proposal draw64(const Lattice& lat,uint64_t seed,uint64_t chain,uint64_t index,bool local) {
  Proposal p;p.local=local;p.rng.key=proposalKey(seed,chain,index,local);
  p.a=int(uniformIndex(p.rng,uint32_t(lat.N),lat.lemN));
  p.b=local?lat.nb(p.a,int(uniformIndex(p.rng,6,lat.lem6))):int(uniformIndex(p.rng,uint32_t(lat.N),lat.lemN));
  p.valid=local||p.a!=p.b;
  if(p.valid)p.u_hi=p.rng.next();
  return p;
}
int sweep64(Lattice& lat,uint64_t seed,uint64_t chain,uint64_t sweep,bool local,const std::vector<Threshold>& th,int& energy) {
  int accepted=0;
  const uint64_t base=(2*sweep+uint64_t(local))*uint64_t(lat.N);
  for(int j=0;j<lat.N;++j){auto p=draw64(lat,seed,chain,base+uint64_t(j),local);accepted+=tryProposal(lat,p,th,energy);}
  return accepted;
}
const std::vector<Threshold>& thresholdTable(int beta) {
  static std::map<int,std::vector<Threshold>> tables;
  auto it=tables.find(beta);if(it!=tables.end())return it->second;
  std::ostringstream text;text<<std::fixed<<std::setprecision(3)<<double(beta)/1000;
  return tables.emplace(beta,makeThresholds(text.str())).first->second;
}
Fixture fixture(uint32_t kind,int L,int chains,int mode,int beta,uint64_t offset,uint64_t seed) {
  Fixture f;f.kind=kind;f.L=L;f.chains=chains;f.mode=mode;f.beta=beta;f.offset=offset;f.seed=seed;
  uint32_t per=uint32_t(L*L*L)*(mode==0?2:1);
  if(kind==2)f.checkpoints={1,2};
  else if(kind==4)f.checkpoints={1,2,3};
  else if(kind==3)f.checkpoints={1,2,(100000000u+per-1)/per};
  else f.checkpoints={1,2,(10000u+per-1)/per,(100000u+per-1)/per,(1000000u+per-1)/per};
  std::sort(f.checkpoints.begin(),f.checkpoints.end());
  f.checkpoints.erase(std::unique(f.checkpoints.begin(),f.checkpoints.end()),f.checkpoints.end());
  f.sweeps=f.checkpoints.back();
  f.traceLimit=kind==2?0:uint32_t(std::min<uint64_t>(100000,uint64_t(f.sweeps)*per));
  f.start=(kind==0||kind==1)&&offset?128:0;
  f.thresholds=thresholdTable(beta);
  size_t N=size_t(L)*L*L;
  f.initial.resize(size_t(chains)*N);f.states.resize(f.checkpoints.size());for(auto& v:f.states)v.resize(size_t(chains)*N);
  f.energy.resize(size_t(chains)*f.sweeps);f.local.resize(f.energy.size());f.nonlocal.resize(f.energy.size());
  f.trace.resize(size_t(chains)*f.traceLimit);
  return f;
}
void reference(Fixture& f,uint32_t c) {
  Lattice lat(f.L);const int N=lat.N;const uint64_t id=f.offset+c;
  lat.init(f.seed+id*EFMA_GOLDEN,int(std::llround(.30*N)),int(std::llround(.45*N)));int e=lat.energy();
  // Boundary fixtures start at a high counter without iterating from zero.
  const uint64_t warmup=f.kind==4?8:f.start;
  for(uint64_t sw=0;sw<warmup;++sw){if(f.mode!=2)sweep64(lat,f.seed,id,sw,false,f.thresholds,e);if(f.mode!=1)sweep64(lat,f.seed,id,sw,true,f.thresholds,e);}
  std::copy(lat.st.begin(),lat.st.end(),f.initial.begin()+size_t(c)*N);
  uint64_t ordinal=0;size_t cp=0;
  for(uint32_t sw=0;sw<f.sweeps;++sw) {
    int an=0,al=0;
    for(int which=0;which<2;++which) {
      bool local=which==1;if((local&&f.mode==1)||(!local&&f.mode==2))continue;
      const uint64_t base=(2*(f.start+sw)+uint64_t(local))*uint64_t(N);
      for(int j=0;j<N;++j,++ordinal) {
        Proposal p=draw64(lat,f.seed,id,base+uint64_t(j),local);
        bool noop=p.valid&&siteType(lat.st[p.a])==siteType(lat.st[p.b]);
        int d=ordinal<f.traceLimit?stencilDelta(lat,p):0;
        bool accepted=tryProposal(lat,p,f.thresholds,e);if(accepted){if(local)++al;else ++an;}
        if(ordinal<f.traceLimit)f.trace[size_t(c)*f.traceLimit+ordinal]={base+uint64_t(j),uint32_t(p.a),uint32_t(p.b),d,uint32_t(p.valid)|(uint32_t(noop)<<1)|(uint32_t(accepted)<<2)};
      }
    }
    int full=lat.energy();if(e!=full)throw std::runtime_error("CPU incremental/packed energy mismatch while preparing");
    size_t index=size_t(c)*f.sweeps+sw;f.energy[index]=full;f.local[index]=al;f.nonlocal[index]=an;
    if(cp<f.checkpoints.size()&&sw+1==f.checkpoints[cp]) {
      std::vector<uint8_t> masks;
      if(rebuild(lat.st,lat.L,&masks)!=e||masks!=lat.st)throw std::runtime_error("CPU independent H/mask mismatch at fixture checkpoint");
      std::copy(lat.st.begin(),lat.st.end(),f.states[cp].begin()+size_t(c)*N);++cp;
    }
  }
}
void prepare(const std::string& path,bool final,uint64_t finalSeed,uint64_t finalChain) {
  std::vector<Fixture> fs;
  if(final) {
    if(finalChain<1000000||finalChain>uint64_t(INT_MAX))throw std::runtime_error("final chain must be fresh, >=1000000 and <=INT_MAX");
    if(finalSeed==15637821895788547078ULL||finalSeed==4585611281380521951ULL||finalSeed==1||finalSeed==7||finalSeed==0xfedcba9876543210ULL||finalSeed==0x123456789abcdef0ULL||finalSeed==123)
      throw std::runtime_error("final seed must not reuse a known development/historical final stream");
    for(int mode:{1,2})fs.push_back(fixture(3,21,1,mode,3200,finalChain,finalSeed));
  } else {
    for(int L:{8,21})for(int b:{2800,3200})for(uint64_t id:{0ULL,1ULL,12345ULL})for(int mode:{1,2})
      fs.push_back(fixture(0,L,1,mode,b,id,id==1?0xfedcba9876543210ULL:1));
    for(int L:{8,21})for(uint64_t id:{0ULL,12345ULL})for(int mode:{0,1,2})fs.push_back(fixture(1,L,3,mode,3200,id,0x123456789abcdef0ULL));
    for(int mode:{1,2})fs.push_back(fixture(2,21,1536,mode,3200,0,123));
    const uint64_t starts[]={uint64_t(UINT32_MAX)-1,(1ULL<<63)-1,UINT64_MAX-1};
    const uint64_t offsets[]={uint64_t(UINT32_MAX)+7,(1ULL<<63)+7,UINT64_MAX-3};
    for(int L:{4,8,21})for(int mode:{0,1,2})for(int i=0;i<3;++i) {
      auto f=fixture(4,L,2,mode,3200,offsets[i],0xfedcba9876543210ULL);f.start=starts[i];fs.push_back(std::move(f));
    }
  }
  std::vector<std::pair<size_t,uint32_t>> jobs;
  for(size_t i=0;i<fs.size();++i)for(uint32_t c=0;c<fs[i].chains;++c)jobs.emplace_back(i,c);
  std::atomic<size_t> next{0};std::atomic<bool> failed{false};std::exception_ptr error;std::mutex errorMutex;
  std::vector<std::thread> workers;
  for(unsigned t=0;t<std::min(8u,std::max(1u,std::thread::hardware_concurrency()));++t)workers.emplace_back([&]{
    try {
      while(!failed.load()) {size_t j=next.fetch_add(1);if(j>=jobs.size())break;reference(fs[jobs[j].first],jobs[j].second);}
    } catch(...) {std::lock_guard<std::mutex> lock(errorMutex);if(!error)error=std::current_exception();failed=true;}
  });
  for(auto& t:workers)t.join();if(error)std::rethrow_exception(error);
  save(path,fs);
  std::cout<<"{\"kind\":\"prepared\",\"fixtures\":"<<fs.size()<<",\"final\":"<<(final?"true":"false")<<",\"trace_record_bytes\":24,\"cpu_authority\":\"solid-opt2/cpu unchanged\",\"full64_chain_reference\":\"natural counter-key extension beyond INT_MAX\"";
  if(final)std::cout<<",\"final_seed\":"<<finalSeed<<",\"final_chain\":"<<finalChain;
  std::cout<<"}\n";
}
void require(bool condition,const std::string& message) {if(!condition)throw std::runtime_error(message);}
void audit() {
  uint64_t cases=0,normalAccepts=0,uphill=0;
  std::vector<Threshold> force(1024,Threshold{UINT32_MAX,UINT32_MAX});
  for(int L:{4,8,21})for(int beta:{10,1000,2000,2500,2800,3200,3400})for(int warmed:{0,1}) {
    Lattice state(L);state.init(0x123456789abcdef0ULL+uint64_t(beta),int(std::llround(.30*state.N)),int(std::llround(.45*state.N)));
    const auto& th=thresholdTable(beta);int energy=state.energy();
    if(warmed)for(uint64_t s=0;s<16;++s){nonlocalSweep(state,7,12345,s,th,energy);localSweep(state,7,12345,s,th,energy);}
    std::vector<uint8_t> rebuilt;
    require(rebuild(state.st,L,&rebuilt)==energy&&rebuilt==state.st,"audit warmed-state independent H/mask mismatch");
    for(int anchor:{0,state.N-1,state.N/2}) {
      std::set<int> partners;
      for(int d=0;d<6;++d) {
        int b=neighbor(anchor,d,L);partners.insert(b);
        for(int e=0;e<6;++e)partners.insert(neighbor(b,e,L));
      }
      partners.insert((anchor+(L/2)+L*(L/2)+L*L*(L/2))%state.N);
      partners.insert(anchor);
      for(int b:partners)for(int ta=0;ta<3;++ta)for(int tb=0;tb<3;++tb) {
        if(anchor==b&&ta!=tb)continue;
        Lattice before=state;before.st[anchor]=uint8_t(ta);before.st[b]=uint8_t(tb);
        int old=rebuild(before.st,L,&rebuilt);before.st=rebuilt;
        bool adjacent=false;for(int d=0;d<6;++d)adjacent|=neighbor(anchor,d,L)==b;
        for(int local=0;local<=(adjacent?1:0);++local) {
          Proposal p;p.a=anchor;p.b=b;p.valid=anchor!=b;p.local=bool(local);
          p.rng.key=proposalKey(7,12345,cases,local);p.u_hi=p.rng.next();
          std::vector<uint8_t> swapped=before.st,after;
          if(p.valid)std::swap(swapped[p.a],swapped[p.b]);
          const int newEnergy=rebuild(swapped,L,&after),delta=newEnergy-old;
          require(stencilDelta(before,p)==delta,"audit independent stencil/full H delta mismatch");
          auto forced=p;forced.u_hi=0;Lattice committed=before;int incremental=old;
          bool did=tryProposal(committed,forced,force,incremental);
          bool changing=p.valid&&ta!=tb;
          require(did==changing&&incremental==newEnergy&&committed.st==after,"audit production incremental delta/commit differs from full H/recomputed masks");
          bool expected=false;
          if(changing) {
            if(delta<=0)expected=true;
            else if(delta<int(th.size())) {
              const auto t=th[delta];auto words=p.rng;
              expected=p.u_hi<t.hi||(p.u_hi==t.hi&&words.next()<t.lo);
            }
          }
          Lattice actual=before;incremental=old;auto normal=p;
          bool accepted=tryProposal(actual,normal,th,incremental);
          require(accepted==expected&&incremental==(expected?newEnergy:old)&&actual.st==(expected?after:before.st),"audit EFMA proposal outcome differs from independent full H");
          normalAccepts+=accepted;uphill+=delta>0;++cases;
        }
      }
    }
  }
  // Check word cursor, Lemire rejection and uint64 proposal-index wrap around
  // boundaries against the actual unchanged int-chain CPU entry point.
  uint64_t draws=0;
  for(int L:{4,8,21}) {
    Lattice lat(L);
    for(int chain:{0,1,12345,INT_MAX})for(uint64_t index:{0ULL,1ULL,0xffffffffULL,0x100000000ULL,0x7fffffffffffffffULL,0xfffffffffffffffeULL,0xffffffffffffffffULL})for(bool local:{false,true}) {
      auto a=draw64(lat,0xfedcba9876543210ULL,uint64_t(chain),index,local);
      auto b=drawProposal(lat,0xfedcba9876543210ULL,chain,index,local);
      require(a.a==b.a&&a.b==b.b&&a.valid==b.valid&&a.u_hi==b.u_hi,"audit full64 adapter differs from frozen CPU draw");
      for(int i=0;i<8;++i)require(a.rng.next()==b.rng.next(),"audit RNG continuation differs");
      ++draws;
    }
    for(uint64_t chain:{0ULL,12345ULL,0x100000007ULL,0x8000000000000007ULL,0xfffffffffffffffcULL})for(uint64_t start:{0ULL,0xfffffffeULL,0x7fffffffffffffffULL,0xfffffffffffffffeULL})for(int mode:{0,1,2}) {
      Lattice whole(L);whole.init(7,int(std::llround(.30*whole.N)),int(std::llround(.45*whole.N)));Lattice split=whole;
      int ew=whole.energy(),es=ew;const auto& th=thresholdTable(3200);
      for(uint64_t j=0;j<3;++j){if(mode!=2)sweep64(whole,7,chain,start+j,false,th,ew);if(mode!=1)sweep64(whole,7,chain,start+j,true,th,ew);}
      for(uint64_t j=0;j<3;++j) {
        // Recreate lattice caches from checkpoint bytes at every restart.
        Lattice restarted(L);restarted.st=split.st;split=std::move(restarted);
        if(mode!=2)sweep64(split,7,chain,start+j,false,th,es);
        if(mode!=1)sweep64(split,7,chain,start+j,true,th,es);
      }
      std::vector<uint8_t> masks;
      require(whole.st==split.st&&ew==es&&rebuild(split.st,L,&masks)==es&&masks==split.st,"audit counter/restart independent H/mask mismatch");
    }
  }
  // Explicitly record the inherited beta=0 defect without changing authority.
  auto zero=makeThresholds("0");
  require(zero[1].hi==0&&zero[1].lo==0,"inherited beta-zero threshold behavior changed");
  std::cout<<"{\"kind\":\"audit\",\"ok\":true,\"full_H_cases\":"<<cases<<",\"normal_accepts\":"<<normalAccepts<<",\"uphill_cases\":"<<uphill<<",\"CPU_adapter_draws\":"<<draws<<",\"L\":[4,8,21],\"beta\":[0.01,1,2,2.5,2.8,3.2,3.4],\"states\":[\"random\",\"warmed\"],\"geometry\":[\"adjacent\",\"distance2\",\"far\",\"self\"],\"beta0_gibbs_validated\":false,\"beta0_inherited_defect\":\"floor(2^64) wraps to zero split words; uphill acceptance zero\",\"full64_chain_reference\":\"natural counter-key extension beyond INT_MAX\"}\n";
}
#if SOLID_HAVE_CUDA
template<class A,class B> [[noreturn]] void mismatch(const Fixture& f,const CudaLaunch& launch,int chain,uint64_t proposal,const char* field,A expected,B got) {
  std::cout<<"{\"kind\":\"mismatch\",\"L\":"<<f.L<<",\"mode\":"<<f.mode<<",\"beta_milli\":"<<f.beta<<",\"chain\":"<<f.offset+uint64_t(chain)<<",\"window\":"<<launch.window<<",\"threads\":"<<launch.threads<<",\"ppt\":"<<launch.proposals_per_thread<<",\"proposal_or_index\":"<<proposal<<",\"field\":"<<std::quoted(field)<<",\"expected\":"<<+expected<<",\"got\":"<<+got<<"}\n";
  throw std::runtime_error("CUDA/CPU first-field mismatch");
}
void checkOne(const Fixture& f,int chains,const CudaLaunch& launch,bool smoke) {
  int N=f.L*f.L*f.L,moves=f.mode==0?2:1;
  std::vector<uint8_t> initial(f.initial.begin(),f.initial.begin()+size_t(chains)*N);
  CudaBackend gpu(f.L,chains,f.seed,initial,f.thresholds,launch,f.offset);
  gpu.reserve(f.sweeps,f.traceLimit);uint32_t before=0;
  for(size_t cp=0;cp<f.checkpoints.size();++cp) {
    uint32_t end=f.checkpoints[cp];if(smoke&&end>2)break;
    uint32_t tracedBefore=uint32_t(std::min<uint64_t>(f.traceLimit,uint64_t(before)*N*moves));
    uint32_t traceCount=uint32_t(std::min<uint64_t>(f.traceLimit-tracedBefore,uint64_t(end-before)*N*moves));
    CudaResult r;gpu.run(f.start+before,end-before,f.mode,r,traceCount);
    require(r.state.size()==size_t(chains)*N&&r.energy.size()==size_t(chains)*(end-before)&&r.acceptLocal.size()==r.energy.size()&&r.acceptNonlocal.size()==r.energy.size()&&r.trace.size()==size_t(chains)*traceCount,"CUDA result shape mismatch");
    // Compare proposal fields BEFORE aggregate state/history so the trace gives
    // the first differing proposal, not an opaque later checkpoint mismatch.
    for(uint32_t j=0;j<traceCount;++j)for(int c=0;c<chains;++c) {
      const auto& a=f.trace[size_t(c)*f.traceLimit+tracedBefore+j];const auto& b=r.trace[size_t(c)*traceCount+j];
      if(a.proposal!=b.proposal)mismatch(f,launch,c,a.proposal,"proposal",a.proposal,b.proposal);
      if(a.a!=b.a)mismatch(f,launch,c,a.proposal,"a",a.a,b.a);
      if(a.b!=b.b)mismatch(f,launch,c,a.proposal,"b",a.b,b.b);
      if(a.delta!=b.delta)mismatch(f,launch,c,a.proposal,"delta",a.delta,b.delta);
      if(a.flags!=b.flags)mismatch(f,launch,c,a.proposal,"valid/noop/accept",a.flags,b.flags);
    }
    for(uint32_t j=0;j<end-before;++j)for(int c=0;c<chains;++c) {
      size_t src=size_t(c)*f.sweeps+before+j,dst=size_t(c)*(end-before)+j;
      if(r.energy[dst]!=f.energy[src])mismatch(f,launch,c,before+j,"energy",f.energy[src],r.energy[dst]);
      if(r.acceptLocal[dst]!=f.local[src])mismatch(f,launch,c,before+j,"local_acceptance",f.local[src],r.acceptLocal[dst]);
      if(r.acceptNonlocal[dst]!=f.nonlocal[src])mismatch(f,launch,c,before+j,"nonlocal_acceptance",f.nonlocal[src],r.acceptNonlocal[dst]);
    }
    for(int c=0;c<chains;++c)for(int j=0;j<N;++j)if(r.state[size_t(c)*N+j]!=f.states[cp][size_t(c)*N+j])
      mismatch(f,launch,c,uint64_t(j),"state_byte_at_checkpoint",f.states[cp][size_t(c)*N+j],r.state[size_t(c)*N+j]);
    std::cout<<"{\"kind\":\"checkpoint\",\"ok\":true,\"fixture_kind\":"<<f.kind<<",\"L\":"<<f.L<<",\"beta_milli\":"<<f.beta<<",\"mode\":"<<f.mode<<",\"seed\":"<<f.seed<<",\"chain_offset\":"<<f.offset<<",\"start_sweep\":"<<f.start<<",\"chains\":"<<chains<<",\"window\":"<<launch.window<<",\"threads\":"<<launch.threads<<",\"ppt\":"<<launch.proposals_per_thread<<",\"profile\":"<<(launch.profile?"true":"false")<<",\"sweeps_per_chain\":"<<end<<",\"attempts_per_chain\":"<<uint64_t(end)*N*moves<<",\"aggregate_attempts\":"<<uint64_t(end)*N*moves*chains<<",\"trace_compared_per_chain\":"<<traceCount<<"}\n";
    before=end;
  }
}
void check(const std::string& file,const std::string& stage,const std::vector<int>& windows,const std::vector<int>& threads,const std::vector<int>& ppt,const std::string& move,bool profile) {
  require(stage=="smoke"||stage=="single"||stage=="multi"||stage=="saturated"||stage=="final","unknown correctness stage");
  auto fs=load(file);auto info=cudaDeviceInfo();uint64_t count=0;
  for(const auto& f:fs) {
    bool selected=(stage=="single"&&(f.kind==0||f.kind==4))||(stage=="multi"&&f.kind==1)||(stage=="saturated"&&f.kind==2)||(stage=="final"&&f.kind==3)||(stage=="smoke"&&((f.kind==0&&f.offset==0&&f.beta==2800)||f.kind==4||f.kind==3));
    if(!selected||(move=="local"&&f.mode!=2)||(move=="nonlocal"&&f.mode!=1))continue;
    for(int p:ppt)for(int t:threads)for(int w:windows) {
      // The generic Cartesian launch domain is W<=T*PPT. Any otherwise
      // unsupported setting reaches the backend, which MUST reject it.
      if(int64_t(w)>int64_t(t)*int64_t(p))continue;
      CudaLaunch launch;launch.window=w;launch.threads=t;launch.proposals_per_thread=p;launch.profile=profile;
      if(f.kind==2)for(int factor:{1,2,4,8}) {
        int chains=info.sms*factor;require(chains>0&&chains<=int(f.chains),"fixture too small for saturation gate");
        checkOne(f,chains,launch,false);++count;
      } else {checkOne(f,f.chains,launch,stage=="smoke");++count;}
    }
  }
  require(count>0,"empty correctness stage or launch matrix");
  std::cout<<"{\"kind\":\"correctness\",\"stage\":"<<std::quoted(stage)<<",\"move\":"<<std::quoted(move)<<",\"checks\":"<<count<<",\"ok\":true,\"sms\":"<<info.sms<<",\"name\":"<<std::quoted(info.name)<<"}\n";
}
std::string fingerprint(const std::vector<uint8_t>& bytes) {
  uint64_t hash=14695981039346656037ULL;for(uint8_t byte:bytes){hash^=byte;hash*=1099511628211ULL;}
  std::ostringstream text;text<<std::hex<<std::setw(16)<<std::setfill('0')<<hash;return text.str();
}
void identity(const char* key,const char* variable) {
  std::cout<<",\""<<key<<"\":";const char* value=std::getenv(variable);
  if(value&&*value)std::cout<<std::quoted(value);else std::cout<<"null";
}
void bench(int L,const std::string& beta,int chains,CudaLaunch launch,const std::string& move,uint32_t warmup,uint32_t sweeps,int repeats) {
  require(move=="local"||move=="nonlocal","benchmark requires one move");
  require(chains>0&&sweeps>0&&repeats>0,"benchmark counts must be positive");
  const boost::multiprecision::cpp_dec_float_100 betaValue(beta);
  require(betaValue>0&&betaValue<std::numeric_limits<boost::multiprecision::cpp_dec_float_100>::infinity(),"benchmark beta must be finite and positive; inherited beta0 is not production physics");
  // A tiny CPU/GPU smoke is mandatory even for standalone benchmark calls.
  auto tiny=fixture(4,4,1,move=="local"?2:1,3200,0,1);reference(tiny,0);checkOne(tiny,1,launch,true);
  int mode=move=="local"?2:1,N=L*L*L;auto th=makeThresholds(beta);
  std::vector<uint8_t> initial(size_t(chains)*N);
  for(int c=0;c<chains;++c) {
    Lattice lat(L);lat.init(1+uint64_t(c)*EFMA_GOLDEN,int(std::llround(.30*N)),int(std::llround(.45*N)));
    std::copy(lat.st.begin(),lat.st.end(),initial.begin()+size_t(c)*N);
  }
  CudaLaunch warmLaunch=launch;warmLaunch.profile=false;
  if(warmup) {
    CudaBackend warmer(L,chains,1,initial,th,warmLaunch);warmer.reserve(warmup);
    CudaResult aged;warmer.run(0,warmup,mode,aged);initial=std::move(aged.state);
  }
  const std::string checkpoint=fingerprint(initial);const auto info=cudaDeviceInfo();
  const uint64_t attempts=uint64_t(sweeps)*N,totalAttempts=attempts*uint64_t(chains);
  for(int repeat=0;repeat<repeats;++repeat) {
    // Construction, upload and reserve are outside the backend's CUDA events.
    // No repeat inherits the previous repeat's final state or counter.
    CudaBackend gpu(L,chains,1,initial,th,launch);gpu.reserve(sweeps);CudaResult r;
    gpu.run(uint64_t(warmup),sweeps,mode,r);
    require(r.milliseconds>0&&std::isfinite(r.milliseconds),"invalid CUDA event time");
    require(r.energy.size()==size_t(chains)*sweeps&&r.acceptLocal.size()==r.energy.size()&&r.acceptNonlocal.size()==r.energy.size(),"benchmark result history shape mismatch");
    uint64_t al=0,an=0;for(uint32_t n:r.acceptLocal)al+=n;for(uint32_t n:r.acceptNonlocal)an+=n;
    require(al+an<=totalAttempts,"benchmark acceptance exceeds attempts");
    std::set<uint32_t> smids;bool measured=r.stats.size()==size_t(chains);
    uint64_t rounds=0,evaluated=0,finalized=0,replay=0,cycles[6]={};
    for(const auto& stat:r.stats) {
      smids.insert(stat.smid);measured=measured&&stat.measured;
      if(stat.measured){rounds+=stat.rounds;evaluated+=stat.evaluated;finalized+=stat.finalized;replay+=stat.replay_rounds;for(int i=0;i<6;++i)cycles[i]+=stat.cycles[i];}
    }
    require(!launch.profile||measured,"profile run did not return measured stage counters");
    double seconds=r.milliseconds*.001;
    std::cout<<std::setprecision(17)<<"{\"kind\":\"benchmark\",\"L\":"<<L<<",\"beta\":"<<std::quoted(beta)<<",\"move\":"<<std::quoted(move)<<",\"chains\":"<<chains<<",\"window\":"<<launch.window<<",\"threads\":"<<launch.threads<<",\"ppt\":"<<launch.proposals_per_thread<<",\"profile\":"<<(launch.profile?"true":"false")<<",\"ordinary_throughput\":"<<(launch.profile?"false":"true")<<",\"repeat\":"<<repeat<<",\"repeats\":"<<repeats<<",\"seed\":1,\"start_sweep\":"<<warmup<<",\"warmup_sweeps\":"<<warmup<<",\"sweeps_per_chain\":"<<sweeps<<",\"attempts_per_chain\":"<<attempts<<",\"aggregate_attempts\":"<<totalAttempts<<",\"cuda_seconds\":"<<seconds<<",\"per_chain_attempts_s\":";
    if(launch.profile)std::cout<<"null";else std::cout<<attempts/seconds;
    std::cout<<",\"aggregate_attempts_s\":";if(launch.profile)std::cout<<"null";else std::cout<<totalAttempts/seconds;
    std::cout<<",\"accepted_local\":"<<al<<",\"accepted_nonlocal\":"<<an<<",\"unaccepted_including_noops\":"<<totalAttempts-al-an<<",\"checkpoint_fnv1a64\":"<<std::quoted(checkpoint)<<",\"checkpoint_digest_kind\":\"noncryptographic byte fingerprint\",\"sms\":"<<info.sms<<",\"name\":"<<std::quoted(info.name)<<",\"clock_khz\":"<<info.clockKhz<<",\"clock_kind\":\"device-reported nominal, not sampled\",\"registers\":null,\"shared_bytes\":null,\"occupancy\":null,\"resource_metadata\":\"not exposed by stable backend API\",\"smids\":[";
    bool comma=false;for(auto smid:smids){if(comma)std::cout<<',';std::cout<<smid;comma=true;}std::cout<<']';
    identity("source_hash","SOLID_SOURCE_HASH");identity("binary_hash","SOLID_BINARY_HASH");
    identity("backend_source","SOLID_BACKEND");identity("build_flags","SOLID_BUILD_FLAGS");
    std::cout<<",\"compiler\":"<<std::quoted(__VERSION__);
    std::cout<<",\"measured\":"<<(measured?"true":"false")<<",\"rounds\":";
    if(measured)std::cout<<rounds;else std::cout<<"null";
    std::cout<<",\"evaluated\":";if(measured)std::cout<<evaluated;else std::cout<<"null";
    std::cout<<",\"finalized\":";if(measured)std::cout<<finalized;else std::cout<<"null";
    std::cout<<",\"replay_rounds\":";if(measured)std::cout<<replay;else std::cout<<"null";
    std::cout<<",\"cycles\":";
    if(measured){std::cout<<'[';for(int i=0;i<6;++i){if(i)std::cout<<',';std::cout<<cycles[i];}std::cout<<']';}else std::cout<<"null";
    std::cout<<",\"cycle_units\":\"sum of CTA elapsed stage intervals including synchronization; CTAs can overlap\",\"cycle_stages\":[\"Rng\",\"Address\",\"LoadDelta\",\"Conflict\",\"Commit\",\"Round\"],\"timing\":\"CUDA events, fixed work; excludes warmup, construction and transfers\"}\n";
  }
}
#endif
uint64_t unsignedValue(const std::string& text) {
  if(text.empty()||text[0]=='-')throw std::runtime_error("expected unsigned integer: "+text);
  size_t used=0;uint64_t value=std::stoull(text,&used,10);
  if(used!=text.size())throw std::runtime_error("invalid unsigned integer: "+text);
  return value;
}
uint32_t u32(const std::string& text) {
  uint64_t value=unsignedValue(text);if(value>UINT32_MAX)throw std::runtime_error("integer exceeds uint32 range");return uint32_t(value);
}
std::vector<int> list(const std::string& text) {
  std::vector<int> out;size_t start=0;
  do {
    size_t comma=text.find(',',start);std::string part=text.substr(start,comma==std::string::npos?comma:comma-start);
    uint64_t n=unsignedValue(part);if(!n||n>INT_MAX)throw std::runtime_error("launch values must be positive int");
    if(std::find(out.begin(),out.end(),int(n))!=out.end())throw std::runtime_error("duplicate launch value");
    out.push_back(int(n));if(comma==std::string::npos)break;start=comma+1;
  }while(true);
  return out;
}
}
int main(int argc,char** argv) {
  try {
    std::map<std::string,std::string> args;
    const std::set<std::string> switches={"--bench","--device-info","--audit"};
    const std::set<std::string> values={"--prepare","--prepare-final","--final-seed","--final-chain","--check","--stage","--windows","--threads","--ppt","--move","--profile","--L","--beta","--chains","--window","--warmup","--sweeps","--repeats"};
    for(int i=1;i<argc;++i) {
      std::string key=argv[i];if(args.count(key))throw std::runtime_error("duplicate option: "+key);
      if(switches.count(key))args[key]="1";
      else {if(!values.count(key))throw std::runtime_error("unknown option: "+key);if(i+1==argc)throw std::runtime_error("missing value: "+key);args[key]=argv[++i];}
    }
    auto value=[&](const std::string& key,const std::string& fallback){auto it=args.find(key);return it==args.end()?fallback:it->second;};
    int actions=0;for(const auto& key:{"--prepare","--prepare-final","--audit","--check","--bench","--device-info"})actions+=args.count(key);
    require(actions==1,"choose exactly one of --audit, --prepare FILE, --prepare-final FILE, --check FILE, --bench, --device-info");
    if(args.count("--prepare")) {
      require(!args.count("--final-seed")&&!args.count("--final-chain"),"development preparation must not consume a final stream");
      prepare(args["--prepare"],false,0,0);return 0;
    }
    if(args.count("--prepare-final")) {
      require(args.count("--final-seed")&&args.count("--final-chain"),"final preparation requires explicit --final-seed and --final-chain");
      prepare(args["--prepare-final"],true,unsignedValue(args["--final-seed"]),unsignedValue(args["--final-chain"]));return 0;
    }
    require(!args.count("--final-seed")&&!args.count("--final-chain"),"final stream options only belong to --prepare-final");
    if(args.count("--audit")){audit();return 0;}
#if SOLID_HAVE_CUDA
    if(args.count("--device-info")) {
      auto info=cudaDeviceInfo();std::cout<<"{\"kind\":\"device_info\",\"sms\":"<<info.sms<<",\"name\":"<<std::quoted(info.name)<<",\"clock_khz\":"<<info.clockKhz<<",\"clock_kind\":\"device-reported nominal, not sampled\"}\n";return 0;
    }
    std::string move=value("--move","both");require(move=="local"||move=="nonlocal"||move=="both","unknown move");
    std::string profile=value("--profile","0");require(profile=="0"||profile=="1","--profile requires 0 or 1");
    if(args.count("--check")) {
      check(args["--check"],value("--stage","single"),list(value("--windows","1,32,64,128,256")),list(value("--threads","32,64,128,256")),list(value("--ppt","1")),move,profile=="1");return 0;
    }
    if(args.count("--bench")) {
      auto one=[&](const std::string& key,const std::string& fallback){auto parsed=list(value(key,fallback));require(parsed.size()==1,"benchmark launch must be scalar");return parsed[0];};
      CudaLaunch launch;launch.window=one("--window","128");launch.threads=one("--threads","128");launch.proposals_per_thread=one("--ppt","1");launch.profile=profile=="1";
      int L=one("--L","21");require(L==4||L==8||L==21,"supported L are 4,8,21");
      bench(L,value("--beta","3.2"),one("--chains","1"),launch,value("--move","nonlocal"),u32(value("--warmup","1024")),u32(value("--sweeps","256")),one("--repeats","4"));return 0;
    }
#else
    throw std::runtime_error("this is the CPU-only checker; CUDA actions require SOLID_HAVE_CUDA=1");
#endif
    return 0;
  } catch(const std::exception& e) {
    std::cerr<<"{\"kind\":\"error\",\"message\":"<<std::quoted(e.what())<<"}\n";return 1;
  }
}
