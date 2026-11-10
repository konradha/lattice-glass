#include "cuda_backend.hpp"
#include "sweeps_cpu.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <thread>

using namespace solid;
namespace {
struct Fixture {
  uint32_t kind=0, L=21, chains=1, mode=1, beta=3200, sweeps=0, traceLimit=0;
  uint64_t seed=1, offset=0, start=0;
  std::vector<Threshold> thresholds;
  std::vector<uint8_t> initial;
  std::vector<uint32_t> checkpoints, energy, local, nonlocal;
  std::vector<std::vector<uint8_t>> states;
  std::vector<CudaTrace> trace;
};
template<class T> void put(std::ostream& s, const T& x) { s.write(reinterpret_cast<const char*>(&x),sizeof(x)); }
template<class T> void get(std::istream& s, T& x) { s.read(reinterpret_cast<char*>(&x),sizeof(x)); if(!s) throw std::runtime_error("truncated fixture"); }
template<class T> void putv(std::ostream& s,const std::vector<T>& v) {
  uint64_t n=v.size(); put(s,n); if(n) s.write(reinterpret_cast<const char*>(v.data()),n*sizeof(T));
}
template<class T> void getv(std::istream& s,std::vector<T>& v) {
  uint64_t n; get(s,n); if(n>(1ULL<<31)/sizeof(T)) throw std::runtime_error("invalid fixture length");
  v.resize(n); if(n) s.read(reinterpret_cast<char*>(v.data()),n*sizeof(T)); if(!s) throw std::runtime_error("truncated fixture vector");
}
void save(const std::string& path,const std::vector<Fixture>& fs) {
  std::ofstream s(path,std::ios::binary); put(s,uint64_t(0x5343554441303031ULL)); put(s,uint32_t(fs.size()));
  for(const auto& f:fs) {
    put(s,f.kind);put(s,f.L);put(s,f.chains);put(s,f.mode);put(s,f.beta);put(s,f.sweeps);put(s,f.traceLimit);
    put(s,f.seed);put(s,f.offset);put(s,f.start);putv(s,f.thresholds);putv(s,f.initial);putv(s,f.checkpoints);
    for(const auto& v:f.states) putv(s,v);
    putv(s,f.energy);putv(s,f.local);putv(s,f.nonlocal);putv(s,f.trace);
  }
  if(!s) throw std::runtime_error("cannot write fixtures");
}
std::vector<Fixture> load(const std::string& path) {
  std::ifstream s(path,std::ios::binary); uint64_t magic;get(s,magic);
  if(magic!=0x5343554441303031ULL) throw std::runtime_error("invalid fixture magic");
  uint32_t n;get(s,n);if(n>1000) throw std::runtime_error("invalid fixture count");std::vector<Fixture> fs(n);
  for(auto& f:fs) {
    get(s,f.kind);get(s,f.L);get(s,f.chains);get(s,f.mode);get(s,f.beta);get(s,f.sweeps);get(s,f.traceLimit);
    get(s,f.seed);get(s,f.offset);get(s,f.start);getv(s,f.thresholds);getv(s,f.initial);getv(s,f.checkpoints);
    f.states.resize(f.checkpoints.size());for(auto& v:f.states)getv(s,v);
    getv(s,f.energy);getv(s,f.local);getv(s,f.nonlocal);getv(s,f.trace);
  }
  return fs;
}
// Independent test-side stencil evaluator; production CPU routines remain untouched.
int stencilDelta(const Lattice& lat,const Proposal& p) {
  Aff aff[14];int n=0;addAff(aff,n,p.a,2,0);addAff(aff,n,p.b,2,0);
  for(int d=0;d<6;++d)addAff(aff,n,lat.nb(p.a,d),0,d^1);
  for(int d=0;d<6;++d)addAff(aff,n,lat.nb(p.b,d),1,d^1);
  int delta=0;
  for(int i=0;i<n;++i) {
    int site=aff[i].s,v=lat.st[site],t=v&3,m=v>>2;
    delta-=lat.localEnergy(t,m);
    if(aff[i].ba){int bit=aff[i].ba-1;m=(m&~(1<<bit))|((lat.type(p.b)!=0)<<bit);}
    if(aff[i].bb){int bit=aff[i].bb-1;m=(m&~(1<<bit))|((lat.type(p.a)!=0)<<bit);}
    int nt=site==p.a?lat.type(p.b):(site==p.b?lat.type(p.a):t);
    delta+=lat.localEnergy(nt,m);
  }
  return delta;
}
Fixture fixture(uint32_t kind,int L,int chains,int mode,int beta,uint64_t offset,uint64_t seed) {
  Fixture f;f.kind=kind;f.L=L;f.chains=chains;f.mode=mode;f.beta=beta;f.offset=offset;f.seed=seed;
  uint32_t per=L*L*L*(mode==0?2:1);
  if(kind==2) f.checkpoints={2};
  else if(kind==3) f.checkpoints={(100000000u+per-1)/per};
  else f.checkpoints={1,2,(10000u+per-1)/per,(100000u+per-1)/per,(1000000u+per-1)/per};
  std::sort(f.checkpoints.begin(),f.checkpoints.end());f.checkpoints.erase(std::unique(f.checkpoints.begin(),f.checkpoints.end()),f.checkpoints.end());
  f.sweeps=f.checkpoints.back();f.traceLimit=kind==2?0:(kind==1?10000:100000);
  f.start=kind==2||kind==3?0:(offset?128:0);
  f.thresholds=makeThresholds(beta==2800?"2.8":"3.2");
  size_t N=L*L*L;f.initial.resize(chains*N);f.states.resize(f.checkpoints.size());for(auto& v:f.states)v.resize(chains*N);
  f.energy.resize(size_t(chains)*f.sweeps);f.local.resize(f.energy.size());f.nonlocal.resize(f.energy.size());f.trace.resize(size_t(chains)*f.traceLimit);
  return f;
}
void reference(Fixture& f,uint32_t c) {
  Lattice lat(f.L);int N=lat.N;uint64_t id=f.offset+c;
  lat.init(f.seed+id*EFMA_GOLDEN,int(std::llround(.30*N)),int(std::llround(.45*N)));int e=lat.energy();
  for(uint64_t sw=0;sw<f.start;++sw){if(f.mode!=2)nonlocalSweep(lat,f.seed,int(id),sw,f.thresholds,e);if(f.mode!=1)localSweep(lat,f.seed,int(id),sw,f.thresholds,e);}
  std::copy(lat.st.begin(),lat.st.end(),f.initial.begin()+size_t(c)*N);
  uint64_t ordinal=0;size_t cp=0;
  for(uint32_t sw=0;sw<f.sweeps;++sw) {
    int an=0,al=0;
    for(int which=0;which<2;++which) {
      bool local=which==1;if((local&&f.mode==1)||(!local&&f.mode==2))continue;
      uint64_t base=(2*(f.start+sw)+uint64_t(local))*uint64_t(N);
      for(int j=0;j<N;++j,++ordinal) {
        Proposal p=drawProposal(lat,f.seed,int(id),base+j,local);
        bool noop=p.valid&&lat.type(p.a)==lat.type(p.b);
        int d=ordinal<f.traceLimit&&p.valid&&!noop?stencilDelta(lat,p):0;
        bool accepted=tryProposal(lat,p,f.thresholds,e);if(accepted){if(local)++al;else ++an;}
        if(ordinal<f.traceLimit) f.trace[size_t(c)*f.traceLimit+ordinal]={base+uint64_t(j),uint32_t(p.a),uint32_t(p.b),d,uint32_t(p.valid)|(uint32_t(noop)<<1)|(uint32_t(accepted)<<2)};
      }
    }
    int full=lat.energy();if(e!=full)throw std::runtime_error("CPU incremental/full energy mismatch");
    size_t index=size_t(c)*f.sweeps+sw;f.energy[index]=full;f.local[index]=al;f.nonlocal[index]=an;
    if(cp<f.checkpoints.size()&&sw+1==f.checkpoints[cp]){std::copy(lat.st.begin(),lat.st.end(),f.states[cp].begin()+size_t(c)*N);++cp;}
  }
}
void prepare(const std::string& path,uint64_t finalSeed) {
  std::vector<Fixture> fs;
  for(int L:{8,21})for(int b:{2800,3200})for(uint64_t id:{0ULL,1ULL,12345ULL})for(int mode:{1,2})
    fs.push_back(fixture(0,L,1,mode,b,id,id==1?0xfedcba9876543210ULL:1));
  for(int L:{8,21})for(uint64_t id:{0ULL,12345ULL})for(int mode:{0,1,2})fs.push_back(fixture(1,L,3,mode,3200,id,0x123456789abcdef0ULL));
  for(int mode:{1,2})fs.push_back(fixture(2,21,1536,mode,3200,0,123));
  for(int mode:{1,2})fs.push_back(fixture(3,21,1,mode,3200,1000000,finalSeed));
  std::vector<std::pair<size_t,uint32_t>> jobs;for(size_t i=0;i<fs.size();++i)for(uint32_t c=0;c<fs[i].chains;++c)jobs.emplace_back(i,c);
  std::atomic<size_t> next{0};std::vector<std::thread> workers;
  for(unsigned t=0;t<std::min(8u,std::max(1u,std::thread::hardware_concurrency()));++t)workers.emplace_back([&]{for(;;){size_t j=next.fetch_add(1);if(j>=jobs.size())break;reference(fs[jobs[j].first],jobs[j].second);}});
  for(auto& t:workers)t.join();save(path,fs);
  std::cout<<"{\"kind\":\"prepared\",\"fixtures\":"<<fs.size()<<",\"final_seed\":"<<finalSeed<<",\"cpu_commit\":\"50fb57a7ac15b521f4b6e7d7677b88ece900dd69\"}\n";
}
#if SOLID_HAVE_CUDA
[[noreturn]] void mismatch(const Fixture& f,int w,int c,uint64_t p,const char* field,uint64_t expected,uint64_t got) {
  std::cerr<<"FAIL L="<<f.L<<" mode="<<f.mode<<" beta="<<f.beta<<" chain="<<f.offset+c<<" window="<<w<<" proposal_or_index="<<p<<" field="<<field<<" expected="<<expected<<" got="<<got<<"\n";throw std::runtime_error("CUDA/CPU mismatch");
}
void checkOne(const Fixture& f,int chains,int window,bool smoke) {
  int N=f.L*f.L*f.L,moves=f.mode==0?2:1;std::vector<uint8_t> initial(f.initial.begin(),f.initial.begin()+size_t(chains)*N);
  CudaBackend gpu(f.L,chains,f.seed,initial,f.thresholds,window,f.offset);uint32_t before=0;
  for(size_t cp=0;cp<f.checkpoints.size();++cp) {
    uint32_t end=f.checkpoints[cp];if(smoke&&end>2)break;
    uint64_t previousAttempts=uint64_t(before)*N*moves;
    uint32_t tracedBefore=std::min<uint64_t>(f.traceLimit,previousAttempts);
    uint32_t traceCount=std::min<uint64_t>(f.traceLimit-tracedBefore,uint64_t(end-before)*N*moves);
    CudaResult r;gpu.run(f.start+before,end-before,f.mode,r,traceCount);
    if(r.state.size()!=size_t(chains)*N||r.energy.size()!=size_t(chains)*(end-before)||r.acceptLocal.size()!=r.energy.size()||r.acceptNonlocal.size()!=r.energy.size()||r.trace.size()!=size_t(chains)*traceCount)throw std::runtime_error("CUDA result shape mismatch");
    for(int c=0;c<chains;++c) {
      for(int j=0;j<N;++j)if(r.state[size_t(c)*N+j]!=f.states[cp][size_t(c)*N+j])mismatch(f,window,c,end,"state",f.states[cp][size_t(c)*N+j],r.state[size_t(c)*N+j]);
      for(uint32_t j=0;j<end-before;++j) {
        size_t src=size_t(c)*f.sweeps+before+j,dst=size_t(c)*(end-before)+j;
        if(r.energy[dst]!=f.energy[src])mismatch(f,window,c,before+j,"energy",f.energy[src],r.energy[dst]);
        if(r.acceptLocal[dst]!=f.local[src])mismatch(f,window,c,before+j,"local_acceptance",f.local[src],r.acceptLocal[dst]);
        if(r.acceptNonlocal[dst]!=f.nonlocal[src])mismatch(f,window,c,before+j,"nonlocal_acceptance",f.nonlocal[src],r.acceptNonlocal[dst]);
      }
      for(uint32_t j=0;j<traceCount;++j) {
        const auto& a=f.trace[size_t(c)*f.traceLimit+tracedBefore+j];const auto& b=r.trace[size_t(c)*traceCount+j];
        if(a.proposal!=b.proposal)mismatch(f,window,c,a.proposal,"proposal",a.proposal,b.proposal);
        if(a.a!=b.a)mismatch(f,window,c,a.proposal,"a",a.a,b.a);
        if(a.b!=b.b)mismatch(f,window,c,a.proposal,"b",a.b,b.b);
        if(a.delta!=b.delta)mismatch(f,window,c,a.proposal,"delta",uint64_t(int64_t(a.delta)),uint64_t(int64_t(b.delta)));
        if(a.flags!=b.flags)mismatch(f,window,c,a.proposal,"valid/noop/accept",a.flags,b.flags);
      }
    }
    before=end;
  }
}
void check(const std::string& file,const std::string& stage,const std::vector<int>& windows,const std::string& move) {
  auto fs=load(file);auto info=cudaDeviceInfo();uint64_t count=0;
  for(const auto& f:fs) {
    bool selected=(stage=="single"&&f.kind==0)||(stage=="multi"&&f.kind==1)||(stage=="saturated"&&f.kind==2)||(stage=="final"&&f.kind==3)||(stage=="smoke"&&f.kind==0&&f.offset==0&&f.beta==2800);
    if(!selected||(move=="local"&&f.mode==1)||(move=="nonlocal"&&f.mode==2))continue;
    for(int w:windows) {
      if(f.kind==2){for(int factor:{1,2,4,8}){int chains=info.sms*factor;if(chains>int(f.chains))throw std::runtime_error("fixture too small for saturation gate");checkOne(f,chains,w,false);++count;}}
      else{checkOne(f,f.chains,w,stage=="smoke");++count;}
    }
  }
  if(!count)throw std::runtime_error("empty correctness stage");
  std::cout<<"{\"kind\":\"correctness\",\"stage\":"<<std::quoted(stage)<<",\"move\":"<<std::quoted(move)<<",\"checks\":"<<count<<",\"ok\":true,\"sms\":"<<info.sms<<",\"name\":"<<std::quoted(info.name)<<"}\n";
}
void bench(int L,const std::string& beta,int chains,int window,const std::string& move,double seconds) {
  int mode=move=="local"?2:1,N=L*L*L;auto th=makeThresholds(beta);std::vector<uint8_t> initial(size_t(chains)*N);
  for(int c=0;c<chains;++c){Lattice lat(L);lat.init(1+uint64_t(c)*EFMA_GOLDEN,int(std::llround(.30*N)),int(std::llround(.45*N)));std::copy(lat.st.begin(),lat.st.end(),initial.begin()+size_t(c)*N);}
  CudaBackend gpu(L,chains,1,initial,th,window);CudaResult result;uint64_t start=0;
  gpu.run(start,1024,mode,result);start+=1024;double ms=0;uint64_t sweeps=0;
  while(ms<seconds*1000){gpu.run(start,256,mode,result);start+=256;sweeps+=256;ms+=result.milliseconds;if(result.milliseconds<=0)throw std::runtime_error("invalid CUDA event time");}
  double perChain=double(sweeps)*N/(ms*.001);auto info=cudaDeviceInfo();
  std::cout<<std::setprecision(17)<<"{\"kind\":\"benchmark\",\"move\":"<<std::quoted(move)<<",\"L\":"<<L<<",\"beta\":"<<std::quoted(beta)<<",\"chains\":"<<chains<<",\"window\":"<<window<<",\"sms\":"<<info.sms<<",\"name\":"<<std::quoted(info.name)<<",\"clock_khz\":"<<info.clockKhz<<",\"warmup_sweeps\":1024,\"sweeps_per_chain\":"<<sweeps<<",\"cuda_seconds\":"<<ms*.001<<",\"per_chain_attempts_s\":"<<perChain<<",\"aggregate_attempts_s\":"<<perChain*chains<<"}\n";
}
#endif
}
int main(int argc,char** argv) {
  try {
    std::map<std::string,std::string> args;for(int i=1;i<argc;++i){std::string key=argv[i];if(key=="--bench"||key=="--device-info")args[key]="1";else{if(i+1==argc)throw std::runtime_error("missing option value");args[key]=argv[++i];}}
    auto value=[&](const std::string& key,const std::string& fallback){auto it=args.find(key);return it==args.end()?fallback:it->second;};
    if(args.count("--prepare")){prepare(args["--prepare"],std::stoull(value("--final-seed","13619432045832709139")));return 0;}
#if SOLID_HAVE_CUDA
    if(args.count("--device-info")){auto info=cudaDeviceInfo();std::cout<<"{\"sms\":"<<info.sms<<",\"name\":"<<std::quoted(info.name)<<"}\n";return 0;}
    if(args.count("--check")){std::vector<int> windows;std::string list=value("--windows","1,32,64,128,256");size_t pos=0;while(pos<list.size()){size_t comma=list.find(',',pos);windows.push_back(std::stoi(list.substr(pos,comma-pos)));if(comma==std::string::npos)break;pos=comma+1;}check(args["--check"],value("--stage","single"),windows,value("--move","both"));return 0;}
    if(args.count("--bench")){bench(std::stoi(value("--L","21")),value("--beta","3.2"),std::stoi(value("--chains","1")),std::stoi(value("--window","128")),value("--move","nonlocal"),std::stod(value("--seconds","1")));return 0;}
#endif
    throw std::runtime_error("use --prepare FILE, --check FILE --stage STAGE, or --bench");
  }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
