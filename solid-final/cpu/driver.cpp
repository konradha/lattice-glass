#include "lattice.hpp"
#include "sweeps_cpu.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifndef SOLID_CPU_CACHED
#if defined(__APPLE__)
#define SOLID_CPU_CACHED 0
#else
#define SOLID_CPU_CACHED 1
#endif
#endif
#if SOLID_CPU_CACHED
#include "fast.hpp"
#define SOLID_CPU_BACKEND "opt3-type-fast-K"
#else
#define SOLID_CPU_BACKEND "opt2-physical-O2"
#endif
#ifndef SOLID_SOURCE_HASH
#define SOLID_SOURCE_HASH "unrecorded"
#endif
using namespace solid;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
// ---------------------------------------------------------------------------
// Minimal stored-ZIP .npz writer.
// ---------------------------------------------------------------------------
static uint32_t crc32_tab[256];
static void init_crc() {
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
    crc32_tab[i] = c;
  }
}
static uint32_t crc32(const uint8_t* p, size_t n) {
  uint32_t c = 0xffffffffu;
  for (size_t i = 0; i < n; ++i) c = crc32_tab[(c ^ p[i]) & 255] ^ (c >> 8);
  return c ^ 0xffffffffu;
}
static void put16(std::vector<uint8_t>& b, uint16_t v) {
  b.push_back(uint8_t(v)); b.push_back(uint8_t(v >> 8));
}
static void put32(std::vector<uint8_t>& b, uint32_t v) {
  b.push_back(uint8_t(v)); b.push_back(uint8_t(v >> 8));
  b.push_back(uint8_t(v >> 16)); b.push_back(uint8_t(v >> 24));
}
struct ZipEntry { std::string name; uint32_t crc, size, offset; };
static std::vector<uint8_t> npyHeader(const std::string& descr,
                                      const std::vector<size_t>& shape) {
  std::string s = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) s += ", ";
    s += std::to_string(shape[i]);
  }
  if (shape.size() == 1) s += ",";
  s += "), }";
  const size_t base = 10;
  size_t pad = (16 - ((base + s.size() + 1) % 16)) % 16;
  s.append(pad, ' ');
  s += '\n';
  std::vector<uint8_t> h = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  put16(h, uint16_t(s.size()));
  h.insert(h.end(), s.begin(), s.end());
  return h;
}
static void zipAdd(std::ofstream& f, std::vector<ZipEntry>& entries,
                   const std::string& name, const std::string& descr,
                   const std::vector<size_t>& shape,
                   const std::vector<uint8_t>& payload) {
  std::vector<uint8_t> data = npyHeader(descr, shape);
  const auto offset=std::streamoff(f.tellp());
  if (payload.size()>UINT32_MAX-data.size() || offset<0 ||
      uint64_t(offset)+30+name.size()+data.size()+payload.size()>UINT32_MAX)
    throw std::length_error("run.npz exceeds stored-ZIP 4GiB limit; use smaller campaign chunks");
  data.insert(data.end(), payload.begin(), payload.end());
  ZipEntry e; e.name = name; e.crc = crc32(data.data(), data.size());
  e.size = uint32_t(data.size()); e.offset = uint32_t(f.tellp());
  // Write local header separately before data.
  std::vector<uint8_t> lh;
  put32(lh, 0x04034b50u); put16(lh, 20); put16(lh, 0); put16(lh, 0);
  put16(lh, 0); put16(lh, 0); put32(lh, e.crc); put32(lh, e.size); put32(lh, e.size);
  put16(lh, uint16_t(name.size())); put16(lh, 0);
  lh.insert(lh.end(), name.begin(), name.end());
  f.write((const char*)lh.data(), lh.size());
  f.write((const char*)data.data(), data.size());
  entries.push_back(e);
}
static void zipFinish(std::ofstream& f, const std::vector<ZipEntry>& entries) {
  uint32_t start = uint32_t(f.tellp());
  for (auto& e : entries) {
    std::vector<uint8_t> c;
    put32(c, 0x02014b50u); put16(c, 20); put16(c, 20); put16(c, 0); put16(c, 0);
    put16(c, 0); put16(c, 0); put32(c, e.crc); put32(c, e.size); put32(c, e.size);
    put16(c, uint16_t(e.name.size())); put16(c, 0); put16(c, 0);
    put16(c, 0); put16(c, 0); put32(c, 0); put32(c, e.offset);
    c.insert(c.end(), e.name.begin(), e.name.end());
    f.write((const char*)c.data(), c.size());
  }
  if (std::streamoff(f.tellp())>std::streamoff(UINT32_MAX)-22)
    throw std::length_error("run.npz exceeds stored-ZIP 4GiB limit");
  uint32_t end = uint32_t(f.tellp());
  std::vector<uint8_t> e;
  put32(e, 0x06054b50u); put16(e, 0); put16(e, 0);
  put16(e, uint16_t(entries.size())); put16(e, uint16_t(entries.size()));
  put32(e, end - start); put32(e, start); put16(e, 0);
  f.write((const char*)e.data(), e.size());
}
static std::vector<uint8_t> asBytes(const void* p, size_t n) {
  if (!n) return {};
  const uint8_t* q = (const uint8_t*)p; return std::vector<uint8_t>(q, q + n);
}

static uint64_t u64(const std::string& text) {
  if (text.empty() || text[0] == '-') throw std::invalid_argument("invalid unsigned integer: " + text);
  char* end = nullptr; errno = 0;
  auto n = std::strtoull(text.c_str(), &end, 10);
  if (errno == ERANGE || end == text.c_str() || *end) throw std::invalid_argument("invalid unsigned integer: " + text);
  return uint64_t(n);
}
static int positiveInt(const std::string& text) {
  uint64_t n = u64(text);
  if (!n || n > uint64_t(INT32_MAX)) throw std::invalid_argument("positive int required: " + text);
  return int(n);
}
static double number(const std::string& text) {
  size_t n = 0; double v = std::stod(text, &n);
  if (n != text.size() || !std::isfinite(v)) throw std::invalid_argument("finite number required: " + text);
  return v;
}
static int moveMode(const std::string& s) {
  if (s == "both") return 0;
  if (s == "nonlocal") return 1;
  if (s == "local") return 2;
  throw std::invalid_argument("move must be both, nonlocal, or local");
}
static void range(uint64_t start, uint64_t count) {
  if (count > UINT64_MAX-start) throw std::overflow_error("sweep range overflow");
}
static size_t sizeProduct(size_t a, size_t b) {
  if (b && a > SIZE_MAX/b) throw std::length_error("host array size overflow");
  return a*b;
}
static std::vector<uint8_t> rawState(const std::string& path, size_t count) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open state: "+path);
  std::vector<uint8_t> v(count);
  f.read(reinterpret_cast<char*>(v.data()), count);
  if (!f || f.peek() != std::char_traits<char>::eof()) throw std::runtime_error("state size mismatch: "+path);
  return v;
}
static void validateState(const std::vector<uint8_t>& state, int L, int chains) {
  Lattice lat(L);
  for (int c=0; c<chains; ++c) {
    const uint8_t* p=state.data()+size_t(c)*lat.N;
    for (int i=0; i<lat.N; ++i) {
      if ((p[i]&3u)==3u) throw std::invalid_argument("invalid type in state");
      int mask=0;
      for (int d=0; d<6; ++d) mask |= int((p[lat.nb(i,d)]&3u)!=0)<<d;
      if ((p[i]>>2)!=mask) throw std::invalid_argument("inconsistent directional state mask");
    }
  }
}

// Chain objects never move after Fast's geometry references are established.
// Both the packed cache and exact integer energy persist across all segments.
struct Segment {
  std::vector<uint8_t> state;
  std::vector<uint32_t> energy, local, nonlocal;
};
class Engine {
  int chains_, N_;
  uint64_t seed_;
  std::vector<Lattice> lats_;
  std::vector<Threshold> thresholds_;
#if SOLID_CPU_CACHED
  std::vector<std::unique_ptr<Fast<true>>> fast_;
#else
  std::vector<int> energy_;
#endif
public:
  Engine(int L, int chains, uint64_t seed, const std::vector<uint8_t>& initial,
         const std::string& beta)
      : chains_(chains), N_(L*L*L), seed_(seed), thresholds_(makeThresholds(beta)) {
    lats_.reserve(chains);
    for (int c=0;c<chains;++c) {
      lats_.emplace_back(L);
      std::copy(initial.begin()+size_t(c)*N_, initial.begin()+size_t(c+1)*N_, lats_.back().st.begin());
    }
#if SOLID_CPU_CACHED
    fast_.reserve(chains);
    for (const auto& lat:lats_) fast_.emplace_back(new Fast<true>(lat));
#else
    energy_.reserve(chains);
    for (const auto& lat:lats_) energy_.push_back(lat.energy());
#endif
  }
  void run(uint64_t start, uint32_t count, int mode, Segment& out) {
    range(start,count);
    size_t samples=sizeProduct(size_t(chains_),count);
    out.energy.resize(samples); out.local.resize(samples); out.nonlocal.resize(samples);
    out.state.resize(sizeProduct(size_t(chains_),size_t(N_)));
    std::atomic<int> next{0};
    std::vector<std::thread> workers;
    for (int w=0;w<std::min(chains_,64);++w) workers.emplace_back([&] {
      for (;;) {
        int c=next.fetch_add(1); if(c>=chains_) break;
        Lattice& lat=lats_[c];
#if SOLID_CPU_CACHED
        Fast<true>& f=*fast_[c];
        auto sweep=[&](uint64_t sw,bool local) {
          const uint64_t before=f.accepted;
          const uint64_t index0=indexAt(sw,local,N_,0);
          for (int t=0;t<N_;++t) {
            Proposal p=draw64(lat,seed_,uint64_t(c),index0+uint64_t(t),local);
            f.attempt(p,thresholds_);
          }
          return uint32_t(f.accepted-before);
        };
#else
        int& e=energy_[c];
#endif
        for (uint32_t k=0;k<count;++k) {
#if SOLID_CPU_CACHED
          uint32_t an=mode!=2?sweep(start+k,false):0;
          uint32_t al=mode!=1?sweep(start+k,true):0;
          int e=f.energy;
#else
          uint32_t an=mode!=2?nonlocalSweep(lat,seed_,c,start+k,thresholds_,e):0;
          uint32_t al=mode!=1?localSweep(lat,seed_,c,start+k,thresholds_,e):0;
#endif
          size_t h=size_t(c)*count+k;
          out.energy[h]=uint32_t(e); out.local[h]=al; out.nonlocal[h]=an;
        }
#if SOLID_CPU_CACHED
        // Canonical bytes are materialized only at a host segment boundary.
        for(int s=0;s<N_;++s) out.state[size_t(c)*N_+s]=f.byte(s);
#else
        std::copy(lat.st.begin(),lat.st.end(),out.state.begin()+size_t(c)*N_);
#endif
      }
    });
    for(auto& w:workers) w.join();
  }
};
static std::vector<uint8_t> initialized(int L,int chains,uint64_t seed,double rho1,double rho2) {
  const int N=L*L*L;
  if (rho1<0 || rho2<0 || rho1>1 || rho2>1 || rho1+rho2>1)
    throw std::invalid_argument("invalid composition");
  const int n1=int(std::llround(rho1*N)), n2=int(std::llround(rho2*N));
  if (n1+n2>N || n1+n2<=0) throw std::invalid_argument("invalid composition");
  std::vector<uint8_t> state(sizeProduct(size_t(chains),size_t(N)));
  for(int c=0;c<chains;++c) {
    Lattice lat(L); lat.init(seed+uint64_t(c)*EFMA_GOLDEN,n1,n2);
    std::copy(lat.st.begin(),lat.st.end(),state.begin()+size_t(c)*N);
  }
  return state;
}
struct History {
  std::vector<uint64_t> times;
  std::vector<uint32_t> energy, local, nonlocal;
};
static void saveRun(const fs::path& path,int L,int chains,const std::string& beta,
                    double rho,double rho1,double rho2,uint64_t seed,uint64_t start,
                    uint64_t sweeps,uint64_t equil,int mode,uint64_t every,
                    const std::vector<uint8_t>& initial,const std::vector<uint8_t>& equilibrated,
                    const std::vector<uint8_t>& final,const History& h,
                    const std::vector<uint64_t>& snapTimes,const std::vector<uint8_t>& snaps) {
  fs::path temp=path.string()+".tmp";
  std::ofstream z(temp,std::ios::binary); z.exceptions(std::ios::failbit|std::ios::badbit);
  std::vector<ZipEntry> entries;
  const size_t N=size_t(L)*L*L, samples=h.times.size();
  auto add=[&](const std::string& name,const std::string& dtype,const std::vector<size_t>& shape,const void* p,size_t bytes) {
    zipAdd(z,entries,name+".npy",dtype,shape,asBytes(p,bytes));
  };
  add("initial","<u1",{size_t(chains),N},initial.data(),initial.size());
  add("equilibrated","<u1",{size_t(chains),N},equilibrated.data(),equilibrated.size());
  add("final","<u1",{size_t(chains),N},final.data(),final.size());
  add("snapshot_sweeps","<u8",{snapTimes.size()},snapTimes.data(),snapTimes.size()*8);
  add("snapshots","<u1",{snapTimes.size(),size_t(chains),N},snaps.data(),snaps.size());
  add("energy_int","<u4",{size_t(chains),samples},h.energy.data(),h.energy.size()*4);
  add("accept_local_count","<u4",{size_t(chains),samples},h.local.data(),h.local.size()*4);
  add("accept_nonlocal_count","<u4",{size_t(chains),samples},h.nonlocal.data(),h.nonlocal.size()*4);
  std::vector<float> f(h.energy.size());
  for(size_t i=0;i<f.size();++i)f[i]=float(h.energy[i]);
  add("energy","<f4",{size_t(chains),samples},f.data(),f.size()*4);
  for(size_t i=0;i<f.size();++i)f[i]=float(h.local[i])/float(N);
  add("accept_local","<f4",{size_t(chains),samples},f.data(),f.size()*4);
  for(size_t i=0;i<f.size();++i)f[i]=float(h.nonlocal[i])/float(N);
  add("accept_nonlocal","<f4",{size_t(chains),samples},f.data(),f.size()*4);
  add("history_sweeps","<u8",{samples},h.times.data(),samples*8);
  std::vector<uint64_t> absoluteTimes=h.times;
  for (auto& t:absoluteTimes) t+=start+equil;
  add("history_sweeps_absolute","<u8",{samples},absoluteTimes.data(),samples*8);
  add("history_every","<u8",{1},&every,8); add("seed","<u8",{1},&seed,8);
  add("start_sweep","<u8",{1},&start,8);
  uint64_t next=start+equil+sweeps;
  add("next_sweep","<u8",{1},&next,8);
  double scal[4]={number(beta),rho,rho1,rho2}; add("scalars","<f8",{4},scal,sizeof(scal));
  uint64_t meta[5]={uint64_t(L),uint64_t(chains),sweeps,uint64_t(mode),equil}; add("meta","<u8",{5},meta,sizeof(meta));
  const std::string backend=SOLID_CPU_BACKEND;
  add("cpu_backend","|u1",{backend.size()},backend.data(),backend.size());
  add("beta_text","|u1",{beta.size()},beta.data(),beta.size());
  const std::string hash=SOLID_SOURCE_HASH;
  add("source_hash","|u1",{hash.size()},hash.data(),hash.size());
  zipFinish(z,entries); z.close(); fs::rename(temp,path);
}
static int standalone(int argc,char** argv) {
  if(argc<9)throw std::invalid_argument("usage: driver L beta rho rho1 rho2 SWEEPS CHAINS CPU [both|nonlocal|local] [EQUIL] [STATE_RAW] [--seed U64] [--start-sweep U64] [--history-every N] [--max-history-samples N]");
  int L=positiveInt(argv[1]);if(L<4||L>21)throw std::invalid_argument("L must be 4..21");
  std::string beta=argv[2];if(number(beta)<0)throw std::invalid_argument("negative beta");
  double rho=number(argv[3]),rho1=number(argv[4]),rho2=number(argv[5]);
  uint64_t sweeps=u64(argv[6]);int chains=positiveInt(argv[7]);
  std::string backend=argv[8];
  if(backend!="CPU"&&backend!="cpu"&&backend!="0")
    throw std::invalid_argument("this executable supports CPU only; no GPU fallback");
  uint64_t seed=1,start=0,every=1,cap=100000;std::vector<std::string> positional;
  for(int i=9;i<argc;++i) {
    std::string arg=argv[i];if(arg.rfind("--",0)!=0){positional.push_back(arg);continue;}
    if(++i==argc)throw std::invalid_argument("missing option value");std::string value=argv[i];
    if(arg=="--seed")seed=u64(value);else if(arg=="--start-sweep")start=u64(value);
    else if(arg=="--history-every")every=u64(value);else if(arg=="--max-history-samples")cap=u64(value);
    else throw std::invalid_argument("unknown option: "+arg);
  }
  if(!every||!cap||positional.size()>3)throw std::invalid_argument("history interval/cap must be positive; at most three positionals");
  int mode=positional.empty()?0:moveMode(positional[0]);uint64_t equil=positional.size()>1?u64(positional[1]):0;
  range(start,equil);range(start+equil,sweeps);
  every=std::max(every,sweeps/cap+uint64_t(sweeps%cap!=0));
  auto initial=positional.size()>2?rawState(positional[2],sizeProduct(chains,size_t(L)*L*L)):initialized(L,chains,seed,rho1,rho2);
  validateState(initial,L,chains);
  Engine engine(L,chains,seed,initial,beta);Segment seg;
  std::vector<uint8_t> state=initial;
  for(uint64_t k=0;k<equil;){uint32_t n=uint32_t(std::min<uint64_t>(4096,equil-k));engine.run(start+k,n,mode,seg);state=seg.state;k+=n;}
  const auto equilibrated=state;History h;
  size_t samples=size_t(sweeps/every+uint64_t(sweeps%every!=0));
  h.times.reserve(samples);h.energy.resize(sizeProduct(chains,samples));h.local.resize(h.energy.size());h.nonlocal.resize(h.energy.size());
  for(uint64_t t=every;t<=sweeps&&t;t+=every){h.times.push_back(t);if(t>UINT64_MAX-every)break;}
  if(sweeps && (h.times.empty()||h.times.back()!=sweeps))h.times.push_back(sweeps);
  std::vector<uint64_t> snapTimes,planned;
  for(uint64_t t=1;t<=sweeps;){planned.push_back(t);if(t>UINT64_MAX/2)break;t*=2;}
  if(sweeps&&(planned.empty()||planned.back()!=sweeps))planned.push_back(sweeps);
  std::vector<uint8_t> snaps;size_t nextSnap=0,nextHist=0;uint64_t cur=0;auto wall0=Clock::now();
  while(cur<sweeps) {
    uint64_t stop=std::min(sweeps,cur+std::min<uint64_t>(4096,sweeps-cur));if(nextSnap<planned.size())stop=std::min(stop,planned[nextSnap]);
    uint32_t n=uint32_t(stop-cur);engine.run(start+equil+cur,n,mode,seg);
    while(nextHist<h.times.size()&&h.times[nextHist]<=stop) {
      size_t j=size_t(h.times[nextHist]-cur-1);
      for(int c=0;c<chains;++c){size_t dst=size_t(c)*samples+nextHist,src=size_t(c)*n+j;h.energy[dst]=seg.energy[src];h.local[dst]=seg.local[src];h.nonlocal[dst]=seg.nonlocal[src];}
      ++nextHist;
    }
    cur=stop;state=seg.state;
    if(nextSnap<planned.size()&&cur==planned[nextSnap]){snapTimes.push_back(cur);snaps.insert(snaps.end(),state.begin(),state.end());++nextSnap;}
  }
  saveRun("run.npz",L,chains,beta,rho,rho1,rho2,seed,start,sweeps,equil,mode,every,initial,equilibrated,state,h,snapTimes,snaps);
  double wall=std::chrono::duration<double>(Clock::now()-wall0).count();
  std::cout<<"done L="<<L<<" chains="<<chains<<" sweeps="<<sweeps<<" wall="<<wall<<" backend="<<SOLID_CPU_BACKEND<<" npz=run.npz\n";
  return 0;
}
int main(int argc,char** argv) {
  try { init_crc(); return standalone(argc,argv); }
  catch(const std::exception& e) { std::cerr<<"driver: "<<e.what()<<'\n'; return 2; }
}
