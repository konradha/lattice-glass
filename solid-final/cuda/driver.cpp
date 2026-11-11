#include "../common/lattice.hpp"
#include "../common/sweeps_cpu.hpp"
#ifndef SOLID_HAVE_CUDA
#define SOLID_HAVE_CUDA 0
#endif
#if SOLID_HAVE_CUDA
#include "backend.hpp"
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifndef SOLID_SOURCE_HASH
#define SOLID_SOURCE_HASH "unavailable"
#endif
#ifndef SOLID_BUILD_FLAGS
#define SOLID_BUILD_FLAGS "unavailable"
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
  data.insert(data.end(), payload.begin(), payload.end());
  if (data.size()>UINT32_MAX || f.tellp()<0 || uint64_t(f.tellp())>UINT32_MAX)
    throw std::length_error("NPZ exceeds stored ZIP32 limit; reduce chains/history cap");
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
static void atomicText(const fs::path& path, const std::string& text) {
  fs::path temp = path.string()+".tmp";
  { std::ofstream f(temp); f.exceptions(std::ios::failbit|std::ios::badbit); f << text; }
  fs::rename(temp, path);
}
static void atomicRaw(const fs::path& path, const std::vector<uint8_t>& state) {
  fs::path temp = path.string()+".tmp";
  { std::ofstream f(temp, std::ios::binary); f.exceptions(std::ios::failbit|std::ios::badbit);
    f.write(reinterpret_cast<const char*>(state.data()), state.size()); }
  fs::rename(temp, path);
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
struct Launch { int window=256, threads=256, ppt=1; bool profile=false; };
static void validateLaunch(const Launch& l) {
  if ((l.window!=1 && l.window!=32 && l.window!=64 && l.window!=128 && l.window!=256) ||
      (l.threads!=32 && l.threads!=64 && l.threads!=128 && l.threads!=256) ||
      l.window>l.threads || l.ppt!=1)
    throw std::invalid_argument("baseline requires W1/32/64/128/256, T32/64/128/256, W<=T, ppt1");
}
struct Segment {
  std::vector<uint8_t> state;
  std::vector<uint32_t> energy, local, nonlocal;
  double kernelSeconds=0;
};
class Engine {
  int L_, chains_, N_; uint64_t seed_; bool cuda_;
  std::vector<Lattice> lats_;
  std::vector<Threshold> thresholds_;
#if SOLID_HAVE_CUDA
  std::unique_ptr<CudaBackend> gpu_;
  CudaResult result_;
#endif
public:
  Engine(int L, int chains, uint64_t seed, const std::vector<uint8_t>& initial,
         const std::string& beta, bool cuda, Launch launch, uint32_t chunk)
      : L_(L), chains_(chains), N_(L*L*L), seed_(seed), cuda_(cuda), thresholds_(makeThresholds(beta)) {
    if (cuda) {
#if SOLID_HAVE_CUDA
      gpu_.reset(new CudaBackend(L, chains, seed, initial, thresholds_,
                     CudaLaunch{launch.window,launch.threads,launch.ppt,launch.profile}));
      gpu_->reserve(chunk);
      result_.state.reserve(initial.size());
      size_t samples=sizeProduct(size_t(chains),chunk);
      result_.energy.reserve(samples); result_.acceptLocal.reserve(samples); result_.acceptNonlocal.reserve(samples);
#else
      throw std::runtime_error("CUDA backend not built; no fallback is permitted");
#endif
    } else {
      lats_.reserve(chains);
      for (int c=0;c<chains;++c) {
        lats_.emplace_back(L);
        std::copy(initial.begin()+size_t(c)*N_, initial.begin()+size_t(c+1)*N_,lats_.back().st.begin());
      }
    }
  }
  void run(uint64_t start, uint32_t count, int mode, Segment& out) {
    range(start,count);
    if (cuda_) {
#if SOLID_HAVE_CUDA
      gpu_->run(start,count,mode,result_);
      out.state=result_.state; out.energy=result_.energy;
      out.local=result_.acceptLocal; out.nonlocal=result_.acceptNonlocal;
      out.kernelSeconds=result_.milliseconds*.001;
#endif
      return;
    }
    size_t samples=sizeProduct(size_t(chains_),count);
    out.energy.resize(samples); out.local.resize(samples); out.nonlocal.resize(samples);
    out.state.resize(sizeProduct(size_t(chains_),size_t(N_))); out.kernelSeconds=0;
    std::atomic<int> next{0};
    std::vector<std::thread> workers;
    for (int w=0;w<std::min(chains_,64);++w) workers.emplace_back([&] {
      for (;;) {
        int c=next.fetch_add(1); if(c>=chains_) break;
        Lattice& lat=lats_[c]; int e=lat.energy();
        for(uint32_t k=0;k<count;++k) {
          int an=mode!=2?nonlocalSweep(lat,seed_,c,start+k,thresholds_,e):0;
          int al=mode!=1?localSweep(lat,seed_,c,start+k,thresholds_,e):0;
          size_t h=size_t(c)*count+k;
          out.energy[h]=uint32_t(lat.energy()); out.local[h]=al; out.nonlocal[h]=an;
        }
        std::copy(lat.st.begin(),lat.st.end(),out.state.begin()+size_t(c)*N_);
      }
    });
    for(auto& w:workers) w.join();
  }
};
static std::vector<uint8_t> initialized(int L,int chains,uint64_t seed,double rho1,double rho2) {
  int N=L*L*L, n1=int(std::llround(rho1*N)), n2=int(std::llround(rho2*N));
  if (rho1<0 || rho2<0 || n1+n2>N || n1+n2<=0) throw std::invalid_argument("invalid composition");
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
static History decimate(const Segment& seg,int chains,uint32_t count,uint64_t start,uint64_t every) {
  History h; std::vector<uint32_t> indices;
  for(uint32_t i=0;i<count;++i) if ((start+i+1)%every==0 || i+1==count) {
    indices.push_back(i); h.times.push_back(start+i+1);
  }
  size_t n=indices.size(); h.energy.resize(sizeProduct(chains,n)); h.local.resize(h.energy.size()); h.nonlocal.resize(h.energy.size());
  for(int c=0;c<chains;++c) for(size_t j=0;j<n;++j) {
    size_t dst=size_t(c)*n+j, src=size_t(c)*count+indices[j];
    h.energy[dst]=seg.energy[src]; h.local[dst]=seg.local[src]; h.nonlocal[dst]=seg.nonlocal[src];
  }
  return h;
}
static void saveRun(const fs::path& path,int L,int chains,const std::string& beta,
                    double rho,double rho1,double rho2,uint64_t seed,uint64_t start,
                    uint64_t sweeps,uint64_t equil,int mode,uint64_t every,const Launch& launch,
                    const std::vector<uint8_t>& initial,const std::vector<uint8_t>& equilibrated,
                    const std::vector<uint8_t>& final,const History& h,
                    const std::vector<uint64_t>& snapTimes,const std::vector<uint8_t>& snaps,
                    bool absoluteHistory=false) {
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
  if (!absoluteHistory) for (auto& t:absoluteTimes) t+=start+equil;
  add("history_sweeps_absolute","<u8",{samples},absoluteTimes.data(),samples*8);
  add("history_every","<u8",{1},&every,8); add("seed","<u8",{1},&seed,8);
  add("start_sweep","<u8",{1},&start,8);
  uint64_t next=start+equil+sweeps;
  add("next_sweep","<u8",{1},&next,8);
  double scal[4]={number(beta),rho,rho1,rho2}; add("scalars","<f8",{4},scal,sizeof(scal));
  uint64_t meta[5]={uint64_t(L),uint64_t(chains),sweeps,uint64_t(mode),equil}; add("meta","<u8",{5},meta,sizeof(meta));
  uint64_t config[4]={uint64_t(launch.window),uint64_t(launch.threads),uint64_t(launch.ppt),uint64_t(launch.profile)};
  add("cuda_launch","<u8",{4},config,sizeof(config));
  add("beta_text","|u1",{beta.size()},beta.data(),beta.size());
  const std::string hash=SOLID_SOURCE_HASH;
  add("source_hash","|u1",{hash.size()},hash.data(),hash.size());
  zipFinish(z,entries); z.close(); fs::rename(temp,path);
}
struct Job {
  std::string id,beta,modeText,initial;
  int mode=0,chains=0; uint64_t seed=0,every=1,start=0,next=0,chunks=0;
  uint32_t chunk=1; Launch launch;
  std::vector<uint8_t> state;
  std::unique_ptr<Engine> engine;
  double kernel=0,wall=0; std::string status="prepared",error;
};
static std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> v; size_t a=0;
  for(;;){size_t b=line.find('\t',a);v.push_back(line.substr(a,b-a));if(b==std::string::npos)break;a=b+1;}
  return v;
}
static std::string checkpoint(const Job& j) {
  std::ostringstream o; o << "key\tvalue\nseed\t"<<j.seed<<"\nbeta\t"<<j.beta<<"\nmode\t"<<j.modeText
    <<"\nL\t21\nchains\t"<<j.chains<<"\nwindow\t"<<j.launch.window<<"\nthreads\t"<<j.launch.threads
    <<"\nppt\t"<<j.launch.ppt<<"\nhistory_every\t"<<j.every<<"\nchunk_sweeps\t"<<j.chunk
    <<"\nnext_sweep\t"<<j.next<<"\nchunks\t"<<j.chunks<<"\nstate\tcheckpoint-"<<(j.chunks%2)<<".raw\nrho\t0.75\nrho1\t0.30\nrho2\t0.45\nell1\t3\nell2\t5\nA\t1\nsource_hash\t"<<SOLID_SOURCE_HASH<<"\n";
  return o.str();
}
static std::string binaryMetadata() {
  const fs::path binary=fs::read_symlink("/proc/self/exe");
  std::ostringstream o;
  o<<"binary_path\t"<<binary.string()<<"\nbinary_size\t"<<fs::file_size(binary)
   <<"\nbinary_mtime_ticks\t"<<fs::last_write_time(binary).time_since_epoch().count()
   <<"\ncompiler\t"<<__VERSION__<<"\n";
  return o.str();
}

// Observation retention never controls the dynamics or stops production.
// Values are append-friendly per-chain rows; output is the existing chain-major ABI.
struct Observations {
  int chains;
  size_t historyCap, snapshotCap;
  uint64_t every;
  std::vector<uint64_t> times, snapshotTimes;
  std::vector<std::vector<uint32_t>> energy, local, nonlocal;
  std::vector<std::vector<uint8_t>> snapshots;
  std::vector<uint32_t> lastEnergy, lastLocal, lastNonlocal;
  Observations(int c,size_t hc,size_t sc,uint64_t interval,uint64_t start,
               const std::vector<uint8_t>& initial)
      : chains(c),historyCap(hc),snapshotCap(sc),every(interval),
        energy(c),local(c),nonlocal(c),lastEnergy(c),lastLocal(c),lastNonlocal(c) {
    times.reserve(hc);
    for(int i=0;i<c;++i){energy[i].reserve(hc);local[i].reserve(hc);nonlocal[i].reserve(hc);}
    snapshotTimes.push_back(start);snapshots.push_back(initial);
  }
  void thin() {
    if(every>UINT64_MAX/2)throw std::overflow_error("observation cadence overflow");
    every*=2;size_t keep=0;
    for(size_t i=0;i<times.size();++i)if(times[i]%every==0) {
      times[keep]=times[i];
      for(int c=0;c<chains;++c){energy[c][keep]=energy[c][i];local[c][keep]=local[c][i];nonlocal[c][keep]=nonlocal[c][i];}
      ++keep;
    }
    times.resize(keep);
    for(int c=0;c<chains;++c){energy[c].resize(keep);local[c].resize(keep);nonlocal[c].resize(keep);}
  }
  void append(const Segment& seg,uint64_t start,uint32_t count) {
    for(uint32_t i=0;i<count;++i) {
      uint64_t t=start+i+1;if(t%every)continue;
      // Reserve a row for the actual final boundary, even between cadence ticks.
      while(times.size()>=historyCap-1 && t%every==0)thin();
      if(t%every)continue;
      times.push_back(t);
      for(int c=0;c<chains;++c){size_t j=size_t(c)*count+i;energy[c].push_back(seg.energy[j]);local[c].push_back(seg.local[j]);nonlocal[c].push_back(seg.nonlocal[j]);}
    }
    for(int c=0;c<chains;++c){size_t j=size_t(c)*count+count-1;lastEnergy[c]=seg.energy[j];lastLocal[c]=seg.local[j];lastNonlocal[c]=seg.nonlocal[j];}
  }
  void snapshot(uint64_t t,const std::vector<uint8_t>& state) {
    snapshotTimes.push_back(t);snapshots.push_back(state);
    if(snapshots.size()>snapshotCap){snapshotTimes.erase(snapshotTimes.begin()+1);snapshots.erase(snapshots.begin()+1);}
  }
  History history(uint64_t next) const {
    History h;h.times=times;
    bool terminal=h.times.empty()||h.times.back()!=next;
    if(terminal)h.times.push_back(next);
    size_t n=h.times.size();h.energy.resize(sizeProduct(chains,n));h.local.resize(h.energy.size());h.nonlocal.resize(h.energy.size());
    for(int c=0;c<chains;++c) {
      std::copy(energy[c].begin(),energy[c].end(),h.energy.begin()+size_t(c)*n);
      std::copy(local[c].begin(),local[c].end(),h.local.begin()+size_t(c)*n);
      std::copy(nonlocal[c].begin(),nonlocal[c].end(),h.nonlocal.begin()+size_t(c)*n);
      if(terminal){size_t j=size_t(c)*n+n-1;h.energy[j]=lastEnergy[c];h.local[j]=lastLocal[c];h.nonlocal[j]=lastNonlocal[c];}
    }
    return h;
  }
  void snapshotOutput(uint64_t next,const std::vector<uint8_t>& final,
                      std::vector<uint64_t>& ts,std::vector<uint8_t>& bytes) const {
    ts=snapshotTimes;
    bool terminal=ts.back()!=next;
    size_t skip=terminal&&ts.size()==snapshotCap?1:SIZE_MAX;
    if(skip!=SIZE_MAX)ts.erase(ts.begin()+skip);
    if(terminal)ts.push_back(next);
    bytes.reserve(sizeProduct(ts.size(),final.size()));
    for(size_t i=0;i<snapshots.size();++i)if(i!=skip)bytes.insert(bytes.end(),snapshots[i].begin(),snapshots[i].end());
    if(terminal)bytes.insert(bytes.end(),final.begin(),final.end());
  }
};
// Relative-to-campaign-origin targets preserve exact tw/10 and common lags.
// For h=10^k: n*h (n=1..10), plus (10*m+r)*h for the bands below, r=1..4.
static uint64_t nextSnapshotTarget(uint64_t after) {
  uint64_t best=UINT64_MAX;
  for(uint64_t h=1;;) {
    for(uint64_t n=1;n<=10;++n)if(h<=UINT64_MAX/n){uint64_t t=n*h;if(t>after)best=std::min(best,t);}
    const uint64_t bands[6]={1,2,3,10,20,30};
    for(uint64_t m:bands)for(uint64_t r=1;r<=4;++r) {
      const uint64_t n=10*m+r;
      if(h<=UINT64_MAX/n){uint64_t t=n*h;if(t>after)best=std::min(best,t);}
    }
    if(h>UINT64_MAX/10)break;h*=10;
  }
  return best;
}

static int batch(int argc,char** argv) {
  std::string input,output;double seconds=0;
  uint64_t historyCap=4096,snapshotCap=128,probeEvery=64,probeSweeps=4,replayCap=32,observationsEvery=16;
  for(int i=1;i<argc;++i) {
    std::string arg=argv[i];if(++i==argc)throw std::invalid_argument("missing batch option value");
    if(arg=="--jobs-tsv")input=argv[i];else if(arg=="--seconds")seconds=number(argv[i]);
    else if(arg=="--output-dir")output=argv[i];
    else if(arg=="--max-history-samples")historyCap=u64(argv[i]);
    else if(arg=="--max-snapshots")snapshotCap=u64(argv[i]);
    else if(arg=="--probe-every")probeEvery=u64(argv[i]);
    else if(arg=="--probe-sweeps")probeSweeps=u64(argv[i]);
    else if(arg=="--max-replay-chunks")replayCap=u64(argv[i]);
    else if(arg=="--observations-every")observationsEvery=u64(argv[i]);
    else throw std::invalid_argument("unknown batch option: "+arg);
  }
  if(input.empty()||output.empty()||seconds<=0||historyCap<2||snapshotCap<64||
     !probeEvery||!probeSweeps||probeSweeps>4096||!replayCap||!observationsEvery||
     historyCap>SIZE_MAX||snapshotCap>SIZE_MAX)
    throw std::invalid_argument("batch requires jobs/seconds/output, history cap>=2, snapshot cap>=64, positive probe/retention/flush settings and probe_sweeps<=4096");
#if !SOLID_HAVE_CUDA
  throw std::runtime_error("jobs mode requires compiled CUDA; no CPU fallback");
#endif
  std::ifstream f(input);if(!f)throw std::runtime_error("cannot open jobs TSV");
  std::string line;std::getline(f,line);
  if(!line.empty()&&line.back()=='\r')line.pop_back();
  if(line!="id\tbeta\tmode\tchains\tseed\twindow\tthreads\tppt\thistory_every\tchunk_sweeps\tinitial\tstart_sweep")throw std::invalid_argument("jobs TSV header mismatch");
  std::vector<std::unique_ptr<Job>> jobs;
  while(std::getline(f,line)) {
    if(!line.empty()&&line.back()=='\r')line.pop_back();if(line.empty())continue;
    auto v=split(line);if(v.size()!=12)throw std::invalid_argument("jobs TSV needs 12 fields");
    auto j=std::make_unique<Job>();j->id=v[0];j->beta=v[1];j->modeText=v[2];j->mode=moveMode(v[2]);
    if(j->id.empty()||j->id=="."||j->id==".."||j->id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")!=std::string::npos)throw std::invalid_argument("unsafe job id");
    for(auto& old:jobs)if(old->id==j->id)throw std::invalid_argument("duplicate job id");
    if(number(j->beta)<=0)throw std::invalid_argument("batch beta must be positive");
    j->chains=positiveInt(v[3]);j->seed=u64(v[4]);j->launch.window=positiveInt(v[5]);j->launch.threads=positiveInt(v[6]);j->launch.ppt=positiveInt(v[7]);validateLaunch(j->launch);
    j->every=u64(v[8]);uint64_t chunk=u64(v[9]);if(!j->every||!chunk||chunk>4096)throw std::invalid_argument("history_every positive and chunk_sweeps 1..4096 required");j->chunk=uint32_t(chunk);
    j->initial=v[10];j->start=j->next=u64(v[11]);range(j->start,std::max<uint64_t>(j->chunk,probeSweeps));
    j->state=j->initial=="-"?initialized(21,j->chains,j->seed,.30,.45):rawState(j->initial,sizeProduct(j->chains,9261));
    validateState(j->state,21,j->chains);jobs.push_back(std::move(j));
  }
  if(jobs.empty())throw std::invalid_argument("empty jobs TSV");
  fs::create_directories(output);
  // Resume uses checkpoint.tsv's state/next_sweep in a fresh output directory.
  for(auto& p:jobs)if(fs::exists(fs::path(output)/p->id))throw std::runtime_error("job directory exists; resume into a fresh output directory");
  {std::ifstream in(input,std::ios::binary);std::ostringstream copy;copy<<in.rdbuf();atomicText(fs::path(output)/"jobs.tsv",copy.str());}
  for(auto& p:jobs) {
    auto& j=*p;fs::path dir=fs::path(output)/j.id;fs::create_directory(dir);
    atomicRaw(dir/"initial.raw",j.state);atomicRaw(dir/"checkpoint-0.raw",j.state);atomicText(dir/"checkpoint.tsv",checkpoint(j));
    j.engine=std::make_unique<Engine>(21,j.chains,j.seed,j.state,j.beta,true,j.launch,uint32_t(std::max<uint64_t>(j.chunk,probeSweeps)));
  }
  const auto begin=Clock::now();const auto deadline=begin+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
  std::mutex manifestMutex;
  auto publish=[&] {
    std::ostringstream o;o<<"id\tbeta\tmode\tchains\tseed\tnext_sweep\tchunks\tkernel_seconds\twall_seconds\tstatus\tcheckpoint\n"<<std::setprecision(17);
    for(auto& p:jobs){auto& j=*p;o<<j.id<<'\t'<<j.beta<<'\t'<<j.modeText<<'\t'<<j.chains<<'\t'<<j.seed<<'\t'<<j.next<<'\t'<<j.chunks<<'\t'<<j.kernel<<'\t'<<j.wall<<'\t'<<j.status<<'\t'<<j.id<<"/checkpoint.tsv\n";}
    atomicText(fs::path(output)/"manifest.tsv",o.str());
  };
  publish();std::vector<std::thread> workers;
  for(auto& ptr:jobs) {
    Job* job=ptr.get();
    workers.emplace_back([&,job] {
      Job& j=*job;Segment seg;fs::path dir=fs::path(output)/j.id;
      const std::vector<uint8_t> initial=j.state;
      Observations obs(j.chains,size_t(historyCap),size_t(snapshotCap),j.every,j.start,initial);
      uint64_t cursor=j.start,target=nextSnapshotTarget(0),productionChunks=0,probes=0;
      double secondsPerSweep=0,observationSeconds=0;
      uint32_t previousCount=1;
      // Estimates, not hard bounds on an uninterruptible sweep or filesystem.
      const double shutdownHeadroom=std::max(2.,std::min(30.,seconds*.05));
      auto estimated=[&](uint32_t count){return 2*secondsPerSweep*count;};
      auto remaining=[&]{return std::chrono::duration<double>(deadline-Clock::now()).count();};
      auto shutdownReserve=[&]{return shutdownHeadroom+estimated(uint32_t(probeSweeps))+2*observationSeconds;};
      auto nextChunk=[&]() -> uint32_t {
        const double available=remaining()-shutdownReserve(),one=estimated(1);
        if(one<=0||available<one)return 0;
        const uint32_t growth=productionChunks?std::min(j.chunk,previousCount*2):1;
        // Even a slow single sweep may exceed the subsecond target, but must
        // still fit the available estimate including mandatory final evidence.
        return uint32_t(std::min({double(growth),std::max(1.,std::floor(.75/one)),std::floor(available/one)}));
      };
      auto flushObservations=[&] {
        const auto started=Clock::now();
        History h=obs.history(cursor);std::vector<uint64_t> ts;std::vector<uint8_t> states;
        obs.snapshotOutput(cursor,j.state,ts,states);
        saveRun(dir/"observations.npz",21,j.chains,j.beta,.75,.30,.45,j.seed,j.start,cursor-j.start,0,j.mode,obs.every,j.launch,initial,initial,j.state,h,ts,states,true);
        observationSeconds=std::max(observationSeconds,std::chrono::duration<double>(Clock::now()-started).count());
      };
      auto advance=[&](uint32_t count,const fs::path& replay,bool startup=false) {
        const auto started=Clock::now();
        range(cursor,count);const uint64_t start=cursor;
        std::vector<uint8_t> incoming;if(!replay.empty())incoming=j.state;
        Segment full;
        if(!replay.empty()){size_t n=sizeProduct(j.chains,count);full.energy.resize(n);full.local.resize(n);full.nonlocal.resize(n);}
        double kernel=0;uint32_t done=0;
        while(done<count) {
          uint32_t n=startup?1:count-done;uint64_t elapsed=cursor-j.start;
          if(target>elapsed&&target-elapsed<uint64_t(n))n=uint32_t(target-elapsed);
          j.engine->run(cursor,n,j.mode,seg);obs.append(seg,cursor,n);kernel+=seg.kernelSeconds;
          if(!replay.empty())for(int c=0;c<j.chains;++c) {
            std::copy_n(seg.energy.begin()+size_t(c)*n,n,full.energy.begin()+size_t(c)*count+done);
            std::copy_n(seg.local.begin()+size_t(c)*n,n,full.local.begin()+size_t(c)*count+done);
            std::copy_n(seg.nonlocal.begin()+size_t(c)*n,n,full.nonlocal.begin()+size_t(c)*count+done);
          }
          cursor+=n;done+=n;j.state=seg.state;
          if(cursor-j.start==target){obs.snapshot(cursor,j.state);target=nextSnapshotTarget(target);}
        }
        if(!replay.empty()) {
          History h=decimate(full,j.chains,count,start,1);
          std::vector<uint8_t> states=incoming;states.insert(states.end(),j.state.begin(),j.state.end());
          saveRun(replay,21,j.chains,j.beta,.75,.30,.45,j.seed,start,count,0,j.mode,1,j.launch,incoming,incoming,j.state,h,{start,cursor},states,true);
        }
        uint64_t completed=j.chunks+1;
        atomicRaw(dir/("checkpoint-"+std::to_string(completed%2)+".raw"),j.state);
        {std::lock_guard<std::mutex> lock(manifestMutex);j.next=cursor;j.chunks=completed;j.every=obs.every;j.kernel+=kernel;j.wall=std::chrono::duration<double>(Clock::now()-begin).count();j.status="running";atomicText(dir/"checkpoint.tsv",checkpoint(j));publish();}
        secondsPerSweep=std::max(secondsPerSweep,std::chrono::duration<double>(Clock::now()-started).count()/count);
      };
      try {
        advance(uint32_t(probeSweeps),dir/"probe-first.npz",true);flushObservations();
        for(uint32_t count=nextChunk();count;count=nextChunk()) {
          advance(count,{});previousCount=count;++productionChunks;
          if(productionChunks%probeEvery==0&&remaining()>=shutdownReserve()+estimated(uint32_t(probeSweeps))) {
            std::ostringstream name;name<<"probe-"<<std::setw(6)<<std::setfill('0')<<(probes%replayCap)<<".npz";++probes;
            advance(uint32_t(probeSweeps),dir/name.str());
          }
          // Finalization publishes the newer state instead of rewriting the
          // same full retained archive immediately before the final probe.
          if(productionChunks%observationsEvery==0&&nextChunk())flushObservations();
        }
        // A small actual continuation, not a regenerated or shadow-chain fixture.
        advance(uint32_t(probeSweeps),dir/"probe-last.npz");flushObservations();
        std::lock_guard<std::mutex> lock(manifestMutex);j.status="deadline";j.wall=std::chrono::duration<double>(Clock::now()-begin).count();publish();
      } catch(const std::exception& e) {
        std::lock_guard<std::mutex> lock(manifestMutex);j.status="failed";j.error=e.what();j.wall=std::chrono::duration<double>(Clock::now()-begin).count();atomicText(dir/"error.txt",j.error+"\n");publish();
      }
    });
  }
  for(auto& t:workers)t.join();
  double wall=std::chrono::duration<double>(Clock::now()-begin).count();
  std::ostringstream metadata;metadata<<"key\tvalue\nwall_seconds\t"<<std::setprecision(17)<<wall<<"\nrequested_seconds\t"<<seconds
    <<"\nmax_history_samples\t"<<historyCap<<"\nmax_snapshots\t"<<snapshotCap<<"\nprobe_every\t"<<probeEvery<<"\nprobe_sweeps\t"<<probeSweeps
    <<"\nmax_replay_chunks\t"<<replayCap<<"\nobservations_every\t"<<observationsEvery<<"\nsource_hash\t"<<SOLID_SOURCE_HASH<<"\nbuild_flags\t"<<SOLID_BUILD_FLAGS
    <<"\nprofile\t0\nchunk_schedule\tone-sweep startup; <=2x growth within planned bound; 0.75s target; 2x high-water per-job end-to-end seconds/sweep; estimated final probe and observation flush plus 2..30s shutdown headroom; not a hard deadline\nreplay_coverage\tbounded actual probes only; gaps not replay-verified\nsnapshot_schedule\trelative campaign origin: n*10^k n1..10 plus (10*m+r)*10^k m{1,2,3,10,20,30} r1..4; latest retained plus initial\nprimary_waiting_group\tlargest fully observed h=10^k>=10 with waits10h/20h/30h vs h/2h/3h and exact lags h/10..4h/10,h..4h; select by saved times only,34h<=elapsed\nhistory_retention\tdouble absolute cadence at cap; retain aligned samples and actual final\n"
    <<"event_seconds_semantics\tper-job overlapping CUDA event sum; not global elapsed\n"<<binaryMetadata();
  atomicText(fs::path(output)/"run.tsv",metadata.str());
  bool failed=false;for(auto& j:jobs)if(j->status=="failed"){failed=true;std::cerr<<j->id<<": "<<j->error<<'\n';}
  std::cout<<"batch wall_seconds="<<wall<<" jobs="<<jobs.size()<<" manifest="<<(fs::path(output)/"manifest.tsv").string()<<'\n';
  return failed?1:0;
}
static int standalone(int argc,char** argv) {
  if(argc<9)throw std::invalid_argument("usage: driver L beta rho rho1 rho2 NUM_SWEEPS NUM_CORES CPU|CUDA [both|nonlocal|local] [EQUIL_SWEEPS] [STATE_RAW] [--seed U64] [--start-sweep U64] [--history-every N] [--max-history-samples N] [--cuda-window W] [--cuda-threads T] [--cuda-ppt P] [--cuda-profile 0|1]");
  int L=positiveInt(argv[1]);if(L<4||L>21)throw std::invalid_argument("L must be 4..21");
  std::string beta=argv[2];if(number(beta)<0)throw std::invalid_argument("negative beta");
  double rho=number(argv[3]),rho1=number(argv[4]),rho2=number(argv[5]);
  uint64_t sweeps=u64(argv[6]);int chains=positiveInt(argv[7]);
  std::string backend=argv[8];bool cuda=backend=="CUDA"||backend=="cuda";
  if(!cuda&&backend!="CPU"&&backend!="cpu"&&backend!="0")throw std::invalid_argument("backend must be CPU or CUDA; Metal/fallback unsupported");
  uint64_t seed=1,start=0,every=1,cap=100000;Launch launch;std::vector<std::string> positional;
  for(int i=9;i<argc;++i) {
    std::string arg=argv[i];if(arg.rfind("--",0)!=0){positional.push_back(arg);continue;}
    if(++i==argc)throw std::invalid_argument("missing option value");std::string value=argv[i];
    if(arg=="--seed")seed=u64(value);else if(arg=="--start-sweep")start=u64(value);
    else if(arg=="--history-every")every=u64(value);else if(arg=="--max-history-samples")cap=u64(value);
    else if(arg=="--cuda-window")launch.window=positiveInt(value);else if(arg=="--cuda-threads")launch.threads=positiveInt(value);
    else if(arg=="--cuda-ppt")launch.ppt=positiveInt(value);else if(arg=="--cuda-profile"){uint64_t p=u64(value);if(p>1)throw std::invalid_argument("profile must be 0|1");launch.profile=bool(p);}
    else throw std::invalid_argument("unknown option: "+arg);
  }
  if(!every||!cap||positional.size()>3)throw std::invalid_argument("history interval/cap must be positive; at most three positionals");
  validateLaunch(launch);int mode=positional.empty()?0:moveMode(positional[0]);uint64_t equil=positional.size()>1?u64(positional[1]):0;
  range(start,equil);range(start+equil,sweeps);
  every=std::max(every,sweeps/cap+uint64_t(sweeps%cap!=0));
  auto initial=positional.size()>2?rawState(positional[2],sizeProduct(chains,size_t(L)*L*L)):initialized(L,chains,seed,rho1,rho2);
  validateState(initial,L,chains);
  Engine engine(L,chains,seed,initial,beta,cuda,launch,4096);Segment seg;
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
  std::vector<uint8_t> snaps;size_t nextSnap=0,nextHist=0;uint64_t cur=0;double kernel=0;auto wall0=Clock::now();
  while(cur<sweeps) {
    uint64_t stop=std::min(sweeps,cur+std::min<uint64_t>(4096,sweeps-cur));if(nextSnap<planned.size())stop=std::min(stop,planned[nextSnap]);
    uint32_t n=uint32_t(stop-cur);engine.run(start+equil+cur,n,mode,seg);kernel+=seg.kernelSeconds;
    while(nextHist<h.times.size()&&h.times[nextHist]<=stop) {
      size_t j=size_t(h.times[nextHist]-cur-1);
      for(int c=0;c<chains;++c){size_t dst=size_t(c)*samples+nextHist,src=size_t(c)*n+j;h.energy[dst]=seg.energy[src];h.local[dst]=seg.local[src];h.nonlocal[dst]=seg.nonlocal[src];}
      ++nextHist;
    }
    cur=stop;state=seg.state;
    if(nextSnap<planned.size()&&cur==planned[nextSnap]){snapTimes.push_back(cur);snaps.insert(snaps.end(),state.begin(),state.end());++nextSnap;}
  }
  saveRun("run.npz",L,chains,beta,rho,rho1,rho2,seed,start,sweeps,equil,mode,every,launch,initial,equilibrated,state,h,snapTimes,snaps);
  double wall=std::chrono::duration<double>(Clock::now()-wall0).count();
  std::ostringstream receipt;receipt<<"key\tvalue\nsource_hash\t"<<SOLID_SOURCE_HASH<<"\nbuild_flags\t"<<SOLID_BUILD_FLAGS<<"\nwall_seconds\t"<<wall<<"\nkernel_event_seconds\t"<<kernel<<"\nprofile\t"<<launch.profile<<"\n"<<binaryMetadata();atomicText("run.tsv",receipt.str());
  std::cout<<"done L="<<L<<" chains="<<chains<<" sweeps="<<sweeps<<" wall="<<wall<<" kernel_event_seconds="<<kernel<<" profile="<<launch.profile<<" npz=run.npz\n";
  return 0;
}
int main(int argc,char** argv) {
  try {init_crc();if(argc>1&&std::string(argv[1])=="--jobs-tsv")return batch(argc,argv);return standalone(argc,argv);}
  catch(const std::exception& e){std::cerr<<"driver: "<<e.what()<<'\n';return 2;}
}
