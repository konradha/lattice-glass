// CPU proposal microbenchmark: byte+mask state, arithmetic neighbors,
// distance>=3 nonlocal swaps, fixed 14-site delta, xoshiro256++.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

constexpr int L = 25;
constexpr int N = L * L * L;
constexpr int N1 = 4688;
constexpr int N2 = 7031;

struct Rng {
  uint64_t s[4];
  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  static uint64_t splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  explicit Rng(uint64_t seed) { uint64_t x = seed; for (auto& v : s) v = splitmix(x); }
  uint64_t next() {
    uint64_t r = rotl(s[0] + s[3], 23) + s[0];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return r;
  }
  uint32_t bounded(uint32_t range) {
    uint64_t x = next();
    __uint128_t m = static_cast<__uint128_t>(x) * range;
    uint64_t l = static_cast<uint64_t>(m);
    if (l < range) {
      uint64_t t = (0 - static_cast<uint64_t>(range)) % range;
      while (l < t) { x = next(); m = static_cast<__uint128_t>(x) * range; l = static_cast<uint64_t>(m); }
    }
    return static_cast<uint32_t>(m >> 64);
  }
  double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
};

inline int nb(int s, int j) {
  int k = s % L, t = s / L, jj = t % L, ii = t / L;
  int di = (j == 1) - (j == 0);
  int dj = (j == 3) - (j == 2);
  int dk = (j == 5) - (j == 4);
  ii = (ii + di + L) % L; jj = (jj + dj + L) % L; k = (k + dk + L) % L;
  return k + L * (jj + ii * L);
}
inline int manhattan(int a, int b) {
  int ia = a / (L*L), ja = (a / L) % L, ka = a % L;
  int ib = b / (L*L), jb = (b / L) % L, kb = b % L;
  int dx = ia - ib; if (dx < 0) dx = -dx; if (dx > L - dx) dx = L - dx;
  int dy = ja - jb; if (dy < 0) dy = -dy; if (dy > L - dy) dy = L - dy;
  int dz = ka - kb; if (dz < 0) dz = -dz; if (dz > L - dz) dz = L - dz;
  return dx + dy + dz;
}
inline int le(int t, int m) {
  if (t == 0) return 0;
  int conn = (t == 1) ? 3 : 5;
  int v = __builtin_popcount((unsigned)m) - conn;
  return v * v;
}

struct State {
  std::vector<uint8_t> st;
  State(uint64_t seed) : st(N) {
    std::vector<int> sites(N); std::iota(sites.begin(), sites.end(), 0);
    std::mt19937 gen((uint32_t)seed); std::shuffle(sites.begin(), sites.end(), gen);
    for (int i = 0; i < N1; ++i) st[sites[i]] = 1;
    for (int i = 0; i < N2; ++i) st[sites[N1+i]] = 2;
    for (int s = 0; s < N; ++s) {
      int m = 0;
      for (int j = 0; j < 6; ++j) if ((st[nb(s,j)] & 3) != 0) m |= 1 << j;
      st[s] |= (uint8_t)(m << 2);
    }
  }
  int full_energy() const { int e=0; for (int s=0;s<N;s++) e += le(st[s]&3, st[s]>>2); return e; }
};

struct Aff { int s; uint8_t ba, bb, ns; };

inline int delta(State& S, int a, int b, Aff* aff) {
  int s0=a, s1=b;
  int s2=nb(a,0), s3=nb(a,1), s4=nb(a,2), s5=nb(a,3), s6=nb(a,4), s7=nb(a,5);
  int s8=nb(b,0), s9=nb(b,1), s10=nb(b,2), s11=nb(b,3), s12=nb(b,4), s13=nb(b,5);
  int sites[14] = {s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13};
  uint8_t ba[14] = {0,0,2,1,4,3,6,5,0,0,0,0,0,0};
  uint8_t bb[14] = {0,0,0,0,0,0,0,0,2,1,4,3,6,5};
  int ta = S.st[a]&3, tb = S.st[b]&3;
  int oa = tb != 0, ob = ta != 0;
  int old_e=0, new_e=0;
  for (int k=0;k<14;k++) {
    int s=sites[k]; uint8_t v=S.st[s]; int ts=v&3, ms=v>>2;
    old_e += le(ts,ms);
    int nm=ms;
    if (ba[k]) { int bit=ba[k]-1; nm=(nm&~(1<<bit))|(oa<<bit); }
    if (bb[k]) { int bit=bb[k]-1; nm=(nm&~(1<<bit))|(ob<<bit); }
    int nt=(s==a)?tb:((s==b)?ta:ts);
    aff[k].s=s; aff[k].ba=ba[k]; aff[k].bb=bb[k]; aff[k].ns=(uint8_t)(nt|(nm<<2));
    new_e += le(nt,nm);
  }
  return new_e-old_e;
}
inline void commit(State& S, const Aff* aff) { for (int k=0;k<14;k++) S.st[aff[k].s]=aff[k].ns; }

struct ExpTable {
  double beta;
  std::array<double,1024> t;
  explicit ExpTable(double b): beta(b) { for(int i=0;i<1024;i++) t[i]=std::exp(-b*(i-512)); }
  bool accept(int dE,double u) const { if(dE<=0)return true; int idx=dE+512; if(idx<1024)return u<t[idx]; return u<std::exp(-beta*dE); }
};

inline void pick(Rng& r,int a,int& b) {
  do { b=(a+1+(int)r.bounded(N-1))%N; } while (manhattan(a,b)<3);
}

// K interleaved chains in one thread.
template<int K>
double run(std::vector<State>& states, std::vector<Rng>& rngs, double beta, int sweeps, int mode, int& accepted) {
  ExpTable table(beta);
  Aff aff[K][14];
  int acc=0;
  auto t0=std::chrono::steady_clock::now();
  for (int sw=0; sw<sweeps; ++sw) {
    for (int i=0; i<N; ++i) {
      #pragma unroll
      for (int k=0;k<K;k++) {
        int a=(int)rngs[k].bounded(N), b;
        pick(rngs[k],a,b);
        if ((states[k].st[a]&3)==(states[k].st[b]&3)) continue;
        int dE=delta(states[k],a,b,aff[k]);
        bool ok=(dE<=0);
        if (!ok) ok=table.accept(dE,rngs[k].uniform());
        if (ok) { commit(states[k],aff[k]); ++acc; }
      }
    }
  }
  auto t1=std::chrono::steady_clock::now();
  accepted=acc;
  return std::chrono::duration<double>(t1-t0).count();
}

int verify(State& S, uint64_t seed, int checks) {
  Rng r(seed); Aff aff[14];
  for (int c=0;c<checks;c++) {
    int a=(int)r.bounded(N), b;
    do { b=(a+1+(int)r.bounded(N-1))%N; } while (manhattan(a,b)<3);
    int e0=S.full_energy();
    int dE=delta(S,a,b,aff);
    commit(S,aff);
    int e1=S.full_energy();
    if (e1-e0 != dE) { std::printf("mismatch %d %d\n",e1-e0,dE); return 1; }
    int d2=delta(S,a,b,aff); commit(S,aff);
    if (S.full_energy()!=e0) { std::printf("restore mismatch\n"); return 1; }
    (void)d2;
  }
  return 0;
}

int main(int argc, char** argv) {
  int sweeps = argc>1 ? std::atoi(argv[1]) : 2000;
  double beta = argc>2 ? std::atof(argv[2]) : 2.5;
  State v(12345);
  if (verify(v, 999, 20000)) return 1;
  std::printf("verify PASS\n");
  for (int K : {1,2,4}) {
    std::vector<State> states; std::vector<Rng> rngs;
    for (int k=0;k<K;k++) { states.emplace_back(1000+k); rngs.emplace_back(2000+k); }
    int acc=0;
    double sec;
    if (K==1) sec=run<1>(states,rngs,beta,sweeps,0,acc);
    else if (K==2) sec=run<2>(states,rngs,beta,sweeps,0,acc);
    else sec=run<4>(states,rngs,beta,sweeps,0,acc);
    double proposals=(double)K*sweeps*N;
    std::printf("K=%d sec=%.3f ns/prop/chain=%.2f aggregate=%.3f Mprop/s acc=%.4f\n",
      K, sec, sec*1e9/((double)sweeps*N), proposals/sec/1e6, (double)acc/proposals);
  }
  return 0;
}
