#include <metal_stdlib>
using namespace metal;

inline uint ballotMask(bool x) { return uint(simd_ballot(x).operator simd_vote::vote_t()); }

struct Aff { int s; uchar ba, bb, ns; };

inline ulong mix64(ulong x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9UL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebUL;
    return x ^ (x >> 31);
}
inline uint nb(uint s, uint d, uint L) {
    uint x = s % L, y = (s / L) % L, z = s / (L * L);
    if (d == 0u) x = (x + 1u) % L;
    else if (d == 1u) x = (x + L - 1u) % L;
    else if (d == 2u) y = (y + 1u) % L;
    else if (d == 3u) y = (y + L - 1u) % L;
    else if (d == 4u) z = (z + 1u) % L;
    else z = (z + L - 1u) % L;
    return x + L * y + L * L * z;
}
inline int torusL1(uint a, uint b, uint L) {
    int ax = int(a % L), ay = int((a / L) % L), az = int(a / (L * L));
    int bx = int(b % L), by = int((b / L) % L), bz = int(b / (L * L));
    int dx = abs(ax - bx), dy = abs(ay - by), dz = abs(az - bz);
    dx = min(dx, int(L) - dx); dy = min(dy, int(L) - dy); dz = min(dz, int(L) - dz);
    return dx + dy + dz;
}
inline int le(uint t, int cMask) {
    if (t == 0u) return 0;
    int c = int(popcount(uint(cMask) & 63u));
    int d = c - (t == 1u ? 3 : 5);
    return d*d;
}
inline void addAff(thread Aff* a, thread int& n, int s, int which, int bit) {
    for (int i=0;i<n;i++) if (a[i].s == s) {
        if (which == 0) a[i].ba = uchar(bit+1);
        if (which == 1) a[i].bb = uchar(bit+1);
        return;
    }
    a[n].s=s; a[n].ba=(which==0)?uchar(bit+1):0;
    a[n].bb=(which==1)?uchar(bit+1):0; a[n].ns=0; ++n;
}
inline bool tryProposal(device uchar* st, uint L, uint N, int a, int b, uint u,
                        constant uint* th, thread int& energy) {
    if (a == b) return false;
    int ta = st[a] & 3, tb = st[b] & 3;
    if (ta == tb) return false;
    Aff aff[14]; int na=0;
    addAff(aff,na,a,2,0); addAff(aff,na,b,2,0);
    for (int j=0;j<6;j++) addAff(aff,na,int(nb(uint(a),uint(j),L)),0,j^1);
    for (int j=0;j<6;j++) addAff(aff,na,int(nb(uint(b),uint(j),L)),1,j^1);
    int oa=(tb!=0), ob=(ta!=0), oldE=0, newE=0;
    for (int i=0;i<na;i++) {
        int s=aff[i].s; uchar v=st[s]; int ts=v&3, ms=v>>2;
        oldE += le(uint(ts),ms);
        int nm=ms;
        if (aff[i].ba) { int bit=aff[i].ba-1; nm=(nm&~(1<<bit))|(oa<<bit); }
        if (aff[i].bb) { int bit=aff[i].bb-1; nm=(nm&~(1<<bit))|(ob<<bit); }
        int nt=(s==a)?tb:((s==b)?ta:ts);
        aff[i].ns=uchar(nt|(nm<<2));
        newE += le(uint(nt),nm);
    }
    int dE=newE-oldE; bool ok=dE<=0;
    if (!ok) { int idx=dE+512; if (idx>=0 && idx<1024) ok=u<th[idx]; }
    if (!ok) return false;
    for (int i=0;i<na;i++) st[aff[i].s]=aff[i].ns;
    energy += dE; return true;
}

kernel void run_segment(device uchar* state [[buffer(0)]],
                        device uint* energy_hist [[buffer(1)]],
                        device uint* acc_local [[buffer(2)]],
                        device uint* acc_nonlocal [[buffer(3)]],
                        constant uint& L [[buffer(4)]],
                        constant uint& N [[buffer(5)]],
                        constant uint& chains [[buffer(6)]],
                        constant uint& start [[buffer(7)]],
                        constant uint& count [[buffer(8)]],
                        constant ulong& seed [[buffer(9)]],
                        constant uint* th [[buffer(10)]],
                        constant uint& mode [[buffer(11)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= chains) return;
    device uchar* st = state + (ulong)gid * N;
    int energy=0;
    for (uint s=0;s<N;s++) energy += le(uint(st[s]&3),int(st[s]>>2));
    for (uint k=0;k<count;k++) {
        uint sw=start+k; int al=0, an=0;
        uint64_t idx0=(uint64_t(2u)*sw+0u)*uint64_t(N);
        for (uint t=0;t<N && mode!=2u;t++) {
            uint64_t r0=mix64(seed ^ (uint64_t(gid)*0x9e3779b97f4a7c15UL) ^
                              ((idx0+uint64_t(t))*0xd1342543de82ef95UL) ^ 0x5678UL);
            uint64_t r1=mix64(r0 ^ 0xa0761d6478bd642fUL);
            uint64_t r2=mix64(r1 ^ 0xe7037ed1a0b428dbUL);
            uint a=uint(r0%uint64_t(N)), b=uint(r1%uint64_t(N));
            if (a!=b && torusL1(a,b,L)>=3 && tryProposal(st,L,N,int(a),int(b),uint(r2>>32),th,energy)) an++;
        }
        uint64_t idx1=(uint64_t(2u)*sw+1u)*uint64_t(N);
        for (uint t=0;t<N && mode!=1u;t++) {
            uint64_t r0=mix64(seed ^ (uint64_t(gid)*0x9e3779b97f4a7c15UL) ^
                              ((idx1+uint64_t(t))*0xd1342543de82ef95UL) ^ 0x1234UL);
            uint64_t r1=mix64(r0 ^ 0xa0761d6478bd642fUL);
            uint64_t r2=mix64(r1 ^ 0xe7037ed1a0b428dbUL);
            uint a=uint(r0%uint64_t(N)), b=nb(a,uint(r1%6u),L);
            if (tryProposal(st,L,N,int(a),int(b),uint(r2>>32),th,energy)) al++;
        }
        ulong hidx=(ulong)gid*count+k;
        energy_hist[hidx]=uint(energy);
        acc_local[hidx]=uint(al);
        acc_nonlocal[hidx]=uint(an);
    }
}

// ---------------------------------------------------------------------------
// Fast one-warp-per-chain kernel. W=32 proposals per round, exact cut.
// ---------------------------------------------------------------------------
struct WAff { int s; uchar ba, bb, ns; };

inline uint32_t pcoords(uint ax,uint ay,uint az){return ax|(ay<<5)|(az<<10);}
inline int ptdist(uint a,uint b,uint L){
    int ax=int(a&31u),ay=int((a>>5)&31u),az=int((a>>10)&31u);
    int bx=int(b&31u),by=int((b>>5)&31u),bz=int((b>>10)&31u);
    int dx=abs(ax-bx),dy=abs(ay-by),dz=abs(az-bz);
    dx=min(dx,int(L)-dx);dy=min(dy,int(L)-dy);dz=min(dz,int(L)-dz);
    return dx+dy+dz;
}
inline void addW(thread WAff* a, thread int& n, int s, int which, int bit) {
    for (int i=0;i<n;i++) if (a[i].s==s) {
        if (which==0) a[i].ba=uchar(bit+1);
        if (which==1) a[i].bb=uchar(bit+1);
        return;
    }
    a[n].s=s; a[n].ba=(which==0)?uchar(bit+1):0;
    a[n].bb=(which==1)?uchar(bit+1):0; a[n].ns=0; ++n;
}
inline bool tryW(device uchar* st, uint L, uint N, int a, int b, uint u,
                 constant uint* th, thread int& energy) {
    int ta=st[a]&3, tb=st[b]&3;
    if (ta==tb) return false;
    WAff aff[14]; int na=0;
    addW(aff,na,a,2,0); addW(aff,na,b,2,0);
    for(int j=0;j<6;j++) addW(aff,na,int(nb(uint(a),uint(j),L)),0,j^1);
    for(int j=0;j<6;j++) addW(aff,na,int(nb(uint(b),uint(j),L)),1,j^1);
    int oa=(tb!=0),ob=(ta!=0), oldE=0,newE=0;
    for(int i=0;i<na;i++){
        int s=aff[i].s; uchar v=st[s]; int ts=v&3, ms=v>>2;
        oldE += le(uint(ts),ms);
        int nm=ms;
        if(aff[i].ba){int bit=aff[i].ba-1;nm=(nm&~(1<<bit))|(oa<<bit);}
        if(aff[i].bb){int bit=aff[i].bb-1;nm=(nm&~(1<<bit))|(ob<<bit);}
        int nt=(s==a)?tb:((s==b)?ta:ts);
        aff[i].ns=uchar(nt|(nm<<2)); newE += le(uint(nt),nm);
    }
    int dE=newE-oldE; bool ok=dE<=0;
    if(!ok){int idx=dE+512;if(idx>=0&&idx<1024)ok=u<th[idx];}
    if(!ok)return false;
    for(int i=0;i<na;i++) st[aff[i].s]=aff[i].ns;
    energy += dE; return true;
}

kernel void run_segment_warp_legacy(device uchar* gstate [[buffer(0)]],
                             device uint* energy_hist [[buffer(1)]],
                             device uint* acc_local [[buffer(2)]],
                             device uint* acc_nonlocal [[buffer(3)]],
                             constant uint& L [[buffer(4)]],
                             constant uint& N [[buffer(5)]],
                             constant uint& chains [[buffer(6)]],
                             constant uint& start [[buffer(7)]],
                             constant uint& count [[buffer(8)]],
                             constant ulong& seed [[buffer(9)]],
                             constant uint* th [[buffer(10)]],
                             constant uint& mode [[buffer(11)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_threadgroup]]) {
    if (tg >= chains) return;
    threadgroup uchar st[9261];
    threadgroup uint ea[32], eb[32];
    device uchar* base = gstate + (ulong)tg * N;
    for (uint i=lane;i<N;i+=32u) st[i]=base[i];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    int energy=0;
    for(uint i=0;i<N;i++) energy += le(uint(st[i]&3),int(st[i]>>2));
    for(uint si=0; si<count; si++){
        uint sw=start+si;
        int an=0,al=0;
        uint nph=(mode==0u)?2u:1u;
        for(uint ph=0; ph<nph; ++ph){
        uint mm=(mode==0u)?(ph==0u?1u:2u):mode;
        uint64_t indexBase=(uint64_t(2u)*sw + (mm==2u?1u:0u))*uint64_t(N);
        uint k=0;
        while(k<N){
            uint p=uint(indexBase)+k+lane;
            bool active=k+lane<N;
            uint64_t r0=mix64(seed ^ (uint64_t(tg)*0x9e3779b97f4a7c15ULL) ^
                              ((uint64_t(p))*0xd1342543de82ef95ULL) ^ (mm==2u?0x1234ULL:0x5678ULL));
            uint64_t r1=mix64(r0 ^ 0xa0761d6478bd642fULL);
            uint64_t r2=mix64(r1 ^ 0xe7037ed1a0b428dbULL);
            uint ax=0,ay=0,az=0,bx=0,by=0,bz=0; int a=0,b=0; bool valid=false,acc=false,noop=false;
            if(active){
                if(mm==2u){
                    uint aa=uint(r0%uint64_t(N)); uint dir=uint(r1%6u);
                    uint bb=nb(aa,dir,L);
                    a=int(aa);b=int(bb);valid=true;
                }else{
                    uint aa=uint(r0%uint64_t(N)),bb=uint(r1%uint64_t(N));
                    if(aa!=bb && torusL1(aa,bb,L)>=3){a=int(aa);b=int(bb);valid=true;}
                }
                if(valid){
                    if ((st[a]&3)==(st[b]&3)) noop=true;
                    else {
                        int ta=st[a]&3,tb=st[b]&3;
                        // recompute dE/accept exactly, then commit only after cut
                        WAff aff[14]; int na=0;
                        addW(aff,na,a,2,0); addW(aff,na,b,2,0);
                        for(int j=0;j<6;j++)addW(aff,na,int(nb(uint(a),uint(j),L)),0,j^1);
                        for(int j=0;j<6;j++)addW(aff,na,int(nb(uint(b),uint(j),L)),1,j^1);
                        int oa=(tb!=0),ob=(ta!=0),oldE=0,newE=0;
                        for(int i=0;i<na;i++){int s=aff[i].s;uchar v=st[s];int ts=v&3,ms=v>>2;oldE+=le(uint(ts),ms);int nm=ms;
                            if(aff[i].ba){int bit=aff[i].ba-1;nm=(nm&~(1<<bit))|(oa<<bit);}
                            if(aff[i].bb){int bit=aff[i].bb-1;nm=(nm&~(1<<bit))|(ob<<bit);}
                            int nt=(s==a)?tb:((s==b)?ta:ts);newE+=le(uint(nt),nm);}
                        int dE=newE-oldE; bool ok=dE<=0;
                        if(!ok){int idx=dE+512;if(idx>=0&&idx<1024)ok=uint(r2>>32)<th[idx];}
                        acc=ok;
                    }
                }
            }
            uint am=ballotMask(acc);
            if(acc){ea[lane]=pcoords(uint(a%L),uint((a/L)%L),uint(a/(L*L)));
                    eb[lane]=pcoords(uint(b%L),uint((b/L)%L),uint(b/(L*L)));}
            simdgroup_barrier(mem_flags::mem_threadgroup);
            uint bits=am & ((1u<<lane)-1u);
            uint pa=pcoords(uint(a%L),uint((a/L)%L),uint(a/(L*L)));
            uint pb=pcoords(uint(b%L),uint((b/L)%L),uint(b/(L*L)));
            bool conflict=false;
            if(active && valid){
                while(bits){uint i=uint(ctz(bits));bits&=bits-1u;
                    conflict=conflict||(ptdist(pa,ea[i],L)<=2)||(ptdist(pa,eb[i],L)<=2)||
                             (ptdist(pb,ea[i],L)<=2)||(ptdist(pb,eb[i],L)<=2);}
            }
            uint cand=(active && valid && conflict)?lane:32u;
            if(!active) cand=min(cand,lane);
            uint cut=simd_min(cand);
            cut=min(cut,N-k);
            if(acc && lane<cut){
                // same exact update as tryW, accepted lanes only
                int ta=st[a]&3,tb=st[b]&3;
                WAff aff[14];int na=0;addW(aff,na,a,2,0);addW(aff,na,b,2,0);
                for(int j=0;j<6;j++)addW(aff,na,int(nb(uint(a),uint(j),L)),0,j^1);
                for(int j=0;j<6;j++)addW(aff,na,int(nb(uint(b),uint(j),L)),1,j^1);
                int oa=(tb!=0),ob=(ta!=0),oldE=0,newE=0;
                for(int i=0;i<na;i++){int s=aff[i].s;uchar v=st[s];int ts=v&3,ms=v>>2;oldE+=le(uint(ts),ms);int nm=ms;
                    if(aff[i].ba){int bit=aff[i].ba-1;nm=(nm&~(1<<bit))|(oa<<bit);}
                    if(aff[i].bb){int bit=aff[i].bb-1;nm=(nm&~(1<<bit))|(ob<<bit);}
                    int nt=(s==a)?tb:((s==b)?ta:ts);aff[i].ns=uchar(nt|(nm<<2));newE+=le(uint(nt),nm);}
                for(int i=0;i<na;i++) st[aff[i].s]=aff[i].ns;
                energy += newE-oldE;
            }
            uint ac=ballotMask(acc && lane<cut);
            if(lane==0){ if(mm==2u) al+=uint(popcount(ac)); else an+=uint(popcount(ac)); }
            k += cut;
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        }
        int full=0;
        for (uint z=0; z<N; z++) full += le(uint(st[z]&3), int(st[z]>>2));
        energy=full;
        ulong hidx=(ulong)tg*count+si;
        energy_hist[hidx]=uint(full);
        acc_local[hidx]=uint(al);
        acc_nonlocal[hidx]=uint(an);
    }
    for(uint i=lane;i<N;i+=32u) base[i]=st[i];
}

// ---------------------------------------------------------------------------
// EFMA fast kernel. One warp per chain. 64-bit threshold table, lazy low word,
// exact Lemire indices, shared thresholds, coordinate/neighbor lookup tables.
// ---------------------------------------------------------------------------
struct Th { uint hi; uint lo; };

inline uint32_t efWord(uint64_t key, thread uint32_t& k,
                       thread uint64_t& cache, thread bool& have) {
    const uint32_t pair = k >> 1;
    ++k;
    if (have) {
        have = false;
        return uint32_t(cache);
    }
    cache = mix64(key + uint64_t(pair) * 0x9e3779b97f4a7c15ULL);
    have = true;
    return uint32_t(cache >> 32);
}
inline uint32_t efIndex(uint64_t key, thread uint32_t& k,
                        thread uint64_t& cache, thread bool& have,
                        uint32_t n, uint32_t t) {
    for (;;) {
        uint32_t r = efWord(key, k, cache, have);
        uint64_t m = uint64_t(r) * uint64_t(n);
        uint32_t lo = uint32_t(m);
        if (lo < t) continue;
        return uint32_t(m >> 32);
    }
}
inline uint64_t efKey(uint64_t seed, uint64_t chain, uint64_t index, uint32_t local) {
    return seed ^ (chain * 0x9e3779b97f4a7c15ULL) ^
           (index * 0xd1342543de82ef95ULL) ^ (local ? 0x1234ULL : 0x5678ULL);
}
inline void addEF(thread WAff* a, thread int& n, int s, int which, int bit) {
    a[n].s = s;
    a[n].ba = (which == 0) ? uchar(bit + 1) : 0;
    a[n].bb = (which == 1) ? uchar(bit + 1) : 0;
    a[n].ns = 0;
    ++n;
}

kernel void run_segment_warp(device uchar* gstate [[buffer(0)]],
                             device uint* energy_hist [[buffer(1)]],
                             device uint* acc_local [[buffer(2)]],
                             device uint* acc_nonlocal [[buffer(3)]],
                             constant uint& L [[buffer(4)]],
                             constant uint& N [[buffer(5)]],
                             constant uint& chains [[buffer(6)]],
                             constant uint& start [[buffer(7)]],
                             constant uint& count [[buffer(8)]],
                             constant ulong& seed [[buffer(9)]],
                             device const Th* th [[buffer(10)]],
                             constant uint& mode [[buffer(11)]],
                             device const uint* pc [[buffer(12)]],
                             device const ushort* nn [[buffer(13)]],
                             constant uint& lemN [[buffer(14)]],
                             constant uint& lem6 [[buffer(15)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_threadgroup]]) {
    if (tg >= chains) return;
    threadgroup uchar st[9261];
    threadgroup uint ea[32], eb[32];
    threadgroup Th tab[1024];
    threadgroup int teng;

    device uchar* base = gstate + (ulong)tg * N;
    for (uint i = lane; i < N; i += 32u) st[i] = base[i];
    for (uint i = lane; i < 1024u; i += 32u) tab[i] = th[i];
    simdgroup_barrier(mem_flags::mem_threadgroup);

    int e0 = 0;
    for (uint i = lane; i < N; i += 32u) e0 += le(uint(st[i] & 3), int(st[i] >> 2));
    int e0s = simd_sum(e0);
    if (lane == 0) teng = e0s;
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint si = 0; si < count; ++si) {
        uint sw = start + si;
        int an = 0, al = 0;
        uint nph = (mode == 0u) ? 2u : 1u;
        for (uint ph = 0; ph < nph; ++ph) {
            uint mm = (mode == 0u) ? (ph == 0u ? 1u : 2u) : mode;
            uint64_t indexBase = (uint64_t(2u) * sw + (mm == 2u ? 1u : 0u)) * uint64_t(N);
            uint k = 0;
            while (k < N) {
                uint64_t p = indexBase + uint64_t(k) + uint64_t(lane);
                bool active = (k + lane) < N;
                uint32_t wc = 0;
                uint64_t wcache = 0; bool whave = false;
                uint64_t key = efKey(seed, uint64_t(tg), p, mm == 2u);
                int a = 0, b = 0;
                bool valid = false, acc = false, noop = false;
                uint32_t u_hi = 0;

                if (active) {
                    uint32_t aa = efIndex(key, wc, wcache, whave, N, lemN);
                    if (mm == 2u) {
                        uint32_t dir = efIndex(key, wc, wcache, whave, 6u, lem6);
                        a = int(aa);
                        b = int(nn[aa * 6u + dir]);
                        valid = true;
                    } else {
                        uint32_t bb = efIndex(key, wc, wcache, whave, N, lemN);
                        if (aa != bb && ptdist(pc[aa], pc[bb], L) >= 3) {
                            a = int(aa);
                            b = int(bb);
                            valid = true;
                        }
                    }
                    if (valid) {
                        u_hi = efWord(key, wc, wcache, whave);
                        if ((st[a] & 3) == (st[b] & 3)) {
                            noop = true;
                        } else {
                            int ta = st[a] & 3, tb = st[b] & 3;
                            WAff aff[14]; int na = 0;
                            if (mm == 2u) {
                                addEF(aff, na, a, 2, 0); addEF(aff, na, b, 2, 0);
                                for (int j = 0; j < 6; ++j) {
                                    int q = int(nn[uint(a) * 6u + uint(j)]);
                                    if (q == b) aff[1].ba = uchar((j ^ 1) + 1);
                                    else addEF(aff, na, q, 0, j ^ 1);
                                }
                                for (int j = 0; j < 6; ++j) {
                                    int q = int(nn[uint(b) * 6u + uint(j)]);
                                    if (q == a) aff[0].bb = uchar((j ^ 1) + 1);
                                    else addEF(aff, na, q, 1, j ^ 1);
                                }
                            } else {
                                addEF(aff, na, a, 2, 0); addEF(aff, na, b, 2, 0);
                                for (int j = 0; j < 6; ++j) addEF(aff, na, int(nn[uint(a) * 6u + uint(j)]), 0, j ^ 1);
                                for (int j = 0; j < 6; ++j) addEF(aff, na, int(nn[uint(b) * 6u + uint(j)]), 1, j ^ 1);
                            }
                            int oa = (tb != 0), ob = (ta != 0), oldE = 0, newE = 0;
                            for (int i = 0; i < na; ++i) {
                                int s = aff[i].s;
                                uchar v = st[s];
                                int ts = v & 3, ms = v >> 2;
                                oldE += le(uint(ts), ms);
                                int nm = ms;
                                if (aff[i].ba) { int bit = aff[i].ba - 1; nm = (nm & ~(1 << bit)) | (oa << bit); }
                                if (aff[i].bb) { int bit = aff[i].bb - 1; nm = (nm & ~(1 << bit)) | (ob << bit); }
                                int nt = (s == a) ? tb : ((s == b) ? ta : ts);
                                newE += le(uint(nt), nm);
                            }
                            int dE = newE - oldE;
                            if (dE <= 0) {
                                acc = true;
                            } else if (dE >= 1024) {
                                acc = false;
                            } else {
                                Th tt = tab[dE];
                                if (u_hi < tt.hi) acc = true;
                                else if (u_hi > tt.hi) acc = false;
                                else {
                                    uint32_t u_lo = efWord(key, wc, wcache, whave);
                                    acc = (u_lo < tt.lo);
                                }
                            }
                        }
                    }
                }

                uint am = ballotMask(acc);
                if (acc) { ea[lane] = pc[a]; eb[lane] = pc[b]; }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                uint bits = am & ((1u << lane) - 1u);
                uint pa = pc[a], pb = pc[b];
                bool conflict = false;
                if (active && valid) {
                    while (bits) {
                        uint i = uint(ctz(bits));
                        bits &= bits - 1u;
                        conflict = conflict || (ptdist(pa, ea[i], L) <= 2) || (ptdist(pa, eb[i], L) <= 2) ||
                                   (ptdist(pb, ea[i], L) <= 2) || (ptdist(pb, eb[i], L) <= 2);
                    }
                }
                uint cand = (active && valid && conflict) ? lane : 32u;
                if (!active) cand = min(cand, lane);
                uint cut = simd_min(cand);
                cut = min(cut, N - k);

                int dcommit = 0;
                if (acc && lane < cut) {
                    int ta = st[a] & 3, tb = st[b] & 3;
                    WAff aff[14]; int na = 0;
                    if (mm == 2u) {
                        addEF(aff, na, a, 2, 0); addEF(aff, na, b, 2, 0);
                        for (int j = 0; j < 6; ++j) {
                            int q = int(nn[uint(a) * 6u + uint(j)]);
                            if (q == b) aff[1].ba = uchar((j ^ 1) + 1);
                            else addEF(aff, na, q, 0, j ^ 1);
                        }
                        for (int j = 0; j < 6; ++j) {
                            int q = int(nn[uint(b) * 6u + uint(j)]);
                            if (q == a) aff[0].bb = uchar((j ^ 1) + 1);
                            else addEF(aff, na, q, 1, j ^ 1);
                        }
                    } else {
                        addEF(aff, na, a, 2, 0); addEF(aff, na, b, 2, 0);
                        for (int j = 0; j < 6; ++j) addEF(aff, na, int(nn[uint(a) * 6u + uint(j)]), 0, j ^ 1);
                        for (int j = 0; j < 6; ++j) addEF(aff, na, int(nn[uint(b) * 6u + uint(j)]), 1, j ^ 1);
                    }
                    int oa = (tb != 0), ob = (ta != 0), oldE = 0, newE = 0;
                    for (int i = 0; i < na; ++i) {
                        int s = aff[i].s;
                        uchar v = st[s];
                        int ts = v & 3, ms = v >> 2;
                        oldE += le(uint(ts), ms);
                        int nm = ms;
                        if (aff[i].ba) { int bit = aff[i].ba - 1; nm = (nm & ~(1 << bit)) | (oa << bit); }
                        if (aff[i].bb) { int bit = aff[i].bb - 1; nm = (nm & ~(1 << bit)) | (ob << bit); }
                        int nt = (s == a) ? tb : ((s == b) ? ta : ts);
                        aff[i].ns = uchar(nt | (nm << 2));
                        newE += le(uint(nt), nm);
                    }
                    for (int i = 0; i < na; ++i) st[aff[i].s] = aff[i].ns;
                    dcommit = newE - oldE;
                }
                int dsum = simd_sum(dcommit);
                if (lane == 0) teng += dsum;

                uint ac = ballotMask(acc && lane < cut);
                if (lane == 0) {
                    if (mm == 2u) al += uint(popcount(ac));
                    else an += uint(popcount(ac));
                }
                k += cut;
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        ulong hidx = (ulong)tg * count + si;
        if (lane == 0) energy_hist[hidx] = uint(teng);
        if (lane == 0) {
            acc_local[hidx] = uint(al);
            acc_nonlocal[hidx] = uint(an);
        }
    }
    for (uint i = lane; i < N; i += 32u) base[i] = st[i];
}
