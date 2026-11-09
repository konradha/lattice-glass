#include <metal_stdlib>
using namespace metal;

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
