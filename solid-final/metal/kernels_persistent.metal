#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Persistent multi-warp sampler.
//
// One threadgroup (TG = 4 warps) owns one chain and keeps its lattice in
// threadgroup memory for the whole run.  A host-supplied segment list lets the
// driver run equilibration, every planned snapshot interval, and the tail in a
// single dispatch; only the >1M-sweep wall-clock checkpoints need a new
// dispatch.
//
// Two local-move schedules are implemented:
//
//   * Nonlocal proposals (any two distinct sites) keep the reference index
//     stream and the "exact cut": a speculative window is evaluated against
//     the state at the start of the round.  The first valid proposal (whether
//     it was accepted or rejected in the speculative pass) whose affected set
//     overlaps an earlier accepted proposal ends the window.  A rejection is
//     only trustworthy while no earlier accepted move can change it, so the
//     trigger must not be restricted to candidates that looked accepted.
//     Proposals at or after the cut are not dropped; the next round restarts
//     the index stream at the cut position, so the trajectory is identical to
//     a sequential scan.  The cut is therefore exact.
//
//   * The local path replaces the old 32-proposal conservative cut with the
//     21-colour edge schedule validated in golden/color_validate.cpp.  For a
//     fixed axis the colour of the +axis edge anchored at (x,y,z) is
//
//         axis 0 (+x): (x + 3y + 5z) mod 21
//         axis 1 (+y): (y + 3z + 5x) mod 21
//         axis 2 (+z): (z + 3x + 5y) mod 21
//
//     Every pair of same-(axis,colour) bonds has disjoint 12-site affected
//     sets, so all edges of a colour class commute and may be applied in
//     parallel.  The sweep draws one axis uniformly, walks the 21 colours in a
//     random forward/reverse order, and attempts exactly N moves (N/21 per
//     colour).  This is a *different trajectory* from the reference
//     random-site/random-direction local sweep: it preserves the stationary
//     distribution but not the reference trajectory or autocorrelation time.
//     It requires a stationarity gate rather than the bitwise CPU/Metal gate.
//     The colour proof needs L == 21 (the only GPU-supported size here), so
//     other L values keep the exact-cut local path.
//
// Nonlocal moves continue to use any two distinct sites and EFMA threshold
// semantics, so nonlocal-only runs remain bitwise identical to the CPU path.
// ---------------------------------------------------------------------------

#define TG 128u
#define WARPS 4u
#define ST_CAP 9261u
#define NOSNAP 0xffffffffu

inline uint ballotMask(bool x) { return uint(simd_ballot(x).operator simd_vote::vote_t()); }

struct WAff { int s; uchar ba, bb, ns; };
struct Th { uint hi; uint lo; };

// One contiguous run of sweeps executed by the persistent kernel.  snapSlot is
// NOSNAP when the segment does not need a state copy; otherwise the kernel
// writes the post-segment state to that slot of the snapshot buffer.  hist
// enables per-sweep history sampling for main-phase segments (it is off during
// equilibration, whose history is discarded).
struct Seg { ulong start; ulong count; uint snapSlot; uint hist; };

inline ulong mix64(ulong x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9UL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebUL;
    return x ^ (x >> 31);
}
inline int le(uint t, int cMask) {
    if (t == 0u) return 0;
    int c = int(popcount(uint(cMask) & 63u));
    int d = c - (t == 1u ? 3 : 5);
    return d*d;
}
inline int ptdist(uint a, uint b, uint L) {
    int ax = int(a & 31u), ay = int((a >> 5) & 31u), az = int((a >> 10) & 31u);
    int bx = int(b & 31u), by = int((b >> 5) & 31u), bz = int((b >> 10) & 31u);
    int dx = abs(ax - bx), dy = abs(ay - by), dz = abs(az - bz);
    dx = min(dx, int(L) - dx); dy = min(dy, int(L) - dy); dz = min(dz, int(L) - dz);
    return dx + dy + dz;
}
inline void addW(thread WAff* a, thread int& n, int s, int which, int bit) {
    for (int i = 0; i < n; i++) if (a[i].s == s) {
        if (which == 0) a[i].ba = uchar(bit + 1);
        if (which == 1) a[i].bb = uchar(bit + 1);
        return;
    }
    a[n].s = s; a[n].ba = (which == 0) ? uchar(bit + 1) : 0;
    a[n].bb = (which == 1) ? uchar(bit + 1) : 0; a[n].ns = 0; ++n;
}
inline void addEF(thread WAff* a, thread int& n, int s, int which, int bit) {
    a[n].s = s;
    a[n].ba = (which == 0) ? uchar(bit + 1) : 0;
    a[n].bb = (which == 1) ? uchar(bit + 1) : 0;
    a[n].ns = 0;
    ++n;
}
inline uint32_t efWord(ulong key, thread uint32_t& k,
                       thread ulong& cache, thread bool& have) {
    const uint32_t pair = k >> 1;
    ++k;
    if (have) {
        have = false;
        return uint32_t(cache);
    }
    cache = mix64(key + ulong(pair) * 0x9e3779b97f4a7c15UL);
    have = true;
    return uint32_t(cache >> 32);
}
inline uint32_t efIndex(ulong key, thread uint32_t& k,
                        thread ulong& cache, thread bool& have,
                        uint32_t n, uint32_t t) {
    for (;;) {
        uint32_t r = efWord(key, k, cache, have);
        ulong m = ulong(r) * ulong(n);
        uint32_t lo = uint32_t(m);
        if (lo < t) continue;
        return uint32_t(m >> 32);
    }
}
inline ulong efKey(ulong seed, ulong chain, ulong index, uint32_t localWord) {
    return seed ^ (chain * 0x9e3779b97f4a7c15UL) ^
           (index * 0xd1342543de82ef95UL) ^ (localWord ? 0x1234UL : 0x5678UL);
}

// Build the 12-/14-site affected set for one proposal and return its exact
// delta energy.  The caller owns the commit: exact-cut rounds evaluate this
// speculatively and re-evaluate accepted lanes at commit time, while the
// coloured local schedule commits immediately because same-colour affected
// sets are disjoint.
inline int evalMove(threadgroup const uchar* st, uint L, uint N,
                    uint a, uint b, uint isLocal,
                    device const uint* pc, device const ushort* nn,
                    thread WAff* aff, thread int& na) {
    const int ta = int(st[a] & 3u), tb = int(st[b] & 3u);
    na = 0;
    if (isLocal != 0u) {
        // Adjacent endpoints have no common neighbour, so after special-casing
        // the two endpoints the other ten affected sites are distinct.
        addEF(aff, na, int(a), 2, 0); addEF(aff, na, int(b), 2, 0);
        for (uint j = 0; j < 6u; ++j) {
            uint q = uint(nn[a * 6u + j]);
            if (q == b) aff[1].ba = uchar((j ^ 1u) + 1u);
            else addEF(aff, na, int(q), 0, int(j ^ 1u));
        }
        for (uint j = 0; j < 6u; ++j) {
            uint q = uint(nn[b * 6u + j]);
            if (q == a) aff[0].bb = uchar((j ^ 1u) + 1u);
            else addEF(aff, na, int(q), 1, int(j ^ 1u));
        }
    } else if (ptdist(pc[a], pc[b], L) > 2) {
        addEF(aff, na, int(a), 2, 0); addEF(aff, na, int(b), 2, 0);
        for (uint j = 0; j < 6u; ++j) addEF(aff, na, int(nn[a * 6u + j]), 0, int(j ^ 1u));
        for (uint j = 0; j < 6u; ++j) addEF(aff, na, int(nn[b * 6u + j]), 1, int(j ^ 1u));
    } else {
        addW(aff, na, int(a), 2, 0); addW(aff, na, int(b), 2, 0);
        for (uint j = 0; j < 6u; ++j) addW(aff, na, int(nn[a * 6u + j]), 0, int(j ^ 1u));
        for (uint j = 0; j < 6u; ++j) addW(aff, na, int(nn[b * 6u + j]), 1, int(j ^ 1u));
    }
    const int oa = (tb != 0), ob = (ta != 0);
    int oldE = 0, newE = 0;
    for (int i = 0; i < na; ++i) {
        int s = aff[i].s;
        uchar v = st[s];
        int ts = int(v & 3u), ms = int(v >> 2);
        oldE += le(uint(ts), ms);
        int nm = ms;
        if (aff[i].ba) { int bit = int(aff[i].ba) - 1; nm = (nm & ~(1 << bit)) | (oa << bit); }
        if (aff[i].bb) { int bit = int(aff[i].bb) - 1; nm = (nm & ~(1 << bit)) | (ob << bit); }
        int nt = (s == int(a)) ? tb : ((s == int(b)) ? ta : ts);
        aff[i].ns = uchar(nt | (nm << 2));
        newE += le(uint(nt), nm);
    }
    return newE - oldE;
}
inline void applyAff(threadgroup uchar* st, thread WAff* aff, int na) {
    for (int i = 0; i < na; ++i) st[aff[i].s] = aff[i].ns;
}
inline bool efAccept(int dE, uint32_t u_hi, ulong key,
                     thread uint32_t& wc, thread ulong& cache, thread bool& have,
                     threadgroup const Th* tab) {
    if (dE <= 0) return true;
    if (dE >= 1024) return false;
    Th t = tab[dE];
    if (u_hi < t.hi) return true;
    if (u_hi > t.hi) return false;
    return efWord(key, wc, cache, have) < t.lo;
}

// Exact-cut phase.  A window of TG consecutive proposals is evaluated against
// the state at the start of the round.  The cut is the smallest window
// position of a valid proposal that conflicts with an accepted move at a
// smaller position (or the first position past N).  Rejected proposals also
// stop the window: an earlier accepted move could flip their rejection.
// Accepted lanes below the cut are pairwise non-conflicting, so committing
// them together is identical to committing them in order; everything from the
// cut onward is re-proposed against the updated state, exactly like a
// sequential scan.
inline void exactPhase(threadgroup uchar* st,
                       threadgroup uint* ea, threadgroup uint* eb,
                       threadgroup uint* masks, threadgroup uint* cuts,
                       threadgroup uint* cutArr,
                       device const uint* pc, device const ushort* nn,
                       threadgroup const Th* tab,
                       uint L, uint N, uint isLocal, ulong indexBase,
                       ulong seed, uint tg, uint lemN, uint lem6,
                       uint tid, uint lane, uint warp,
                       thread int& edelta, thread uint& accCount) {
    uint k = 0;
    while (k < N) {
        uint32_t wc = 0;
        ulong cache = 0;
        bool have = false;
        uint a = 0u, b = 0u, u_hi = 0u;
        const bool active = (k + tid < N);
        bool valid = false, acc = false;
        if (active) {
            ulong key = efKey(seed, ulong(tg), indexBase + ulong(k) + ulong(tid), isLocal);
            if (isLocal != 0u) {
                uint aa = efIndex(key, wc, cache, have, N, lemN);
                uint dir = efIndex(key, wc, cache, have, 6u, lem6);
                a = aa; b = uint(nn[aa * 6u + dir]); valid = true;
            } else {
                uint aa = efIndex(key, wc, cache, have, N, lemN);
                uint bb = efIndex(key, wc, cache, have, N, lemN);
                if (aa != bb) { a = aa; b = bb; valid = true; }
            }
            if (valid) {
                u_hi = efWord(key, wc, cache, have);
                if ((st[a] & 3u) != (st[b] & 3u)) {
                    WAff aff[14]; int na = 0;
                    int dE = evalMove(st, L, N, a, b, isLocal, pc, nn, aff, na);
                    acc = efAccept(dE, u_hi, key, wc, cache, have, tab);
                }
            }
        }
        const uint pa = pc[a], pb = pc[b];
        if (acc) { ea[tid] = pa; eb[tid] = pb; }
        uint am = ballotMask(acc);
        if (lane == 0u) masks[warp] = am;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The cut stops at the first valid proposal (accepted or not) whose
        // affected set overlaps an earlier accepted proposal.  A rejection is
        // only trustworthy while no earlier accepted move can change it.
        bool conflict = false;
        if (active && valid) {
            for (uint ww = 0; ww <= warp; ++ww) {
                uint bits = masks[ww];
                if (ww == warp) bits &= ((1u << lane) - 1u);
                while (bits) {
                    uint i = ww * 32u + uint(ctz(bits));
                    bits &= bits - 1u;
                    conflict = conflict ||
                        (ptdist(pa, ea[i], L) <= 2) || (ptdist(pa, eb[i], L) <= 2) ||
                        (ptdist(pb, ea[i], L) <= 2) || (ptdist(pb, eb[i], L) <= 2);
                }
            }
        }
        uint cand = TG;
        if (!active) cand = tid;
        if (valid && conflict) cand = tid;
        uint wmin = simd_min(cand);
        if (lane == 0u) cuts[warp] = wmin;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) cutArr[0] = min(min(cuts[0], cuts[1]), min(cuts[2], cuts[3]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint cut = cutArr[0];
        if (acc && tid < cut) {
            WAff aff[14]; int na = 0;
            int dE = evalMove(st, L, N, a, b, isLocal, pc, nn, aff, na);
            applyAff(st, aff, na);
            edelta += dE;
            accCount += 1u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k += cut;
    }
}

// 21-colour local sweep (L == 21).  All edges of one (axis, colour) class have
// disjoint affected sets, so they commit in parallel with no cut and no
// conflict scan.  One sweep attempts N = 21 * 441 moves along a single axis;
// the axis and the forward/reverse colour order are drawn from the state-
// independent sweep key.
inline void coloredPhase(threadgroup uchar* st,
                         device const uint* pc, device const ushort* nn,
                         threadgroup const Th* tab,
                         uint L, uint N, ulong seed, uint tg, ulong sw,
                         uint tid, thread int& edelta, thread uint& accCount) {
    ulong rk = mix64(seed ^ (ulong(tg) * 0x9e3779b97f4a7c15UL) ^
                     (sw * 0xd1342543de82ef95UL) ^ 0x6c6f63616cUL);
    uint axis = uint(rk % 3UL);
    bool rev = ((rk >> 32) & 1UL) != 0UL;
    uint cls = N / 21u;
    ulong baseIdx = (3UL * sw + ulong(axis)) * ulong(N);
    for (uint ci = 0; ci < 21u; ++ci) {
        uint col = rev ? (20u - ci) : ci;
        for (uint t = tid; t < cls; t += TG) {
            uint p1 = t / 21u, p2 = t % 21u;
            int p0 = int(col) - 3 * int(p1) - 5 * int(p2);
            p0 %= 21; if (p0 < 0) p0 += 21;
            uint q0, q1, q2;
            if (axis == 0u) { q0 = uint(p0); q1 = p1; q2 = p2; }
            else if (axis == 1u) { q0 = p2; q1 = uint(p0); q2 = p1; }
            else { q0 = p1; q1 = p2; q2 = uint(p0); }
            uint a = q0 + L * q1 + L * L * q2;
            uint b = uint(nn[a * 6u + 2u * axis]);
            if ((st[a] & 3u) == (st[b] & 3u)) continue;
            ulong key = efKey(seed, ulong(tg),
                              baseIdx + ulong(col) * ulong(cls) + ulong(t), 1u);
            uint32_t wc = 0;
            ulong cache = 0;
            bool have = false;
            uint u_hi = efWord(key, wc, cache, have);
            WAff aff[14]; int na = 0;
            int dE = evalMove(st, L, N, a, b, 1u, pc, nn, aff, na);
            if (efAccept(dE, u_hi, key, wc, cache, have, tab)) {
                applyAff(st, aff, na);
                edelta += dE;
                accCount += 1u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void run_persistent(device uchar* gstate [[buffer(0)]],
                           device uint* energy_hist [[buffer(1)]],
                           device uint* acc_local [[buffer(2)]],
                           device uint* acc_nonlocal [[buffer(3)]],
                           constant uint& L [[buffer(4)]],
                           constant uint& N [[buffer(5)]],
                           constant uint& chains [[buffer(6)]],
                           device const Seg* segs [[buffer(7)]],
                           constant uint& nsegs [[buffer(8)]],
                           constant ulong& seed [[buffer(9)]],
                           device const Th* th [[buffer(10)]],
                           constant uint& mode [[buffer(11)]],
                           device const uint* pc [[buffer(12)]],
                           device const ushort* nn [[buffer(13)]],
                           constant uint& lemN [[buffer(14)]],
                           constant uint& lem6 [[buffer(15)]],
                           constant ulong& histEvery [[buffer(16)]],
                           constant ulong& histStart [[buffer(17)]],
                           constant uint& samples [[buffer(18)]],
                           device uchar* snaps [[buffer(19)]],
                           constant ulong& totalSweeps [[buffer(20)]],
                           uint tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    if (tg >= chains) return;
    threadgroup uchar st[ST_CAP];
    threadgroup uint ea[TG], eb[TG];
    threadgroup uint masks[WARPS], cuts[WARPS];
    threadgroup uint cutArr[1];
    threadgroup Th tab[1024];
    threadgroup int tengArr[1];
    threadgroup int ered[WARPS];
    threadgroup uint lred[WARPS], nred[WARPS];

    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    device uchar* base = gstate + (ulong)tg * (ulong)N;
    for (uint i = tid; i < N; i += TG) st[i] = base[i];
    for (uint i = tid; i < 1024u; i += TG) tab[i] = th[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int e0 = 0;
    for (uint i = tid; i < N; i += TG) e0 += le(uint(st[i] & 3u), int(st[i] >> 2));
    int e0s = simd_sum(e0);
    if (lane == 0u) ered[warp] = e0s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) tengArr[0] = ered[0] + ered[1] + ered[2] + ered[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint sg = 0; sg < nsegs; ++sg) {
        Seg seg = segs[sg];
        for (ulong s = 0; s < seg.count; ++s) {
            ulong sw = seg.start + s;
            int edelta = 0;
            uint alA = 0u, anA = 0u;
            uint nph = (mode == 0u) ? 2u : 1u;
            for (uint ph = 0; ph < nph; ++ph) {
                uint mm = (mode == 0u) ? (ph == 0u ? 1u : 2u) : mode;
                if (mm == 1u) {
                    ulong indexBase = (2UL * sw) * ulong(N);
                    exactPhase(st, ea, eb, masks, cuts, cutArr, pc, nn, tab,
                               L, N, 0u, indexBase, seed, tg, lemN, lem6,
                               tid, lane, warp, edelta, anA);
                } else if (L == 21u) {
                    coloredPhase(st, pc, nn, tab, L, N, seed, tg, sw,
                                 tid, edelta, alA);
                } else {
                    ulong indexBase = (2UL * sw + 1UL) * ulong(N);
                    exactPhase(st, ea, eb, masks, cuts, cutArr, pc, nn, tab,
                               L, N, 1u, indexBase, seed, tg, lemN, lem6,
                               tid, lane, warp, edelta, alA);
                }
            }
            int ese = simd_sum(edelta);
            int ale = simd_sum(int(alA)), ane = simd_sum(int(anA));
            if (lane == 0u) { ered[warp] = ese; lred[warp] = uint(ale); nred[warp] = uint(ane); }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                tengArr[0] += ered[0] + ered[1] + ered[2] + ered[3];
                if (seg.hist != 0u && histEvery > 0UL) {
                    ulong elapsed = seg.start + s - histStart + 1UL;
                    if ((elapsed % histEvery) == 0UL || elapsed == totalSweeps) {
                        ulong idx = (elapsed - 1UL) / histEvery;
                        ulong hidx = (ulong)tg * (ulong)samples + idx;
                        energy_hist[hidx] = uint(tengArr[0]);
                        acc_local[hidx] = lred[0] + lred[1] + lred[2] + lred[3];
                        acc_nonlocal[hidx] = nred[0] + nred[1] + nred[2] + nred[3];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (seg.snapSlot != NOSNAP) {
            ulong sbase = (ulong)seg.snapSlot * (ulong)chains * (ulong)N +
                          (ulong)tg * (ulong)N;
            for (uint i = tid; i < N; i += TG) snaps[sbase + (ulong)i] = st[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = tid; i < N; i += TG) base[i] = st[i];
}
