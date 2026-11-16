"""Leiden community detection (modularity) in numba, for any number of resolutions at once.

The objective is igraph's ``community_leiden(objective_function="modularity")`` as Scanpy calls
it: node weights are strengths, the resolution is divided by the total node weight, and every
stored entry of the adjacency is an edge (so ``A + A.T`` is clustered). Each Leiden pass is local
moving, refinement with randomness ``beta``, and aggregation on the refined partition with
the moved partition as initial membership; ``n_iterations`` passes each start from the previous
partition (negative: until the partition stops changing, at most ``_MAX_ITERATIONS`` passes, with
a warning when that cap is hit).

Parallel and deterministic:

- local moving goes colour class by colour class of a greedy colouring (vertices visited in a
  seeded random order, classes processed in a seeded random order on every round). A class has no
  internal edges, so its active vertices find their best moves in parallel against a snapshot of
  community totals, and the moves are applied in a fixed order. Moved vertices re-activate their
  neighbours, as igraph's queue does,
- refinement runs in parallel over clusters, each cluster with its own random stream,
- aggregation builds the collapsed graph in two parallel passes.

Edge weights are summed per neighbouring community in a small hash table per chunk of work (see
"hash tables" below), not in arrays as long as the graph. An item with more neighbouring
communities than its chunk's table holds is summed afterwards, serially, in one dense table.
Either way each community's sum adds the same weights in the same order, and communities are
listed in the order they are first met, so the tables don't change results.

Several resolutions run one after another, each with parallel kernels.

A partition depends only on (graph, resolution, seed, n_iterations, beta): not on the number of
threads, on how work is chunked, on the table sizes, or on which other
resolutions are computed in the same call.
"""

from __future__ import annotations

import types

import numba
import numpy as np

from .._compat import warn
from .._utils._numba import _mix

_MAX_ITERATIONS = 1000  # read at call time and passed to the kernels, so tests can lower it
_MAX_ROUNDS = 1000  # local-moving rounds per level: stale totals within a class could in principle cycle
_SMALL = 512  # below this many vertices, a parallel launch costs more than it saves
_NO_LIMIT = 1 << 62  # the key limit of a dense table
# Slots per chunk table, powers of two (read at call time by `_table_slots`); a graph with
# fewer vertices gets dense tables. 2**16 slots of 16 B are 1 MiB, so 384 chunks hold 0.4 GB.
_TABLE_MIN, _TABLE_MAX = 1 << 16, 1 << 20
# Helpers called only from jitted code. fau's `njit` returns a Python dispatcher that jitted
# callers can't call, and has no `inline=`; these are compiled into each caller's variant.
_sjit = numba.njit(cache=True, nogil=True)  # noqa: TID251 (callee-only, see above)
_ijit = numba.njit(cache=True, nogil=True, inline="always")  # noqa: TID251 (callee-only, see above)

# A workspace `ws` is a flat tuple (numba's parallel lowering rejects nested ones):
#   (idx_buf, w_buf, ws_i, ws_f, ws_b, ws_u, tkey, tval, tlst, bkey, bval, blst)
# idx_buf/w_buf (2, nnz): the aggregated graphs' edges, double-buffered. Rows of ws_i/ws_f/ws_b,
# each n+1 long (unpacked by position in the kernels):
#   ws_i  0-9   local moving: color order cptr mark nbn empty prop perm alist cperm
#        10-17  driver: agg memb refined cptr cnodes tmp prev corder (tmp doubles as _refine's
#               size counts, then its deferred clusters; corder as local moving's deferred vertices)
#        18-19  indptr of the aggregated graph (double-buffered with idx_buf/w_buf), 20 iota
#   ws_f  0 cw, 1 rcw, 2 ext, 3 init, 4 nw0, 5-6 node weights of the aggregated graph (double-buffered)
#   ws_b  0 active, 1 nonsing
#   ws_u  (nch+1 long) 0 the driver's random state, 1.. refinement states, one per chunk
# tkey/tval/tlst (nch, slots): one hash table per chunk; bkey/bval/blst (n+1, or 1 when the chunk
# tables are dense): the dense table for deferred items.
_NWS_I, _NWS_F, _NWS_B = 21, 7, 2


# ---------------------------------------------------------------- random numbers and small helpers
@_ijit
def _uniform(state):
    state[0] += np.uint64(0x9E3779B97F4A7C15)
    return np.float64(_mix(state[0]) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@_sjit
def _shuffle(a, lo, hi, state):
    for i in range(hi - 1, lo, -1):
        j = lo + np.int64(_uniform(state) * (i - lo + 1))
        a[i], a[j] = a[j], a[i]


@_sjit
def _relabel(memb, n, tmp):
    """Renumber memb[:n] to 0..k-1 by first appearance; returns k."""
    for i in range(n):
        tmp[i] = -1
    k = 0
    for v in range(n):
        c = memb[v]
        if tmp[c] < 0:
            tmp[c] = k
            k += 1
        memb[v] = tmp[c]
    return k


@_sjit
def _group(lab, perm, n, k, cptr, cnodes):  # noqa: PLR0917
    """Counting-sort perm[:n] by lab, keeping perm's order: group c is cnodes[cptr[c]:cptr[c+1]]."""
    for c in range(k + 1):
        cptr[c] = 0
    for v in range(n):
        cptr[lab[v] + 1] += 1
    for c in range(k):
        cptr[c + 1] += cptr[c]
    for p in range(n):
        v = perm[p]
        c = lab[v]
        cnodes[cptr[c]] = v
        cptr[c] += 1
    for c in range(k, 0, -1):
        cptr[c] = cptr[c - 1]
    cptr[0] = 0


# ---------------------------------------------------------------- hash tables
# A table t = (tkey, tval, tlst) sums values per key (a community id): tkey holds the keys (-1 for
# an empty slot, whose sum in tval is 0.0), tlst the used slots in insertion order. Its parameters
# tp = (mask, shift, lim): shift 0 is a dense table, where the key is the slot; otherwise an
# open-addressing table of mask+1 slots with linear probing from the key's Fibonacci hash, holding
# at most `lim` keys (half its slots). Users clear the slots they listed, so tables stay empty
# between uses and a smaller table can use the front of a larger one.
@_sjit
def _table(slots, nkeys, space):
    """Parameters of a table for up to `nkeys` keys in [0, space), using at most `slots` slots.

    Dense if `space` fits, else the smallest power of two with room for 2 * nkeys (`slots` is then
    a power of two).
    """
    if space <= slots:
        return 0, 0, _NO_LIMIT
    bits = 1
    while (1 << bits) < min(2 * nkeys, slots):
        bits += 1
    return (1 << bits) - 1, 64 - bits, 1 << (bits - 1)


@_ijit
def _slot(tkey, key, mask, shift):
    """The slot holding `key`, or the empty slot where it goes."""
    if shift == 0:
        return np.int64(key)
    s = np.int64((np.uint64(key) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(shift))
    while tkey[s] >= 0 and tkey[s] != key:
        s = (s + 1) & mask
    return s


@_ijit
def _add(t, tp, nl, key, x):
    """Sum x into key's slot, listing the slot on first touch; returns the new list length."""
    tkey, tval, tlst = t
    s = _slot(tkey, key, tp[0], tp[1])
    if tkey[s] < 0:
        tkey[s] = key
        tlst[nl] = s
        nl += 1
    tval[s] += x
    return nl


@_ijit
def _clear(t, nl):
    tkey, tval, tlst = t
    for i in range(nl):
        s = tlst[i]
        tkey[s] = -1
        tval[s] = 0.0


# ---------------------------------------------------------------- local moving
@_sjit
def _coloring(ptr, idx, n, perm, color, order, cptr, mark):  # noqa: PLR0917
    """Greedy distance-1 colouring visiting vertices in `perm` order; classes in order[cptr[c]:cptr[c+1]]."""
    for v in range(n):
        color[v] = -1
        mark[v] = -1
    nc = 0
    for p in range(n):
        v = perm[p]
        for e in range(ptr[v], ptr[v + 1]):
            c = color[idx[e]]
            if c >= 0:
                mark[c] = v
        c = 0
        while mark[c] == v:
            c += 1
        color[v] = c
        nc = max(nc, c + 1)
    # within a class, keep the random visiting order
    _group(color, perm, n, nc, cptr, order)
    return nc


@_ijit
def _best_move(v, g, memb, cw, gamma, t, tp):  # noqa: PLR0917
    """Best community for v given the others' labels and community totals; -1 means a new singleton."""
    ptr, idx, w, nw = g
    tkey, tval, tlst = t
    cur = memb[v]
    wv = nw[v]
    nl = 0
    for e in range(ptr[v], ptr[v + 1]):
        u = idx[e]
        if u != v:
            nl = _add(t, tp, nl, memb[u], w[e])
    # cur's sum is 0.0 if no neighbour is in it
    maxd = tval[_slot(tkey, cur, tp[0], tp[1])] - wv * (cw[cur] - wv) * gamma
    best = cur
    if maxd < 0.0:  # an empty community has gain 0
        maxd = 0.0
        best = -1
    for i in range(nl):
        s = tlst[i]
        c = np.int64(tkey[s])
        if c != cur:
            d = tval[s] - wv * cw[c] * gamma
            if d > maxd:
                maxd = d
                best = c
    _clear(t, nl)
    return best


def _propose(g, ws, gamma, m, nbig, tp):
    """Best moves of the m active vertices of a colour class (alist), against the current totals.

    The nbig vertices in blist have more neighbours than a chunk table holds; they use the dense
    table afterwards.
    """
    ptr = g[0]
    ws_i, ws_f = ws[2], ws[3]
    tkey, tval, tlst, bkey, bval, blst = ws[6], ws[7], ws[8], ws[9], ws[10], ws[11]
    prop, alist, memb, blist = ws_i[6], ws_i[8], ws_i[11], ws_i[17]
    cw = ws_f[0]
    lim = tp[2]
    if m < _SMALL:
        t0 = (tkey[0], tval[0], tlst[0])
        for p in range(m):
            v = alist[p]
            if ptr[v + 1] - ptr[v] <= lim:
                prop[v] = _best_move(v, g, memb, cw, gamma, t0, tp)
    else:
        nch = tkey.shape[0]
        for ch in numba.prange(nch):
            t = (tkey[ch], tval[ch], tlst[ch])
            for p in range(ch * m // nch, (ch + 1) * m // nch):
                v = alist[p]
                if ptr[v + 1] - ptr[v] <= lim:
                    prop[v] = _best_move(v, g, memb, cw, gamma, t, tp)
    big = (bkey, bval, blst)
    bp = _table(bkey.shape[0], 1, bkey.shape[0])  # dense
    for j in range(nbig):
        u = blist[j]
        prop[u] = _best_move(u, g, memb, cw, gamma, big, bp)


@_sjit
def _apply(g, memb, cw, nbn, empty, ne, alist, m, active, prop):  # noqa: PLR0913, PLR0917
    """Apply the proposals in alist order; returns the new size of the empty-community stack."""
    ptr, idx, _, nw = g
    for p in range(m):
        v = alist[p]
        active[v] = False
        cur, best = memb[v], prop[v]
        # a vertex already alone that asks for a new singleton doesn't move
        if best == cur or (best == -1 and nbn[cur] == 1):
            continue
        # cur keeps a member, so at most n-1 communities are in use: ne >= 1
        if best == -1:
            ne -= 1
            best = empty[ne]
        cw[cur] -= nw[v]
        nbn[cur] -= 1
        if nbn[cur] == 0:
            empty[ne] = cur
            ne += 1
        cw[best] += nw[v]
        nbn[best] += 1
        memb[v] = best
        for e in range(ptr[v], ptr[v + 1]):
            u = idx[e]
            if u != v and memb[u] != best:
                active[u] = True
    return ne


def _move_nodes(g, n, gamma, state, ws, tp, maxdeg):  # noqa: PLR0917
    """Local moving of the n vertices of g, from and into the membership ws_i[11].

    `maxdeg` is g's largest degree: if it is above tp's key limit, vertices with more neighbours
    are listed for the dense table.
    """
    ptr, idx, nw = g[0], g[1], g[3]
    ws_i, ws_f, ws_b = ws[2], ws[3], ws[4]
    color, order, cptr, mark, nbn, empty, prop, perm, alist, cperm = ws_i[:10]
    memb, blist = ws_i[11], ws_i[17]
    cw, active = ws_f[0], ws_b[0]
    lim = tp[2]
    check = maxdeg > lim
    for v in range(n):
        perm[v] = v
    _shuffle(perm, 0, n, state)
    nc = _coloring(ptr, idx, n, perm, color, order, cptr, mark)
    for c in range(n):
        cw[c] = 0.0
        nbn[c] = 0
    for v in range(n):
        cw[memb[v]] += nw[v]
        nbn[memb[v]] += 1
        active[v] = True
    ne = 0
    for c in range(n - 1, -1, -1):
        if nbn[c] == 0:
            empty[ne] = c
            ne += 1
    for c in range(nc):
        cperm[c] = c
    n_active = n
    rounds = 0
    while n_active > 0 and rounds < _MAX_ROUNDS:
        rounds += 1
        _shuffle(cperm, 0, nc, state)
        for q in range(nc):
            c = cperm[q]
            m = 0
            nbig = 0
            for p in range(cptr[c], cptr[c + 1]):
                v = order[p]
                if active[v]:
                    alist[m] = v
                    m += 1
                    if check and ptr[v + 1] - ptr[v] > lim:
                        blist[nbig] = v
                        nbig += 1
            if m == 0:
                continue
            _propose(g, ws, gamma, m, nbig, tp)
            ne = _apply(g, memb, cw, nbn, empty, ne, alist, m, active, prop)
        n_active = 0
        for v in range(n):
            if active[v]:
                n_active += 1


# ---------------------------------------------------------------- refinement
@_ijit
def _refine_cluster(  # noqa: PLR0912, PLR0913, PLR0915, PLR0917
    cl, g, rows, rcw, ext, nonsing, gamma, beta, seed, state, t, tp
):
    """Igraph's refinement of cluster cl, with rows = (memb, refined, cptr, cnodes).

    Returns False without refining if a vertex could list more keys than tp's limit; a rerun
    resets everything this changed.
    """
    ptr, idx, w, nw = g
    memb, refined, cptr, cnodes = rows
    tkey, tval, tlst = t
    state[0] = _mix(seed ^ (np.uint64(cl) * np.uint64(0x9E3779B97F4A7C15)))
    lo, hi = cptr[cl], cptr[cl + 1]
    tot = 0.0
    most = 0  # largest number of neighbours inside cl
    for p in range(lo, hi):
        v = cnodes[p]
        tot += nw[v]
        refined[v] = v
        rcw[v] = nw[v]
        nonsing[v] = False
        s = 0.0
        nin = 0
        for e in range(ptr[v], ptr[v + 1]):
            u = idx[e]
            if u != v and memb[u] == cl:
                s += w[e]
                nin += 1
        ext[v] = s
        most = max(most, nin)
    # a vertex lists its own and its neighbours' refined communities, all members of cl
    if min(hi - lo, most + 1) > tp[2]:
        return False
    _shuffle(cnodes, lo, hi, state)
    for p in range(lo, hi):
        v = cnodes[p]
        cur = refined[v]
        if nonsing[cur] or ext[cur] < rcw[cur] * (tot - rcw[cur]) * gamma:
            continue
        rcw[cur] = 0.0
        ext[cur] = 0.0
        q = _slot(tkey, cur, tp[0], tp[1])  # listed first, with sum 0.0
        tkey[q] = cur
        tlst[0] = q
        nl = 1
        for e in range(ptr[v], ptr[v + 1]):
            u = idx[e]
            if u != v and memb[u] == cl:
                nl = _add(t, tp, nl, refined[u], w[e])
        best = cur
        maxd = 0.0
        total = 0.0
        for i in range(nl):
            q = tlst[i]
            c = np.int64(tkey[q])
            if ext[c] >= rcw[c] * (tot - rcw[c]) * gamma:
                d = tval[q] - nw[v] * rcw[c] * gamma
                if d > maxd:
                    maxd = d
                    best = c
                if d >= 0:
                    total += np.exp(d / beta)
            tval[q] = total  # the slot's sum is used up: it now holds the cumulative total
        if total < np.inf:  # otherwise the greedy choice (exp overflowed)
            r = _uniform(state) * total
            a, b = 0, nl - 1
            while a < b:
                mid = (a + b) // 2
                if tval[tlst[mid]] > r:
                    b = mid
                else:
                    a = mid + 1
            best = np.int64(tkey[tlst[a]])
        _clear(t, nl)
        rcw[best] += nw[v]
        for e in range(ptr[v], ptr[v + 1]):
            u = idx[e]
            if u != v and memb[u] == cl:
                if refined[u] == best:
                    ext[best] -= w[e]
                else:
                    ext[best] += w[e]
        if best != cur:
            refined[v] = best
            nonsing[best] = True
    return True


def _refine(g, k, gamma, beta, seed, ws, tp):  # noqa: PLR0917
    """Igraph's refinement of each of the k clusters; clusters dealt to chunks round-robin, largest first.

    Clusters whose vertices could list more keys than a chunk table holds are refined afterwards
    with the dense table. Every cluster has its own random stream, so the order doesn't matter.
    """
    ws_i, ws_f, ws_b, ws_u = ws[2], ws[3], ws[4], ws[5]
    tkey, tval, tlst, bkey, bval, blst = ws[6], ws[7], ws[8], ws[9], ws[10], ws[11]
    memb, refined, cptr, cnodes, cnt = ws_i[11:16]
    corder = ws_i[17]
    rcw, ext, nonsing, states = ws_f[1], ws_f[2], ws_b[1], ws_u[1:]
    rows = (memb, refined, cptr, cnodes)
    # order by decreasing size, ties by index (a stable counting sort into cnt, no allocation)
    big = 0
    for c in range(k):
        big = max(big, cptr[c + 1] - cptr[c])
    for s in range(big + 1):
        cnt[s] = 0
    for c in range(k):
        cnt[cptr[c + 1] - cptr[c]] += 1
    pos = 0
    for s in range(big, -1, -1):
        t = cnt[s]
        cnt[s] = pos
        pos += t
    for c in range(k):
        s = cptr[c + 1] - cptr[c]
        corder[cnt[s]] = c
        cnt[s] += 1
    nch = tkey.shape[0]
    for ch in numba.prange(nch):
        tab = (tkey[ch], tval[ch], tlst[ch])
        state = states[ch : ch + 1]
        for i in range(ch, k, nch):
            # cnt[i]: whether corder[i] is deferred
            done = _refine_cluster(
                corder[i],
                g,
                rows,
                rcw,
                ext,
                nonsing,
                gamma,
                beta,
                seed,
                state,
                tab,
                tp,
            )
            cnt[i] = 0 if done else 1
    dense = (bkey, bval, blst)
    bp = _table(bkey.shape[0], 1, bkey.shape[0])
    for j in range(k):
        if cnt[j]:
            _refine_cluster(
                corder[j],
                g,
                rows,
                rcw,
                ext,
                nonsing,
                gamma,
                beta,
                seed,
                states[0:1],
                dense,
                bp,
            )


# ---------------------------------------------------------------- aggregation
@_ijit
def _agg_row(c, g, refined, cptr, cnodes, t, tp):  # noqa: PLR0917
    """Sum the edges of refined community c by neighbouring community into t.

    Returns (keys listed, c's node weight), or -1 keys (t cleared) past tp's key limit.
    """
    ptr, idx, w, nw = g
    tkey, tval, tlst = t
    nl = 0
    s = 0.0
    for p in range(cptr[c], cptr[c + 1]):
        v = cnodes[p]
        s += nw[v]
        for e in range(ptr[v], ptr[v + 1]):
            d = refined[idx[e]]
            if d == c:
                continue
            q = _slot(tkey, d, tp[0], tp[1])
            if tkey[q] < 0:
                if nl == tp[2]:
                    _clear(t, nl)
                    return -1, s
                tkey[q] = d
                tlst[nl] = q
                nl += 1
            tval[q] += w[e]
    return nl, s


@_ijit
def _agg_fill(c, g, refined, cptr, cnodes, g2, t, tp):  # noqa: PLR0917
    """Write row c of the aggregated graph g2 = (ptr2, idx2, w2, nw2) at ptr2[c]."""
    ptr2, idx2, w2, nw2 = g2
    tkey, tval, tlst = t
    nl, s = _agg_row(c, g, refined, cptr, cnodes, t, tp)
    nw2[c] = s
    out = ptr2[c]
    for i in range(nl):
        q = tlst[i]
        idx2[out + i] = tkey[q]
        w2[out + i] = tval[q]
    _clear(t, nl)


def _aggregate(g, nr, g2, ws):
    """Collapse g by its nr refined communities (ws_i[12], grouped in ws_i[13:15]) into g2.

    Rows are interleaved over chunks, in two passes: row lengths into ptr2[1:] (then summed), then
    the rows. Rows with more entries than a chunk table holds are done afterwards with the dense
    table. Returns g2's largest degree.
    """
    ws_i = ws[2]
    tkey, tval, tlst, bkey, bval, blst = ws[6], ws[7], ws[8], ws[9], ws[10], ws[11]
    refined, cptr, cnodes = ws_i[12], ws_i[13], ws_i[14]
    ptr2 = g2[0]
    nch, slots = tkey.shape[0], tkey.shape[1]
    tp = _table(slots, slots, nr)
    dense = (bkey, bval, blst)
    bp = _table(bkey.shape[0], 1, bkey.shape[0])
    for ch in numba.prange(nch):
        t = (tkey[ch], tval[ch], tlst[ch])
        for c in range(ch, nr, nch):
            nl = _agg_row(c, g, refined, cptr, cnodes, t, tp)[0]
            _clear(t, nl)  # nothing if deferred (nl = -1)
            ptr2[c + 1] = nl
    ptr2[0] = 0
    deferred = False
    maxdeg = 0
    for r in range(nr):
        ln = ptr2[r + 1]
        if ln < 0:
            ln = _agg_row(r, g, refined, cptr, cnodes, dense, bp)[0]
            _clear(dense, ln)
            deferred = True
        maxdeg = max(maxdeg, ln)
        ptr2[r + 1] = ptr2[r] + ln
    lim = tp[2]
    for ch in numba.prange(nch):
        t = (tkey[ch], tval[ch], tlst[ch])
        for c in range(ch, nr, nch):
            if ptr2[c + 1] - ptr2[c] <= lim:
                _agg_fill(c, g, refined, cptr, cnodes, g2, t, tp)
    if deferred:  # exactly the rows deferred above
        for r in range(nr):
            if ptr2[r + 1] - ptr2[r] > lim:
                _agg_fill(r, g, refined, cptr, cnodes, g2, dense, bp)
    return maxdeg


# ---------------------------------------------------------------- one resolution
def _leiden_one(g0, resolution, res_bits, params, memb_out, capped, ws):  # noqa: PLR0917
    """One resolution on g0 = (indptr, indices, data), with params = (beta, n_iterations, max_iterations, seed).

    `res_bits` are the bits of `resolution`, so each resolution's random stream is keyed by its value.
    `ws` is a workspace (see the table at the top); aggregated graphs alternate between the two
    rows of idx_buf/w_buf (and ws_i[18:20], ws_f[5:7]).
    `capped[0]` is set if negative `n_iterations` stopped at `max_iterations` without converging.
    """
    indptr, indices, data = g0
    beta, n_iterations, max_iterations, seed = params
    idx_buf, w_buf, ws_i, ws_f, ws_u, tkey = ws[0], ws[1], ws[2], ws[3], ws[5], ws[6]
    n0 = indptr.shape[0] - 1
    agg, memb, refined, cptr, cnodes, tmp, prev = ws_i[10:17]
    iota = ws_i[20]
    init, nw0 = ws_f[3], ws_f[4]
    state = ws_u[0:1]
    slots = tkey.shape[1]
    base = np.uint64(seed) * np.uint64(0xD1B54A32D192ED03) ^ res_bits * np.uint64(
        0xABC98388FB8FAC03
    )
    two_m = 0.0
    maxdeg0 = 0
    for v in range(n0):
        s = 0.0
        for e in range(indptr[v], indptr[v + 1]):
            s += data[e]
        nw0[v] = s
        two_m += s
        maxdeg0 = max(maxdeg0, indptr[v + 1] - indptr[v])
        iota[v] = v
        memb_out[v] = v
    gamma = resolution / two_m
    max_it = n_iterations if n_iterations >= 0 else max_iterations
    capped[0] = n_iterations < 0  # cleared on convergence
    for it in range(max_it):
        for v in range(n0):  # memb_out is already numbered by first appearance
            agg[v] = v
            memb[v] = memb_out[v]
            prev[v] = memb_out[v]
        n = n0
        g = (indptr, indices, data, nw0)
        maxdeg = maxdeg0
        level = 0
        while True:
            lseed = _mix(base ^ (np.uint64(it) << np.uint64(32)) ^ np.uint64(level))
            state[0] = lseed ^ np.uint64(0x5851F42D4C957F2D)
            # keys of local moving and refinement: at most one per neighbour, plus the own community
            tp = _table(slots, maxdeg + 1, n)
            _move_nodes(g, n, gamma, state, ws, tp, maxdeg)
            k = _relabel(memb, n, tmp)
            if k >= n:
                break
            _group(memb, iota, n, k, cptr, cnodes)
            _refine(g, k, gamma, beta, lseed, ws, tp)
            nr = _relabel(refined, n, tmp)
            # refinement merged nothing: aggregate by the clustering itself
            if nr >= n:
                for v in range(n):
                    refined[v] = memb[v]
                nr = k
            for v in range(n0):
                agg[v] = refined[agg[v]]
            for v in range(n):
                init[refined[v]] = memb[v]
            _group(refined, iota, n, nr, cptr, cnodes)
            b = level & 1
            g2 = (ws_i[18 + b], idx_buf[b], w_buf[b], ws_f[5 + b])
            maxdeg = _aggregate(g, nr, g2, ws)
            for c in range(nr):
                memb[c] = np.int64(init[c])
            n = nr
            g = g2
            level += 1
        for v in range(n0):
            memb_out[v] = memb[agg[v]]
        _relabel(memb_out, n0, tmp)
        # negative n_iterations: until the partition stops changing
        if n_iterations < 0:
            same = True
            for v in range(n0):
                if memb_out[v] != prev[v]:
                    same = False
                    break
            if same:
                capped[0] = False
                break


# ---------------------------------------------------------------- compiled variants
_CHAIN = (
    "_propose",
    "_move_nodes",
    "_refine",
    "_aggregate",
    "_leiden_one",
)


def _variant(kind: str, *, parallel: bool) -> dict:
    """Compile the chain with callees bound to this variant; distinct qualnames keep numba's caches apart.

    fau's `njit` also makes a parallel and a serial copy, but both copies share the module's
    globals, so a jitted caller would see the module-level callees (fau's Python dispatchers,
    which jitted code can't call). Each variant here gets its own globals instead. Compilation
    is lazy and disk-cached; a cold cache compiles up to two chains (par and ser).
    """
    g = dict(globals())
    for name in _CHAIN:
        f = globals()[name]
        fn = types.FunctionType(
            f.__code__, g, f.__name__, f.__defaults__, f.__closure__
        )
        fn.__qualname__ = f"{f.__qualname__}-{kind}"
        g[name] = numba.njit(cache=True, nogil=True, parallel=parallel)(fn)  # noqa: TID251 (per-variant callees, see docstring)
    return g


_PAR = _variant("par", parallel=True)
_SER = _variant("ser", parallel=False)


# ---------------------------------------------------------------- scheduling and memory
def _serial_only() -> bool:
    """fast-array-utils' checks for where parallel numba kernels are unsafe (non-main thread, forks)."""
    from fast_array_utils import numba as fau_numba

    checks = (
        getattr(fau_numba, name, None)
        for name in ("_is_on_unsafe_thread", "_is_in_unsafe_fork")
    )
    return any(check() for check in checks if check is not None)


def _table_slots(n: int, maxdeg: int) -> int:
    """Slots per chunk table: room for twice the largest degree within [_TABLE_MIN, _TABLE_MAX], as a power of two.

    n + 1 (dense tables) when that isn't larger. Aggregated graphs can have larger degrees; their
    vertices, rows or clusters that don't fit use the dense table.
    """
    slots = min(max(1 << (2 * maxdeg + 1).bit_length(), _TABLE_MIN), _TABLE_MAX)
    return n + 1 if slots > n else slots


def _run(g, res, params, *, serial: bool) -> tuple[np.ndarray, np.ndarray]:
    """Run the resolutions one after another; returns the raw memberships and the capped flags.

    One workspace, with 2 chunk tables per thread of `_table_slots` slots each.
    """
    n, nr = g.shape[0], res.shape[0]
    chunks = 2 * numba.get_num_threads()
    slots = _table_slots(n, int(np.diff(g.indptr).max()))
    g0 = (
        g.indptr.astype(np.int64, copy=False),
        g.indices.astype(np.int32, copy=False),
        g.data,
    )
    shape = (chunks, slots)
    dense = n + 1 if slots <= n else 1  # unused with dense chunk tables
    ws = (
        np.empty((2, g.nnz), np.int32),
        np.empty((2, g.nnz), np.float32),
        np.empty((_NWS_I, n + 1), np.int64),
        np.empty((_NWS_F, n + 1), np.float64),
        np.zeros((_NWS_B, n + 1), np.bool),
        np.empty(chunks + 1, np.uint64),
        np.full(shape, -1, np.int32),  # tables: empty on entry and exit
        np.zeros(shape, np.float64),
        np.empty(shape, np.int32),
        np.full(dense, -1, np.int32),
        np.zeros(dense, np.float64),
        np.empty(dense, np.int32),
    )
    out = np.empty((nr, n), dtype=np.int64)
    capped = np.zeros(nr, dtype=np.bool)
    bits = res.view(np.uint64)
    one = (_SER if serial else _PAR)["_leiden_one"]
    for r in range(nr):
        one(g0, res[r], bits[r], params, out[r], capped[r : r + 1], ws)
    return out, capped


# ---------------------------------------------------------------- entry point (private)
def _modularities(g, memberships, resolutions) -> np.ndarray:
    """Igraph's (generalised) modularity of each membership on the symmetric weighted CSR `g`."""
    k = np.asarray(g.sum(axis=1)).ravel()
    two_m = k.sum()
    if two_m == 0:
        return np.full(len(memberships), np.nan)
    rows = np.repeat(np.arange(g.shape[0]), np.diff(g.indptr))
    out = np.empty(len(memberships))
    for i, (memb, res) in enumerate(zip(memberships, resolutions, strict=True)):
        inside = g.data[memb[rows] == memb[g.indices]].sum()
        kc = np.bincount(memb, weights=k)
        out[i] = (inside - res * (kc**2).sum() / two_m) / two_m
    return out


def leiden(
    graph,
    resolutions=(1.0,),
    *,
    n_iterations: int = 2,
    beta: float = 0.01,
    seed: int = 0,
):
    """Leiden partitions of a weighted graph, one per resolution.

    The graph is taken as undirected like igraph does: ``A[i, j]`` and ``A[j, i]`` both count, and
    self-loops count towards node strengths. Returns ``(membership, modularity)``: an int32 array
    (n_res, n_obs), clusters numbered by decreasing size (ties by first appearance), and a float64
    array (n_res,). Edge weights must be non-negative and resolutions non-negative and finite, as
    igraph's modularity requires; `beta` must be finite and positive. A graph without edges (or
    only zero weights) gives singletons and NaN modularity, as igraph does. Threads come from
    `numba.get_num_threads()`.
    """
    import scipy.sparse as sp

    res = np.atleast_1d(np.asarray(resolutions, dtype=np.float64))
    if not np.isfinite(res).all():
        msg = f"Resolutions must be finite, got {res.tolist()}."
        raise ValueError(msg)
    if (res < 0).any():
        msg = "The resolution parameter must not be negative."  # igraph's message
        raise ValueError(msg)
    if not (np.isfinite(beta) and beta > 0):
        msg = f"`beta` must be finite and positive, got {beta!r}."
        raise ValueError(msg)
    # not scanpy's CSR helpers: this takes any sparse or dense input and needs scipy's `+`/`.T`
    g = sp.csr_matrix(graph, dtype=np.float32)  # noqa: TID251 (generic input coercion, see above)
    # checked before symmetrising, where opposite signs could cancel
    if not np.isfinite(g.data).all():
        msg = "Edge weights must be finite."
        raise ValueError(msg)
    if (g.data < 0).any():
        msg = "Negative weight in the graph; flavor='numba' requires non-negative edge weights."
        raise ValueError(msg)
    g = (g + g.T).tocsr()
    g.eliminate_zeros()
    n, nr = g.shape[0], res.shape[0]
    if nr == 0:
        return np.empty((0, n), dtype=np.int32), np.empty(0)
    # no edges (with non-negative weights, the only way the total weight isn't positive)
    if g.nnz == 0:
        return np.tile(np.arange(n, dtype=np.int32), (nr, 1)), np.full(nr, np.nan)
    params = (float(beta), int(n_iterations), int(_MAX_ITERATIONS), int(seed))
    # the workspaces are freed when `_run` returns, before the modularities are computed
    out, capped = _run(g, res, params, serial=_serial_only())
    if capped.any():
        msg = (
            f"Leiden didn’t converge within {_MAX_ITERATIONS} iterations "
            f"at resolution(s) {res[capped].tolist()}; returning the last partition. "
            "Pass a positive `n_iterations` to set the number of iterations."
        )
        warn(msg, UserWarning)
    memb = np.empty((nr, n), dtype=np.int32)
    for r in range(nr):
        order = np.argsort(-np.bincount(out[r]), kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(order.shape[0])
        memb[r] = rank[out[r]]
    return memb, _modularities(g.astype(np.float64), memb, res)
