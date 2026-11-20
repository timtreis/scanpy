"""Parallel UMAP layout optimisation with numba (``sc.tl.umap(method="numba")``).

Same objective as umap-learn’s legacy optimiser (attraction along graph edges,
``negative_sample_rate`` repulsive samples per accepted edge, both endpoints moved,
learning rate decaying linearly), with two parallel schedules:

``deterministic=True`` (``sc.tl.umap`` with a seed)
    Edges are coloured so that no two edges of a colour share a vertex. Each colour
    is processed in parallel, so every vertex has a single writer, and negative
    samples are read from a snapshot taken at the start of the epoch. Random numbers
    come from a counter-based hash of (seed, epoch, edge), so the result is identical
    for any number of threads.
``deterministic=False`` (``sc.tl.umap`` with ``rng=None``)
    Hogwild: vertex blocks are processed in parallel without locks, like umap-learn’s
    ``parallel=True``. Fastest, but not reproducible.

Edges are accepted with probability ``w / w.max()`` each epoch, which samples them
as often as umap-learn’s ``epochs_per_sample`` schedule does on average.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np
from fast_array_utils.numba import njit
from llvmlite import ir
from numba.core import cgutils, types
from numba.extending import intrinsic
from scipy import sparse

from .._utils._numba import _mix

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .._compat import CSRBase

_inline = numba.njit(cache=True, inline="always")  # noqa: TID251 (fau's njit takes no `inline`; these helpers are inlined into the kernels)
# colours with fewer edges run serially: a parallel region costs more
_SMALL_COLOR = 256


@intrinsic
def _prefetch_row(typingctx, arr, row):
    """Hint the CPU to load ``arr[row, 0]`` into cache; the layout is bound by these random reads."""

    def codegen(context, builder, signature, args):
        aryty, _ = signature.args
        ary, r = args
        array = context.make_array(aryty)(context, builder, ary)
        zero = context.get_constant(types.intp, 0)
        ptr = cgutils.get_item_pointer(
            context, builder, aryty, array, [r, zero], wraparound=False
        )
        i8p = ir.IntType(8).as_pointer()
        i32 = ir.IntType(32)
        fnty = ir.FunctionType(ir.VoidType(), [i8p, i32, i32, i32])
        fn = builder.module.declare_intrinsic("llvm.prefetch", [i8p], fnty)
        builder.call(fn, [builder.bitcast(ptr, i8p), i32(0), i32(3), i32(1)])
        return context.get_dummy_value()

    return types.void(arr, row), codegen


# splitmix64 increment (golden ratio) and xorshift64* output multiplier
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_XS_MUL = np.uint64(0x2545F4914F6CDD1D)


@_inline
def _hash(seed, epoch, edge):
    z = _mix(seed * _GOLDEN ^ (np.uint64(epoch) * np.uint64(0xD1B54A32D192ED03)))
    return _mix(z ^ (np.uint64(edge) * np.uint64(0xABC98388FB8FAC03)))


@_inline
def _xorshift(state):
    state ^= state >> np.uint64(12)
    state ^= state << np.uint64(25)
    state ^= state >> np.uint64(27)
    return state


@_inline
def _randint(state, n):
    """Map a xorshift state to 0..n-1 (multiply-shift)."""
    return np.int64(
        ((state * _XS_MUL) >> np.uint64(32)) * np.uint64(n) >> np.uint64(32)
    )


@_inline
def _unit(x):
    """Top 24 bits of a uint64 as a float32 in [0, 1)."""
    return np.float32(x >> np.uint64(40)) * np.float32(1.0 / (1 << 24))


@_inline
def _clip(v):
    return min(np.float32(4.0), max(np.float32(-4.0), v))


@_inline
def _attract(dx, dy, d2, a, b, att_c, alpha):  # noqa: PLR0917
    """Clipped attractive step along an edge with offset (dx, dy) and d2 > 0."""
    db = d2 ** (b - np.float32(1.0))
    s = att_c * db / (np.float32(1.0) + a * db * d2)
    return alpha * _clip(s * dx), alpha * _clip(s * dy)


@_inline
def _repel(dx, dy, d2, a, b, rep_c, alpha):  # noqa: PLR0917
    """Clipped repulsive step from a negative sample with offset (dx, dy) and d2 > 0."""
    s = rep_c / ((np.float32(0.001) + d2) * (np.float32(1.0) + a * d2**b))
    return alpha * _clip(s * dx), alpha * _clip(s * dy)


# ---------------------------------------------------------------- deterministic: edge-coloured SGD
@njit
def _edge_coloring(heads, tails, deg):
    """Greedy colouring so that no two edges of a colour share an endpoint.

    Edges are coloured in the given order, each with the lowest colour free at both
    endpoints; ``deg`` is the degree of each vertex.
    Returns the edges grouped by colour and the colour offsets.
    """
    n, m = deg.shape[0], heads.shape[0]
    # Edge (u, v) gets a colour below deg[u] + deg[v] - 1, since the other edges at u and v
    # use at most that many. So each vertex keeps a bitset of used colours with
    # `words[v] * 64 > deg[u] + deg[v]` for all its edges (u, v), which bounds every colour
    # set there, and the search below finds a free bit within both endpoints’ words.
    words = np.ones(n, dtype=np.int64)
    for e in range(m):
        u, v = heads[e], tails[e]
        w = (deg[u] + deg[v]) // 64 + 1
        words[u] = max(words[u], w)
        words[v] = max(words[v], w)
    start = np.zeros(n + 1, dtype=np.int64)
    start[1:] = np.cumsum(words)
    used = np.zeros(start[n], dtype=np.uint64)
    color = np.empty(m, dtype=np.int64)
    n_colors = 0
    for e in range(m):
        bu, bv = start[heads[e]], start[tails[e]]
        # lowest colour free at both endpoints: skip full words, then the bits of the first free one
        wd = 0
        free = ~(used[bu] | used[bv])
        while free == 0:
            wd += 1
            free = ~(used[bu + wd] | used[bv + wd])
        t = 0
        while not (free >> np.uint64(t)) & np.uint64(1):
            t += 1
        c = wd * 64 + t
        used[bu + wd] |= np.uint64(1) << np.uint64(t)
        used[bv + wd] |= np.uint64(1) << np.uint64(t)
        color[e] = c
        n_colors = max(n_colors, c + 1)
    offsets = np.zeros(n_colors + 1, dtype=np.int64)
    for e in range(m):
        offsets[color[e] + 1] += 1
    offsets = np.cumsum(offsets)
    fill = offsets[:-1].copy()
    order = np.empty(m, dtype=np.int64)
    for e in range(m):
        order[fill[color[e]]] = e
        fill[color[e]] += 1
    return order, offsets


@_inline
def _edge_event(pos, snap, u, v, eid, prob, p):  # noqa: PLR0917
    a, b, att_c, rep_c, alpha, neg_rate, seed, epoch = p
    n = pos.shape[0]
    h = _hash(seed, epoch, eid)
    if _unit(h) >= prob:
        return
    ux = pos[u, 0]
    uy = pos[u, 1]
    dx = ux - pos[v, 0]
    dy = uy - pos[v, 1]
    d2 = dx * dx + dy * dy
    if d2 > 0:
        gx, gy = _attract(dx, dy, d2, a, b, att_c, alpha)
        ux += gx
        uy += gy
        pos[v, 0] -= gx
        pos[v, 1] -= gy
    z = h
    for _ in range(neg_rate):
        z += _GOLDEN
        w = np.int64(_mix(z) % np.uint64(n))
        if w == u:
            continue
        dx = ux - snap[w, 0]
        dy = uy - snap[w, 1]
        d2 = dx * dx + dy * dy
        if d2 > 0:
            rx, ry = _repel(dx, dy, d2, a, b, rep_c, alpha)
            ux += rx
            uy += ry
    pos[u, 0] = ux
    pos[u, 1] = uy


@_inline
def _edge_pair(pos, snap, r, us, vs, eids, prob, p):  # noqa: PLR0917
    """Both directions of edge r (in colour order); `eids` are the original edge ids the hash uses.

    `p` is ``(a, b, att_c, rep_c, alpha, neg_rate, seed, epoch)``, a flat tuple:
    numba's parallel lowering rejects nested ones.
    """
    u, v, e = us[r], vs[r], eids[r]
    _edge_event(pos, snap, u, v, np.uint64(2 * e), prob[r], p)
    _edge_event(pos, snap, v, u, np.uint64(2 * e + 1), prob[r], p)


@njit
def _colored_epochs(  # noqa: PLR0913, PLR0917
    pos, offsets, us, vs, eids, prob, a, b, gamma, alpha, n_epochs, neg_rate, seed
):
    """All epochs; edges of a colour share no vertex, so each colour runs in parallel (small ones serially)."""
    att_c = np.float32(-2.0) * a * b
    rep_c = np.float32(2.0) * gamma * b
    for ep in range(n_epochs):
        al = np.float32(alpha * (1.0 - ep / n_epochs))
        p = (a, b, att_c, rep_c, al, neg_rate, seed, ep)
        snap = pos.copy()
        for c in range(offsets.shape[0] - 1):
            lo, hi = offsets[c], offsets[c + 1]
            if hi - lo < _SMALL_COLOR:
                for r in range(lo, hi):
                    _edge_pair(pos, snap, r, us, vs, eids, prob, p)
            else:
                for r in numba.prange(lo, hi):
                    _edge_pair(pos, snap, r, us, vs, eids, prob, p)


# ---------------------------------------------------------------- non-deterministic: Hogwild vertex blocks
@njit
def _hogwild_epoch(  # noqa: PLR0913, PLR0917
    pos, indptr, indices, prob, a, b, gamma, alpha, neg_rate, seed, epoch, block, queue
):
    n = pos.shape[0]
    m = indices.shape[0]
    att_c = np.float32(-2.0) * a * b
    rep_c = np.float32(2.0) * gamma * b
    for blk in numba.prange(queue.shape[0]):
        state = _hash(seed, epoch, np.uint64(blk)) | np.uint64(1)
        q = queue[blk]
        # negatives are drawn one edge ahead and prefetched, which breaks the dependency on memory latency
        for k in range(neg_rate):
            state = _xorshift(state)
            q[k] = _randint(state, n)
        for u in range(blk * block, min(n, (blk + 1) * block)):
            ux = pos[u, 0]
            uy = pos[u, 1]
            for e in range(indptr[u], indptr[u + 1]):
                if e + 8 < m:
                    _prefetch_row(pos, indices[e + 8])
                state = _xorshift(state)
                if _unit(state * _XS_MUL) >= prob[e]:
                    continue
                v = indices[e]
                vx = pos[v, 0]
                vy = pos[v, 1]
                dx = ux - vx
                dy = uy - vy
                d2 = dx * dx + dy * dy
                if d2 > 0:
                    gx, gy = _attract(dx, dy, d2, a, b, att_c, alpha)
                    ux += gx
                    uy += gy
                    pos[v, 0] = vx - gx
                    pos[v, 1] = vy - gy
                # repulsion from all negatives is computed from the same position and applied jointly
                sx = np.float32(0.0)
                sy = np.float32(0.0)
                for k in range(neg_rate):
                    w = q[k]
                    state = _xorshift(state)
                    q[k] = _randint(state, n)
                    _prefetch_row(pos, q[k])
                    if w == u:
                        continue
                    dx = ux - pos[w, 0]
                    dy = uy - pos[w, 1]
                    d2 = dx * dx + dy * dy
                    if d2 > 0:
                        rx, ry = _repel(dx, dy, d2, a, b, rep_c, alpha)
                        sx += rx
                        sy += ry
                ux += sx
                uy += sy
            pos[u, 0] = ux
            pos[u, 1] = uy


def default_epochs(n_obs: int) -> int:
    """umap-learn’s default number of epochs."""
    return 500 if n_obs <= 10000 else 200


def optimize_layout(
    init: NDArray[np.floating],
    graph: CSRBase,
    *,
    n_epochs: int,
    a: float,
    b: float,
    gamma: float = 1.0,
    alpha: float = 1.0,
    negative_sample_rate: int = 5,
    seed: int = 0,
    deterministic: bool = True,
) -> NDArray[np.float32]:
    """Optimise a 2-D UMAP layout of the symmetric fuzzy ``graph`` starting from ``init``."""
    # 2-D only: the kernels unroll x/y
    if init.shape[1] != 2:
        msg = "method='numba' currently supports n_components=2 only."
        raise NotImplementedError(msg)
    pos = np.array(init, dtype=np.float32, order="C")
    g = sparse.csr_matrix(graph, dtype=np.float32, copy=True)  # noqa: TID251 (a private float32 copy we modify, whatever the input class)
    g.eliminate_zeros()  # explicitly stored zeros are no edges
    if n_epochs <= 0 or g.nnz == 0:
        return pos
    # as umap-learn: drop edges sampled less than once; few epochs prune by the default count
    cut = n_epochs if n_epochs > 10 else default_epochs(g.shape[0])
    g.data[g.data < g.data.max() / cut] = 0
    g.eliminate_zeros()
    g.sort_indices()
    g.data /= g.data.max()  # per-epoch acceptance probability of each edge
    a32, b32, g32 = np.float32(a), np.float32(b), np.float32(gamma)
    seed64 = np.uint64(seed)
    n = pos.shape[0]
    if deterministic:
        up = sparse.triu(g, k=1).tocoo()
        us, vs, prob = up.row.astype(np.int64), up.col.astype(np.int64), up.data
        deg = np.bincount(us, minlength=n) + np.bincount(vs, minlength=n)
        # colour edges at high-degree vertices first: fewer colours
        first = np.argsort(-np.maximum(deg[us], deg[vs]), kind="stable")
        order, offsets = _edge_coloring(us[first], vs[first], deg)
        order = first[order]
        _colored_epochs(
            pos,
            offsets,
            us[order],
            vs[order],
            order,
            prob[order],
            a32,
            b32,
            g32,
            float(alpha),
            n_epochs,
            negative_sample_rate,
            seed64,
        )
    else:
        block = 256
        indptr, indices = g.indptr.astype(np.int64), g.indices.astype(np.int64)
        queue = np.empty(
            ((n + block - 1) // block, negative_sample_rate), dtype=np.int64
        )
        for ep in range(n_epochs):
            al = np.float32(alpha * (1.0 - ep / n_epochs))
            _hogwild_epoch(
                pos,
                indptr,
                indices,
                g.data,
                a32,
                b32,
                g32,
                al,
                negative_sample_rate,
                seed64,
                ep,
                block,
                queue,
            )
    return pos
