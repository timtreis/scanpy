from __future__ import annotations

import math
from contextlib import contextmanager
from typing import TYPE_CHECKING, NamedTuple

import numba
import numpy as np
from fast_array_utils.numba import njit

from .._compat import CSBase, CSRBase, warn

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import TypeIs

    from numpy.typing import NDArray

    from .._compat import CSCBase


# Floating-point reductions whose partial sums are added in a fixed order split their input into
# this many chunks (see `_chunk_bounds`), independent of the thread count, so the result is too.
# This also caps such a reduction at `N_REDUCE_CHUNKS` threads.
N_REDUCE_CHUNKS = 32

# The smallest chunk of a boolean scan (see `_scan_chunks`).
_MIN_SCAN_CHUNK = 4096

FLOAT_DTYPES = frozenset({np.dtype(np.float32), np.dtype(np.float64)})
"""Float dtypes the numba kernels handle (not float16 or long double)."""


def _chunk_bounds(n: int, n_chunks: int) -> NDArray[np.int64]:
    """Bounds of contiguous, near-equal chunks of ``range(n)``: chunk `c` is ``bounds[c]:bounds[c + 1]``.

    Chunk `c` starts at ``c * n // n_chunks``. There are ``min(n_chunks, n)`` chunks, so none is
    empty (but at least one, for ``n == 0``).
    """
    n_chunks = max(1, min(n_chunks, n))
    return np.arange(n_chunks + 1, dtype=np.int64) * n // n_chunks


def _nnz_row_blocks(indptr: NDArray[np.integer], n_blocks: int) -> NDArray[np.int64]:
    """Bounds of up to `n_blocks` contiguous row blocks with about equal numbers of stored entries.

    Only for kernels whose result per row doesn't depend on the blocks, which may then depend on
    the thread count. Equal row counts would leave threads idle when row lengths vary.
    """
    n_rows = indptr.shape[0] - 1
    bounds = np.searchsorted(indptr, np.linspace(0, indptr[-1], max(1, n_blocks) + 1))
    bounds[0], bounds[-1] = 0, n_rows
    return np.unique(np.clip(bounds, 0, n_rows)).astype(np.int64)


def _is_numba_dtype(dtype: np.dtype) -> bool:
    """Whether the numba kernels in Scanpy handle `dtype` (not float16, long double or complex)."""
    return dtype.kind in "biu" or dtype in FLOAT_DTYPES


def _is_numba_csr(x: object) -> TypeIs[CSRBase]:
    """Whether `x` is a CSR matrix or array with a dtype the numba kernels handle."""
    return isinstance(x, CSRBase) and _is_numba_dtype(x.dtype)


class CompressedArrays(NamedTuple):
    """The arrays of a CSR/CSC matrix, for numba kernels.

    numba can only disk-cache functions whose argument types it can pickle, which
    scipy matrices aren't. Same attribute names, so kernel bodies needn't change.
    """

    indptr: NDArray[np.integer]
    indices: NDArray[np.integer]
    data: NDArray[np.number]
    shape: NDArray[
        np.int64
    ]  # an array: numba's parallel lowering rejects nested tuples


def arrays(x: CSRBase | CSCBase) -> CompressedArrays:
    return CompressedArrays(
        x.indptr, x.indices, x.data, np.array(x.shape, dtype=np.int64)
    )


def _resolve_n_threads(n_threads: int) -> int:
    n_max = numba.config.NUMBA_NUM_THREADS
    if n_threads < -1:
        msg = f"n_threads < -1 is not supported for this operation; treating {n_threads} as -1 ({n_max} threads, as per numba.config.NUMBA_NUM_THREADS)."
        warn(msg, UserWarning)
    if n_threads < 0:
        return n_max
    return max(1, min(n_threads, n_max))


@contextmanager
def _numba_thread_limit(n_threads: int | None) -> Generator[int | None]:
    """Temporarily set Numba's thread count and restore it on exit.

    Resolves ``n_threads`` (``-1`` selects ``numba.config.NUMBA_NUM_THREADS``)
    and yields the resolved count. ``None`` leaves Numba unchanged and yields ``None``.
    """
    if n_threads is None:
        yield None
        return

    resolved = _resolve_n_threads(n_threads)
    previous = numba.get_num_threads()
    numba.set_num_threads(resolved)
    try:
        yield resolved
    finally:
        numba.set_num_threads(previous)


def _scan_chunks(n: int) -> int:
    """Chunk count for boolean scans that stop at the first hit, whose result doesn't depend on it.

    Depends on the thread count: many more chunks than threads, so a chunk that exits early frees
    its thread. Never use it for floating-point reductions (see `N_REDUCE_CHUNKS`).
    """
    return max(1, min(n // _MIN_SCAN_CHUNK, 64 * numba.get_num_threads()))


@njit
def _all_nonnegative_integers(data: NDArray[np.floating], n_chunks: int) -> bool:
    """Whether all values are finite, non-negative (no ``-0.0``) and integral."""
    n = data.shape[0]
    bad = np.zeros(n_chunks, dtype=np.bool)
    for c in numba.prange(n_chunks):
        for i in range(c * n // n_chunks, (c + 1) * n // n_chunks):
            v = data[i]
            if not math.isfinite(v) or math.copysign(1.0, v) < 0 or v != math.floor(v):
                bad[c] = True
                break
    return not bad.any()


@njit
def _any_zero(data: NDArray[np.number], n_chunks: int) -> bool:
    n = data.shape[0]
    found = np.zeros(n_chunks, dtype=np.bool)
    for c in numba.prange(n_chunks):
        for i in range(c * n // n_chunks, (c + 1) * n // n_chunks):
            if data[i] == 0:
                found[c] = True
                break
    return found.any()


def eliminate_zeros(x: object) -> None:
    """`x.eliminate_zeros()` for a sparse `x` (other arrays are left alone).

    Skips scipy's serial compaction pass if there are no explicit zeros.
    """
    if not isinstance(x, CSBase):
        return
    if not _is_numba_dtype(x.data.dtype) or _any_zero(
        x.data, _scan_chunks(x.data.shape[0])
    ):
        x.eliminate_zeros()


@njit
def _is_canonical(
    indptr: NDArray[np.integer], indices: NDArray[np.integer], n_chunks: int
) -> bool:
    n = indptr.shape[0] - 1
    ok = np.ones(n_chunks, dtype=np.bool)
    for c in numba.prange(n_chunks):
        for i in range(c * n // n_chunks, (c + 1) * n // n_chunks):
            for k in range(indptr[i] + 1, indptr[i + 1]):
                if indices[k] <= indices[k - 1]:
                    ok[c] = False
                    break
            if not ok[c]:
                break
    return ok.all()


def has_canonical_format(x: CSBase) -> bool:
    """`x.has_canonical_format` (sorted indices, no duplicates), computed in parallel and cached on `x`.

    scipy's check is a serial pass over all indices the first time it's asked.
    """
    if not getattr(x, "_has_sorted_indices", True):
        # as scipy: indices known to be unsorted mean not canonical, whatever was cached
        x.has_canonical_format = False
        return False
    if (flag := getattr(x, "_has_canonical_format", None)) is None:
        flag = bool(
            _is_canonical(x.indptr, x.indices, _scan_chunks(x.indptr.shape[0] - 1))
        )
        x.has_canonical_format = flag
    return flag


# numpy's pairwise summation, which `np.add.reduce` and `np.add.reduceat` (so scipy's sparse row
# sums) use for float32 and float64: `@TYPE@_pairwise_sum` in numpy/_core/src/umath/loops_utils.h.src,
# called by the float `add` loop when it reduces (`*io1 += pairwise_sum(rest)`). Up to 8 values
# are added in order; up to 128 into 8 interleaved accumulators that are then combined as a tree,
# plus the remainder in order; above that, the sum of the two halves, split at a multiple of 8.
# Every addition is in the input's dtype, and numba neither reassociates nor vectorises across
# the accumulators without fastmath, so the replica gives the same bits.
_PW_BLOCKSIZE = 128
# deep enough for any `n` below 2**63: each level at least halves it
_PW_STACK = 64


@numba.njit(cache=True)  # noqa: TID251 (a helper called from fast-array-utils kernels)
def _pairwise_leaf(a: NDArray[np.floating], s: int, n: int) -> np.floating:
    """numpy's pairwise sum of ``a[s:s + n]`` for ``1 <= n <= 128``."""
    if n < 8:
        # numpy starts at -0.0 here, and -0.0 + v is v for every v
        res = a[s]
        for i in range(1, n):
            res += a[s + i]
        return res
    r0, r1, r2, r3 = a[s], a[s + 1], a[s + 2], a[s + 3]
    r4, r5, r6, r7 = a[s + 4], a[s + 5], a[s + 6], a[s + 7]
    end8 = n - n % 8
    for i in range(s + 8, s + end8, 8):
        r0 += a[i]
        r1 += a[i + 1]
        r2 += a[i + 2]
        r3 += a[i + 3]
        r4 += a[i + 4]
        r5 += a[i + 5]
        r6 += a[i + 6]
        r7 += a[i + 7]
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    for i in range(s + end8, s + n):
        res += a[i]
    return res


@numba.njit(cache=True)  # noqa: TID251 (a helper called from fast-array-utils kernels)
def _pairwise_sum(
    a: NDArray[np.floating],
    s: int,
    n: int,
    frames: NDArray[np.int64],
    lefts: NDArray[np.floating],
) -> np.floating:
    """numpy's pairwise sum of ``a[s:s + n]`` (``n >= 1``), its recursion unrolled.

    `frames` (``(_PW_STACK, 3)`` int64: start, length, whether the left half is done) and `lefts`
    (``_PW_STACK`` values of `a`'s dtype: the left halves' sums) are scratch.
    """
    if n <= _PW_BLOCKSIZE:
        return _pairwise_leaf(a, s, n)
    top = 0
    frames[0, 0], frames[0, 1], frames[0, 2] = s, n, 0
    while True:
        start, length = frames[top, 0], frames[top, 1]
        if length > _PW_BLOCKSIZE:  # descend into the left half
            half = length // 2
            top += 1
            frames[top, 0], frames[top, 1], frames[top, 2] = start, half - half % 8, 0
            continue
        v = _pairwise_leaf(a, start, length)
        while True:  # hand `v` to the parent frames
            top -= 1
            if frames[top, 2] == 0:  # `v` is the left half: keep it, then descend into the right
                lefts[top] = v
                frames[top, 2] = 1
                half = frames[top, 1] // 2
                half -= half % 8
                top += 1
                frames[top, 0] = frames[top - 1, 0] + half
                frames[top, 1] = frames[top - 1, 1] - half
                frames[top, 2] = 0
                break
            v = lefts[top] + v
            if top == 0:
                return v


@numba.njit(cache=True)  # noqa: TID251 (a helper called from fast-array-utils kernels)
def _reduceat_add(
    a: NDArray[np.floating],
    s: int,
    e: int,
    frames: NDArray[np.int64],
    lefts: NDArray[np.floating],
) -> np.floating:
    """``np.add.reduceat``'s value for the segment ``a[s:e]`` (``e > s``).

    numpy's reduceat copies the segment's first value and reduces the rest into it
    (`PyUFunc_Reduceat` in numpy/_core/src/umath/ufunc_object.c), i.e. ``a[s] + pairwise_sum(a[s + 1:e])``.
    """
    if e - s == 1:
        return a[s]
    return a[s] + _pairwise_sum(a, s + 1, e - s - 1, frames, lefts)
