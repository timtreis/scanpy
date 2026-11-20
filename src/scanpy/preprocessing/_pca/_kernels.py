from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING

import numba
import numpy as np
import scipy.linalg
from fast_array_utils.numba import njit as fau_njit
from numba import njit  # noqa: TID251
from sklearn.utils.extmath import svd_flip
from threadpoolctl import threadpool_limits

from ..._utils._numba import _chunk_bounds

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from scanpy._compat import CSRBase


@njit(nogil=True)
def _csr_gram_upper_triangular(
    mat: CSRBase,
    out: NDArray,
) -> None:
    """Accumulate upper triangle of ``X.T @ X`` into ``out``, then write over the lower."""
    n_rows = mat.indptr.shape[0] - 1
    for i in range(n_rows):
        row_start = mat.indptr[i]
        row_end = mat.indptr[i + 1]
        for k in range(row_start, row_end):
            jk = mat.indices[k]
            vk = mat.data[k]
            for l in range(k, row_end):
                out[jk, mat.indices[l]] += vk * mat.data[l]
    for i in range(out.shape[0]):
        for j in range(i):
            out[i, j] = out[j, i]


@njit(nogil=True)
def _csr_gram_full(
    mat: CSRBase,
    out: NDArray,
) -> None:
    """Accumulate ``X.T @ X`` into ``out``."""
    n_rows = mat.indptr.shape[0] - 1
    for i in range(n_rows):
        row_start = mat.indptr[i]
        row_end = mat.indptr[i + 1]
        for k in range(row_start, row_end):
            jk = mat.indices[k]
            vk = mat.data[k]
            for l in range(row_start, row_end):
                out[jk, mat.indices[l]] += vk * mat.data[l]


def csr_gram_dense(mat: CSRBase) -> NDArray:
    """Compute the full dense gram matrix ``X.T @ X`` for a CSR ``X``."""
    out = np.zeros((mat.shape[1], mat.shape[1]), dtype=mat.dtype)
    if mat.has_sorted_indices:
        _csr_gram_upper_triangular(mat, out)
    else:
        _csr_gram_full(mat, out)
    return out


@fau_njit
def _csr_dense_matmul(  # noqa: PLR0917
    indptr: NDArray[np.integer],
    indices: NDArray[np.integer],
    data: NDArray[np.number],
    slot: NDArray[np.intp],
    b: NDArray[np.floating],
    out: NDArray[np.floating],
) -> None:
    """``out = X[:, cols] @ b``, reading column ``j`` of the full CSR ``X`` as row ``slot[j]`` of ``b`` (skip if -1).

    Per row, the same operations as scipy's ``csr_matvecs`` on ``X[:, cols]``:
    ``out[i] += x_ij * b[slot[j]]`` over the row's entries in stored order, in the dtype of ``data * b``.
    """
    for i in numba.prange(indptr.shape[0] - 1):
        row = out[i]
        row[:] = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            j = slot[indices[k]]
            if j < 0:
                continue
            v = data[k]
            for c in range(b.shape[1]):
                row[c] += v * b[j, c]


def _slot_map(n_cols: int, cols: NDArray[np.intp] | None) -> NDArray[np.intp]:
    """Position of each column within `cols` (-1 if not selected)."""
    if cols is None:
        return np.arange(n_cols, dtype=np.intp)
    slot = np.full(n_cols, -1, dtype=np.intp)
    slot[cols] = np.arange(len(cols), dtype=np.intp)
    return slot


@fau_njit
def _mirror_upper(a: NDArray[np.floating]) -> None:
    """Copy the upper triangle of square `a` into its lower one, in place."""
    for j in numba.prange(a.shape[1]):
        for i in range(j + 1, a.shape[0]):
            a[i, j] = a[j, i]


# Reference path (`settings.reference_numerics`): sklearn's arithmetic, in parallel.
#
# sklearn's `PCA(svd_solver="covariance_eigh")` on CSR `X` (`sklearn/decomposition/_pca.py`, `_fit_full`)
# computes `mean_ = X.mean(axis=0)`, `C = X.T @ X` with scipy, `C -= n * mean_[:, None] * mean_[None, :]`,
# `C /= n - 1`, `numpy.linalg.eigh(C)`, and later `X @ components_.T - mean_ @ components_.T`.
# The sparse steps are serial scipy loops, all in the input dtype:
# - `X.T @ X` converts `X` to CSC and runs `csr_matmat`: entry `(i, l)` starts at +0.0 and adds
#   `x[k, i] * x[k, l]` for every cell `k` that stores both, in increasing `k`.
# - `X.mean(axis=0)` multiplies the data by `1.0 / n` (cast to the dtype), then `csc_matvec` adds the
#   scaled values to each column's +0.0 in increasing `k`.
# - `X @ B` (`csr_matvecs`) adds `x[k, j] * B[j]` to row `k` of +0.0s, in stored order.
# The fixed order of each entry lets the kernels below split the work over output entries instead of
# cells and still give the same bits. numba rounds each product and sum separately (no fastmath), like
# scipy's x86-64 builds. `_replays_sklearn` checks that once per dtype.

# Selected entries copied per row block, so that the gram kernel skips no unselected columns.
_REF_BLOCK_ENTRIES = 1 << 22


@fau_njit
def _scan_selected(  # noqa: PLR0917
    indptr: NDArray[np.integer],
    indices: NDArray[np.integer],
    data: NDArray[np.floating],
    slot: NDArray[np.intp],
    chunks: NDArray[np.int64],
    counts: NDArray[np.int64],
    cost: NDArray[np.int64],
    ok: NDArray[np.bool],
) -> None:
    """Per row, count the selected entries into `counts`.

    Per row chunk ``c``, add each column's share of the gram work to ``cost[c]`` (the entries from it to
    the row's end) and set ``ok[c]`` if all selected entries are finite and in strictly increasing column order.
    """
    for c in numba.prange(chunks.shape[0] - 1):
        part = cost[c]
        good = True
        for r in range(chunks[c], chunks[c + 1]):
            m = 0
            prev = -1
            for p in range(indptr[r], indptr[r + 1]):
                s = slot[indices[p]]
                if s >= 0:
                    good = good and s > prev and np.isfinite(data[p])
                    prev = s
                    m += 1
            counts[r] = m
            for p in range(indptr[r], indptr[r + 1]):
                s = slot[indices[p]]
                if s >= 0:
                    part[s] += m
                    m -= 1
        ok[c] = good


@fau_njit
def _gather_selected(  # noqa: PLR0917
    indptr: NDArray[np.integer],
    indices: NDArray[np.integer],
    data: NDArray[np.floating],
    slot: NDArray[np.intp],
    lo: int,
    out_ptr: NDArray[np.int64],
    out_slot: NDArray[np.int32],
    out_data: NDArray[np.floating],
) -> None:
    """Copy the selected entries of rows ``lo:lo + len(out_ptr) - 1`` as a CSR of slots, in stored order."""
    for r in numba.prange(out_ptr.shape[0] - 1):
        q = out_ptr[r]
        for p in range(indptr[lo + r], indptr[lo + r + 1]):
            s = slot[indices[p]]
            if s >= 0:
                out_slot[q] = s
                out_data[q] = data[p]
                q += 1


@fau_njit
def _gram_upper_tiles(  # noqa: PLR0917
    indptr: NDArray[np.int64],
    slots: NDArray[np.int32],
    data: NDArray[np.floating],
    tiles: NDArray[np.int64],
    scale: np.floating,
    gram: NDArray[np.floating],
    col_sum: NDArray[np.floating],
) -> None:
    """Add a row block to the upper triangle of ``X.T @ X`` and to ``(X * scale).sum(axis=0)``, as scipy does.

    Tile ``t`` owns gram rows and columns ``tiles[t]:tiles[t + 1]``. It walks the block's rows in order,
    so each entry gets scipy's additions in scipy's order. Needs strictly increasing slots in each row.
    """
    for t in numba.prange(tiles.shape[0] - 1):
        lo, hi = tiles[t], tiles[t + 1]
        for r in range(indptr.shape[0] - 1):
            end = indptr[r + 1]
            p = indptr[r] + np.searchsorted(slots[indptr[r] : end], lo)
            while p < end and slots[p] < hi:
                i = slots[p]
                v = data[p]
                col_sum[i] += v * scale
                row = gram[i]
                for q in range(p, end):
                    row[slots[q]] += v * data[q]
                p += 1


def _tile_bounds(cost: NDArray[np.int64], n_tiles: int) -> NDArray[np.int64]:
    """Contiguous ranges of columns with about equal total `cost`: tile `t` is ``bounds[t]:bounds[t + 1]``."""
    cum = np.cumsum(cost)
    targets = cum[-1] * np.arange(1, n_tiles, dtype=np.int64) // n_tiles
    cuts = np.searchsorted(cum, targets, side="right")
    return np.unique(np.concatenate(([0], cuts, [len(cost)]))).astype(np.int64)


def _gram_mean_replay(
    x: CSRBase, slot: NDArray[np.intp], n_sel: int
) -> tuple[NDArray[np.floating], NDArray[np.floating]] | None:
    """Scipy's ``(x_sel.T @ x_sel).toarray()`` and ``x_sel.mean(axis=0)`` for ``x_sel = x[:, cols]``, bit for bit.

    `None` if a selected entry isn't finite (sklearn raises) or a row's selected columns aren't strictly
    increasing (duplicates are summed in an order this doesn't replay).
    """
    n = x.shape[0]
    n_threads = numba.get_num_threads()
    chunks = _chunk_bounds(n, n_threads)
    ptr = np.zeros(n + 1, dtype=np.int64)
    cost = np.zeros((len(chunks) - 1, n_sel), dtype=np.int64)
    ok = np.zeros(len(chunks) - 1, dtype=np.bool)
    _scan_selected(x.indptr, x.indices, x.data, slot, chunks, ptr[1:], cost, ok)
    if not ok.all():
        return None
    max_row = int(ptr.max())  # `ptr[1:]` holds the row counts here
    np.cumsum(ptr, out=ptr)
    size = max(max_row, min(_REF_BLOCK_ENTRIES, int(ptr[-1])))
    tiles = _tile_bounds(cost.sum(axis=0), n_threads)
    del cost
    gram = np.zeros((n_sel, n_sel), dtype=x.dtype)
    col_sum = np.zeros(n_sel, dtype=x.dtype)
    scale = x.dtype.type(
        1.0 / n
    )  # scipy's `mean` scales the data by `1.0 / n` in its dtype
    b_slot = np.empty(size, dtype=np.int32)
    b_data = np.empty(size, dtype=x.dtype)
    lo = 0
    while lo < n:
        hi = max(lo + 1, int(np.searchsorted(ptr, ptr[lo] + size, side="right")) - 1)
        b_ptr = ptr[lo : hi + 1] - ptr[lo]
        _gather_selected(x.indptr, x.indices, x.data, slot, lo, b_ptr, b_slot, b_data)
        _gram_upper_tiles(b_ptr, b_slot, b_data, tiles, scale, gram, col_sum)
        lo = hi
    _mirror_upper(gram)  # products commute exactly, so scipy's gram is symmetric
    return gram, col_sum


def _covariance_eigh_replay(
    x: CSRBase, cols: NDArray[np.intp] | None, k: int
) -> tuple[NDArray[np.floating], ...] | None:
    """Sklearn's ``PCA(k, svd_solver="covariance_eigh").fit_transform(x[:, cols])``, bit for bit.

    Returns the projection, ``components_``, ``explained_variance_``, ``explained_variance_ratio_``
    and ``mean_``, or `None` where :func:`_gram_mean_replay` doesn't apply.
    """
    n = x.shape[0]
    slot = _slot_map(x.shape[1], cols)
    n_sel = len(slot) if cols is None else len(cols)
    if (res := _gram_mean_replay(x, slot, n_sel)) is None:
        return None
    c, mean = res
    # From here on, sklearn's own numpy calls (`_fit_full`, `_transform`) on identical arrays.
    c -= n * mean.reshape(-1, 1) * mean.reshape(1, -1)
    c /= n - 1
    eigenvals, eigenvecs = np.linalg.eigh(c)
    del c
    eigenvals = np.flip(eigenvals, axis=0)
    eigenvecs = np.flip(eigenvecs, axis=1)
    eigenvals[eigenvals < 0.0] = 0.0
    _, vt = svd_flip(None, eigenvecs.T, u_based_decision=False)
    ratio = eigenvals / np.sum(eigenvals)
    components = np.array(vt[:k], copy=True)
    x_pca = np.empty((n, k), dtype=x.dtype)
    _csr_dense_matmul(
        x.indptr, x.indices, x.data, slot, np.ascontiguousarray(components.T), x_pca
    )
    x_pca -= mean.reshape(1, -1) @ components.T
    var = np.array(eigenvals[:k], copy=True)
    return x_pca, components, var, np.array(ratio[:k], copy=True), mean


@cache
def _replays_sklearn(dtype: np.dtype) -> bool:
    """Whether :func:`_covariance_eigh_replay` gives sklearn's bits for CSR input of `dtype` here.

    That needs scipy's compiled loops to round each product and sum separately, as numba does.
    A build that fuses them into multiply-add instructions (likely on arm64) gives different bits.
    Checked once per dtype on a small matrix with mixed signs.
    """
    from scipy import sparse
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(0)
    dense = rng.standard_normal((200, 12)).astype(dtype)
    dense[rng.random(dense.shape) < 0.6] = 0
    x = sparse.csr_matrix(dense)  # noqa: TID251
    ref = PCA(4, svd_solver="covariance_eigh")
    expected = (
        ref.fit_transform(x),
        ref.components_,
        ref.explained_variance_,
        ref.explained_variance_ratio_,
        ref.mean_,
    )
    got = _covariance_eigh_replay(x, None, 4)
    return got is not None and all(
        a.dtype == b.dtype and np.array_equal(a, b)
        for a, b in zip(got, expected, strict=True)
    )


@dataclass
class PCAEighSparseReference:
    """:class:`sklearn.decomposition.PCA` with ``svd_solver='covariance_eigh'`` on in-memory CSR, bit for bit.

    sklearn runs scipy's serial sparse ``X.T @ X``, column mean and projection. Here they are parallel
    kernels that do each output entry's float operations in scipy's order, followed by sklearn's own
    dense numpy steps. Where that replay doesn't apply (see :func:`_replays_sklearn` and
    :func:`_gram_mean_replay`), this runs sklearn.
    """

    n_components: int
    cols: NDArray[np.intp] | None = None
    """Columns to fit on (all if `None`), read from the full matrix without subsetting it."""
    components_: NDArray[np.floating] = field(init=False)
    explained_variance_: NDArray[np.floating] = field(init=False)
    explained_variance_ratio_: NDArray[np.floating] = field(init=False)
    mean_: NDArray[np.floating] = field(init=False)

    def fit_transform(self, x: CSRBase) -> NDArray[np.floating]:
        """Fit on ``x[:, self.cols]`` and return its projection."""
        n = x.shape[0]
        n_sel = x.shape[1] if self.cols is None else len(self.cols)
        res = None
        if (
            0 < self.n_components <= min(n, n_sel)
            and n > 1
            and _replays_sklearn(x.dtype)
        ):
            res = _covariance_eigh_replay(x, self.cols, self.n_components)
        if res is None:
            from sklearn.decomposition import PCA

            pca = PCA(self.n_components, svd_solver="covariance_eigh")
            x_pca = pca.fit_transform(x if self.cols is None else x[:, self.cols])
            res = (
                x_pca,
                pca.components_,
                pca.explained_variance_,
                pca.explained_variance_ratio_,
                pca.mean_,
            )
        (
            x_pca,
            self.components_,
            self.explained_variance_,
            self.explained_variance_ratio_,
            self.mean_,
        ) = res
        return x_pca


# Fast path (`settings.reference_numerics` off): float64 gram from densified row blocks and BLAS.

# Rows densified per block in `blocked_gram`: a (rows x columns) float64 buffer of this size.
_BLOCK_BYTES = 64 * 2**20


@fau_njit
def _densify_rows(  # noqa: PLR0917
    indptr: NDArray[np.integer],
    indices: NDArray[np.integer],
    data: NDArray[np.number],
    lo: int,
    slot: NDArray[np.intp],
    out: NDArray[np.float64],
) -> None:
    """Write rows ``lo:lo + len(out)`` of a CSR matrix into dense ``out``, column ``j`` to ``slot[j]`` (skip if -1)."""
    for r in numba.prange(out.shape[0]):
        row = out[r]
        row[:] = 0.0
        for k in range(indptr[lo + r], indptr[lo + r + 1]):
            c = slot[indices[k]]
            if c >= 0:
                row[c] += data[k]  # `+=`: sums duplicate entries, as scipy does


@fau_njit
def _add_col_sums(block: NDArray[np.float64], out: NDArray[np.float64]) -> None:
    """``out += block.sum(axis=0)``, rows added in order as numpy does, in parallel over columns."""
    n_rows, n_cols = block.shape
    for c in numba.prange((n_cols + 63) // 64):
        lo, hi = 64 * c, min(n_cols, 64 * c + 64)
        part = block[0, lo:hi].copy()
        for r in range(1, n_rows):
            part += block[r, lo:hi]
        out[lo:hi] += part


@fau_njit
def _center_gram(gram: NDArray[np.float64], mean: NDArray[np.float64], n: int) -> None:
    """``gram = (gram - n * np.outer(mean, mean)) / (n - 1)`` in place, with the same arithmetic."""
    for j in numba.prange(gram.shape[1]):
        for i in range(gram.shape[0]):
            gram[i, j] = (gram[i, j] - n * (mean[i] * mean[j])) / (n - 1)


def blocked_gram(
    mat: CSRBase, slot: NDArray[np.intp], n_sel: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``X[:, cols].T @ X[:, cols]`` and column sums in float64, from densified row blocks and BLAS ``dsyrk``.

    Reading the full-width CSR through the column map `slot` avoids materialising the column subset.
    Each output entry is summed over blocks in a fixed order. The gram matrix is Fortran-ordered
    and the only `n_sel` × `n_sel` array allocated.
    """
    from scipy.linalg.blas import dsyrk

    step = max(1, min(mat.shape[0], _BLOCK_BYTES // (8 * max(n_sel, 1))))
    buf = np.empty((step, n_sel))
    gram = np.zeros((n_sel, n_sel), order="F")
    col_sums = np.zeros(n_sel)
    for lo in range(0, mat.shape[0], step):
        block = buf[: min(step, mat.shape[0] - lo)]
        _densify_rows(mat.indptr, mat.indices, mat.data, lo, slot, block)
        # C-order (rows x n_sel) is a Fortran (n_sel x rows) matrix B; dsyrk gives B @ B.T.
        # dsyrk keeps all BLAS threads: pinning it would make this O(n_obs * n_sel²) step about as
        # many times slower. OpenBLAS splits the output between threads, so each entry is summed by
        # one thread in a fixed order and the result doesn't depend on the thread count. MKL may split
        # a block's rows between threads, so with MKL the last bits can depend on the thread count.
        # TODO: measure dsyrk pinned to one thread vs. all threads on the cluster, with MKL and OpenBLAS.
        gram = dsyrk(1.0, block.T, beta=1.0, c=gram, trans=0, lower=0, overwrite_c=1)
        _add_col_sums(block, col_sums)
    _mirror_upper(gram)  # dsyrk fills the upper triangle
    return gram, col_sums


@dataclass
class PCAEighSparse:
    """Exact PCA of an in-memory CSR matrix via eigendecomposition of its covariance.

    Matches :class:`sklearn.decomposition.PCA` with ``svd_solver='covariance_eigh'`` to rounding
    (and in dtypes), but computes the gram matrix (densified row blocks, BLAS ``dsyrk``)
    and the projection in parallel, and in float64.
    """

    n_components: int
    cols: NDArray[np.intp] | None = None
    """Columns to fit on (all if `None`), read from the full matrix without subsetting it."""
    components_: NDArray[np.floating] = field(init=False)
    explained_variance_: NDArray[np.floating] = field(init=False)
    explained_variance_ratio_: NDArray[np.floating] = field(init=False)
    mean_: NDArray[np.floating] = field(init=False)

    def fit_transform(self, x: CSRBase) -> NDArray[np.float64]:
        """Fit on ``x[:, self.cols]`` and return its projection."""
        n = x.shape[0]
        n_sel = x.shape[1] if self.cols is None else len(self.cols)
        if not 0 < self.n_components <= min(n, n_sel):
            msg = (
                f"n_components={self.n_components} must be between 0 and "
                f"min(n_samples, n_features)={min(n, n_sel)} with svd_solver='covariance_eigh'"
            )
            raise ValueError(msg)
        slot = _slot_map(x.shape[1], self.cols)
        cov, col_sums = blocked_gram(x, slot, n_sel)
        mean = col_sums / n
        _center_gram(cov, mean, n)  # the gram matrix becomes the covariance
        total_var = np.trace(cov)
        d = cov.shape[0]
        k = self.n_components
        # Only the top `k` eigenpairs; the total variance is the trace (sum of all eigenvalues).
        # One BLAS thread keeps the eigenvectors independent of the thread count, which
        # `settings.reference_numerics=False` promises. That costs time: the tridiagonal reduction is
        # mostly level-2 BLAS, about 1 to 2 s on one thread for 2,000 columns (not measured yet), and
        # far more for tens of thousands of columns without a mask.
        with threadpool_limits(1, user_api="blas"):
            evals, evecs = scipy.linalg.eigh(
                cov, subset_by_index=[d - k, d - 1], driver="evr", overwrite_a=True
            )
        del cov  # overwritten by LAPACK
        order = np.argsort(evals)[::-1]
        # clip tiny negative eigenvalues from rounding, as sklearn does
        evals = np.maximum(evals, 0.0)
        # largest-magnitude loading positive, as sklearn
        _, components = svd_flip(None, evecs[:, order].T, u_based_decision=False)
        x_pca = np.empty((n, len(order)))
        _csr_dense_matmul(
            x.indptr, x.indices, x.data, slot, np.ascontiguousarray(components.T), x_pca
        )
        x_pca -= mean @ components.T
        # sklearn fits float32 input in float32 and anything else in float64
        dtype = (
            x.dtype
            if x.dtype in {np.dtype(np.float32), np.dtype(np.float64)}
            else np.dtype(np.float64)
        )
        self.mean_ = mean.astype(dtype, copy=False)
        self.components_ = components.astype(dtype, copy=False)
        self.explained_variance_ = evals[order].astype(dtype, copy=False)
        ratio = evals[order] / max(total_var, 0.0)
        self.explained_variance_ratio_ = ratio.astype(dtype, copy=False)
        return x_pca
