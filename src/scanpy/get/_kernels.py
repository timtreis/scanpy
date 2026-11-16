from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np
from fast_array_utils.numba import njit

from .._compat import warn
from .._utils._numba import (
    CompressedArrays,
    _chunk_bounds,
    arrays,
    has_canonical_format,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .._compat import CSCBase, CSRBase


@njit
def agg_sum_csr(indicator: CSRBase, data: CSRBase, out: NDArray) -> None:
    for cat_num in numba.prange(indicator.shape[0]):
        start_cat_idx = indicator.indptr[cat_num]
        stop_cat_idx = indicator.indptr[cat_num + 1]
        for row_num in range(start_cat_idx, stop_cat_idx):
            obs_per_cat = indicator.indices[row_num]

            start_obs = data.indptr[obs_per_cat]
            end_obs = data.indptr[obs_per_cat + 1]

            for j in range(start_obs, end_obs):
                col = data.indices[j]
                out[cat_num, col] += data.data[j]


@njit
def agg_sum_csc(indicator: CSRBase, data: CSCBase, out: np.ndarray) -> None:
    obs_to_cat = np.full(data.shape[0], -1, dtype=np.int64)

    for cat in range(indicator.shape[0]):
        for k in range(indicator.indptr[cat], indicator.indptr[cat + 1]):
            obs_to_cat[indicator.indices[k]] = cat

    for col in numba.prange(data.shape[1]):
        start = data.indptr[col]
        end = data.indptr[col + 1]

        for j in range(start, end):
            obs = data.indices[j]
            cat = obs_to_cat[obs]

            if cat != -1:
                out[cat, col] += data.data[j]


@njit
def mean_var_dense(
    indicator: CSRBase, data: NDArray
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Welford's online algorithm, parallelized over categories. The indicator
    # CSR lists which observations belong to each category, allowing mask
    # handling to be folded in naturally.
    n_cats = indicator.shape[0]
    n_features = data.shape[1]
    mean = np.zeros((n_cats, n_features), dtype="float64")
    var = np.zeros((n_cats, n_features), dtype="float64")

    for cat in numba.prange(n_cats):
        start = indicator.indptr[cat]
        stop = indicator.indptr[cat + 1]
        n = 0
        for row_num in range(start, stop):
            obs = indicator.indices[row_num]
            n += 1
            for col in range(n_features):
                value = np.float64(data[obs, col])
                delta = value - mean[cat, col]
                mean[cat, col] += delta / n
                delta2 = value - mean[cat, col]
                var[cat, col] += delta * delta2
        if n > 0:
            for col in range(n_features):
                var[cat, col] /= n
    return mean, var


@njit
def mean_var_csr(
    indicator: CSRBase,
    data: CSCBase,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Welford's online algorithm over nonzeros, then merge with the block of
    # implicit zeros per (category, feature). Merging a Welford accumulator
    # (n_A, mean_A, M2_A) with k zeros gives:
    #   mean   = mean_A * n_A / (n_A + k)
    #   M2_new = M2_A + mean_A^2 * n_A * k / (n_A + k)
    n_cats = indicator.shape[0]
    n_features = data.shape[1]
    mean = np.zeros((n_cats, n_features), dtype="float64")
    var = np.zeros((n_cats, n_features), dtype="float64")

    for cat_num in numba.prange(n_cats):
        start_cat_idx = indicator.indptr[cat_num]
        stop_cat_idx = indicator.indptr[cat_num + 1]
        n_obs = stop_cat_idx - start_cat_idx
        if n_obs == 0:
            continue

        n_nonzero = np.zeros(n_features, dtype=np.int64)

        for row_num in range(start_cat_idx, stop_cat_idx):
            obs_per_cat = indicator.indices[row_num]
            start_obs = data.indptr[obs_per_cat]
            end_obs = data.indptr[obs_per_cat + 1]

            for j in range(start_obs, end_obs):
                col = data.indices[j]
                value = np.float64(data.data[j])
                n = n_nonzero[col] + 1
                n_nonzero[col] = n
                m = mean[cat_num, col]
                delta = value - m
                m += delta / n
                mean[cat_num, col] = m
                var[cat_num, col] += delta * (value - m)

        for col in range(n_features):
            n_nz = n_nonzero[col]
            k = n_obs - n_nz
            if k > 0 and n_nz > 0:
                mean_a = mean[cat_num, col]
                mean[cat_num, col] = mean_a * n_nz / n_obs
                var[cat_num, col] += mean_a * mean_a * n_nz * k / n_obs
            var[cat_num, col] /= n_obs
    return mean, var


@njit
def mean_var_csc(
    indicator: CSRBase, data: CSCBase
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Welford's online algorithm, parallelized over columns. For each column
    # we accumulate per-category over the explicit nonzeros, then merge each
    # category's accumulator with its block of implicit zeros (see merge
    # formula in `mean_var_csr`).
    n_cats = indicator.shape[0]
    n_features = data.shape[1]
    obs_to_cat = np.full(data.shape[0], -1, dtype=np.int64)
    n_obs_per_cat = np.zeros(n_cats, dtype=np.int64)
    for cat in range(n_cats):
        n_obs_per_cat[cat] = indicator.indptr[cat + 1] - indicator.indptr[cat]
        for k in range(indicator.indptr[cat], indicator.indptr[cat + 1]):
            obs_to_cat[indicator.indices[k]] = cat

    mean = np.zeros((n_cats, n_features), dtype="float64")
    var = np.zeros((n_cats, n_features), dtype="float64")

    for col in numba.prange(n_features):
        n_nonzero = np.zeros(n_cats, dtype=np.int64)
        start = data.indptr[col]
        end = data.indptr[col + 1]

        for j in range(start, end):
            obs = data.indices[j]
            cat = obs_to_cat[obs]
            if cat == -1:
                continue
            value = np.float64(data.data[j])
            n = n_nonzero[cat] + 1
            n_nonzero[cat] = n
            m = mean[cat, col]
            delta = value - m
            m += delta / n
            mean[cat, col] = m
            var[cat, col] += delta * (value - m)

        for cat in range(n_cats):
            n_obs = n_obs_per_cat[cat]
            if n_obs == 0:
                continue
            n_nz = n_nonzero[cat]
            k = n_obs - n_nz
            if k > 0 and n_nz > 0:
                mean_a = mean[cat, col]
                mean[cat, col] = mean_a * n_nz / n_obs
                var[cat, col] += mean_a * mean_a * n_nz * k / n_obs
            var[cat, col] /= n_obs
    return mean, var


# Column blocks of the blocked group kernels. Any count gives the same result: each (block, group)
# task owns its cells and walks the group's rows in order, as the per-group kernels above do.
# This many blocks times the groups gives enough tasks to balance a dominant group.
_N_COL_BLOCKS = 64


@njit
def _mean_var_group_blocks(  # noqa: PLR0917
    indptr: NDArray[np.integer],
    indices: NDArray[np.integer],
    data: NDArray[np.number],
    rows: NDArray[np.int64],
    group_ptr: NDArray[np.int64],
    block_ptr: NDArray[np.int64],
    mean: NDArray[np.float64],
    var: NDArray[np.float64],
    nnz: NDArray[np.int64],
) -> None:
    """:func:`mean_var_csr`'s arithmetic, parallel over (column block, group) pairs.

    Bit-identical to it, but a dominant group no longer serialises the pass. Each pair
    writes its own cells; rows reach their block's first column by binary search,
    so `indices` must be sorted.
    """
    n_groups = group_ptr.shape[0] - 1
    for t in numba.prange((block_ptr.shape[0] - 1) * n_groups):
        b, g = t // n_groups, t % n_groups  # block-major: each thread sees every group
        j0, j1 = block_ptr[b], block_ptr[b + 1]
        lo, hi = group_ptr[g], group_ptr[g + 1]
        n_obs = hi - lo
        if n_obs == 0:
            continue
        for p in range(lo, hi):
            r = rows[p]
            start, end = indptr[r], indptr[r + 1]
            k = start + np.searchsorted(indices[start:end], j0)
            while k < end:
                col = indices[k]
                if col >= j1:
                    break
                value = np.float64(data[k])
                n = nnz[g, col] + 1
                nnz[g, col] = n
                m = mean[g, col]
                delta = value - m
                m += delta / n
                mean[g, col] = m
                var[g, col] += delta * (value - m)
                k += 1
        for col in range(j0, j1):
            n_nz = nnz[g, col]
            n_zeros = n_obs - n_nz
            if n_zeros > 0 and n_nz > 0:
                mean_a = mean[g, col]
                mean[g, col] = mean_a * n_nz / n_obs
                var[g, col] += mean_a * mean_a * n_nz * n_zeros / n_obs
            var[g, col] /= n_obs


@njit
def _sum_group_blocks(  # noqa: PLR0917
    indptr: NDArray[np.integer],
    indices: NDArray[np.integer],
    data: NDArray[np.number],
    rows: NDArray[np.int64],
    group_ptr: NDArray[np.int64],
    block_ptr: NDArray[np.int64],
    sums: NDArray[np.float64 | np.int64],
    nnz: NDArray[np.int64],
) -> None:
    """:func:`agg_sum_csr`'s arithmetic, and the count of non-zeros, parallel over (column block, group) pairs.

    Each sum adds the group's values in row order into the `sums` dtype, as :func:`agg_sum_csr` does,
    so it is bit-identical to it. `indices` must be sorted.
    """
    n_groups = group_ptr.shape[0] - 1
    for t in numba.prange((block_ptr.shape[0] - 1) * n_groups):
        b, g = t // n_groups, t % n_groups
        j0, j1 = block_ptr[b], block_ptr[b + 1]
        for p in range(group_ptr[g], group_ptr[g + 1]):
            r = rows[p]
            start, end = indptr[r], indptr[r + 1]
            k = start + np.searchsorted(indices[start:end], j0)
            while k < end:
                col = indices[k]
                if col >= j1:
                    break
                sums[g, col] += data[k]
                nnz[g, col] += data[k] != 0
                k += 1


def _group_layout(
    codes: NDArray[np.integer], n_groups: int
) -> tuple[NDArray[np.intp], NDArray[np.int64], NDArray[np.int64]]:
    """Rows grouped by `codes` in stable order (negative codes dropped), group offsets and sizes.

    The rows of each group are in ascending order, as in :func:`~scanpy.get._aggregated.sparse_indicator`.
    """
    counts = np.bincount(codes[codes >= 0], minlength=n_groups)
    rows = np.argsort(codes, kind="stable")[np.count_nonzero(codes < 0) :]
    return rows, np.concatenate([[0], np.cumsum(counts)]), counts


def _scale_dof(
    var: NDArray[np.float64], counts: NDArray[np.integer], dof: int
) -> NDArray[np.bool]:
    """Turn biased per-group variances `var` into `dof`-corrected ones in place.

    Returns which groups have too few observations (their variance becomes NaN).
    """
    denom = np.where(counts > dof, counts - dof, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        var *= (counts / denom)[:, None]
    return np.isnan(denom)


def _warn_nan_var(groups: list) -> None:
    """Warn that the variance of `groups` (labels as `aggregate` lists them) is NaN, as `aggregate` does."""
    msg = f"Group counts matches dof, resulting var for groups {groups} will be nan"
    warn(msg, RuntimeWarning)


def grouped_mean_var(
    x: CSRBase, rows: NDArray[np.integer], group_ptr: NDArray[np.integer]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64] | None]:
    """Per-group mean, biased variance and (if cheap) non-zero count of CSR `x`.

    Group `g` is the rows ``rows[group_ptr[g]:group_ptr[g + 1]]``. Canonical input takes the
    blocked kernel; otherwise :func:`mean_var_csr`, which treats duplicate entries as observations.
    """
    if has_canonical_format(x):
        return mean_var_csr_blocked(x, rows, group_ptr)
    shape = np.array([group_ptr.shape[0] - 1, x.shape[0]], dtype=np.int64)
    indicator = CompressedArrays(group_ptr, rows, np.empty(0), shape)  # data is unused
    return (*mean_var_csr(indicator, arrays(x)), None)


def mean_var_csr_blocked(
    x: CSRBase,
    rows: NDArray[np.integer],
    group_ptr: NDArray[np.integer],
    *,
    n_blocks: int = _N_COL_BLOCKS,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Per-group mean, biased variance and stored-entry count of canonical CSR `x`.

    Group `g` is the rows ``rows[group_ptr[g]:group_ptr[g + 1]]``, i.e. an indicator matrix's
    `indices` and `indptr`. Bit-identical to :func:`mean_var_csr` for any `n_blocks`.
    """
    n_groups, n_genes = group_ptr.shape[0] - 1, x.shape[1]
    mean = np.zeros((n_groups, n_genes))
    var = np.zeros((n_groups, n_genes))
    nnz = np.zeros((n_groups, n_genes), dtype=np.int64)
    _mean_var_group_blocks(
        x.indptr,
        x.indices,
        x.data,
        rows.astype(np.int64),
        group_ptr.astype(np.int64),
        _chunk_bounds(n_genes, n_blocks),
        mean,
        var,
        nnz,
    )
    return mean, var, nnz


def sum_csr_blocked(
    x: CSRBase,
    rows: NDArray[np.integer],
    group_ptr: NDArray[np.integer],
    *,
    n_blocks: int = _N_COL_BLOCKS,
) -> tuple[NDArray[np.float64 | np.int64], NDArray[np.int64]]:
    """Per-group sum (int64 for integer `x`, else float64) and non-zero count of canonical CSR `x`.

    Groups as in :func:`mean_var_csr_blocked`. Bit-identical to :func:`agg_sum_csr` (what
    :meth:`~scanpy.get._aggregated.Aggregate.sum` uses) for any `n_blocks`.
    """
    n_groups, n_genes = group_ptr.shape[0] - 1, x.shape[1]
    # as `Aggregate._sum`
    dtype = np.int64 if np.issubdtype(x.dtype, np.integer) else np.float64
    sums = np.zeros((n_groups, n_genes), dtype=dtype)
    nnz = np.zeros((n_groups, n_genes), dtype=np.int64)
    _sum_group_blocks(
        x.indptr,
        x.indices,
        x.data,
        rows.astype(np.int64),
        group_ptr.astype(np.int64),
        _chunk_bounds(n_genes, n_blocks),
        sums,
        nnz,
    )
    return sums, nnz
