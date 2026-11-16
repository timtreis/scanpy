from __future__ import annotations

from functools import singledispatch, wraps
from typing import TYPE_CHECKING, NamedTuple, TypedDict

import numba
import numpy as np
import pandas as pd
from fast_array_utils import stats
from fast_array_utils.numba import njit
from scipy import sparse
from scverse_misc import Deprecation, deprecated_arg

from scanpy.get._out import _out_dests, _write_dests
from scanpy.get._use import _get_arr, _resolve_layer
from scanpy.preprocessing._distributed import materialize_as_ndarray

from .._compat import CSBase, CSRBase, DaskArray, warn
from .._docs import DEPR_RAW
from .._settings import Default, settings
from .._utils import _doc_params, axis_nnz
from .._utils._numba import (
    N_REDUCE_CHUNKS,
    _PW_STACK,
    _chunk_bounds,
    _is_numba_csr,
    _nnz_row_blocks,
    _reduceat_add,
    eliminate_zeros,
    has_canonical_format,
)
from ._docs import (
    doc_adata_basic,
    doc_expr_reps,
    doc_obs_qc_args,
    doc_obs_qc_returns,
    doc_qc_metric_naming,
    doc_var_qc_returns,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from typing import Literal

    from anndata import AnnData
    from numpy._typing._array_like import NDArray

    from ..get._out import OutKind
    from ..get._types import LayerAcc, MultiAcc


@_doc_params(
    doc_adata_basic=doc_adata_basic,
    doc_expr_reps=doc_expr_reps,
    doc_obs_qc_args=doc_obs_qc_args,
    doc_qc_metric_naming=doc_qc_metric_naming,
    doc_obs_qc_returns=doc_obs_qc_returns,
)
def describe_obs(  # noqa: PLR0913
    adata: AnnData,
    *,
    expr_type: str = "counts",
    var_type: str = "genes",
    qc_vars: Collection[str] = (),
    percent_top: Collection[int] | None = (50, 100, 200, 500),
    use: LayerAcc | str | None = None,
    layer: str | None = None,
    use_raw: bool = False,
    log1p: bool | None = True,
    inplace: bool = False,
    x=None,
    parallel=None,
) -> pd.DataFrame | None:
    """Describe observations of anndata.

    Calculates a number of qc metrics for observations in AnnData object. See
    section `Returns` for a description of those metrics.

    Note that this method can take a while to compile on the first call. That
    result is then cached to disk to be used later.

    Params
    ------
    {doc_adata_basic}
    {doc_qc_metric_naming}
    {doc_obs_qc_args}
    {doc_expr_reps}
    log1p
        Add `log1p` transformed metrics.
    inplace
        Whether to place calculated metrics in `adata.obs`.
    X
        Matrix to calculate values on. Meant for internal usage.

    Returns
    -------
    QC metrics for observations in adata. If inplace, values are placed into
    the AnnData's `.obs` dataframe.

    {doc_obs_qc_returns}

    """
    if parallel is not None:
        msg = "Argument `parallel` is deprecated, and currently has no effect."
        warn(msg, FutureWarning)
    # Handle whether X is passed
    if x is None:
        x = _get_arr(adata, _resolve_layer(use), use_raw=use_raw, layer=layer)
        eliminate_zeros(x)
    obs_metrics = pd.DataFrame(index=adata.obs_names)
    qc_masks = [adata.var[v].to_numpy() for v in qc_vars]
    qc_totals = {}
    if kernels := _qc_kernels(x, qc_masks):
        # one parallel pass instead of a pass per metric and a column subset per qc_var
        nnz, totals, qc_tot = _obs_qc_csr(x, qc_masks, exact=kernels == "exact")
        qc_totals = dict(zip(qc_vars, qc_tot.T, strict=True))
    else:
        nnz, totals = materialize_as_ndarray(axis_nnz(x, axis=1)), stats.sum(x, axis=1)
    obs_metrics[f"n_{var_type}_by_{expr_type}"] = nnz
    if log1p:
        obs_metrics[f"log1p_n_{var_type}_by_{expr_type}"] = np.log1p(
            obs_metrics[f"n_{var_type}_by_{expr_type}"]
        )
    obs_metrics[f"total_{expr_type}"] = totals
    if log1p:
        obs_metrics[f"log1p_total_{expr_type}"] = np.log1p(
            obs_metrics[f"total_{expr_type}"]
        )
    if percent_top:
        percent_top = sorted(percent_top)
        proportions = top_segment_proportions(x, percent_top)
        for i, n in enumerate(percent_top):
            obs_metrics[f"pct_{expr_type}_in_top_{n}_{var_type}"] = (
                proportions[:, i] * 100
            )
    for qc_var, mask in zip(qc_vars, qc_masks, strict=True):
        obs_metrics[f"total_{expr_type}_{qc_var}"] = (
            qc_totals[qc_var] if qc_var in qc_totals else stats.sum(x[:, mask], axis=1)
        )
        if log1p:
            obs_metrics[f"log1p_total_{expr_type}_{qc_var}"] = np.log1p(
                obs_metrics[f"total_{expr_type}_{qc_var}"]
            )
        obs_metrics[f"pct_{expr_type}_{qc_var}"] = (
            obs_metrics[f"total_{expr_type}_{qc_var}"]
            / obs_metrics[f"total_{expr_type}"]
            * 100
        )
    if inplace:
        adata.obs[obs_metrics.columns] = obs_metrics
    else:
        return obs_metrics
    return None


@_doc_params(
    doc_adata_basic=doc_adata_basic,
    doc_expr_reps=doc_expr_reps,
    doc_qc_metric_naming=doc_qc_metric_naming,
    doc_var_qc_returns=doc_var_qc_returns,
)
def describe_var(
    adata: AnnData,
    *,
    expr_type: str = "counts",
    var_type: str = "genes",
    use: LayerAcc | str | None = None,
    layer: str | None = None,
    use_raw: bool = False,
    inplace: bool = False,
    log1p: bool = True,
    x: CSBase | np.ndarray | None = None,
) -> pd.DataFrame | None:
    """Describe variables of anndata.

    Calculates a number of qc metrics for variables in AnnData object. See
    section `Returns` for a description of those metrics.

    Params
    ------
    {doc_adata_basic}
    {doc_qc_metric_naming}
    {doc_expr_reps}
    inplace
        Whether to place calculated metrics in `adata.var`.
    X
        Matrix to calculate values on. Meant for internal usage.

    Returns
    -------
    QC metrics for variables in adata. If inplace, values are placed into the
    AnnData's `.var` dataframe.

    {doc_var_qc_returns}

    """
    # Handle whether X is passed
    if x is None:
        x = _get_arr(adata, _resolve_layer(use), use_raw=use_raw, layer=layer)
        eliminate_zeros(x)
    var_metrics = pd.DataFrame(index=adata.var_names)
    if kernels := _qc_kernels(x):
        n_cells, totals = _var_qc_csr(x, exact=kernels == "exact")
        mean = totals / x.shape[0]  # `stats.mean` is `stats.sum` divided by this
    else:
        n_cells, mean = materialize_as_ndarray((
            axis_nnz(x, axis=0),
            stats.mean(x, axis=0),
        ))
        totals = stats.sum(x, axis=0)
    var_metrics[f"n_cells_by_{expr_type}"] = n_cells
    var_metrics[f"mean_{expr_type}"] = mean
    if log1p:
        var_metrics[f"log1p_mean_{expr_type}"] = np.log1p(
            var_metrics[f"mean_{expr_type}"]
        )
    var_metrics[f"pct_dropout_by_{expr_type}"] = (
        1 - var_metrics[f"n_cells_by_{expr_type}"] / x.shape[0]
    ) * 100
    var_metrics[f"total_{expr_type}"] = totals
    if log1p:
        var_metrics[f"log1p_total_{expr_type}"] = np.log1p(
            var_metrics[f"total_{expr_type}"]
        )
    if inplace:
        adata.var[var_metrics.columns] = var_metrics
        return None
    return var_metrics


class QcMetricsResult(NamedTuple):
    """Outputs of :func:`~scanpy.pp.calculate_qc_metrics`."""

    obs: pd.DataFrame
    """Per-observation metrics, see :func:`~scanpy.pp.calculate_qc_metrics`."""
    var: pd.DataFrame
    """Per-variable metrics, see :func:`~scanpy.pp.calculate_qc_metrics`."""


class QcMetricsOut(TypedDict, total=False):
    """Where a mapping `out` of :func:`~scanpy.pp.calculate_qc_metrics` writes each output, see :class:`~scanpy.pp.QcMetricsResult`."""

    obs: MultiAcc | str
    var: MultiAcc | str


QC_METRICS_KINDS: QcMetricsResult[set[OutKind], set[OutKind]] = QcMetricsResult(
    {"obsm"}, {"varm"}
)
"""Where a mapping `out` can write the outputs of `pp.calculate_qc_metrics`."""


@_doc_params(
    doc_adata_basic=doc_adata_basic,
    doc_expr_reps=doc_expr_reps,
    doc_obs_qc_args=doc_obs_qc_args,
    doc_qc_metric_naming=doc_qc_metric_naming,
    doc_obs_qc_returns=doc_obs_qc_returns,
    doc_var_qc_returns=doc_var_qc_returns,
)
@deprecated_arg("layer", Deprecation("1.13.0", "Use `use` instead."))
@deprecated_arg("use_raw", DEPR_RAW)
@deprecated_arg(
    "inplace", Deprecation("1.13.0", "Use `out=False` instead of `inplace=True`.")
)
def calculate_qc_metrics(  # noqa: PLR0913
    adata: AnnData,
    *,
    expr_type: str = "counts",
    var_type: str = "genes",
    qc_vars: Collection[str] | str = (),
    percent_top: Collection[int] | None = (50, 100, 200, 500),
    use: LayerAcc | str | None = None,
    out: MultiAcc | str | QcMetricsOut | bool | Default = Default(
        preset=("calculate_qc_metrics", "out")
    ),
    log1p: bool = True,
    parallel: bool | None = None,
    # deprecated
    layer: str | None = None,
    use_raw: bool = False,
    inplace: bool | None = None,
) -> QcMetricsResult | None:
    r"""Calculate quality control metrics.

    Calculates a number of qc metrics for an AnnData object, see section
    `Returns` for specifics. Largely based on `calculateQCMetrics` from scater
    :cite:p:`McCarthy2017`. Currently is most efficient on a sparse CSR or dense matrix.

    Note that this method can take a while to compile on the first call. That
    result is then cached to disk to be used later.

    .. array-support:: pp.calculate_qc_metrics

    Parameters
    ----------
    {doc_adata_basic}
    {doc_qc_metric_naming}
    {doc_obs_qc_args}
    {doc_expr_reps}
    out
        Where to write the metrics.
        If :data:`False`, they are written as columns of `.obs` and `.var` (see below).
        An accessor like `A.obsm['qc']` writes the metrics tables as whole :class:`~pandas.DataFrame`\ s
        to `.obsm['qc']` and `.varm['qc']` instead.
        A mapping writes only the tables it names, e.g. `{{'obs': A.obsm['qc']}}`.
        :class:`str`\ s are :meth:`anndata.acc.AdAcc.resolve`\ d.
        If :data:`True`, they are returned as a :class:`~scanpy.pp.QcMetricsResult` instead of written.
        The default depends on :attr:`scanpy.settings.preset`:
        :data:`True` for :attr:`~scanpy.Preset.ScanpyV1`, :data:`False` for :attr:`~scanpy.Preset.ScanpyV2Preview`.
    log1p
        Set to `False` to skip computing `log1p` transformed annotations.
    inplace
        Whether to place calculated metrics in `adata`'s `.obs` and `.var`.

    Returns
    -------
    Returns the calculated metrics (as :class:`~scanpy.pp.QcMetricsResult`) if `out=True`, else `None`.
    With `out=False`, sets the following columns of `adata.obs` and `adata.var`
    (an accessor or mapping writes them as tables instead, see `out`):

    {doc_obs_qc_returns}

    {doc_var_qc_returns}

    Example
    -------
    Calculate qc metrics for visualization.

    ..  exec-jupyter::

        import scanpy as sc
        import seaborn as sns

        pbmc = sc.datasets.pbmc3k()
        pbmc.var["mito"] = pbmc.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(pbmc, qc_vars=["mito"], out=False)
        sns.jointplot(
            data=pbmc.obs,
            x="log1p_total_counts",
            y="log1p_n_genes_by_counts",
            kind="hex",
        )

    ..  exec-jupyter::

        sns.histplot(pbmc.obs["pct_counts_mito"])

    """
    if parallel is not None:
        msg = "Argument `parallel` is deprecated, and currently has no effect."
        warn(msg, FutureWarning)
    if isinstance(out, Default):
        out = (
            settings.preset.calculate_qc_metrics.out if inplace is None else not inplace
        )
    elif inplace is not None:
        msg = "Pass either `out` or `inplace`, not both."
        raise TypeError(msg)
    # Pass X so I only have to do it once
    x = _get_arr(adata, _resolve_layer(use), use_raw=use_raw, layer=layer)
    eliminate_zeros(x)

    # Convert qc_vars to list if str
    if isinstance(qc_vars, str):
        qc_vars = [qc_vars]

    obs_metrics = describe_obs(
        adata,
        expr_type=expr_type,
        var_type=var_type,
        qc_vars=qc_vars,
        percent_top=percent_top,
        inplace=False,
        x=x,
        log1p=log1p,
    )
    var_metrics = describe_var(
        adata,
        expr_type=expr_type,
        var_type=var_type,
        inplace=False,
        x=x,
        log1p=log1p,
    )

    vals = QcMetricsResult(obs_metrics, var_metrics)
    match out:
        case True:
            return vals
        case False:
            for df, annot in zip(vals, [adata.obs, adata.var], strict=True):
                annot[df.columns] = df
        case _:
            _, dests = _out_dests(
                out,
                None,
                # what a single accessor names after itself (`out=False` writes columns instead)
                dests=lambda _: QcMetricsResult(dict(obsm="qc"), dict(varm="qc")),
                kinds=QC_METRICS_KINDS,
            )
            _write_dests(adata, vals, dests)
    return None


def top_proportions(
    mtx: np.ndarray | CSBase | sparse.coo_matrix | sparse.coo_array, n: int
):
    """Calculate cumulative proportions of top expressed genes.

    Parameters
    ----------
    mtx
        Matrix, where each row is a sample, each column a feature.
    n
        Rank to calculate proportions up to. Value is treated as 1-indexed,
        `n=50` will calculate cumulative proportions up to the 50th most
        expressed gene.

    """
    if isinstance(mtx, CSBase | sparse.coo_matrix | sparse.coo_array):
        if not isinstance(mtx, CSRBase):
            mtx = mtx.tocsr()
        # Allowing numba to do more
        return top_proportions_sparse_csr(mtx, np.array(n))
    else:
        return top_proportions_dense(mtx, n)


def top_proportions_dense(mtx: np.ndarray, n: int) -> NDArray[np.float64]:
    sums = mtx.sum(axis=1)
    partitioned = np.apply_along_axis(np.argpartition, 1, -mtx, n - 1)
    partitioned = partitioned[:, :n]
    values = np.zeros_like(partitioned, dtype=np.float64)
    for i in range(partitioned.shape[0]):
        vec = mtx[i, partitioned[i, :]]  # Not a view
        vec[::-1].sort()  # Sorting on a reversed view (e.g. a descending sort)
        vec = np.cumsum(vec) / sums[i]
        values[i, :] = vec
    return values


def top_proportions_sparse_csr(mtx: CSRBase, n: int) -> NDArray[np.float64]:
    values = np.zeros((mtx.indptr.size - 1, n), dtype=np.float64)
    for i in numba.prange(mtx.indptr.size - 1):
        start, end = mtx.indptr[i], mtx.indptr[i + 1]
        vec = np.zeros(n, dtype=np.float64)
        if end - start <= n:
            vec[: end - start] = mtx.data[start:end]
            total = vec.sum()
        else:
            vec[:] = -(np.partition(-mtx.data[start:end], n - 1)[:n])
            total = (mtx.data[start:end]).sum()  # Is this not just vec.sum()?
        vec[::-1].sort()
        values[i, :] = vec.cumsum() / total
    return values


def check_ns(func):
    @wraps(func)
    def check_ns_inner(
        mtx: np.ndarray | CSBase | sparse.coo_matrix | DaskArray, ns: Collection[int]
    ):
        if not (max(ns) <= mtx.shape[1] and min(ns) > 0):
            msg = "Positions outside range of features."
            raise IndexError(msg)
        return func(mtx, ns)

    return check_ns_inner


@singledispatch
@check_ns
def top_segment_proportions(mtx: np.ndarray, ns: Collection[int]) -> np.ndarray:
    """Calculate total percentage of counts in top ns genes.

    Parameters
    ----------
    mtx
        Matrix, where each row is a sample, each column a feature.
    ns
        Positions to calculate cumulative proportion at. Values are considered
        1-indexed, e.g. `ns=[50]` will calculate cumulative proportion up to
        the 50th most expressed gene.

    """
    # Currently ns is considered to be 1 indexed
    ns = np.sort(ns)
    sums = mtx.sum(axis=1)
    partitioned = np.apply_along_axis(np.partition, 1, mtx, mtx.shape[1] - ns)[:, ::-1][
        :, : ns[-1]
    ]
    values = np.zeros((mtx.shape[0], len(ns)))
    acc = np.zeros(mtx.shape[0])
    prev = 0
    for j, n in enumerate(ns):
        acc += partitioned[:, prev:n].sum(axis=1)
        values[:, j] = acc
        prev = n
    return values / sums[:, None]


@top_segment_proportions.register(DaskArray)
@check_ns
def _(mtx: DaskArray, ns: Collection[int]) -> DaskArray:
    if not isinstance(mtx._meta, CSRBase | np.ndarray):
        msg = f"DaskArray must have csr matrix or ndarray meta, got {mtx._meta}."
        raise ValueError(msg)
    if mtx.chunksize[1] != mtx.shape[1]:
        msg = f"{mtx} must not be chunked along the feature axis"
        raise ValueError(msg)
    return mtx.map_blocks(
        lambda x: top_segment_proportions(x, ns), meta=np.array([])
    ).compute()


@top_segment_proportions.register(CSBase)
@top_segment_proportions.register(sparse.coo_matrix)
@check_ns
def _(mtx: CSBase | sparse.coo_matrix, ns: Collection[int]) -> DaskArray:
    if not isinstance(mtx, CSRBase):
        mtx = mtx.tocsr()
    return top_segment_proportions_sparse_csr(mtx.data, mtx.indptr, np.array(ns))


@njit
def top_segment_proportions_sparse_csr(data, indptr, ns):
    # work around https://github.com/numba/numba/issues/5056
    indptr = indptr.astype(np.int64)
    ns = ns.astype(np.int64)
    ns = np.sort(ns)
    maxidx = ns[-1]
    sums = np.zeros((indptr.size - 1), dtype=np.float64)
    values = np.zeros((indptr.size - 1, len(ns)), dtype=np.float64)
    # Just to keep it simple, as a dense matrix
    partitioned = np.zeros((indptr.size - 1, maxidx), dtype=data.dtype)
    for i in numba.prange(indptr.size - 1):
        start, end = indptr[i], indptr[i + 1]
        sums[i] = np.sum(data[start:end])
        if end - start <= maxidx:
            partitioned[i, : end - start] = data[start:end]
        elif (end - start) > maxidx:
            partitioned[i, :] = -(np.partition(-data[start:end], maxidx))[:maxidx]
        partitioned[i, :] = np.partition(partitioned[i, :], maxidx - ns)
    partitioned = partitioned[:, ::-1][:, : ns[-1]]
    acc = np.zeros((indptr.size - 1), dtype=np.float64)
    prev = 0
    # can’t use enumerate due to https://github.com/numba/numba/issues/2625
    for j in range(ns.size):
        acc += partitioned[:, prev : ns[j]].sum(axis=1)
        values[:, j] = acc
        prev = ns[j]
    return values / sums.reshape((indptr.size - 1, 1))


# qc_vars are packed into the bits of one int64 per gene
_MAX_QC_FLAGS = 64


def _qc_kernels(
    x: CSBase | np.ndarray | DaskArray, qc_masks: Sequence[np.ndarray] = ()
) -> Literal["exact", "fast"] | None:
    """Which CSR kernels compute the metrics of `x`, if any (else the scipy code path does).

    - "exact": float data with :attr:`~scanpy.settings.reference_numerics`. The kernels repeat
      scipy's sums operation by operation, so the results are bit-identical.
    - "fast": integer data (integer sums are exact in any order), or float data without
      :attr:`~scanpy.settings.reference_numerics` (sums in float64, in another order).

    The `qc_masks` must be boolean: scipy indexes columns with any other array by its values.
    """
    if not _is_numba_csr(x) or x.dtype.kind == "b":
        return None
    if len(qc_masks) > _MAX_QC_FLAGS or any(m.dtype != np.bool for m in qc_masks):
        return None
    if x.dtype.kind in "iu" or not settings.reference_numerics:
        return "fast"
    return "exact"


def _sum_duplicates(x: CSRBase) -> None:
    """Sum duplicate entries in place, as scipy's `count_nonzero` does first in the scipy code path.

    Unlike :func:`~scanpy._utils._numba.eliminate_zeros`, this keeps the zeros of duplicates that
    cancel out. Afterwards, `x` has sorted indices, which the kernels below rely on.
    """
    if not has_canonical_format(x):
        x.sum_duplicates()


def _sum_dtype(dtype: np.dtype) -> np.dtype:
    """dtype of scipy's (and so `stats.sum`'s) sparse sums: the widest integer of the same kind, else `dtype`."""
    if dtype.kind in "iu":
        return np.result_type(dtype, np.uint64 if dtype.kind == "u" else np.int64)
    return dtype


def _acc_dtype(dtype: np.dtype) -> np.dtype:
    """dtype the "fast" kernels add up in: float64 for floats, else as scipy."""
    return np.dtype(np.float64) if dtype.kind == "f" else _sum_dtype(dtype)


def _obs_qc_csr(
    x: CSRBase, qc_masks: Sequence[NDArray[np.bool]], *, exact: bool
) -> tuple[NDArray[np.integer], NDArray, NDArray]:
    """Per-row nonzero count, total and total over each mask in `qc_masks` (see :func:`_qc_kernels`)."""
    _sum_duplicates(x)
    flags = np.zeros(x.shape[1], dtype=np.int64)
    for q, mask in enumerate(qc_masks):
        flags |= mask.astype(np.int64) << q
    acc = x.dtype if exact else _acc_dtype(x.dtype)
    nnz = np.zeros(x.shape[0], dtype=np.int64)
    totals = np.zeros(x.shape[0], dtype=acc)
    qc_tot = np.zeros((x.shape[0], len(qc_masks)), dtype=acc)
    if exact:
        blocks = _nnz_row_blocks(x.indptr, 8 * numba.get_num_threads())
        _obs_qc_csr_exact_kernel(
            x.indptr, x.indices, x.data, flags, blocks, nnz, totals, qc_tot
        )
    else:
        _obs_qc_csr_kernel(x.indptr, x.indices, x.data, flags, nnz, totals, qc_tot)
    dtype = _sum_dtype(x.dtype)
    # dtypes as from `axis_nnz` (scipy's `count_nonzero`: `np.diff(indptr)` if all stored values
    # are non-zero, which only fails for duplicates that cancel out) and `stats.sum`
    nnz_dtype = x.indptr.dtype if int(nnz.sum()) == x.indptr[-1] else np.int64
    return (
        nnz.astype(nnz_dtype, copy=False),
        totals.astype(dtype, copy=False),
        qc_tot.astype(dtype, copy=False),
    )


@njit
def _obs_qc_csr_kernel(  # noqa: PLR0917
    indptr: NDArray[np.integer],
    indices: NDArray[np.integer],
    data: NDArray[np.number],
    flags: NDArray[np.int64],
    nnz: NDArray[np.int64],
    totals: NDArray[np.number],
    qc_tot: NDArray[np.number],
) -> None:
    for i in numba.prange(indptr.shape[0] - 1):
        for k in range(indptr[i], indptr[i + 1]):
            v = data[k]
            nnz[i] += v != 0
            totals[i] += v  # in `totals`' dtype
            f = flags[indices[k]]
            if f == 0:
                continue
            for q in range(qc_tot.shape[1]):
                if (f >> q) & 1:
                    qc_tot[i, q] += v


# Why this is bit-identical to the scipy code path (`axis_nnz(x, axis=1)`, `stats.sum(x, axis=1)`
# and `stats.sum(x[:, mask], axis=1)` per qc_var on a canonical float CSR `x`):
# - fast-array-utils' `stats.sum` is `csr_array(x).sum(axis=1)`, which scipy computes as
#   `np.add.reduceat(x.data, x.indptr[nonempty rows])` in `x.dtype` (`_compressed.py`, `sum` and
#   `_minor_reduce`), with 0 for empty rows. Each segment is one row's stored values in stored
#   order; `_reduceat_add` repeats numpy's arithmetic for a segment.
# - `x[:, mask]` keeps each row's selected values in stored order (`csr_column_index2` copies them
#   in one pass over `x.indices`), so its row sums are `_reduceat_add` of that subsequence.
# - Each row is reduced on its own, so splitting the rows over threads doesn't change any result,
#   and the row blocks may depend on the thread count. Counts are integers.
@njit
def _obs_qc_csr_exact_kernel(  # noqa: PLR0917
    indptr: NDArray[np.integer],
    indices: NDArray[np.integer],
    data: NDArray[np.floating],
    flags: NDArray[np.int64],
    row_blocks: NDArray[np.int64],
    nnz: NDArray[np.int64],
    totals: NDArray[np.floating],
    qc_tot: NDArray[np.floating],
) -> None:
    n_qc = qc_tot.shape[1]
    for b in numba.prange(row_blocks.shape[0] - 1):
        r0, r1 = row_blocks[b], row_blocks[b + 1]
        longest = 0
        if n_qc > 0:
            for i in range(r0, r1):
                longest = max(longest, indptr[i + 1] - indptr[i])
        masked = np.empty(longest, dtype=data.dtype)  # a row's values in a qc_var's genes
        frames = np.empty((_PW_STACK, 3), dtype=np.int64)
        lefts = np.empty(_PW_STACK, dtype=data.dtype)
        for i in range(r0, r1):
            s, e = indptr[i], indptr[i + 1]
            for k in range(s, e):
                nnz[i] += data[k] != 0
            if e > s:
                totals[i] = _reduceat_add(data, s, e, frames, lefts)
            for q in range(n_qc):
                m = 0
                for k in range(s, e):
                    if (flags[indices[k]] >> q) & 1:
                        masked[m] = data[k]
                        m += 1
                if m > 0:
                    qc_tot[i, q] = _reduceat_add(masked, 0, m, frames, lefts)


def _var_qc_csr(x: CSRBase, *, exact: bool) -> tuple[NDArray[np.int64], NDArray]:
    """Per-column nonzero count and total (see :func:`_qc_kernels`)."""
    _sum_duplicates(x)
    nnz = np.zeros(x.shape[1], dtype=np.int64)
    if exact:
        totals = np.zeros(x.shape[1], dtype=x.dtype)
        blocks = _chunk_bounds(x.shape[1], 4 * numba.get_num_threads())
        _var_qc_csr_exact_kernel(x.indptr, x.indices, x.data, blocks, nnz, totals)
        return nnz, totals
    bounds = _chunk_bounds(x.shape[0], N_REDUCE_CHUNKS)
    acc = _acc_dtype(x.dtype)
    part_nnz = np.zeros((bounds.shape[0] - 1, x.shape[1]), dtype=np.int64)
    part_tot = np.zeros((bounds.shape[0] - 1, x.shape[1]), dtype=acc)
    totals = np.zeros(x.shape[1], dtype=acc)
    _var_qc_csr_kernel(
        x.indptr, x.indices, x.data, bounds, part_nnz, part_tot, nnz, totals
    )
    return nnz, totals.astype(_sum_dtype(x.dtype), copy=False)


@njit
def _var_qc_csr_kernel(  # noqa: PLR0917
    indptr: NDArray[np.integer],
    indices: NDArray[np.integer],
    data: NDArray[np.number],
    bounds: NDArray[np.int64],
    part_nnz: NDArray[np.int64],
    part_tot: NDArray[np.number],
    nnz: NDArray[np.int64],
    totals: NDArray[np.number],
) -> None:
    # one partial per row chunk, added up in chunk order
    for b in numba.prange(bounds.shape[0] - 1):
        for i in range(bounds[b], bounds[b + 1]):
            for k in range(indptr[i], indptr[i + 1]):
                v = data[k]
                j = indices[k]
                part_nnz[b, j] += v != 0
                part_tot[b, j] += v
    for b in range(bounds.shape[0] - 1):
        for j in range(nnz.shape[0]):
            nnz[j] += part_nnz[b, j]
            totals[j] += part_tot[b, j]


# Why this is bit-identical to the scipy code path (`axis_nnz(x, axis=0)` and
# `stats.sum(x, axis=0)` on a canonical float CSR `x`; `stats.mean` is that sum divided by `n_obs`):
# - scipy's `sum(axis=0)` is `ones @ x` with `ones` in `x.dtype` (`_base.py`, `sum`), which it
#   computes as `x.T @ ones.T` with `csc_matvec` on the transpose (`sparsetools/csc.h`): for each
#   row of `x` in order and each stored value in order, `y[col] += value * 1` in `x.dtype`,
#   starting from 0. Multiplying by 1 is exact, so each column is a sequential sum in row order.
# - Here each column block walks all rows in order and adds the same values in the same order;
#   a row's first value in the block is found by binary search (the indices are sorted).
# - A column is summed by one block alone, so the blocks may depend on the thread count.
#   `count_nonzero(axis=0)` is a `bincount` of the columns of the non-zero values: integers.
@njit
def _var_qc_csr_exact_kernel(  # noqa: PLR0917
    indptr: NDArray[np.integer],
    indices: NDArray[np.integer],
    data: NDArray[np.floating],
    col_blocks: NDArray[np.int64],
    nnz: NDArray[np.int64],
    totals: NDArray[np.floating],
) -> None:
    n_rows = indptr.shape[0] - 1
    for b in numba.prange(col_blocks.shape[0] - 1):
        j0, j1 = col_blocks[b], col_blocks[b + 1]
        for i in range(n_rows):
            s, e = indptr[i], indptr[i + 1]
            k = s + np.searchsorted(indices[s:e], j0)
            while k < e:
                j = indices[k]
                if j >= j1:
                    break
                v = data[k]
                nnz[j] += v != 0
                totals[j] += v  # in `x.dtype`, as scipy
                k += 1
