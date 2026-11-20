from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from anndata.tests.helpers import assert_equal
from fast_array_utils import stats
from scipy import sparse

import scanpy as sc
from scanpy._compat import CSCBase, DaskArray
from scanpy.preprocessing._qc import (
    describe_obs,
    describe_var,
    top_proportions,
    top_segment_proportions,
)
from testing.scanpy._helpers import (
    as_sparse_dask_matrix,
    maybe_dask_process_context,
    with_duplicates,
)
from testing.scanpy._pytest.marks import needs
from testing.scanpy._pytest.params import ARRAY_TYPES, ARRAY_TYPES_MEM

if TYPE_CHECKING:
    from scanpy._compat import CSRBase


@pytest.fixture
def adata() -> AnnData:
    rng = np.random.default_rng()
    a = rng.binomial(100, 0.005, (1000, 1000))
    adata = AnnData(
        sparse.csr_matrix(a),  # noqa: TID251
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(a.shape[0])]),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(a.shape[1])]),
    )
    return adata


def prepare_adata(adata: AnnData) -> AnnData:
    if isinstance(adata.X, DaskArray):
        adata.X = adata.X.rechunk((10, -1))
    adata.var["mito"] = np.concatenate((
        np.ones(100, dtype=bool),
        np.zeros(900, dtype=bool),
    ))
    adata.var["negative"] = False
    return adata


@pytest.fixture(params=ARRAY_TYPES)
def adata_prepared(request: pytest.FixtureRequest, adata: AnnData) -> AnnData:
    adata.X = request.param(adata.X)
    return prepare_adata(adata)


@pytest.mark.parametrize(
    "a",
    [np.ones((100, 100)), sparse.csr_matrix(np.ones((100, 100)))],  # noqa: TID251
    ids=["dense", "sparse"],
)
def test_proportions(a):
    prop = top_proportions(a, 100)
    assert (prop[:, -1] == 1).all()
    assert np.array_equal(np.sort(prop, axis=1), prop)
    assert np.apply_along_axis(lambda x: len(np.unique(x)) == 1, 0, prop).all()
    assert (prop[:, 49] == 0.5).all()


def test_segments_binary():
    a = np.concatenate([np.zeros((300, 50)), np.ones((300, 50))], 1)
    a = np.apply_along_axis(np.random.permutation, 1, a)
    seg = top_segment_proportions(a, [25, 50, 100])
    assert (seg[:, 0] == 0.5).all()
    assert (top_segment_proportions(a, [25]) == 0.5).all()
    assert (seg[:, 1] == 1.0).all()
    assert (seg[:, 2] == 1.0).all()
    segfull = top_segment_proportions(a, np.arange(100) + 1)
    propfull = top_proportions(a, 100)
    assert (segfull == propfull).all()


@pytest.mark.parametrize(
    "array_type", [*ARRAY_TYPES, pytest.param(sparse.coo_matrix, id="scipy_coo")]
)
def test_top_segments(request: pytest.FixtureRequest, array_type):
    if "dask" in array_type.__name__ and "1d_chunked" not in array_type.__name__:
        reason = "DaskArray with feature axis chunking not yet supported"
        request.applymarker(pytest.mark.xfail(reason=reason))
    a = array_type(np.ones((300, 100)))
    is_csc_dask = isinstance(a, DaskArray) and isinstance(a._meta, CSCBase)
    with (
        maybe_dask_process_context(),
        pytest.raises(ValueError, match=r"DaskArray must have csr")
        if is_csc_dask
        else nullcontext(),
    ):
        seg = top_segment_proportions(a, [50, 100])
    if is_csc_dask:
        return
    assert (seg[:, 0] == 0.5).all()
    assert (seg[:, 1] == 1.0).all()


@pytest.mark.parametrize(
    "array_type", [*ARRAY_TYPES, pytest.param(sparse.coo_matrix, id="scipy_coo")]
)
def test_top_proportions(request: pytest.FixtureRequest, array_type):
    if "dask" in array_type.__name__:
        reason = "DaskArray not yet supported"
        request.applymarker(pytest.mark.xfail(reason=reason))
    a = array_type(np.ones((300, 100)))
    # while dask is xfailed, since this relies on numba, we need to run it in the threaded context.
    with maybe_dask_process_context():
        segfull = top_segment_proportions(a, np.arange(100) + 1)
    propfull = top_proportions(a, 100)
    assert (segfull == propfull).all()


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int16, np.uint8, np.uint16, np.uint32],
    ids=lambda d: d.__name__,
)
def test_top_segments_sparse_small_int(dtype):
    """Row sums must not overflow or fail for small integer dtypes."""
    dense = (
        np.random
        .default_rng(0)
        .integers(1, min(np.iinfo(dtype).max, 1000), (3, 60))
        .astype(dtype)
    )
    ns = np.array([5, 10])
    expected = top_segment_proportions(dense, ns)
    result = top_segment_proportions(sparse.csr_matrix(dense), ns)  # noqa: TID251
    np.testing.assert_allclose(result, expected)


# While many of these are trivial,
# they’re also just making sure the metrics are there
def test_qc_metrics(adata_prepared: AnnData):
    is_csc_dask = isinstance(adata_prepared.X, DaskArray) and isinstance(
        adata_prepared.X._meta, CSCBase
    )
    with (
        maybe_dask_process_context(),
        (
            pytest.raises(ValueError, match=r"DaskArray must have csr")
            if is_csc_dask
            else nullcontext()
        ),
    ):
        sc.pp.calculate_qc_metrics(
            adata_prepared, qc_vars=["mito", "negative"], out=False
        )
    if is_csc_dask:
        return
    x = (
        adata_prepared.X.compute()
        if isinstance(adata_prepared.X, DaskArray)
        else adata_prepared.X
    )
    max_x = x.max(axis=0)
    if isinstance(max_x, sparse.coo_matrix | sparse.coo_array):
        max_x = max_x.toarray()
    elif isinstance(max_x, DaskArray):
        max_x = max_x.compute()
    assert (adata_prepared.obs["n_genes_by_counts"] < adata_prepared.shape[1]).all()
    assert (
        adata_prepared.obs["n_genes_by_counts"]
        >= adata_prepared.obs["log1p_n_genes_by_counts"]
    ).all()
    assert (
        adata_prepared.obs["total_counts"] == stats.sum(adata_prepared.X, axis=1)
    ).all()
    assert (
        adata_prepared.obs["total_counts"] >= adata_prepared.obs["log1p_total_counts"]
    ).all()
    assert (
        adata_prepared.obs["total_counts_mito"]
        >= adata_prepared.obs["log1p_total_counts_mito"]
    ).all()
    assert (adata_prepared.obs["total_counts_negative"] == 0).all()
    assert (
        adata_prepared.obs["pct_counts_in_top_50_genes"]
        <= adata_prepared.obs["pct_counts_in_top_100_genes"]
    ).all()
    for col in filter(lambda x: "negative" not in x, adata_prepared.obs.columns):
        assert (adata_prepared.obs[col] >= 0).all()  # Values should be positive or zero
        assert (adata_prepared.obs[col] != 0).any().all()  # Nothing should be all zeros
        if col.startswith("pct_counts_in_top"):
            assert (adata_prepared.obs[col] <= 100).all()
            assert (adata_prepared.obs[col] >= 0).all()
    for col in adata_prepared.var.columns:
        assert (adata_prepared.var[col] >= 0).all()
    assert (adata_prepared.var["mean_counts"] < np.ravel(max_x)).all()
    assert (
        adata_prepared.var["mean_counts"] >= adata_prepared.var["log1p_mean_counts"]
    ).all()
    assert (
        adata_prepared.var["total_counts"] >= adata_prepared.var["log1p_total_counts"]
    ).all()


def test_qc_metrics_idempotent(adata_prepared: AnnData):
    is_csc_dask = isinstance(adata_prepared.X, DaskArray) and isinstance(
        adata_prepared.X._meta, CSCBase
    )
    with (
        maybe_dask_process_context(),
        (
            pytest.raises(ValueError, match=r"DaskArray must have csr")
            if is_csc_dask
            else nullcontext()
        ),
    ):
        sc.pp.calculate_qc_metrics(
            adata_prepared, qc_vars=["mito", "negative"], out=False
        )
        old_obs, old_var = adata_prepared.obs.copy(), adata_prepared.var.copy()
        sc.pp.calculate_qc_metrics(
            adata_prepared, qc_vars=["mito", "negative"], out=False
        )
    if is_csc_dask:
        return
    assert set(adata_prepared.obs.columns) == set(old_obs.columns)
    assert set(adata_prepared.var.columns) == set(old_var.columns)
    for col in adata_prepared.obs:
        assert np.allclose(adata_prepared.obs[col], old_obs[col])
    for col in adata_prepared.var:
        assert np.allclose(adata_prepared.var[col], old_var[col])


def test_qc_metrics_no_log1p(adata_prepared: AnnData):
    with (
        maybe_dask_process_context(),
        (
            pytest.raises(ValueError, match=r"DaskArray must have csr")
            if isinstance(adata_prepared.X, DaskArray)
            and isinstance(adata_prepared.X._meta, CSCBase)
            else nullcontext()
        ),
    ):
        sc.pp.calculate_qc_metrics(
            adata_prepared, qc_vars=["mito", "negative"], log1p=False, out=False
        )
    assert not np.any(adata_prepared.obs.columns.str.startswith("log1p_"))
    assert not np.any(adata_prepared.var.columns.str.startswith("log1p_"))


@needs.dask
@pytest.mark.parametrize("log1p", [True, False], ids=["log1p", "no_log1p"])
def test_dask_against_in_memory(adata, log1p):
    adata_as_dask = adata.copy()
    adata_as_dask.X = as_sparse_dask_matrix(adata.X)
    adata = prepare_adata(adata)
    adata_as_dask = prepare_adata(adata_as_dask)
    with maybe_dask_process_context():
        sc.pp.calculate_qc_metrics(
            adata_as_dask, qc_vars=["mito", "negative"], log1p=log1p, out=False
        )
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mito", "negative"], log1p=log1p, out=False
    )
    assert_equal(adata, adata_as_dask)


@pytest.fixture
def adata_mito() -> AnnData:
    rng = np.random.default_rng()
    return AnnData(
        X=rng.binomial(100, 0.005, (1000, 1000)),
        var=dict(
            mito=np.concatenate((np.ones(100, dtype=bool), np.zeros(900, dtype=bool)))
        ),
    )


@pytest.mark.parametrize("cls", ARRAY_TYPES_MEM)
@pytest.mark.parametrize("qc_var_param", ["mito", ["mito"]], ids=["str", "list"])
def test_qc_metrics_format(
    cls, adata_mito: AnnData, qc_var_param: list[str] | str
) -> None:
    var = adata_mito.var.copy()
    sc.pp.calculate_qc_metrics(adata_mito, qc_vars=qc_var_param, out=False)
    adata = AnnData(X=cls(adata_mito.X), var=var)
    sc.pp.calculate_qc_metrics(adata, qc_vars=qc_var_param, out=False)
    assert np.allclose(adata.obs, adata_mito.obs)
    for col in adata.var:  # np.allclose doesn't like mix of types
        assert np.allclose(adata.var[col], adata_mito.var[col])


def test_qc_metrics_percentage(adata_mito: AnnData) -> None:  # In response to #421
    sc.pp.calculate_qc_metrics(adata_mito, percent_top=[])
    sc.pp.calculate_qc_metrics(adata_mito, percent_top=())
    sc.pp.calculate_qc_metrics(adata_mito, percent_top=None)
    sc.pp.calculate_qc_metrics(adata_mito, percent_top=[1, 2, 3, 10])
    sc.pp.calculate_qc_metrics(adata_mito, percent_top=[1])
    with pytest.raises(IndexError):
        sc.pp.calculate_qc_metrics(adata_mito, percent_top=[1, 2, 3, -5])
    with pytest.raises(IndexError):
        sc.pp.calculate_qc_metrics(adata_mito, percent_top=[20, 30, 1001])


@needs.anndata_acc
@pytest.mark.filterwarnings("ignore:.*use_raw is deprecated:FutureWarning")
def test_layer_raw(adata: AnnData):
    adata = adata.copy()
    adata.raw = adata.copy()
    adata.layers["counts"] = adata.X.copy()
    obs_orig, var_orig = sc.pp.calculate_qc_metrics(adata)
    sc.pp.log1p(adata)  # To be sure they aren't reusing it
    obs_layer, var_layer = sc.pp.calculate_qc_metrics(adata, use="layers.counts")
    obs_raw, var_raw = sc.pp.calculate_qc_metrics(adata, use_raw=True)
    assert np.allclose(obs_orig, obs_layer)
    assert np.allclose(obs_orig, obs_raw)
    assert np.allclose(var_orig, var_layer)
    assert np.allclose(var_orig, var_raw)


def test_inner_methods(adata: AnnData):
    adata = adata.copy()
    full_inplace = adata.copy()
    partial_inplace = adata.copy()
    obs_orig, var_orig = sc.pp.calculate_qc_metrics(adata)
    assert np.all(obs_orig == describe_obs(adata))
    assert np.all(var_orig == describe_var(adata))
    sc.pp.calculate_qc_metrics(full_inplace, out=False)
    describe_obs(partial_inplace, inplace=True)
    describe_var(partial_inplace, inplace=True)
    assert np.all(full_inplace.obs == partial_inplace.obs)
    assert np.all(full_inplace.var == partial_inplace.var)
    assert np.all(partial_inplace.obs[obs_orig.columns] == obs_orig)
    assert np.all(partial_inplace.var[var_orig.columns] == var_orig)


@needs.anndata_acc
def test_qc_metrics_out_acc(adata: AnnData) -> None:
    """A single accessor writes both tables as whole DataFrames named after it."""
    from anndata.acc import A

    adata = prepare_adata(adata)
    obs, var = sc.pp.calculate_qc_metrics(adata, qc_vars=["mito"], out=True)
    obs_cols, var_cols = set(adata.obs), set(adata.var)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mito"], out=A.obsm["qc"])
    pd.testing.assert_frame_equal(adata.obsm["qc"], obs)
    pd.testing.assert_frame_equal(adata.varm["qc"], var)
    assert set(adata.obs) == obs_cols
    assert set(adata.var) == var_cols


@needs.anndata_acc
def test_qc_metrics_out_mapping(adata: AnnData) -> None:
    """A mapping writes only the tables it names, as whole `.obsm`/`.varm` DataFrames."""
    from anndata.acc import A

    adata = prepare_adata(adata)
    obs_before, var_before = adata.obs.copy(), adata.var.copy()
    obs, var = sc.pp.calculate_qc_metrics(adata, qc_vars=["mito"], out=True)
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mito"], out={"obs": A.obsm["qc"], "var": "varm.qc"}
    )
    pd.testing.assert_frame_equal(adata.obsm["qc"], obs)
    pd.testing.assert_frame_equal(adata.varm["qc"], var)
    pd.testing.assert_frame_equal(adata.obs, obs_before)
    pd.testing.assert_frame_equal(adata.var, var_before)

    sc.pp.calculate_qc_metrics(adata, out={"obs": A.obsm["only_obs"]})
    assert "only_obs" not in adata.varm
    with pytest.raises(TypeError, match=r"must be written to one of \['obsm'\]"):
        sc.pp.calculate_qc_metrics(adata, out={"obs": A.obs["x"]})


def test_qc_metrics_fast_thread_count_independent() -> None:
    """Without reference numerics, float QC sums use fixed chunks: the bits don't depend on the thread count."""
    import numba

    from scanpy._utils._numba import _numba_thread_limit

    n_threads = min(4, numba.config.NUMBA_NUM_THREADS)
    if n_threads == 1:
        pytest.skip("needs several threads")
    x = sparse.random(5000, 60, density=0.1, format="csr", dtype=np.float32, rng=0)
    var = pd.DataFrame({"mt": np.arange(60) < 5}, index=[f"g{i}" for i in range(60)])
    results = []
    for n in (1, n_threads):
        with sc.settings.override(reference_numerics=False), _numba_thread_limit(n):
            results.append(
                sc.pp.calculate_qc_metrics(
                    AnnData(x.copy(), var=var), qc_vars=["mt"], percent_top=None
                )
            )
    for one, many in zip(*results, strict=True):
        pd.testing.assert_frame_equal(one, many, check_exact=True)


@pytest.mark.parametrize("deterministic", [True, False], ids=["det", "fast"])
def test_qc_metrics_sum_duplicate_entries(*, deterministic: bool) -> None:
    x = sparse.random(300, 40, density=0.2, format="csr", dtype=np.float64, rng=0)
    x.data = np.ceil(x.data * 10)
    dup = with_duplicates(x)
    assert not dup.has_canonical_format
    var = pd.DataFrame({"mt": np.arange(40) < 5}, index=[f"g{i}" for i in range(40)])
    with sc.settings.override(reference_numerics=deterministic):
        expected, got = (
            sc.pp.calculate_qc_metrics(
                AnnData(m, var=var), qc_vars=["mt"], percent_top=[5]
            )
            for m in (x, dup)
        )
    for e, g in zip(expected, got, strict=True):
        pd.testing.assert_frame_equal(e, g)


def _qc_reference(
    x: CSRBase, mask: np.ndarray
) -> tuple[dict[str, np.ndarray], CSRBase]:
    """Compute the metrics as `calculate_qc_metrics` did before its kernels, in the same order."""
    from scanpy._utils import axis_nnz

    x = x.copy()
    x.eliminate_zeros()
    ref = dict(
        n_genes_by_counts=axis_nnz(x, axis=1),  # scipy sums duplicates in place here
        total_counts=stats.sum(x, axis=1),
        total_counts_mt=stats.sum(x[:, mask], axis=1),
        n_cells_by_counts=axis_nnz(x, axis=0),
        mean_counts=stats.mean(x, axis=0),
        total_counts_var=stats.sum(x, axis=0),
    )
    return ref, x


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.uint16])
@pytest.mark.parametrize("duplicates", [False, True], ids=["canonical", "dups"])
def test_qc_metrics_deterministic_matches_scipy(
    dtype: type, *, duplicates: bool
) -> None:
    """With `settings.reference_numerics`, metrics are bit-identical to the scipy sums, in their dtypes.

    Integer sums are exact in any order; float sums repeat scipy's operations in its order.
    Fast mode agrees to rounding.
    """
    rng = np.random.default_rng(0)
    x = sparse.random(400, 60, density=0.2, format="csr", dtype=np.float64, rng=rng)
    x.data = np.ceil(x.data * 30)
    if dtype not in {np.int32, np.uint16}:  # values that aren't exact sums of counts
        x.data *= rng.random(x.data.shape) + 0.5
    x = x.astype(dtype)
    x.data[::11] = 0  # explicit zeros
    if duplicates:
        x = with_duplicates(x, split=False)
    mask = np.arange(60) < 7
    var = pd.DataFrame({"mt": mask}, index=[f"g{i}" for i in range(60)])
    ref, x_ref = _qc_reference(x, mask)

    adata = AnnData(x.copy(), var=var)
    obs, var_df = sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False
    )
    got = dict(
        n_genes_by_counts=obs["n_genes_by_counts"],
        total_counts=obs["total_counts"],
        total_counts_mt=obs["total_counts_mt"],
        n_cells_by_counts=var_df["n_cells_by_counts"],
        mean_counts=var_df["mean_counts"],
        total_counts_var=var_df["total_counts"],
    )
    for key, expected in ref.items():
        actual = got[key].to_numpy()
        assert actual.dtype == expected.dtype, key
        np.testing.assert_array_equal(actual, expected, err_msg=key)
    # X is changed in place exactly as the scipy code path changed it
    for attr in ("data", "indices", "indptr"):
        np.testing.assert_array_equal(getattr(adata.X, attr), getattr(x_ref, attr))

    with sc.settings.override(reference_numerics=False):
        obs_f, var_f = sc.pp.calculate_qc_metrics(
            AnnData(x.copy(), var=var), qc_vars=["mt"], percent_top=None, log1p=False
        )
    rtol = 1e-5
    for col in ("n_genes_by_counts", "total_counts", "total_counts_mt"):
        np.testing.assert_allclose(obs_f[col], obs[col], rtol=rtol)
    for col in ("n_cells_by_counts", "mean_counts", "total_counts"):
        np.testing.assert_allclose(var_f[col], var_df[col], rtol=rtol)


@pytest.mark.parametrize("deterministic", [True, False], ids=["det", "fast"])
@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
def test_qc_metrics_many_qc_vars(*, deterministic: bool) -> None:
    """More `qc_vars` than fit in the kernel's int64 bit flags fall back correctly."""
    x = sparse.random(50, 80, density=0.3, format="csr", dtype=np.float64, rng=0)
    var = pd.DataFrame(
        {f"q{i}": np.arange(80) == i for i in range(70)},
        index=[f"g{i}" for i in range(80)],
    )
    adata = AnnData(x, var=var)
    with sc.settings.override(reference_numerics=deterministic):
        obs, _ = sc.pp.calculate_qc_metrics(
            adata, qc_vars=list(var.columns), percent_top=None
        )
    for i in (0, 63, 64, 69):
        np.testing.assert_allclose(obs[f"total_counts_q{i}"], x[:, i].toarray().ravel())


def _ragged_csr(
    dtype: type, container: type, *, rng: np.random.Generator, n_vars: int = 4000
):
    """Rows of every length around numpy's pairwise-summation blocks (8, 128 and their multiples).

    Values span orders of magnitude, so any other summation order changes the last bits.
    Every tenth column stays empty.
    """
    special = [0, 1, 2, 7, 8, 9, 16, 17, 127, 128, 129, 130, 136, 255, 256, 257]
    special += [258, 1000, 1031, 3000, 0, 5]
    lengths = np.r_[special, rng.integers(0, 400, size=300)]
    pool = np.flatnonzero(np.arange(n_vars) % 10 != 3)
    indices = [np.sort(rng.choice(pool, size=n, replace=False)) for n in lengths]
    data = rng.lognormal(0, 3, size=lengths.sum()).astype(dtype)
    indptr = np.r_[0, np.cumsum(lengths)]
    return container(
        (data, np.concatenate(indices), indptr), shape=(len(lengths), n_vars)
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("container", [sparse.csr_matrix, sparse.csr_array])  # noqa: TID251
@pytest.mark.parametrize("n_threads", [1, 3, None], ids=["1", "3", "all"])
def test_qc_metrics_reference_kernels_match_scipy(
    dtype: type, container: type, n_threads: int | None
) -> None:
    """With `settings.reference_numerics`, float sums are bit-identical to scipy's, for any thread count.

    Row sums are `np.add.reduceat` (pairwise summation), column sums sequential in row order.
    """
    from scanpy._utils import axis_nnz
    from scanpy._utils._numba import _numba_thread_limit

    rng = np.random.default_rng(0)
    x = _ragged_csr(dtype, container, rng=rng)
    n_vars = x.shape[1]
    masks = dict(
        few=np.isin(np.arange(n_vars), rng.choice(n_vars, 40, replace=False)),
        many=rng.random(n_vars) < 0.8,  # long masked rows: pairwise blocks too
        none=np.zeros(n_vars, dtype=bool),
    )
    ref = x.copy()
    expected_obs = dict(
        n_genes_by_counts=axis_nnz(ref, axis=1),
        total_counts=stats.sum(ref, axis=1),
        **{f"total_counts_{k}": stats.sum(ref[:, m], axis=1) for k, m in masks.items()},
    )
    expected_var = dict(
        n_cells_by_counts=axis_nnz(ref, axis=0),
        mean_counts=stats.mean(ref, axis=0),
        total_counts=stats.sum(ref, axis=0),
    )

    adata = AnnData(x, var=pd.DataFrame(masks, index=[f"g{i}" for i in range(n_vars)]))
    with _numba_thread_limit(n_threads):
        obs, var = sc.pp.calculate_qc_metrics(
            adata, qc_vars=list(masks), percent_top=None, log1p=False
        )
    for df, expected in ((obs, expected_obs), (var, expected_var)):
        for key, e in expected.items():
            actual = df[key].to_numpy()
            assert actual.dtype == e.dtype, key
            np.testing.assert_array_equal(actual, e, err_msg=key)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_qc_metrics_reference_non_bool_qc_var(dtype: type) -> None:
    """A non-boolean qc_var column indexes genes by value in scipy: the reference path keeps that."""
    x = sparse.random(50, 20, density=0.5, format="csr", dtype=dtype, rng=0)
    flags = (np.arange(20) < 3).astype(
        np.int64
    )  # as indices: gene 1 three times, gene 0 17 times
    adata = AnnData(
        x, var=pd.DataFrame({"q": flags}, index=[f"g{i}" for i in range(20)])
    )
    obs, _ = sc.pp.calculate_qc_metrics(adata, qc_vars=["q"], percent_top=None)
    np.testing.assert_array_equal(
        obs["total_counts_q"].to_numpy(), stats.sum(x[:, flags], axis=1)
    )


@pytest.mark.parametrize("dtype", [np.float32, np.int64])
@pytest.mark.parametrize("n_threads", [1, 3, None], ids=["1", "3", "all"])
def test_qc_metrics_fused_pass(
    monkeypatch: pytest.MonkeyPatch, dtype: type, n_threads: int | None
) -> None:
    """With the fast kernels, one pass over `X` gives exactly what the per-table passes give."""
    from scanpy._utils._numba import _numba_thread_limit
    from scanpy.preprocessing import _qc

    monkeypatch.setattr(sc.settings, "reference_numerics", False)
    rng = np.random.default_rng(0)
    x = _ragged_csr(dtype, sparse.csr_matrix, rng=rng)  # noqa: TID251
    var = pd.DataFrame(
        dict(mt=rng.random(x.shape[1]) < 0.1, ribo=rng.random(x.shape[1]) < 0.3),
        index=[f"g{i}" for i in range(x.shape[1])],
    )
    kw = dict(qc_vars=["mt", "ribo"], percent_top=None)
    with _numba_thread_limit(n_threads):
        fused = sc.pp.calculate_qc_metrics(AnnData(x.copy(), var=var), **kw, out=True)
        monkeypatch.setattr(_qc, "_qc_csr_fast", lambda *_: pytest.fail("fused"))
        adata = AnnData(x.copy(), var=var)
        obs = _qc.describe_obs(adata, **kw)
        var_metrics = _qc.describe_var(adata)
    pd.testing.assert_frame_equal(fused.obs, obs, check_exact=True)
    pd.testing.assert_frame_equal(fused.var, var_metrics, check_exact=True)
