from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_raises

import scanpy as sc
from testing.scanpy._helpers.data import pbmc68k_reduced
from testing.scanpy._pytest.marks import needs

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from anndata import AnnData


@pytest.mark.parametrize(
    ("out", "key_obsm", "key_uns"),
    [
        pytest.param(False, "X_tsne", "tsne", id="default"),
        pytest.param(
            "obsm.custom", "custom", "custom", marks=needs.anndata_acc, id="acc"
        ),
    ],
)
def test_tsne(*, out: str | bool, key_obsm: str, key_uns: str):
    pbmc = pbmc68k_reduced()[:200].copy()

    euclidean1, euclidean2, cosine = (pbmc.copy() for _ in range(3))
    sc.tl.tsne(euclidean1, metric="euclidean")
    sc.tl.tsne(euclidean2, metric="euclidean", n_jobs=2, out=out)
    sc.tl.tsne(cosine, metric="cosine")

    # Reproducibility
    np.testing.assert_equal(euclidean1.obsm["X_tsne"], euclidean2.obsm[key_obsm])
    # Metric has some effect
    assert not np.array_equal(euclidean1.obsm["X_tsne"], cosine.obsm["X_tsne"])

    # Params are recorded
    assert euclidean1.uns["tsne"]["params"]["n_jobs"] == sc.settings.n_jobs
    assert euclidean2.uns[key_uns]["params"]["n_jobs"] == 2
    assert cosine.uns["tsne"]["params"]["n_jobs"] == sc.settings.n_jobs
    assert euclidean1.uns["tsne"]["params"]["metric"] == "euclidean"
    assert euclidean2.uns[key_uns]["params"]["metric"] == "euclidean"
    assert cosine.uns["tsne"]["params"]["metric"] == "cosine"


@pytest.mark.parametrize(
    ("out", "key_obsm", "key_uns"),
    [
        pytest.param(False, "X_umap", "umap", id="default"),
        pytest.param(
            "obsm.custom", "custom", "custom", marks=needs.anndata_acc, id="acc"
        ),
    ],
)
def test_umap_init_dtype(*, out: str | bool, key_obsm: str, key_uns: str):
    pbmc1 = pbmc68k_reduced()[:100, :].copy()
    pbmc2 = pbmc1.copy()
    for pbmc, dtype, o in [(pbmc1, np.float32, False), (pbmc2, np.float64, out)]:
        sc.tl.umap(pbmc, init_pos=pbmc.obsm["X_pca"][:, :2].astype(dtype), out=o)

    # check that embeddings are close for different dtypes
    assert_array_almost_equal(pbmc1.obsm["X_umap"], pbmc2.obsm[key_obsm])

    # check that params are recorded
    assert pbmc1.uns["umap"]["params"]["a"] == pbmc2.uns[key_uns]["params"]["a"]
    assert pbmc1.uns["umap"]["params"]["b"] == pbmc2.uns[key_uns]["params"]["b"]


@pytest.mark.parametrize(
    "layout",
    [
        pytest.param("fa", marks=needs.fa2),
        pytest.param("fr", marks=needs.igraph),
    ],
)
def test_umap_init_paga(layout):
    pbmc = pbmc68k_reduced()[:100, :].copy()
    sc.tl.paga(pbmc)
    sc.pl.paga(pbmc, layout=layout, show=False)
    sc.tl.umap(pbmc, init_pos="paga")


@pytest.mark.parametrize(
    "preset",
    [
        pytest.param(sc.Preset.ScanpyV1, id="v1"),
        # umap-learn ≥0.6’s new layout is compiled for float32 graphs only
        pytest.param(sc.Preset.ScanpyV2Preview, marks=needs.scanpy2, id="v2"),
    ],
)
def test_umap_preserves_connectivities(preset: sc.Preset) -> None:
    # https://github.com/scverse/scanpy/issues/4028
    pbmc = pbmc68k_reduced()[:100, :].copy()  # load before preset: cached
    conn = pbmc.obsp["connectivities"]
    assert conn.dtype == np.float64
    data_before = conn.data.copy()
    nnz_before = conn.nnz

    sc.settings.preset = preset
    sc.tl.umap(pbmc)

    assert_array_equal(data_before, conn.data)
    assert conn.nnz == nnz_before
    assert (conn.data == 0).sum() == 0, "CSR should have no explicit zeros"
    assert ("X_umap" if preset is sc.Preset.ScanpyV1 else "umap") in pbmc.obsm


@pytest.mark.parametrize(
    "preset",
    [
        pytest.param(sc.Preset.ScanpyV1, id="v1"),
        pytest.param(sc.Preset.ScanpyV2Preview, marks=needs.scanpy2, id="v2"),
    ],
)
def test_umap_init_pos_unknown(preset: sc.Preset) -> None:
    pbmc = pbmc68k_reduced()[:100, :].copy()  # load before preset: cached
    sc.settings.preset = preset
    with pytest.raises(ValueError, match=r"Unknown `init_pos` 'spectal'"):
        sc.tl.umap(pbmc, init_pos="spectal")


def _knn_preservation(emb: np.ndarray, conns, k: int = 10) -> float:
    from sklearn.neighbors import NearestNeighbors

    _, nn = NearestNeighbors(n_neighbors=k + 1).fit(emb).kneighbors(emb)
    hits = [
        len(set(nn[i, 1:]) & set(conns[i].indices[np.argsort(-conns[i].data)[:k]])) / k
        for i in range(emb.shape[0])
    ]
    return float(np.mean(hits))


@pytest.mark.parametrize("rng", [0, None], ids=["deterministic", "hogwild"])
def test_umap_numba(rng: int | None) -> None:
    from scanpy._utils._numba import _numba_thread_limit

    pbmc = pbmc68k_reduced()
    sc.pp.neighbors(pbmc, random_state=0)
    ref = sc.tl.umap(pbmc, rng=0, out=True)
    emb = sc.tl.umap(pbmc, method="numba", rng=rng, out=True)
    assert emb.shape == (pbmc.n_obs, 2)
    assert np.isfinite(emb).all()
    conns = pbmc.obsp["connectivities"]
    # local structure as well preserved as by umap-learn (≈ 0.5 here)
    assert _knn_preservation(emb, conns) > 0.9 * _knn_preservation(ref, conns)
    if rng is not None:
        with _numba_thread_limit(1):
            serial = sc.tl.umap(pbmc, method="numba", rng=rng, out=True)
        assert_array_equal(emb, serial)


def test_umap_numba_array_init() -> None:
    pbmc = pbmc68k_reduced()[:200].copy()
    init = pbmc.obsm["X_pca"][:, :2].copy()
    emb = sc.tl.umap(pbmc, method="numba", init_pos=init, rng=0, out=True)
    assert emb.shape == (pbmc.n_obs, 2)


def test_umap_numba_unsupported_init() -> None:
    pbmc = pbmc68k_reduced()[:100].copy()
    with pytest.raises(ValueError, match=r"does not support init_pos"):
        sc.tl.umap(pbmc, method="numba", init_pos="recursive", rng=0)


@pytest.mark.parametrize(
    ("rng", "expected"),
    [pytest.param(0, True, id="seed"), pytest.param(None, False, id="rng_none")],
)
def test_umap_numba_mode(
    monkeypatch: pytest.MonkeyPatch, *, rng: int | None, expected: bool
) -> None:
    """A seed selects the edge-coloured (deterministic) layout, `rng=None` Hogwild."""
    from scanpy.tools import _umap_layout

    seen: list[bool] = []
    orig = _umap_layout.optimize_layout

    def spy(*args, deterministic: bool, **kwargs):
        seen.append(deterministic)
        return orig(*args, deterministic=deterministic, **kwargs)

    monkeypatch.setattr(_umap_layout, "optimize_layout", spy)
    pbmc = pbmc68k_reduced()[:100].copy()
    sc.tl.umap(pbmc, method="numba", rng=rng, maxiter=5)
    assert seen == [expected]


def _connected_knn(x: np.ndarray, *, k: int = 10):
    """A symmetric float32 kNN graph of `x` (like `pp.neighbors`’), plus a ring through all points so it’s connected."""
    from scipy import sparse
    from sklearn.neighbors import kneighbors_graph

    n = x.shape[0]
    g = kneighbors_graph(x, k, mode="distance")
    g.data = np.exp(-g.data / g.data.mean())
    i = np.arange(n)
    ring = sparse.coo_matrix((np.full(n, 0.1), (i, (i + 1) % n)), shape=(n, n))
    g = g + ring
    return ((g + g.T) / 2).astype(np.float32).tocsr()


# blob sizes and number of connected components
_SPECTRAL_CASES = {
    "connected": ([300], 1),
    "two_components": ([200, 150], 2),
    # plus a 3-vertex path and an isolated vertex, which get random positions
    "many_components": ([200, 150, 120], 5),
}


def _spectral_inputs(case: str):
    """Data and its graph, with components as in `_SPECTRAL_CASES`."""
    from scipy import sparse

    rng = np.random.default_rng(0)
    sizes, _ = _SPECTRAL_CASES[case]
    blobs = [rng.normal(size=(n, 5)) + 100 * c for c, n in enumerate(sizes)]
    graphs = [_connected_knn(b) for b in blobs]
    if case == "many_components":
        path = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)
        single = sparse.coo_matrix((1, 1), dtype=np.float32)
        graphs += [sparse.coo_matrix(path), single]
        blobs += [rng.normal(size=(3, 5)) - 100, rng.normal(size=(1, 5)) - 200]
    return np.vstack(blobs), sparse.block_diag(graphs, format="csr")


@pytest.mark.parametrize("case", _SPECTRAL_CASES)
def test_spectral_layout_matches_umap(case: str) -> None:
    from scipy.sparse.csgraph import connected_components
    from umap.spectral import spectral_layout as umap_spectral_layout

    from scanpy.tools._spectral import spectral_layout

    x, g = _spectral_inputs(case)
    assert connected_components(g)[0] == _SPECTRAL_CASES[case][1]
    expected = umap_spectral_layout(
        x, g, 2, np.random.RandomState(0), compatibility_layout=False
    )
    actual = spectral_layout(x, g, 2, np.random.RandomState(0))
    assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


def test_spectral_layout_lobpcg_matches_umap() -> None:
    """LOBPCG, which umap-learn uses from 2M vertices on."""
    from umap.spectral import _spectral_layout as umap_spectral_layout

    from scanpy.tools._spectral import spectral_layout

    x, g = _spectral_inputs("connected")
    expected = umap_spectral_layout(
        x, g, 2, np.random.RandomState(0), method="lobpcg", compatibility_layout=False
    )
    actual = spectral_layout(x, g, 2, np.random.RandomState(0), method="lobpcg")
    assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("shape", [(500,), (500, 3)], ids=["vector", "block"])
def test_csr_operator_matches_scipy(
    dtype: type, index_dtype: type, shape: tuple[int, ...]
) -> None:
    """Parallel products give scipy’s bits, also for unsorted and duplicate indices and negative values."""
    import platform

    from scipy import sparse

    from scanpy.tools._spectral import _CsrOperator, _rounds_like_scipy

    if platform.machine().lower() in {"x86_64", "amd64"}:
        # scipy’s x86-64 builds don’t fuse multiply-adds, so the parallel path must be used
        assert _rounds_like_scipy(np.dtype(dtype))
    rng = np.random.default_rng(0)
    n, per_row = 500, 20
    m = sparse.csr_matrix(  # noqa: TID251 (building a test input)
        (
            rng.standard_normal(n * per_row).astype(dtype),
            rng.integers(0, n, n * per_row),
            np.arange(0, n * per_row + 1, per_row),
        ),
        shape=(n, n),
    )
    # the constructor picks the smallest index dtype
    m.indices, m.indptr = m.indices.astype(index_dtype), m.indptr.astype(index_dtype)
    x = rng.standard_normal(shape).astype(dtype)
    op = _CsrOperator(m)
    actual = op @ x
    assert_array_equal(actual, m @ x)
    assert actual.dtype == dtype
    assert op.n_products == 1


@pytest.mark.parametrize(
    "init_pos",
    [
        pytest.param("spectral", id="spectral"),
        pytest.param(np.zeros((100, 2), np.float32), id="array"),
    ],
)
def test_umap_numba_n_components(
    monkeypatch: pytest.MonkeyPatch, init_pos: str | np.ndarray
) -> None:
    from scanpy.tools import _umap

    def no_init(*_, **__):
        pytest.fail("init computed before validating n_components")

    monkeypatch.setattr(_umap, "_init_coords", no_init)
    pbmc = pbmc68k_reduced()[:100].copy()
    with pytest.raises(ValueError, match=r"n_components=2 only"):
        sc.tl.umap(pbmc, method="numba", n_components=3, init_pos=init_pos, rng=0)


def test_umap_numba_init_columns() -> None:
    pbmc = pbmc68k_reduced()[:100].copy()
    init = pbmc.obsm["X_pca"][:, :3].copy()
    with pytest.raises(ValueError, match=r"`init_pos` has 3 columns"):
        sc.tl.umap(pbmc, method="numba", init_pos=init, rng=0)


def test_umap_numba_compatibility_layout_warns() -> None:
    pbmc = pbmc68k_reduced()[:100].copy()
    with pytest.warns(UserWarning, match=r"`compatibility_layout` has no effect"):
        sc.tl.umap(pbmc, method="numba", compatibility_layout=True, rng=0, maxiter=5)


def test_umap_numba_constant_init() -> None:
    pbmc = pbmc68k_reduced()[:100].copy()
    init = np.ones((pbmc.n_obs, 2), np.float32)
    init[:, 1] = np.arange(pbmc.n_obs)  # one constant column, one not
    emb = sc.tl.umap(pbmc, method="numba", init_pos=init, rng=0, maxiter=5, out=True)
    assert np.isfinite(emb).all()
    emb = sc.tl.umap(
        pbmc, method="numba", init_pos=np.ones_like(init), rng=0, maxiter=5, out=True
    )
    assert np.isfinite(emb).all()


@pytest.mark.parametrize("deterministic", [True, False])
def test_umap_layout_all_explicit_zeros(*, deterministic: bool) -> None:
    from scipy import sparse

    from scanpy.tools._umap_layout import optimize_layout

    # 0 – 1 – 2 path, all values stored as explicit zeros
    g = sparse.csr_matrix(  # noqa: TID251 (building a test input)
        (np.zeros(4, np.float32), [1, 0, 2, 1], [0, 1, 3, 4]), shape=(3, 3)
    )
    init = np.random.default_rng(0).random((3, 2)).astype(np.float32)
    pos = optimize_layout(
        init, g, n_epochs=10, a=1.0, b=1.0, deterministic=deterministic
    )
    assert_array_equal(pos, init)


def test_edge_coloring_matches_greedy() -> None:
    """Per-vertex bitsets give the colours of a plain greedy colouring, on a graph with hubs."""
    from itertools import count

    from scanpy.tools._umap_layout import _edge_coloring

    rng = np.random.default_rng(0)
    n = 400
    hubs = {(h, v) for h in range(3) for v in range(h + 1, n) if rng.random() < 0.8}
    rand = {
        (int(min(u, v)), int(max(u, v)))
        for u, v in rng.integers(0, n, (800, 2))
        if u != v
    }
    us, vs = np.array(sorted(hubs | rand), dtype=np.int64).T
    deg = np.bincount(us, minlength=n) + np.bincount(vs, minlength=n)
    first = np.argsort(-np.maximum(deg[us], deg[vs]), kind="stable")
    heads, tails = us[first], vs[first]

    order, offsets = _edge_coloring(heads, tails, deg)
    colors = np.empty(len(heads), dtype=np.int64)
    colors[order] = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))

    used: list[set[int]] = [set() for _ in range(n)]
    expected = []
    for u, v in zip(heads.tolist(), tails.tolist(), strict=True):
        c = next(c for c in count() if c not in used[u] and c not in used[v])
        used[u].add(c)
        used[v].add(c)
        expected.append(c)
    assert_array_equal(colors, expected)
    assert max(expected) >= 2 * 64  # the colours span several bitset words
    # within a colour, edges keep their input order
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        assert (np.diff(order[lo:hi]) > 0).all()


@pytest.mark.parametrize("rng_arg", ["rng", "random_state"])
def test_diffmap(
    subtests: pytest.Subtests, rng_arg: Literal["rng", "random_state"]
) -> None:
    pbmc = pbmc68k_reduced()

    d1, d2, d3 = (
        sc.tl.diffmap(pbmc, out=True, **{rng_arg: seed}) for seed in (0, 0, 1234)
    )

    with subtests.test("reproducible"):
        assert_array_equal(d1, d2)
    with subtests.test("different embedding"):
        assert_raises(AssertionError, assert_array_equal, d1, d3)


@pytest.mark.parametrize(
    ("out", "key_obsm", "key_uns", "is_dict"),
    [
        pytest.param(False, "X_diffmap", "diffmap_evals", False, id="default"),
        pytest.param(
            *("obsm.custom", "custom", "custom", True),
            marks=needs.anndata_acc,
            id="acc",
        ),
        pytest.param(sc.Preset.ScanpyV1, "X_diffmap", "diffmap_evals", False, id="v1"),
        pytest.param(
            *(sc.Preset.ScanpyV2Preview, "diffmap", "diffmap", True),
            marks=needs.scanpy2,
            id="v2",
        ),
    ],
)
def test_diffmap_out(
    *,
    out: str | sc.Preset | bool,
    key_obsm: str,
    key_uns: str,
    is_dict: bool,
) -> None:
    adata = pbmc68k_reduced()[:300, :100].copy()
    if isinstance(out, sc.Preset):
        sc.settings.preset = out
        out = False
    sc.tl.diffmap(adata, out=out)
    assert key_obsm in adata.obsm
    assert key_uns in adata.uns
    assert isinstance(adata.uns[key_uns], dict if is_dict else np.ndarray)


@needs.igraph
@pytest.mark.parametrize(
    ("out", "key_obsm", "key_uns"),
    [
        pytest.param(False, "X_draw_graph_fr", "draw_graph", id="default"),
        pytest.param(
            "obsm.custom", "custom", "custom", marks=needs.anndata_acc, id="acc"
        ),
        pytest.param(sc.Preset.ScanpyV1, "X_draw_graph_fr", "draw_graph", id="v1"),
        pytest.param(
            *(sc.Preset.ScanpyV2Preview, "graph_fr", "graph_fr"),
            marks=needs.scanpy2,
            id="v2",
        ),
    ],
)
def test_draw_graph_out(
    *, out: str | sc.Preset | bool, key_obsm: str, key_uns: str
) -> None:
    adata = pbmc68k_reduced()[:100, :100].copy()
    if isinstance(out, sc.Preset):
        sc.settings.preset = out
        out = False
    sc.tl.draw_graph(adata, layout="fr", out=out)
    assert key_obsm in adata.obsm
    assert key_uns in adata.uns


EMBEDDINGS = [
    pytest.param(sc.tl.tsne, id="tsne"),
    pytest.param(sc.tl.diffmap, id="diffmap"),
    pytest.param(
        lambda a, **kw: sc.tl.draw_graph(a, layout="fr", **kw),
        marks=needs.igraph,
        id="draw_graph",
    ),
]


@pytest.fixture
def pbmc_small() -> AnnData:
    adata = pbmc68k_reduced()[:100, :50].copy()
    sc.pp.neighbors(adata)  # the subset’s graph has isolated cells
    return adata


@needs.anndata_acc
@pytest.mark.parametrize("func", EMBEDDINGS)
@pytest.mark.parametrize("as_str", [True, False], ids=["str", "acc"])
def test_embedding_out_acc(
    pbmc_small: AnnData, func: Callable[..., object], *, as_str: bool
) -> None:
    """An accessor `out` is used literally, and `.uns` follows it."""
    from anndata.acc import A

    obsm_before = set(pbmc_small.obsm)
    func(pbmc_small, out="obsm.m_{layout}" if as_str else A.obsm["m_{layout}"])
    assert set(pbmc_small.obsm) - obsm_before == {"m_{layout}"}
    assert "m_{layout}" in pbmc_small.uns


@pytest.mark.parametrize("func", EMBEDDINGS)
def test_embedding_out_true(pbmc_small: AnnData, func: Callable[..., object]) -> None:
    """`out=True` returns the embedding and writes nothing."""
    before = pbmc_small.copy()
    emb = func(pbmc_small, out=True)
    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] == pbmc_small.n_obs
    assert pbmc_small.obsm.keys() == before.obsm.keys()
    assert pbmc_small.uns.keys() == before.uns.keys()
    with (
        pytest.warns(FutureWarning, match=r"argument copy is deprecated"),
        pytest.raises(ValueError, match=r"`copy=True` cannot be used"),
    ):
        func(pbmc_small, out=True, copy=True)


@needs.anndata_acc
def test_tsne_key_added_deprecated(pbmc_small: AnnData) -> None:
    """`key_added='k'` works like `out=A.obsm['k']`, but warns."""
    from anndata.acc import A

    expected = pbmc_small.copy()
    sc.tl.tsne(expected, out=A.obsm["k"])
    with pytest.warns(FutureWarning, match=r"argument key_added is deprecated"):
        sc.tl.tsne(pbmc_small, key_added="k")
    assert pbmc_small.obsm.keys() == expected.obsm.keys()
    assert pbmc_small.uns.keys() == expected.uns.keys()
