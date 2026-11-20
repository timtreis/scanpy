from __future__ import annotations

import warnings
from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from sklearn.metrics.cluster import normalized_mutual_info_score

import scanpy as sc
from testing.scanpy._helpers.data import pbmc68k_reduced
from testing.scanpy._pytest.marks import needs

if TYPE_CHECKING:
    from typing import Literal

    from pytest_mock import MockerFixture


@pytest.fixture
def adata_neighbors() -> AnnData:
    return pbmc68k_reduced()


def _leiden_copy(adata: AnnData, **kwargs) -> AnnData:
    adata = adata.copy()
    sc.tl.leiden(adata, **kwargs)
    return adata


@pytest.fixture(
    params=[
        pytest.param("igraph", marks=needs.igraph),
        pytest.param("leidenalg", marks=needs.leidenalg),
        pytest.param("networkit", marks=needs.networkit),
    ]
)
def flavor(
    request: pytest.FixtureRequest,
) -> Literal["igraph", "leidenalg", "networkit"]:
    return request.param


@needs.leidenalg
@needs.igraph
@pytest.mark.parametrize("resolution", [1, 2])
@pytest.mark.parametrize("n_iterations", [-1, 3])
def test_leiden_basic(
    adata_neighbors: AnnData,
    flavor: Literal["igraph", "leidenalg", "networkit"],
    resolution: float,
    n_iterations: int,
) -> None:
    with (
        nullcontext()
        if flavor in {"igraph", "networkit"}
        else pytest.warns(
            UserWarning,
            match=r"The `igraph` and `networkit` implementations of leiden clustering",
        )
    ):
        sc.tl.leiden(
            adata_neighbors,
            flavor=flavor,
            resolution=resolution,
            n_iterations=n_iterations,
            directed=(flavor == "leidenalg"),
        )
    assert "leiden" in adata_neighbors.obs
    assert adata_neighbors.uns["leiden"]["params"]["resolution"] == resolution
    assert adata_neighbors.uns["leiden"]["params"]["n_iterations"] == n_iterations


@needs.leidenalg
@needs.igraph
@pytest.mark.parametrize("rng_arg", ["rng", "random_state"])
def test_leiden_random_state(
    subtests: pytest.Subtests,
    adata_neighbors: AnnData,
    flavor: Literal["igraph", "leidenalg", "networkit"],
    rng_arg: Literal["rng", "random_state"],
) -> None:
    is_leiden_alg = flavor == "leidenalg"
    n_iterations = 2 if is_leiden_alg else -1
    adata_1, adata_1_again, adata_2 = (
        _leiden_copy(
            adata_neighbors,
            flavor=flavor,
            directed=is_leiden_alg,
            n_iterations=n_iterations,
            **{rng_arg: seed},
        )
        # some seeds give the same partition as 1 for some flavor, 8 differs for all
        for seed in (1, 1, 8)
    )
    with subtests.test("reproducible"):
        pd.testing.assert_series_equal(
            adata_1.obs["leiden"], adata_1_again.obs["leiden"]
        )
        assert (
            pytest.approx(adata_1.uns["leiden"]["modularity"])
            == adata_1_again.uns["leiden"]["modularity"]
        )
    with subtests.test("different clustering"):
        assert not adata_2.obs["leiden"].equals(adata_1_again.obs["leiden"])
        assert (
            pytest.approx(adata_2.uns["leiden"]["modularity"])
            != adata_1_again.uns["leiden"]["modularity"]
        )


@needs.anndata_acc
@needs.igraph
@needs.leidenalg
def test_leiden_igraph_directed():
    from anndata.acc import A

    graph = np.array([
        [0, 0, 0, 3, 0, 0, 0, 2],
        [0, 0, 1, 0, 0, 0, 3, 0],
        [2, 0, 0, 1, 2, 0, 1, 2],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0],
    ])
    adata = AnnData(np.zeros((8, 1)), obsp=dict(graph=graph))
    kw = dict(use=A.obsp["graph"], directed=True, out=True)
    igraph = sc.tl.leiden(adata, flavor="igraph", **kw)
    leidenalg = sc.tl.leiden(adata, flavor="leidenalg", **kw)
    assert normalized_mutual_info_score(igraph, leidenalg) == 1


@needs.networkit
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param({}, 1, id="default"),
        pytest.param({"rng": 0}, 1, id="seeded"),
        pytest.param({"rng": None}, 3, id="unseeded"),
        pytest.param({"rng": 0, "n_jobs": 2}, 2, id="explicit"),
    ],
)
def test_leiden_networkit_n_jobs(
    mocker: MockerFixture,
    adata_neighbors: AnnData,
    kwargs: dict[str, int | None],
    expected: int,
) -> None:
    import networkit

    sc.settings.n_jobs = 3
    spy = mocker.spy(networkit, "setNumberOfThreads")
    sc.tl.leiden(adata_neighbors, flavor="networkit", **kwargs)
    assert spy.call_args_list[0].args == (expected,)


@needs.networkit
def test_leiden_networkit_directed(adata_neighbors):
    with pytest.raises(ValueError, match=r"Cannot use NetworKit's leiden.*directed"):
        sc.tl.leiden(adata_neighbors, flavor="networkit", directed=True)


@pytest.mark.parametrize("flavor", ["igraph", "leidenalg", "networkit"])
@pytest.mark.parametrize(
    ("preset", "directed_flavors"),
    [
        pytest.param(sc.Preset.ScanpyV1, {"leidenalg"}, id="v1"),
        pytest.param(
            sc.Preset.ScanpyV2Preview,
            {"igraph", "leidenalg"},
            marks=needs.scanpy2,
            id="v2",
        ),
    ],
)
def test_leiden_directed_default(
    preset: sc.Preset, directed_flavors: set[str], flavor: str
) -> None:
    from scanpy._settings import Default
    from scanpy.tools._leiden import _resolve_directed

    sc.settings.preset = preset
    assert _resolve_directed(Default(), flavor=flavor) is (flavor in directed_flavors)


@needs.igraph
def test_leiden_wrong_flavor(adata_neighbors):
    with pytest.raises(ValueError, match=r"flavor must be.*'igraph'.*'leidenalg'.*but"):
        sc.tl.leiden(adata_neighbors, flavor="foo")


@needs.igraph
@needs.leidenalg
def test_leiden_igraph_partition_type(adata_neighbors):
    import leidenalg

    with pytest.raises(ValueError, match=r"Do not pass in partition_type"):
        sc.tl.leiden(
            adata_neighbors,
            flavor="igraph",
            partition_type=leidenalg.RBConfigurationVertexPartition,
        )


@pytest.fixture(
    params=[
        pytest.param("igraph", marks=needs.igraph),
        pytest.param("networkit", marks=needs.networkit),
    ]
)
def fast_flavor(request: pytest.FixtureRequest) -> Literal["igraph", "networkit"]:
    return request.param


@needs.leidenalg
def test_leiden_equal_defaults_same_args(
    adata_neighbors: AnnData, fast_flavor: Literal["igraph", "networkit"]
) -> None:
    """Ensure the implementations are the same for the same args."""
    leiden_alg_clustered = _leiden_copy(
        adata_neighbors, flavor="leidenalg", n_iterations=2
    )
    fast_clustered = _leiden_copy(
        adata_neighbors, flavor=fast_flavor, directed=False, n_iterations=2
    )
    assert (
        normalized_mutual_info_score(
            leiden_alg_clustered.obs["leiden"], fast_clustered.obs["leiden"]
        )
        > 0.9
    )


@needs.leidenalg
def test_leiden_equal_defaults(
    adata_neighbors: AnnData, fast_flavor: Literal["igraph", "networkit"]
) -> None:
    """Ensure that the old leidenalg defaults are close enough to the current default outputs."""
    leiden_alg_clustered = _leiden_copy(
        adata_neighbors, flavor="leidenalg", directed=True
    )
    fast_clustered = _leiden_copy(
        adata_neighbors, flavor=fast_flavor, n_iterations=2, directed=False
    )
    assert (
        normalized_mutual_info_score(
            leiden_alg_clustered.obs["leiden"], fast_clustered.obs["leiden"]
        )
        > 0.9
    )


@needs.igraph
def test_leiden_objective_function(adata_neighbors):
    """Ensure that popping this as a `clustering_kwargs` and using it does not error out."""
    sc.tl.leiden(
        adata_neighbors,
        objective_function="modularity",
        flavor="igraph",
        directed=False,
    )


@needs.anndata_acc
@needs.igraph
@pytest.mark.parametrize(
    ("clustering", "key"),
    [
        pytest.param(
            partial(sc.tl.leiden, flavor="leidenalg"),
            "leiden",
            marks=needs.leidenalg,
            id="leiden",
        ),
    ],
)
def test_clustering_subset(adata_neighbors, clustering, key):
    from anndata.acc import A

    clustering(adata_neighbors, out=A.obs[key])

    for c in adata_neighbors.obs[key].unique():
        print("Analyzing cluster ", c)
        cells_in_c = adata_neighbors.obs[key] == c
        ncells_in_c = adata_neighbors.obs[key].value_counts().loc[c]
        key_sub = f"{key}_sub"
        clustering(
            adata_neighbors,
            restrict_to=(key, [c]),
            out=A.obs[key_sub],
        )
        # Get new clustering labels
        new_partition = adata_neighbors.obs[key_sub]

        cat_counts = new_partition[cells_in_c].value_counts()

        # Only original cluster's cells assigned to new categories
        assert cat_counts.sum() == ncells_in_c

        # Original category's cells assigned only to new categories
        nonzero_cat = cat_counts[cat_counts > 0].index
        common_cat = nonzero_cat.intersection(adata_neighbors.obs[key].cat.categories)
        assert len(common_cat) == 0


@needs.anndata_acc
@pytest.mark.parametrize(
    ("clustering", "default_key", "default_res", "custom_resolutions"),
    [
        pytest.param(
            partial(sc.tl.leiden, flavor="leidenalg"),
            "leiden",
            0.8,
            [0.9, 1.1],
            marks=needs.leidenalg,
            id="leiden",
        ),
    ],
)
def test_clustering_custom_key(
    adata_neighbors, clustering, default_key, default_res, custom_resolutions
):
    from anndata.acc import A

    custom_keys = [f"{default_key}_{res}" for res in custom_resolutions]

    # Run clustering with default key, then custom keys
    clustering(adata_neighbors, resolution=default_res)
    for key, res in zip(custom_keys, custom_resolutions, strict=True):
        clustering(adata_neighbors, resolution=res, out=A.obs[key])

    # ensure that all clustering parameters are added to user provided keys and not overwritten
    assert adata_neighbors.uns[default_key]["params"]["resolution"] == default_res
    for key, res in zip(custom_keys, custom_resolutions, strict=True):
        assert adata_neighbors.uns[key]["params"]["resolution"] == res


@needs.anndata_acc
@needs.igraph
def test_leiden_out_acc(adata_neighbors: AnnData) -> None:
    """An accessor `out` names the column, and `.uns` follows it."""
    from anndata.acc import A

    sc.tl.leiden(adata_neighbors, flavor="igraph", out=A.obs["mine"])
    assert "mine" in adata_neighbors.obs
    assert "modularity" in adata_neighbors.uns["mine"]
    assert "leiden" not in adata_neighbors.uns
    # an explicit accessor isn’t renamed when restricting
    sc.tl.leiden(
        adata_neighbors, flavor="igraph", restrict_to=("mine", ["0"]), out="obs.leiden"
    )
    assert "leiden" in adata_neighbors.obs
    assert "leiden_R" not in adata_neighbors.obs


@needs.igraph
def test_leiden_out_true(adata_neighbors: AnnData) -> None:
    """`out=True` returns the labels and writes nothing."""
    obs_before, uns_before = adata_neighbors.obs.copy(), set(adata_neighbors.uns)
    labels = sc.tl.leiden(adata_neighbors, flavor="igraph", out=True)
    assert isinstance(labels, pd.Categorical)
    assert len(labels) == adata_neighbors.n_obs
    pd.testing.assert_frame_equal(adata_neighbors.obs, obs_before)
    assert set(adata_neighbors.uns) == uns_before


@needs.anndata_acc
@needs.igraph
@pytest.mark.parametrize("restrict", [False, True], ids=["plain", "restrict"])
def test_leiden_key_added_deprecated(
    adata_neighbors: AnnData, *, restrict: bool
) -> None:
    """`key_added='k'` is `out=A.obs['k']`, except for the `_R` suffix for the default."""
    from anndata.acc import A

    kw = dict(flavor="igraph", restrict_to=("louvain", ["0"]) if restrict else None)
    key = "leiden_R" if restrict else "leiden"
    expected = adata_neighbors.copy()
    sc.tl.leiden(expected, out=A.obs[key], **kw)
    with pytest.warns(FutureWarning, match=r"argument key_added is deprecated"):
        sc.tl.leiden(adata_neighbors, key_added="leiden", **kw)
    assert key in adata_neighbors.obs
    assert adata_neighbors.obs.keys().equals(expected.obs.keys())
    assert adata_neighbors.uns.keys() == expected.uns.keys()


def test_leiden_numba_multi_resolution(adata_neighbors: AnnData) -> None:
    resolutions = [0.5, 1, 2.0]
    keys = ["leiden_res0_5", "leiden_res1", "leiden_res2"]
    sc.tl.leiden(adata_neighbors, resolution=resolutions, flavor="numba", rng=0)
    for res, key in zip(resolutions, keys, strict=True):
        assert adata_neighbors.uns[key]["params"]["resolution"] == res
        # a resolution computed alone gives the same clusters as within a sequence
        alone = sc.tl.leiden(
            adata_neighbors, resolution=res, flavor="numba", rng=0, out=True
        )
        pd.testing.assert_series_equal(
            adata_neighbors.obs[key].astype(str),
            pd.Series(alone.astype(str), index=adata_neighbors.obs_names, name=key),
        )
    assert (
        adata_neighbors.obs["leiden_res0_5"].nunique()
        < adata_neighbors.obs["leiden_res2"].nunique()
    )
    serial = sc.tl.leiden(
        adata_neighbors,
        resolution=resolutions,
        flavor="numba",
        rng=0,
        out=True,
        n_jobs=1,
    )
    assert list(serial) == [0.5, 1.0, 2.0]
    for res, key in zip(resolutions, keys, strict=True):
        assert list(serial[res]) == list(adata_neighbors.obs[key])


@pytest.mark.parametrize(
    "flavor",
    [
        pytest.param("igraph", marks=needs.igraph),
        pytest.param("networkit", marks=needs.networkit),
        "numba",
    ],
)
@pytest.mark.parametrize("restrict", [False, True], ids=["plain", "restrict"])
def test_leiden_multi_resolution_keys(
    adata_neighbors: AnnData, flavor: str, *, restrict: bool
) -> None:
    """Every flavor accepts several resolutions; default keys have no dots."""
    kw = dict(restrict_to=("louvain", ["0"])) if restrict else {}
    sc.tl.leiden(
        adata_neighbors, resolution=np.array([0.25, 1.0]), flavor=flavor, rng=0, **kw
    )
    base = "leiden_R" if restrict else "leiden"
    for key, res in [(f"{base}_res0_25", 0.25), (f"{base}_res1", 1.0)]:
        assert adata_neighbors.obs[key].dtype == "category"
        assert adata_neighbors.uns[key]["params"]["resolution"] == res
        assert "modularity" in adata_neighbors.uns[key]
    assert base not in adata_neighbors.obs


@pytest.fixture(params=[pytest.param("igraph", marks=needs.igraph), "numba"])
def multi_flavor(request: pytest.FixtureRequest) -> Literal["igraph", "numba"]:
    """Run the default flavor and Scanpy’s own through the multi-resolution API."""
    return request.param


@needs.anndata_acc
def test_leiden_multi_resolution_out_mapping(
    adata_neighbors: AnnData, multi_flavor: Literal["igraph", "numba"]
) -> None:
    from anndata.acc import A

    obs_before = set(adata_neighbors.obs)
    ret = sc.tl.leiden(
        adata_neighbors,
        resolution=[0.5, 2],
        flavor=multi_flavor,
        out={2.0: A.obs["fine"], 0.5: "obs.coarse"},
    )
    assert ret is None
    assert set(adata_neighbors.obs) - obs_before == {"fine", "coarse"}
    assert adata_neighbors.uns["coarse"]["params"]["resolution"] == 0.5
    assert adata_neighbors.uns["fine"]["params"]["resolution"] == 2.0
    assert (
        adata_neighbors.obs["coarse"].nunique() < adata_neighbors.obs["fine"].nunique()
    )


def test_leiden_multi_resolution_out_true(
    adata_neighbors: AnnData, multi_flavor: Literal["igraph", "numba"]
) -> None:
    """Returns a dict, and a resolution within a sequence gives what it gives alone."""
    from typing import assert_type

    obs_before, uns_before = adata_neighbors.obs.copy(), set(adata_neighbors.uns)
    multi = sc.tl.leiden(
        adata_neighbors, resolution=[0.5, 1, 2], flavor=multi_flavor, rng=0, out=True
    )
    assert_type(multi, dict[float, pd.Categorical])
    assert list(multi) == [0.5, 1.0, 2.0]
    assert all(isinstance(v, pd.Categorical) for v in multi.values())
    for res in [1, 2]:
        single = sc.tl.leiden(
            adata_neighbors, resolution=res, flavor=multi_flavor, rng=0, out=True
        )
        assert_type(single, pd.Categorical)
        assert list(single) == list(multi[res])
    pd.testing.assert_frame_equal(adata_neighbors.obs, obs_before)
    assert set(adata_neighbors.uns) == uns_before


@pytest.mark.parametrize(
    ("resolution", "keys"),
    [
        pytest.param(
            np.arange(
                0.1, 0.35, 0.1
            ),  # 0.30000000000000004 (a stop of 0.4 would include 0.4)
            ["leiden_res0_1", "leiden_res0_2", "leiden_res0_3"],
            id="arange",
        ),
        pytest.param([1e-5, 2.0], ["leiden_res0_00001", "leiden_res2"], id="small"),
        pytest.param([-0.0, 1], ["leiden_res0", "leiden_res1"], id="neg-zero"),
        pytest.param(
            np.array([0.1, 0.5], np.float32),
            ["leiden_res0_1", "leiden_res0_5"],
            id="float32",
        ),
    ],
)
def test_leiden_multi_resolution_key_names(
    adata_neighbors: AnnData, resolution: list[float] | np.ndarray, keys: list[str]
) -> None:
    """Keys are rounded, positional and have no `-`; `uns` keeps the value as given."""
    sc.tl.leiden(adata_neighbors, resolution=resolution, flavor="numba", rng=0)
    for key, res in zip(keys, np.asarray(resolution).tolist(), strict=True):
        assert key in adata_neighbors.obs
        assert adata_neighbors.uns[key]["params"]["resolution"] == pytest.approx(res)


@needs.anndata_acc
def test_leiden_multi_resolution_float32(adata_neighbors: AnnData) -> None:
    """float32 resolutions are taken as written, so `out` and the result use those values."""
    from anndata.acc import A

    res = np.array([0.1, 0.5], np.float32)
    multi = sc.tl.leiden(
        adata_neighbors, resolution=res, flavor="numba", rng=0, out=True
    )
    assert list(multi) == [0.1, 0.5]
    alone = sc.tl.leiden(
        adata_neighbors, resolution=0.1, flavor="numba", rng=0, out=True
    )
    assert list(multi[0.1]) == list(alone)
    sc.tl.leiden(
        adata_neighbors,
        resolution=res,
        flavor="numba",
        rng=0,
        out={0.1: A.obs["a"], 0.5: A.obs["b"]},
    )
    assert adata_neighbors.uns["a"]["params"]["resolution"] == 0.1


@needs.anndata_acc
def test_leiden_multi_resolution_restrict_reads_input(
    adata_neighbors: AnnData, multi_flavor: Literal["igraph", "numba"]
) -> None:
    """With `restrict_to`, every resolution relabels the column as it was before the call."""
    from anndata.acc import A

    kw = dict(
        resolution=[0.5, 1],
        flavor=multi_flavor,
        rng=0,
        restrict_to=("louvain", ["0", "1"]),
    )
    expected = sc.tl.leiden(adata_neighbors.copy(), out=True, **kw)
    # the first result overwrites the restricted column
    sc.tl.leiden(adata_neighbors, out={0.5: A.obs["louvain"], 1: A.obs["fine"]}, **kw)
    assert list(adata_neighbors.obs["louvain"]) == list(expected[0.5])
    assert list(adata_neighbors.obs["fine"]) == list(expected[1.0])


@needs.anndata_acc
@pytest.mark.parametrize(
    ("kwargs", "exc", "match"),
    [
        pytest.param(
            dict(resolution=[0.3, 0.1 + 0.2]), ValueError, r"same key", id="key-clash"
        ),
        pytest.param(dict(beta=0), ValueError, r"`beta`", id="beta-zero"),
        pytest.param(dict(beta=-0.5), ValueError, r"`beta`", id="beta-negative"),
        pytest.param(dict(beta=np.nan), ValueError, r"`beta`", id="beta-nan"),
        pytest.param(dict(beta=np.inf), ValueError, r"`beta`", id="beta-inf"),
        pytest.param(dict(resolution=[]), ValueError, r"empty", id="empty"),
        pytest.param(dict(resolution=[1, 1.0]), ValueError, r"distinct", id="dupe"),
        pytest.param(dict(resolution=[0.5, np.nan]), ValueError, r"finite", id="nan"),
        pytest.param(
            dict(resolution=np.ones((2, 2))), ValueError, r"1-dimensional", id="2d"
        ),
        pytest.param(
            dict(resolution=[0.5, 1], out="obs.k"),
            TypeError,
            r"mapping from each resolution",
            id="single-acc",
        ),
        pytest.param(
            dict(resolution=[0.5, 1], out={0.5: "obs.a"}),
            TypeError,
            r"missing \[1\.0\]",
            id="mapping-missing",
        ),
        pytest.param(
            dict(resolution=[0.5, 1], out={0.5: "obs.a", 1: "obs.b", 2: "obs.c"}),
            TypeError,
            r"aren’t in `resolution`",
            id="mapping-unknown",
        ),
        pytest.param(
            dict(resolution=[0.5, 1], out={0.5: "obs.a", 1: "obs.a"}),
            ValueError,
            r"same column",
            id="mapping-same-dest",
        ),
        pytest.param(
            dict(resolution=[0.5, 1], out={0.5: "obs.a", 1.0: True}),
            TypeError,
            r"map resolutions to accessors",
            id="mapping-bool",
        ),
        pytest.param(
            dict(resolution=1, out={1: "obs.a"}),
            TypeError,
            r"needs a sequence",
            id="mapping-scalar",
        ),
    ],
)
def test_leiden_resolution_errors(
    adata_neighbors: AnnData,
    multi_flavor: Literal["igraph", "numba"],
    kwargs: dict,
    exc: type[Exception],
    match: str,
) -> None:
    obs_before = set(adata_neighbors.obs)
    with pytest.raises(exc, match=match):
        sc.tl.leiden(adata_neighbors, flavor=multi_flavor, **kwargs)
    assert set(adata_neighbors.obs) == obs_before


def test_leiden_numba_negative_resolution(adata_neighbors: AnnData) -> None:
    with pytest.raises(ValueError, match=r"must not be negative"):
        sc.tl.leiden(adata_neighbors, flavor="numba", resolution=-1)


def test_leiden_numba_kernel_beta() -> None:
    """The kernel checks `beta` too, so every schedule rejects it alike."""
    from scipy import sparse

    from scanpy.tools._leiden_numba import leiden

    adj = sparse.csr_array(np.ones((3, 3), np.float32))  # noqa: TID251
    with pytest.raises(ValueError, match=r"`beta`"):
        leiden(adj, [1.0], beta=0.0)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        pytest.param([1.0, -0.5], r"Negative weight", id="negative"),
        # opposite signs would cancel when symmetrising, so they are checked before
        pytest.param([1.0, -1.0], r"Negative weight", id="cancelling"),
        pytest.param([1.0, np.inf], r"finite", id="inf"),
    ],
)
def test_leiden_numba_bad_weights(data: list[float], match: str) -> None:
    from scipy import sparse

    from scanpy.tools._leiden_numba import leiden

    # entries (0, 1) and (1, 0)
    adj = sparse.csr_array((data, [1, 0], [0, 1, 2, 2]), shape=(3, 3))  # noqa: TID251
    with pytest.raises(ValueError, match=match):
        leiden(adj, [1.0])


def test_leiden_numba_no_resolutions(adata_neighbors: AnnData) -> None:
    from scanpy.tools._leiden_numba import leiden

    memb, q = leiden(adata_neighbors.obsp["connectivities"], [])
    assert memb.shape == (0, adata_neighbors.n_obs)
    assert q.shape == (0,)


def test_leiden_numba_n_jobs(mocker: MockerFixture, adata_neighbors: AnnData) -> None:
    """`n_jobs` (default: `settings.n_jobs`, also when seeded) limits numba’s threads."""
    import numba

    sc.settings.n_jobs = 1
    spy = mocker.spy(numba, "set_num_threads")
    sc.tl.leiden(adata_neighbors, flavor="numba", rng=0)
    assert spy.call_args_list[0].args == (1,)
    spy.reset_mock()
    sc.tl.leiden(adata_neighbors, flavor="numba", rng=0, n_jobs=2)
    assert spy.call_args_list[0].args == (min(2, numba.config.NUMBA_NUM_THREADS),)


def test_leiden_numba_iteration_cap_warns(
    adata_neighbors: AnnData, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scanpy.tools import _leiden_numba as ln

    monkeypatch.setattr(ln, "_MAX_ITERATIONS", 1)
    with pytest.warns(UserWarning, match=r"didn’t converge within 1 iterations"):
        sc.tl.leiden(adata_neighbors, flavor="numba", n_iterations=-1)
    monkeypatch.undo()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", r".*converge", UserWarning)
        sc.tl.leiden(adata_neighbors, flavor="numba", n_iterations=-1)


@needs.igraph
def test_leiden_numba_quality_like_igraph(adata_neighbors: AnnData) -> None:
    conns = adata_neighbors.obsp["connectivities"]
    ours = sc.tl.leiden(
        adata_neighbors, flavor="numba", n_iterations=2, rng=0, out=True
    )
    ref = sc.tl.leiden(
        adata_neighbors, flavor="igraph", n_iterations=2, rng=0, out=True
    )
    q_ours = sc.metrics.modularity(conns, ours, is_directed=False)
    q_ref = sc.metrics.modularity(conns, ref, is_directed=False)
    assert q_ours > q_ref - 0.01
    assert normalized_mutual_info_score(ours, ref) > 0.8


@needs.igraph
def test_leiden_numba_modularity_like_metrics(adata_neighbors: AnnData) -> None:
    """The stored modularity (resolution 1) is what `sc.metrics.modularity` computes."""
    sc.tl.leiden(adata_neighbors, flavor="numba", rng=0)
    assert adata_neighbors.uns["leiden"]["modularity"] == pytest.approx(
        sc.metrics.modularity(adata_neighbors, "leiden"), rel=1e-5
    )


@pytest.mark.parametrize("resolution", [1, np.array(1.0)], ids=["int", "0d-array"])
def test_leiden_scalar_resolution_unchanged(
    adata_neighbors: AnnData, resolution: float | np.ndarray
) -> None:
    obs_before = set(adata_neighbors.obs)
    sc.tl.leiden(adata_neighbors, resolution=resolution, flavor="numba", rng=0)
    assert set(adata_neighbors.obs) - obs_before == {"leiden"}
    assert adata_neighbors.uns["leiden"]["params"]["resolution"] is resolution


def test_leiden_numba_apply_alone_is_noop() -> None:
    """Two vertices of a 2-member community both propose a new singleton (path A-D-B, {A,B},{D}).

    A kernel test: `leiden()` reaches this state only from initial memberships it builds itself.
    """
    from scanpy.tools._leiden_numba import _apply

    ptr = np.array([0, 1, 3, 4])  # A=0 - D=1 - B=2
    idx = np.array([1, 0, 2, 1])
    g = (
        ptr,
        idx,
        np.ones(4, np.float32),
        np.ones(3),
    )  # (indptr, indices, weights, node weights)
    memb = np.array([0, 1, 0])
    cw, nbn = np.array([2.0, 1.0, 0.0]), np.array([2, 1, 0])
    empty = np.array([2, -1, -1])  # one free community
    alist, prop = np.array([0, 2]), np.array([-1, 0, -1])
    ne = _apply(g, memb, cw, nbn, empty, 1, alist, 2, np.ones(3, bool), prop)
    assert ne == 0
    assert set(memb.tolist()) == {0, 1, 2}  # A moved out, B stayed alone in 0
    assert nbn.tolist() == [1, 1, 1]


@needs.igraph
def test_leiden_numba_asymmetric_graph_like_igraph() -> None:
    """Asymmetric input and self-loops are read as igraph's undirected graph is."""
    import igraph as ig
    from scipy import sparse

    from scanpy.tools._leiden_numba import leiden

    adj = sparse.random(
        300, 300, density=0.03, format="csr", random_state=0, dtype=np.float32
    )
    adj.setdiag(0.5)
    (memb,), (q,) = leiden(adj, [1.0])
    # one undirected edge per stored entry, as `flavor='igraph'` builds it
    g = ig.Graph(n=300, edges=np.column_stack(adj.nonzero()).tolist(), directed=False)
    assert q == pytest.approx(
        g.modularity(memb.tolist(), weights=adj.data.tolist()), rel=1e-5
    )


def test_leiden_numba_edgeless() -> None:
    from scipy import sparse

    adata = AnnData(np.zeros((5, 2)))
    adata.obsp["connectivities"] = sparse.csr_matrix((5, 5), dtype=np.float32)  # noqa: TID251
    adata.uns["neighbors"] = dict(
        connectivities_key="connectivities", params=dict(method="umap")
    )
    labels = sc.tl.leiden(adata, flavor="numba", out=True)
    assert labels.categories.size == 5


def _planted_graph(n: int = 20_000, *, hub: bool = False):
    """20 planted communities: 8 edges into a vertex’s own, 2 anywhere; `hub`: vertex 0 joins all.

    Large enough that colour classes take the parallel path (`_SMALL`).
    """
    from scipy import sparse

    rng = np.random.default_rng(0)
    k, d_in, d_out = 20, 8, 2
    v = np.arange(n)
    same = k * rng.integers(n // k, size=(n, d_in)) + (v % k)[:, None]
    other = rng.integers(n, size=(n, d_out))
    rows = np.repeat(v, d_in + d_out)
    cols = np.hstack([same, other]).ravel()
    if hub:
        rows, cols = np.r_[rows, np.zeros(n, rows.dtype)], np.r_[cols, v]
    w = rng.uniform(0.1, 1, rows.size).astype(np.float32)
    return sparse.csr_array((w, (rows, cols)), shape=(n, n))  # noqa: TID251


@pytest.mark.parametrize("hub", [False, True], ids=["planted", "hub"])
def test_leiden_numba_schedules_identical(
    monkeypatch: pytest.MonkeyPatch, *, hub: bool
) -> None:
    """Memberships don’t depend on the schedule, thread or chunk count, table size or memory budget.

    "hashed" tables are smaller than the graph, so vertices, clusters and rows above their key
    limit (the hub, aggregated levels) take the dense table; "tiny" sends nearly everything there.
    """
    import numba

    from scanpy._utils import _numba_thread_limit
    from scanpy.tools import _leiden_numba as ln

    graph, res = _planted_graph(hub=hub), [0.5, 1.0, 2.0]
    ref = ln.leiden(graph, res)[0]
    assert all(len(np.unique(m)) > 1 for m in ref)
    modes = ["seq", "outer"] + (["concurrent"] if ln._concurrent_launches_ok() else [])
    threads = sorted({1, min(3, numba.config.NUMBA_NUM_THREADS)})
    tables = dict(
        dense=(ln._TABLE_MIN, ln._TABLE_MAX), hashed=(1 << 10, 1 << 10), tiny=(4, 4)
    )
    results = {}
    for name, (lo, hi) in tables.items():
        monkeypatch.setattr(ln, "_TABLE_MIN", lo)
        monkeypatch.setattr(ln, "_TABLE_MAX", hi)
        for mode in modes:
            monkeypatch.setattr(ln, "_schedule", lambda *_, mode=mode, **__: mode)
            for n_threads in threads:
                with _numba_thread_limit(n_threads):
                    results[name, mode, n_threads] = ln.leiden(graph, res)[0]
    monkeypatch.setattr(ln, "_TABLE_MIN", 1 << 10)
    monkeypatch.setattr(ln, "_TABLE_MAX", 1 << 10)
    monkeypatch.setattr(ln, "_schedule", lambda *_, **__: "seq")
    with monkeypatch.context() as m:
        m.setattr(ln, "_serial_only", lambda: True)  # the serial chain
        results["serial"] = ln.leiden(graph, res)[0]
    # no budget: outer/concurrent fall back to seq with one chunk per thread, as on huge graphs
    monkeypatch.setattr(sc.settings, "max_memory", 0)
    monkeypatch.setattr(ln, "_OUTER", lambda *_: pytest.fail("outer ran over budget"))
    for mode in modes[1:]:
        monkeypatch.setattr(ln, "_schedule", lambda *_, mode=mode, **__: mode)
        results["low_mem", mode] = ln.leiden(graph, res)[0]
    for name, memb in results.items():
        np.testing.assert_array_equal(memb, ref, err_msg=str(name))


def test_leiden_numba_concurrent_shares_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The "concurrent" schedule runs min(resolutions, threads) workers that split the threads."""
    import threading

    import numba

    from scanpy._utils import _numba_thread_limit
    from scanpy.tools import _leiden_numba as ln

    if not ln._concurrent_launches_ok():
        pytest.skip(
            "the threading layer can’t launch parallel kernels from several threads"
        )
    n_threads = min(3, numba.config.NUMBA_NUM_THREADS)
    if n_threads < 2:
        pytest.skip("needs at least 2 numba threads")
    set_num_threads, calls = numba.set_num_threads, []

    def record(k: int) -> None:
        calls.append((threading.current_thread().name, k))
        set_num_threads(k)

    graph = _planted_graph(2000)
    with _numba_thread_limit(n_threads):
        monkeypatch.setattr(numba, "set_num_threads", record)
        monkeypatch.setattr(ln, "_schedule", lambda *_, **__: "concurrent")
        ln.leiden(graph, np.linspace(0.2, 2, 7))
        monkeypatch.undo()
    workers = [k for name, k in calls if name.startswith("leiden")]
    assert len(workers) == n_threads  # min(7, n_threads)
    assert sum(workers) == n_threads


@pytest.mark.parametrize("mode", ["seq", "concurrent", "outer"])
def test_leiden_numba_plan(mode: str) -> None:
    """A plan fits the budget, else falls back to "seq", whose last resort is one chunk per thread."""
    from scanpy.tools import _leiden_numba as ln

    kw = dict(n=10_000, nnz=200_000, maxdeg=60, nr=5, n_threads=8)
    full = ln._plan(mode, **kw, budget=np.inf)
    assert (full.mode, full.workers) == (mode, 1 if mode == "seq" else 5)
    seq = ln._plan("seq", **kw, budget=np.inf)
    for budget in [full.nbytes, full.nbytes - 1, seq.nbytes, seq.nbytes - 1, 0]:
        plan = ln._plan(mode, **kw, budget=budget)
        assert plan == full if budget >= full.nbytes else plan.mode == "seq"
        if plan.nbytes > budget:
            assert (plan.mode, plan.chunks) == ("seq", kw["n_threads"])


def test_leiden_numba_plan_at_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    """At 10M vertices and 192 threads, the chunk tables add little to the workspace."""
    from scanpy.tools import _leiden_numba as ln

    kw = dict(n=10_000_000, nnz=200_000_000, maxdeg=500, nr=1, n_threads=192)
    plan = ln._plan("seq", **kw, budget=np.inf)
    assert plan.slots < kw["n"]  # hashed
    assert plan.nbytes < 7e9  # 5.5 GB per vertex and edge, 0.6 GB of tables
    # dense tables, as before hashing
    monkeypatch.setattr(ln, "_TABLE_MIN", 1 << 40)
    monkeypatch.setattr(ln, "_TABLE_MAX", 1 << 40)
    assert ln._plan("seq", **kw, budget=np.inf).nbytes > 6e10


def test_leiden_numba_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """The workspace budget is `settings.max_memory`, at most half the physical memory."""
    import os

    from scanpy.tools import _leiden_numba as ln

    monkeypatch.setattr(sc.settings, "max_memory", 2)
    assert ln._budget() <= 2e9
    monkeypatch.setattr(sc.settings, "max_memory", 1e12)  # more than any machine has
    try:
        ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        assert ln._budget() == sc.settings.max_memory * 1e9
    else:
        assert ln._budget() == ram / 2


def test_leiden_numba_unknown_argument(adata_neighbors: AnnData) -> None:
    with pytest.raises(TypeError, match=r"objective_function"):
        sc.tl.leiden(adata_neighbors, flavor="numba", objective_function="CPM")
