from __future__ import annotations

import importlib.util
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Real
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast, overload

import numpy as np
import pandas as pd
from natsort import natsorted
from scverse_misc import deprecated_arg

from .. import _utils
from .. import logging as logg
from .._compat import warn
from .._docs import (
    DEPR_ADJACENCY,
    DEPR_KEY_ADDED,
    DEPR_NEIGHBORS_KEY,
    DEPR_OBSP_GRAPH,
    doc_out_key,
    doc_rng,
)
from .._settings import Default, settings
from .._utils import _doc_params, _numba_thread_limit
from .._utils.random import _accepts_legacy_random_state, _LegacyRng, _set_igraph_rng
from ..get._out import _check_legacy, _out_key, deprecated_copy
from ..neighbors._common import _use_graph
from ._docs import (
    doc_adata,
    doc_adjacency,
    doc_neighbors_key,
    doc_obsp,
    doc_restrict_to,
    doc_use_neighbors,
)
from ._utils_clustering import rename_groups, restrict_adjacency

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal

    from anndata import AnnData
    from numpy.typing import NDArray
    from igraph import Graph, VertexClustering

    from .._compat import CSBase
    from .._settings.presets import LeidenFlavor
    from .._utils.random import RNGLike, SeedLike
    from ..get._types import GraphAcc, VecRef

    try:  # sphinx-autodoc-typehints + optional dependency
        from leidenalg.VertexPartition import MutableVertexPartition
    except ImportError:
        if not TYPE_CHECKING:
            MutableVertexPartition = type(
                "MutableVertexPartition",
                (),
                dict(__module__="leidenalg.VertexPartition"),
            )


@overload
def leiden(  # noqa: PLR0913
    adata: AnnData,
    resolution: float | None = 1,
    *,
    restrict_to: tuple[str, Sequence[str]] | None = None,
    rng: SeedLike | RNGLike | None = None,
    directed: bool | Default | None = ...,
    use_weights: bool = True,
    n_iterations: int = -1,
    partition_type: type[MutableVertexPartition] | None = None,
    flavor: LeidenFlavor | Default = ...,
    n_jobs: int | None = None,
    use: GraphAcc | str | None = None,
    out: Literal[True],
    adjacency: CSBase | None = None,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    key_added: str | None = None,
    copy: Literal[False] = False,
    **clustering_args: Any,
) -> pd.Categorical: ...
@overload
def leiden(  # noqa: PLR0913
    adata: AnnData,
    resolution: Sequence[float] | NDArray[np.floating],
    *,
    restrict_to: tuple[str, Sequence[str]] | None = None,
    rng: SeedLike | RNGLike | None = None,
    directed: bool | Default | None = ...,
    use_weights: bool = True,
    n_iterations: int = -1,
    partition_type: type[MutableVertexPartition] | None = None,
    flavor: LeidenFlavor | Default = ...,
    n_jobs: int | None = None,
    use: GraphAcc | str | None = None,
    out: Literal[True],
    adjacency: CSBase | None = None,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    key_added: str | None = None,
    copy: Literal[False] = False,
    **clustering_args: Any,
) -> dict[float, pd.Categorical]: ...
@overload
def leiden(  # noqa: PLR0913
    adata: AnnData,
    resolution: float | Sequence[float] | NDArray[np.floating] | None = 1,
    *,
    restrict_to: tuple[str, Sequence[str]] | None = None,
    rng: SeedLike | RNGLike | None = None,
    directed: bool | Default | None = ...,
    use_weights: bool = True,
    n_iterations: int = -1,
    partition_type: type[MutableVertexPartition] | None = None,
    flavor: LeidenFlavor | Default = ...,
    n_jobs: int | None = None,
    use: GraphAcc | str | None = None,
    out: VecRef | Mapping[float, VecRef] | Literal[False] = False,
    adjacency: CSBase | None = None,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    key_added: str | None = None,
    copy: Literal[False] = False,
    **clustering_args: Any,
) -> None: ...
@overload
def leiden(  # noqa: PLR0913
    adata: AnnData,
    resolution: float | Sequence[float] | NDArray[np.floating] | None = 1,
    *,
    restrict_to: tuple[str, Sequence[str]] | None = None,
    rng: SeedLike | RNGLike | None = None,
    directed: bool | Default | None = ...,
    use_weights: bool = True,
    n_iterations: int = -1,
    partition_type: type[MutableVertexPartition] | None = None,
    flavor: LeidenFlavor | Default = ...,
    n_jobs: int | None = None,
    use: GraphAcc | str | None = None,
    out: VecRef | Mapping[float, VecRef] | Literal[False] = False,
    adjacency: CSBase | None = None,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    key_added: str | None = None,
    copy: Literal[True],
    **clustering_args: Any,
) -> AnnData: ...


@_doc_params(
    doc_adata=doc_adata,
    restrict_to=doc_restrict_to,
    adjacency=doc_adjacency,
    neighbors_key=doc_neighbors_key.format(method="leiden"),
    obsp=doc_obsp,
    use=doc_use_neighbors(),
    rng=doc_rng,
    out=doc_out_key(
        "the cluster labels",
        example="A.obs['clusters']",
        key_added="`key_added='k'` is `out=A.obs['k']`.",
        default=(
            "`.obs['leiden']` and `.uns['leiden']` "
            "(`'leiden_R'` if `restrict_to` is passed)"
        ),
    ),
)
@_accepts_legacy_random_state(0)
@deprecated_arg("key_added", DEPR_KEY_ADDED)
@deprecated_arg("adjacency", DEPR_ADJACENCY)
@deprecated_arg("neighbors_key", DEPR_NEIGHBORS_KEY)
@deprecated_arg("obsp", DEPR_OBSP_GRAPH)
@deprecated_copy
def leiden(  # noqa: PLR0913
    adata: AnnData,
    resolution: float | Sequence[float] | NDArray[np.floating] | None = 1,
    *,
    restrict_to: tuple[str, Sequence[str]] | None = None,
    rng: SeedLike | RNGLike | None = None,
    directed: bool | Default | None = Default(
        repr="None (in 2.0: True for 'igraph' and 'leidenalg')"
    ),
    use_weights: bool = True,
    n_iterations: int = -1,
    partition_type: type[MutableVertexPartition] | None = None,
    flavor: LeidenFlavor | Default = Default(preset=("leiden", "flavor")),
    n_jobs: int | None = None,
    use: GraphAcc | str | None = None,
    out: VecRef | Mapping[float, VecRef] | bool = False,
    # deprecated
    adjacency: CSBase | None = None,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    key_added: str | None = None,
    copy: bool = False,
    **clustering_args,
) -> AnnData | pd.Categorical | dict[float, pd.Categorical] | None:
    r"""Cluster cells into subgroups :cite:p:`Traag2019`.

    Cluster cells using the Leiden algorithm :cite:p:`Traag2019`,
    an improved version of the Louvain algorithm :cite:p:`Blondel2008`.
    It was proposed for single-cell analysis by :cite:t:`Levine2015`.

    This requires having run :func:`~scanpy.pp.neighbors` or
    :func:`~scanpy.external.pp.bbknn` first.

    .. array-support:: tl.leiden

    Parameters
    ----------
    {doc_adata}
    resolution
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
        Set to `None` if overriding `partition_type`
        to one that doesn’t accept a `resolution_parameter`.

        A sequence (or 1-dimensional array) clusters at each resolution.
        Each resolution gives the same clusters as when it is passed alone with the same `rng`
        (for `'networkit'` with several threads, as similar ones).
        Resolutions run one after another.
        Resolutions are compared as :class:`float`\ s and must be distinct and finite;
        float32 values are taken as written (`0.1`, not `0.10000000149011612`).
        By default (`out=False`) each result goes to `.obs` and `.uns` under
        `{{key}}_res{{r}}`, where `{{key}}` is the default key (or `key_added`) and `{{r}}` is
        the resolution rounded to 12 significant digits, written without exponent or trailing zeros,
        with `.` replaced by `_` and a leading `-` by `neg`
        (keys contain no `.`, which accessor strings would split on),
        e.g. `leiden_res0_5` for `0.5`, `leiden_res1` for both `1` and `1.0`,
        `leiden_res0_3` for `0.30000000000000004` (as in `np.arange(0.1, 1, 0.1)`),
        and `leiden_res0_00001` for `1e-5`.
        Resolutions that round to the same key raise a :class:`ValueError`.
        `out` can instead map each resolution to an accessor,
        e.g. `{{0.5: A.obs['coarse'], 2: A.obs['fine']}}`;
        a single accessor can’t name several results.
        `out=True` returns a :class:`dict` from each resolution (as :class:`float`) to its labels.
    {rng}
    {restrict_to}
    directed
        Whether to treat the graph as directed or undirected.
        `None` means directed for `flavor='leidenalg'` and undirected otherwise.
        `flavor='networkit'` only supports undirected graphs.
    use_weights
        If `True`, edge weights from the graph are used in the computation
        (placing more emphasis on stronger edges).
    n_iterations
        How many iterations of the Leiden clustering algorithm to perform.
        Positive values above 2 define the total number of iterations to perform,
        -1 has the algorithm run until it reaches its optimal clustering.
        2 is faster and the default for underlying packages.
    partition_type
        Type of partition to use.
        Defaults to :class:`~leidenalg.RBConfigurationVertexPartition`.
        For the available options, consult the documentation for
        :func:`~leidenalg.find_partition`.
    flavor
        Which package's implementation to use.
        `'networkit'` runs in parallel (see `n_jobs`).
        `'numba'` is Scanpy’s parallel implementation of igraph’s algorithm:
        results are identical for any number of threads and for a resolution
        computed alone or within a sequence, but differ from `'igraph'`’s.
        It needs non-negative edge weights and non-negative resolutions.
    n_jobs
        Number of threads for `flavor='networkit'` and `flavor='numba'` (`-1` means all cores).
        With `'networkit'` and multiple threads, results for the same `rng` are similar but not identical,
        so `None` means :attr:`scanpy.settings.n_jobs` if `rng=None` is passed
        and `1` otherwise, for exact reproducibility.
        `'numba'`’s results don’t depend on the number of threads,
        so `None` always means :attr:`scanpy.settings.n_jobs`.
    {use}
    {out}
    {adjacency}
    {neighbors_key}
    {obsp}
    copy
        Whether to copy `adata` or modify it inplace.
    **clustering_args
        Any further arguments to pass to :func:`~leidenalg.find_partition` (which in turn passes arguments to the `partition_type`)
        or :meth:`igraph.Graph.community_leiden` from `igraph`.
        `flavor='numba'` only accepts `beta` (refinement randomness, default `0.01`).
        `beta`, where passed, must be finite and positive.

    Returns
    -------
    Returns the cluster labels if `out=True`
    (a :class:`dict` from resolution to labels for a sequence of resolutions), else `None`.
    By default sets the following fields (see `out` for others,
    and `resolution` for the keys with several resolutions):

    `adata.obs['leiden']` : :class:`pandas.Series` (dtype ``category``)
        Array of dim (number of samples) that stores the subgroup id
        (``'0'``, ``'1'``, ...) for each cell.

    `adata.uns['leiden']['params']` : :class:`dict`
        A dict with the values for the parameters `resolution`, `n_iterations`,
        and `random_state` (if applicable).

    `adata.uns['leiden']['modularity']` : :class:`float`
        The modularity score of the final clustering,
        as calculated by the `flavor`.
        Use :func:`scanpy.metrics.modularity`\ `(adata, mode='calculate' | 'update')`
        to calculate a score independent of `flavor`.

    """
    flavor, directed = _validate_flavor(
        flavor, partition_type=partition_type, directed=directed
    )
    if flavor != "numba":
        _utils.ensure_igraph()
    clustering_args = dict(clustering_args)
    if "beta" in clustering_args:
        _check_beta(clustering_args["beta"])
    # numba’s results don’t depend on the thread count, so seeded runs can be parallel too
    numba_threads = settings.n_jobs if n_jobs is None else n_jobs
    if n_jobs is None:
        # unseeded runs needn’t be reproducible, so they can be parallel
        n_jobs = settings.n_jobs if rng is None else 1
    seeded = rng is not None
    rng = np.random.default_rng(rng)
    meta_random_state = (
        dict(random_state=rng.arg) if isinstance(rng, _LegacyRng) else {}
    )

    resolutions: list[Any]
    keys: list[str] | Literal[True]
    if multi := _is_multi(resolution):
        resolutions = _multi_resolutions(
            cast("Sequence[float] | np.ndarray", resolution)
        )
        keys = _multi_keys(
            out, key_added, resolutions, restrict=restrict_to is not None
        )
    else:  # exactly as for a single resolution before sequences were accepted
        if isinstance(out, Mapping):
            msg = f"A mapping `out` needs a sequence of resolutions, was {out!r}"
            raise TypeError(msg)
        resolutions = [resolution]
        key = _out_key(out, key_added, kind="obs", default="leiden")
        # like `key_added`, the default key gets a suffix, keys from `out` are literal
        if restrict_to is not None and key == "leiden" and out is False:
            key += "_R"
        keys = True if key is True else [key]

    start = logg.info("running Leiden clustering")
    # are we clustering a user-provided graph or the default AnnData one?
    adjacency = _use_graph(
        adata, use, adjacency=adjacency, obsp=obsp, neighbors_key=neighbors_key
    )
    if restrict_to is not None:
        restrict_key, restrict_categories = restrict_to
        adjacency, restrict_indices = restrict_adjacency(
            adata,
            restrict_key,
            restrict_categories=restrict_categories,
            adjacency=adjacency,
        )
    # Prepare find_partition arguments as a dictionary,
    # appending to whatever the user provided. It needs to be this way
    # as this allows for the accounting of a None resolution
    # (in the case of a partition variant that doesn't take it on input)
    clustering_args["n_iterations"] = n_iterations
    if flavor == "numba":
        with _numba_thread_limit(numba_threads):
            parts = _leiden_numba(
                adjacency,
                resolutions,
                rng=rng,
                use_weights=use_weights,
                **clustering_args,
            )
    else:
        # each resolution starts from the generator's state, as a single call would
        rewind = _rewinder(rng)
        parts = []
        for res in resolutions:
            rewind()
            parts.append(
                _leiden_single(
                    adjacency,
                    res,
                    flavor=flavor,
                    rng=rng,
                    seeded=seeded,
                    directed=directed,
                    use_weights=use_weights,
                    n_iterations=n_iterations,
                    n_jobs=n_jobs,
                    partition_type=partition_type,
                    clustering_args=dict(clustering_args),
                )
            )
    # all labels first: with `restrict_to`, a result may be written to the restricted column
    results: list[pd.Categorical] = []
    for part in parts:
        groups = np.array(part.membership)
        if restrict_to is not None:
            groups = rename_groups(
                adata,
                restrict_key=restrict_key,
                restrict_categories=restrict_categories,
                restrict_indices=restrict_indices,
                groups=groups,
            )
        results.append(
            pd.Categorical(
                values=groups.astype("U"),
                categories=natsorted(map(str, np.unique(groups))),
            )
        )
    if keys is True:
        logg.info("    finished", time=start)
        return dict(zip(resolutions, results, strict=True)) if multi else results[0]
    for key, res, part, labels in zip(keys, resolutions, parts, results, strict=True):
        # store output into adata.obs
        adata.obs[key] = labels
        # store information on the clustering parameters
        adata.uns[key] = {}
        adata.uns[key]["params"] = dict(
            resolution=res, n_iterations=n_iterations, **meta_random_state
        )
        adata.uns[key]["modularity"] = part.modularity
    logg.info(
        "    finished",
        time=start,
        deep=(
            f"found {', '.join(str(len(v.categories)) for v in results)} clusters and added\n"
            f"    {', '.join(map(repr, keys))}, the cluster labels (adata.obs, categorical)"
        ),
    )
    return None


def _is_multi(resolution: object) -> bool:
    """Whether `resolution` is a sequence of resolutions (a 0-d array is a scalar)."""
    if isinstance(resolution, np.ndarray):
        return resolution.ndim > 0
    return isinstance(resolution, Sequence) and not isinstance(resolution, str)


def _multi_resolutions(resolution: Sequence[float] | np.ndarray) -> list[float]:
    if isinstance(resolution, np.ndarray):
        if resolution.ndim != 1:
            msg = f"`resolution` must be a scalar or 1-dimensional, has shape {resolution.shape}."
            raise ValueError(msg)
        resolution = list(resolution)
    res = [_as_float(r) for r in resolution]
    if not res:
        msg = "`resolution` must not be empty."
        raise ValueError(msg)
    if not all(map(math.isfinite, res)):
        msg = f"Resolutions must be finite, got {res}."
        raise ValueError(msg)
    if dupes := sorted(r for r, c in Counter(res).items() if c > 1):
        msg = f"Resolutions must be distinct as floats (e.g. `1` and `1.0` are the same), got {dupes} more than once."
        raise ValueError(msg)
    return res


def _as_float(r: object) -> float:
    """`r` as a float; a float32 (or float16) value as written, e.g. 0.1 rather than 0.10000000149011612."""
    if isinstance(r, np.floating) and r.dtype.itemsize < 8:
        return float(np.format_float_positional(r, unique=True))
    return float(cast("float", r))


def _check_beta(beta: object) -> None:
    if (
        isinstance(beta, bool)
        or not isinstance(beta, Real)
        or not (math.isfinite(beta) and beta > 0)
    ):
        msg = f"`beta` must be finite and positive, got {beta!r}."
        raise ValueError(msg)


def _rewinder(rng: np.random.Generator) -> Callable[[], None]:
    """A function that resets `rng` to its current state."""
    if isinstance(rng, _LegacyRng):
        legacy, state = rng.state, rng.state.get_state()
        return lambda: legacy.set_state(state)
    bit_generator = rng.bit_generator
    state = bit_generator.state
    return lambda: setattr(bit_generator, "state", state)


def _res_key(base: str, resolution: float) -> str:
    """Name a resolution’s result without `.`, e.g. `leiden_res0_5`, `leiden_res1` (for 1 and 1.0).

    The resolution is rounded to 12 significant digits and written without exponent or trailing
    zeros (`np.arange` artefacts like 0.30000000000000004 and 1e-05 give `0_3` and `0_00001`),
    `-0` as `0`, and a leading `-` as `neg`.
    """
    r = float(f"{resolution:.12g}") + 0.0  # + 0.0: -0.0 becomes 0.0
    name = np.format_float_positional(r, trim="-")
    return f"{base}_res{name.replace('-', 'neg').replace('.', '_')}"


def _multi_keys(
    out: VecRef | Mapping[float, VecRef] | bool,  # noqa: FBT001
    key_added: str | None,
    resolutions: list[float],
    *,
    restrict: bool,
) -> list[str] | Literal[True]:
    """Translate `out` into one `.obs` key per resolution, or `True` to return the labels."""
    if out is True:  # like `_out_key`, `key_added` is ignored, as nothing is written
        return True
    if key_added is not None:
        _check_legacy(out, "key_added")
        base = key_added
    elif out is False:
        base = "leiden"
    elif isinstance(out, Mapping):
        return _mapping_keys(out, resolutions)
    else:
        msg = (
            "With several resolutions, `out` must be a `bool` or a mapping from each "
            f"resolution to an accessor, e.g. `{{0.5: A.obs['coarse']}}`, was {out!r}"
        )
        raise TypeError(msg)
    # like `key_added` for one resolution, the default key gets a suffix
    if restrict and base == "leiden":
        base += "_R"
    keys = [_res_key(base, r) for r in resolutions]
    if clashes := [k for k, c in Counter(keys).items() if c > 1]:
        same = [
            [r for r, k in zip(resolutions, keys, strict=True) if k == c]
            for c in clashes
        ]
        msg = (
            f"Resolutions {same} round to the same key(s) {clashes}; "
            "pass `out` as a mapping from each resolution to an accessor to name them."
        )
        raise ValueError(msg)
    return keys


def _mapping_keys(out: Mapping[float, VecRef], resolutions: list[float]) -> list[str]:
    keys: dict[float, str] = {}
    for r, acc in out.items():
        if not isinstance(r, Real) or isinstance(r, bool) or isinstance(acc, bool):
            msg = f"`out` must map resolutions to accessors, was {out!r}"
            raise TypeError(msg)
        if (fr := _as_float(r)) in keys:
            msg = f"`out` names resolution {fr} more than once, was {out!r}"
            raise ValueError(msg)
        keys[fr] = cast("str", _out_key(acc, None, kind="obs"))
    if missing := sorted(set(resolutions) - keys.keys()):
        msg = f"`out` must name every resolution, missing {missing}, was {out!r}"
        raise TypeError(msg)
    if unknown := sorted(keys.keys() - set(resolutions)):
        msg = f"`out` names resolutions {unknown} that aren’t in `resolution`, was {out!r}"
        raise TypeError(msg)
    if dupes := sorted(k for k, c in Counter(keys.values()).items() if c > 1):
        msg = f"Resolutions can’t be written to the same column(s) {dupes}, was {out!r}"
        raise ValueError(msg)
    return [keys[r] for r in resolutions]


def _leiden_single(  # noqa: PLR0913
    adjacency: CSBase,
    resolution: float | None,
    *,
    flavor: LeidenFlavor,
    rng: np.random.Generator,
    seeded: bool,
    directed: bool,
    use_weights: bool,
    n_iterations: int,
    n_jobs: int,
    partition_type: type[MutableVertexPartition] | None,
    clustering_args: dict[str, Any],
) -> MutableVertexPartition | SimpleNamespace:
    """Cluster at one resolution with one of the external backends."""
    if flavor == "leidenalg":
        import leidenalg

        if resolution is not None:
            clustering_args["resolution_parameter"] = resolution
        g = _utils.get_igraph_from_adjacency(adjacency, directed=directed)
        if partition_type is None:
            partition_type = leidenalg.RBConfigurationVertexPartition
        if use_weights:
            clustering_args["weights"] = np.array(g.es["weight"]).astype(np.float64)
        seed = (
            rng.arg
            if isinstance(rng, _LegacyRng) and isinstance(rng.arg, int | np.integer)
            # for some reason leidenalg only accepts int32 (signed) seeds …
            else rng.integers((i := np.iinfo(np.int32)).min, i.max, dtype=np.int32)
        )
        return cast(
            "MutableVertexPartition",
            leidenalg.find_partition(g, partition_type, seed=seed, **clustering_args),
        )
    if flavor == "networkit":
        return _leiden_networkit(
            adjacency,
            rng=rng,
            n_jobs=n_jobs,
            use_weights=use_weights,
            n_iterations=n_iterations,
            resolution=resolution,
        )
    return _leiden_igraph(
        adjacency,
        rng=rng if seeded else None,
        directed=directed,
        use_weights=use_weights,
        resolution=resolution,
        clustering_args=clustering_args,
    )


def _leiden_numba(
    adjacency: CSBase,
    resolutions: Sequence[float | None],
    *,
    rng: np.random.Generator,
    use_weights: bool,
    **kwargs: Any,
) -> list[SimpleNamespace]:
    """Cluster at all resolutions in one call of Scanpy’s numba Leiden (`kwargs` go to the kernel)."""
    from ._leiden_numba import leiden as leiden_numba

    graph = adjacency if use_weights else adjacency.astype(bool).astype(np.float32)
    res = [1.0 if r is None else float(r) for r in resolutions]
    seed = (
        int(rng.arg)
        if isinstance(rng, _LegacyRng) and isinstance(rng.arg, int | np.integer)
        else int(rng.integers(np.iinfo(np.int64).max))
    )
    membership, modularity = leiden_numba(graph, res, seed=seed, **kwargs)
    return [
        SimpleNamespace(membership=m, modularity=float(q))
        for m, q in zip(membership, modularity, strict=True)
    ]


def _leiden_networkit(
    adjacency: CSBase,
    *,
    rng: np.random.Generator,
    n_jobs: int,
    use_weights: bool,
    n_iterations: int,
    resolution: float | None,
) -> SimpleNamespace:
    _utils.ensure_networkit()
    import networkit

    # Seeding controls which nodes are tried and in what order, but not
    # which thread wins when two move neighboring nodes simultaneously.
    # So runs are statistically reproducible, but only bit-for-bit identical
    # when single-threaded (a fixed thread count > 1 is not enough).
    seed = int(rng.integers(np.iinfo(np.int64).max))
    networkit.setSeed(seed, useThreadId=True)

    # only undirected for Parallel Leiden
    g = _utils.get_networkit_from_adjacency(adjacency, weighted=use_weights)
    # NetworKit's `iterations` caps Leiden passes (stops early on convergence; default is 3).
    # We map scanpy's `n_iterations=-1` ("run until done") to this default,
    # while any positive value explicitly overrides it.
    iterations = n_iterations if n_iterations > 0 else 3
    gamma = 1.0 if resolution is None else resolution
    # randomization was removed as an option, so it is randomize = True
    algorithm = networkit.community.ParallelLeiden(
        g, iterations=iterations, gamma=gamma
    )
    prev_n_threads = networkit.getCurrentNumberOfThreads()
    networkit.setNumberOfThreads(n_jobs if n_jobs > 0 else os.process_cpu_count() or 1)
    try:
        algorithm.run()
    finally:
        networkit.setNumberOfThreads(prev_n_threads)
    nk_part = algorithm.getPartition()
    # NetworKit's Partition exposes getVector() for the labels and a
    # separate Modularity measure, rather than .membership / .modularity.
    return SimpleNamespace(
        membership=np.asarray(nk_part.getVector()),
        modularity=networkit.community.Modularity().getQuality(nk_part, g),
    )


def _leiden_igraph(
    adjacency: CSBase,
    *,
    rng: np.random.Generator | None,
    directed: bool,
    use_weights: bool,
    resolution: float | None,
    clustering_args: dict[str, object],
) -> VertexClustering:
    g = _utils.get_igraph_from_adjacency(adjacency, directed=directed)
    _maybe_suggest_networkit(g)
    if use_weights:
        clustering_args["weights"] = "weight"
    if resolution is not None:
        clustering_args["resolution"] = resolution
    clustering_args.setdefault("objective_function", "modularity")
    if (
        directed
        and clustering_args["objective_function"] == "modularity"
        and clustering_args.keys().isdisjoint({"node_weights", "node_in_weights"})
    ):
        weights = clustering_args.get("weights")
        if not (weights is None or isinstance(weights, str | Sequence)):
            msg = f"`weights` must be a str or sequence, got {type(weights).__name__}."
            raise TypeError(msg)
        clustering_args |= _directed_node_weights(g, weights=weights)
    with _set_igraph_rng(rng):
        return g.community_leiden(**clustering_args)


def _directed_node_weights(
    g: Graph, *, weights: str | Sequence[float] | None
) -> dict[str, list[float]]:
    """Get in- and out-degrees as node weights; python-igraph otherwise only counts out-degrees."""
    return dict(
        node_weights=g.strength(mode="out", weights=weights),
        node_in_weights=g.strength(mode="in", weights=weights),
    )


def _validate_flavor(  # noqa: PLR0912
    flavor: str | None,
    *,
    partition_type: object | None,
    directed: bool | Default | None,
) -> tuple[LeidenFlavor, bool]:
    from scanpy import settings

    if was_default := (flavor is None or isinstance(flavor, Default)):
        flavor = settings.preset.leiden.flavor
    is_directed = _resolve_directed(directed, flavor=flavor)
    match flavor:
        case "igraph":
            if partition_type is not None:
                msg = "Do not pass in partition_type argument when using igraph."
                raise ValueError(msg)
        case "numba":
            if is_directed:
                msg = "flavor='numba' only supports undirected graphs."
                raise ValueError(msg)
            if partition_type is not None:
                msg = "Do not pass in partition_type argument when using numba."
                raise ValueError(msg)
        case "networkit":
            if is_directed:
                msg = "Cannot use NetworKit's leiden implementation with a directed graph."
                raise ValueError(msg)
            if partition_type is not None:
                msg = "Do not pass in partition_type argument when using networkit."
                raise ValueError(msg)
        case "leidenalg":
            msg = (
                "The `igraph` and `networkit` implementations of leiden clustering are *orders of magnitude faster*. "
                "Set the flavor argument to (and install if needed) 'igraph' or 'networkit' to use them."
            )
            if was_default:
                msg += (
                    "\nIn the future, the default backend for leiden will be igraph instead of leidenalg. "
                    "To achieve the future defaults please pass: `flavor='igraph'` and `n_iterations=2`."
                )
            warn(msg, FutureWarning if was_default else UserWarning)
            try:
                import leidenalg  # noqa: F401
            except ImportError as e:
                e.add_note(
                    "Please install `scanpy[leiden]` (or `leidenalg` directly) and try again."
                )
                raise
        case _:
            msg = f"flavor must be either 'igraph', 'leidenalg', 'networkit', or 'numba', but {flavor!r} was passed."
            raise ValueError(msg)
    return flavor, is_directed


def _resolve_directed(directed: bool | Default | None, *, flavor: str) -> bool:  # noqa: FBT001
    from scanpy import settings

    resolved = (
        settings.preset.leiden.directed if isinstance(directed, Default) else directed
    )
    if isinstance(resolved, frozenset):
        return flavor in resolved
    return flavor == "leidenalg" if resolved is None else resolved


def _maybe_suggest_networkit(g: Graph) -> None:
    """Encourage users toward the parallel NetworKit backend on large graphs."""
    _networkit_edge_heuristic = 500000
    if g.ecount() < _networkit_edge_heuristic:
        return
    if importlib.util.find_spec("networkit") is None:
        return
    logg.hint(
        f"Graph has {g.ecount():,} edges; NetworKit's parallel Leiden "
        f"(`flavor='networkit'`) is typically several times faster than "
        f"igraph's serial implementation."
    )
