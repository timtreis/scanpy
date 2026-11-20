"""Compile Scanpy’s numba kernels into the disk cache ahead of the first analysis.

``python -m scanpy._warmup [--jobs N]`` runs small synthetic pipelines with the
dtypes real data has, so later sessions load compiled code instead of compiling it.
The cache is specific to the machine (CPU model), so run it where the analysis runs,
e.g. after installing into an environment or when building a container on that hardware.

Independent stages compile in separate processes at the same time: numba holds
the GIL while compiling, so threads would not overlap.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData

_N_OBS = 9000
_N_VARS = 600


def _counts(index_dtype: type) -> AnnData:
    import anndata as ad
    import pandas as pd
    from scipy import sparse

    rng = np.random.default_rng(0)
    x = sparse.random(
        _N_OBS, _N_VARS, density=0.1, format="csr", dtype=np.float32, random_state=rng
    )
    x.data = rng.poisson(3.0, x.nnz).astype(np.float32) + 1
    x.indptr = x.indptr.astype(index_dtype)
    x.indices = x.indices.astype(index_dtype)
    var_names = [
        f"MT-{i}" if i < 10 else ("MALAT1" if i == 10 else f"g{i}")
        for i in range(_N_VARS)
    ]
    obs = pd.DataFrame(
        {
            "donor": pd.Categorical(rng.integers(0, 5, _N_OBS).astype(str)),
            "cell_type": pd.Categorical(rng.integers(0, 8, _N_OBS).astype(str)),
        },
        index=[f"c{i}" for i in range(_N_OBS)],
    )
    return ad.AnnData(x, obs=obs, var=pd.DataFrame(index=var_names))


def _pcs_adata() -> AnnData:
    import anndata as ad

    pcs = np.random.default_rng(0).normal(size=(_N_OBS, 50)).astype(np.float32)
    return ad.AnnData(obs={"i": np.arange(_N_OBS)}, obsm={"X_pca": pcs})


def _graph_adata() -> AnnData:
    """Cells with float32 PCs and a symmetric float32 kNN graph built without numba."""
    from sklearn.neighbors import kneighbors_graph

    a = _pcs_adata()
    g = kneighbors_graph(a.obsm["X_pca"], 15, mode="connectivity")
    g = ((g + g.T) / 2).astype(np.float32).tocsr()
    a.obsp["connectivities"] = g
    a.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "params": {"method": "umap", "use_rep": "X_pca", "metric": "euclidean"},
    }
    return a


def _preprocess(index_dtype: type) -> None:
    import scanpy as sc

    a = _counts(index_dtype)
    a.var["mt"] = a.var_names.str.startswith("MT-")
    a.var["malat1"] = a.var_names == "MALAT1"
    sc.pp.calculate_qc_metrics(
        a, qc_vars=["mt", "malat1"], percent_top=None, log1p=False, inplace=True
    )
    a = a[a.obs["n_genes_by_counts"].to_numpy() > 0].copy()
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    sc.tl.rank_genes_groups(a, groupby="cell_type")
    # QC, batched HVG and PCA compile different kernels without reference numerics
    for reference in (True, False):
        with sc.settings.override(reference_numerics=reference):
            b = a.copy()
            sc.pp.calculate_qc_metrics(b, percent_top=None, log1p=False, inplace=True)
            sc.pp.highly_variable_genes(b, n_top_genes=200, batch_key="donor")
            sc.pp.pca(b, svd_solver="covariance_eigh")


def _neighbors() -> None:
    """Compile the connectivity kernels and, if it applies, the pynndescent cache.

    The pynndescent cache (`neighbors._backends._pynndescent_cache`) is only active for
    pynndescent versions it was tested with, and is compiled per metric: this stage warms
    it for the euclidean metric only.
    """
    import scanpy as sc

    sc.pp.neighbors(_pcs_adata(), use_rep="X_pca", transformer="pynndescent")


def _cluster() -> None:
    """Compile Leiden's kernels for every schedule.

    "seq" and "concurrent" use the parallel kernels. "outer" (serial kernels in a parallel loop
    over resolutions) is chosen only with several threads and a threading layer that isn't
    thread-safe, so it is forced here: with one thread per worker (up to `len(_STAGES)` cores)
    or a thread-safe layer it would otherwise never compile.
    """
    from unittest import mock

    import scanpy as sc
    from scanpy.tools import _leiden_numba

    a = _graph_adata()
    sc.tl.leiden(a, resolution=[0.5, 1.0], flavor="numba", n_iterations=2)
    sc.tl.leiden(a, resolution=1.0, flavor="numba", n_iterations=2)
    with mock.patch.object(_leiden_numba, "_schedule", return_value="outer"):
        sc.tl.leiden(a, resolution=[0.5, 1.0], flavor="numba", n_iterations=2)


def _embed() -> None:
    import scanpy as sc
    from scanpy.tools._spectral import _csr_matmat

    a = _graph_adata()
    for rng in (0, None):  # edge-coloured and Hogwild layouts
        sc.tl.umap(a, method="numba", rng=rng, maxiter=5)
    # the spectral init's products: float32 for ARPACK, float64 for LOBPCG (2M cells and more)
    for dtype in (np.float32, np.float64):
        for index_dtype in (np.int32, np.int64):
            ptr, idx = np.arange(3, dtype=index_dtype), np.arange(2, dtype=index_dtype)
            x, out = np.ones((2, 1), dtype), np.zeros((2, 1), dtype)
            _csr_matmat(ptr, idx, np.ones(2, dtype), x, out)


_STAGES = {
    "preprocess (int32 indices)": (_preprocess, np.int32),
    "preprocess (int64 indices)": (_preprocess, np.int64),
    "neighbors": (_neighbors,),
    "clustering": (_cluster,),
    "embedding": (_embed,),
}


def _run(name: str) -> tuple[str, float]:
    import warnings

    fn, *args = _STAGES[name]
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fn(*args)
    return name, time.perf_counter() - t0


def warmup(n_jobs: int | None = None) -> dict[str, float]:
    """Compile all kernels into numba’s disk cache; returns seconds per stage."""
    n_jobs = len(_STAGES) if n_jobs is None else max(1, n_jobs)
    if n_jobs == 1:
        return dict(_run(name) for name in _STAGES)
    # spawned workers inherit the environment: split the threads among them, or every
    # worker starts a numba thread pool as large as the machine
    import numba

    threads = max(1, numba.config.NUMBA_NUM_THREADS // n_jobs)
    old = os.environ.get("NUMBA_NUM_THREADS")
    os.environ["NUMBA_NUM_THREADS"] = str(threads)
    try:
        with ProcessPoolExecutor(n_jobs, mp_context=get_context("spawn")) as pool:
            return dict(pool.map(_run, _STAGES))
    finally:
        if old is None:
            del os.environ["NUMBA_NUM_THREADS"]
        else:
            os.environ["NUMBA_NUM_THREADS"] = old


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=None,
        help="worker processes (default: one per stage)",
    )
    args = parser.parse_args()
    t0 = time.perf_counter()
    for name, sec in warmup(args.jobs).items():
        print(f"{name:28s} {sec:6.1f}s")
    print(f"{'total wall':28s} {time.perf_counter() - t0:6.1f}s")


if __name__ == "__main__":
    main()
