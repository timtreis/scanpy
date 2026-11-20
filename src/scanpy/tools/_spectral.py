"""Spectral initialisation for ``sc.tl.umap(method="numba")`` with parallel sparse products.

`spectral_layout` and `_multi_component_layout` are ported from umap-learn 0.6.0,
`umap/spectral.py` (`_spectral_layout` and `multi_component_layout`, with
``compatibility_layout=False``, ``init="random"`` and the default ``tol`` and ``maxiter``),
Copyright (c) 2017, Leland McInnes, BSD 3-Clause License
(https://github.com/lmcinnes/umap/blob/master/LICENSE.txt).
Two changes: the eigensolvers get the matrix as a `_CsrOperator`, whose products run in
parallel over rows (scipy's are serial) and round exactly as scipy's, and the Laplacian is
only built where it is used. So the result is bit-identical to umap-learn's
`spectral_layout(..., compatibility_layout=False)` at the same BLAS thread count.
"""

from __future__ import annotations

import warnings
from functools import cache
from typing import TYPE_CHECKING

import numba
import numpy as np
import scipy.linalg
import scipy.sparse.csgraph
import scipy.sparse.linalg
from fast_array_utils.numba import njit
from scipy import sparse
from sklearn.metrics import pairwise_distances

from .. import logging as logg
from .._compat import warn

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any, Literal

    from numpy.typing import NDArray

    from .._compat import CSRBase, SpBase

# umap-learn switches from ARPACK to LOBPCG at this many vertices
_LOBPCG_MIN_N = 2_000_000


@njit
def _csr_matmat(indptr, indices, data, x, out):
    """``out += m @ x``, parallel over rows; each entry sums in stored order in `out`'s dtype, like scipy's ``csr_matvec(s)``."""
    for i in numba.prange(out.shape[0]):
        for v in range(out.shape[1]):
            s = out[i, v]
            for jj in range(indptr[i], indptr[i + 1]):
                s += data[jj] * x[indices[jj], v]
            out[i, v] = s


@cache
def _rounds_like_scipy(dtype: np.dtype) -> bool:
    """Whether `_csr_matmat` gives scipy's bits for `dtype` here.

    It doesn't if scipy was compiled to fuse multiply-adds (FMA), as ARM builds may be:
    this row sums to 0 with separate rounding and to ``d**2`` with fused.
    """
    if dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        return False
    d = dtype.type(2.0 ** -((np.finfo(dtype).nmant + 3) // 2))
    m = sparse.csr_matrix(  # noqa: TID251 (a probe matrix)
        (np.array([-(1 + 2 * d), 1 + d], dtype), [0, 1], [0, 2]), shape=(1, 2)
    )
    x = np.array([[1], [1 + d]], dtype)
    out = np.zeros((1, 1), dtype)
    _csr_matmat(m.indptr, m.indices, m.data, x, out)
    return out.tobytes() == np.asarray(m @ x).tobytes()


class _CsrOperator(scipy.sparse.linalg.LinearOperator):
    """CSR matrix `m` as an operator whose products are parallel over rows and bit-identical to ``m @ x``.

    Falls back to ``m @ x`` where they wouldn't be (see `_rounds_like_scipy`) and for mixed dtypes,
    which scipy upcasts. Counts products, so the eigensolver's work can be logged.
    """

    def __init__(self, m: CSRBase) -> None:
        super().__init__(m.dtype, m.shape)
        self.m = m
        self.parallel = m.format == "csr" and _rounds_like_scipy(m.dtype)
        self.n_products = 0

    def _matvec(self, x: NDArray) -> NDArray:
        return self._matmat(x.reshape(-1, 1)).reshape(-1)

    def _matmat(self, x: NDArray) -> NDArray:
        self.n_products += 1
        if not self.parallel or x.dtype != self.dtype:
            return self.m @ x
        out = np.zeros((self.shape[0], x.shape[1]), dtype=self.dtype)
        m = self.m
        _csr_matmat(m.indptr, m.indices, m.data, np.ascontiguousarray(x), out)
        return out


def _laplacian(a: CSRBase) -> CSRBase:
    return sparse.identity(a.shape[0], dtype=np.float64) - a


def spectral_layout(
    data: Any,
    graph: SpBase,
    dim: int,
    random_state: np.random.RandomState,
    *,
    metric: str = "euclidean",
    metric_kwds: Mapping[str, Any] | None = None,
    method: Literal["eigsh", "lobpcg"] | None = None,
) -> NDArray[np.floating]:
    """Eigenvectors 2 to `dim` + 1 of the normalised Laplacian of `graph`, as umap-learn 0.6 computes them."""
    metric_kwds = {} if metric_kwds is None else metric_kwds
    n_samples = graph.shape[0]
    n_components, labels = scipy.sparse.csgraph.connected_components(graph)
    if n_components > 1:
        return _multi_component_layout(
            data,
            graph,
            n_components,
            labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )

    sqrt_deg = np.sqrt(np.asarray(graph.sum(axis=0)).squeeze())
    # normalised adjacency; umap-learn also builds the Laplacian `I - A` here, but only LOBPCG uses it
    d = sparse.spdiags(1.0 / sqrt_deg, 0, n_samples, n_samples)
    a = d * graph * d
    k = dim + 1
    gen = (
        random_state
        if isinstance(random_state, np.random.Generator | np.random.RandomState)
        else np.random.default_rng(seed=random_state)
    )
    if n_samples < k:
        msg = "Spectral initialisation needs more samples than n_components.\nFalling back to random initialisation!"
        warn(msg, UserWarning)
        return gen.uniform(low=-10.0, high=10.0, size=(n_samples, dim))
    method = method or ("lobpcg" if n_samples >= _LOBPCG_MIN_N else "eigsh")
    # ARPACK solves for A's largest eigenvalues, LOBPCG for the Laplacian's smallest
    op = _CsrOperator(_laplacian(a) if method == "lobpcg" else a)
    start = logg.debug(f"    spectral init of {n_samples} vertices ({method})")
    try:
        if method == "eigsh" and n_samples == k:
            # eigsh needs k < n_samples, so such small graphs are solved densely
            eigenvalues, eigenvectors = scipy.linalg.eigh(_laplacian(a).toarray())
        elif method == "eigsh":
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                op,
                k,
                which="LA",
                ncv=max(2 * k + 1, int(np.sqrt(n_samples))),
                tol=1e-4,
                v0=np.ones(n_samples),
                maxiter=n_samples * 5,
            )
            eigenvalues = [1.0 - eigenvalue for eigenvalue in eigenvalues]
        elif method == "lobpcg":
            x = gen.normal(size=(n_samples, k))
            # the first eigenvector of the normalised Laplacian is proportional to sqrt(degrees)
            x[:, 0] = sqrt_deg / np.linalg.norm(sqrt_deg)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    category=UserWarning,
                    message=r"(?ms).*not reaching the requested tolerance",
                    action="error",
                )
                eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
                    op, x, largest=False, tol=1e-4, maxiter=5 * n_samples
                )
        else:
            msg = "Method should either be None, 'eigsh' or 'lobpcg'"
            raise ValueError(msg)
        logg.debug(f"    {op.n_products} sparse products", time=start)
        order = np.argsort(eigenvalues)[1:k]
        return eigenvectors[:, order]
    except (scipy.sparse.linalg.ArpackError, UserWarning):
        msg = (
            "Spectral initialisation failed! The eigenvector solver\n"
            "failed. This is likely due to too small an eigengap. Consider\n"
            "adding some noise or jitter to your data.\n\n"
            "Falling back to random initialisation!"
        )
        warn(msg, UserWarning)
        return gen.uniform(low=-10.0, high=10.0, size=(n_samples, dim))


def _multi_component_layout(  # noqa: PLR0917
    data: Any,
    graph: SpBase,
    n_components: int,
    component_labels: NDArray[np.integer],
    dim: int,
    random_state: np.random.RandomState,
    *,
    metric: str,
    metric_kwds: Mapping[str, Any],
) -> NDArray[np.float32]:
    """Lay out each component spectrally, placed by a spectral layout of the components’ centroids."""
    from umap.spectral import component_layout

    result = np.empty((graph.shape[0], dim), dtype=np.float32)
    if n_components > 2 * dim:
        meta_embedding = component_layout(
            data,
            n_components,
            component_labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )
    else:
        k = int(np.ceil(n_components / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        meta_embedding = np.vstack([base, -base])[:n_components]

    graph_csr = graph.tocsr()
    for label in range(n_components):
        component_graph = graph_csr[component_labels == label, :].tocsc()
        component_graph = component_graph[:, component_labels == label].tocoo()

        distances = pairwise_distances([meta_embedding[label]], meta_embedding)
        data_range = distances[distances > 0.0].min() / 2.0

        if component_graph.shape[0] < 2 * dim or component_graph.shape[0] <= dim + 1:
            result[component_labels == label] = (
                random_state.uniform(
                    low=-data_range,
                    high=data_range,
                    size=(component_graph.shape[0], dim),
                )
                + meta_embedding[label]
            )
        else:
            component_embedding = spectral_layout(
                None,
                component_graph,
                dim,
                random_state,
                metric=metric,
                metric_kwds=metric_kwds,
            )
            expansion = data_range / np.max(np.abs(component_embedding))
            component_embedding *= expansion
            result[component_labels == label] = (
                component_embedding + meta_embedding[label]
            )
    return result
