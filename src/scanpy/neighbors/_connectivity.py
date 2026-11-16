from __future__ import annotations

import numba
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .._compat import CSRBase
from ._common import (
    _get_indices_distances_from_dense_matrix,
    _get_indices_distances_from_sparse_matrix,
    _get_sparse_matrix_from_indices_distances,
)


def gauss[D: (NDArray[np.float32 | np.float64], CSRBase)](  # noqa: PLR0912
    distances: D, n_neighbors: int, *, knn: bool
) -> D:
    """Derive gaussian connectivities between data points from their distances.

    Parameters
    ----------
    distances
        The input matrix of distances between data points.
    n_neighbors
        The number of nearest neighbors to consider.
    knn
        Specify if the distances have been restricted to k nearest neighbors.

    """
    # init distances
    if isinstance(distances, CSRBase):
        d_sq = distances.power(2)
        indices, distances_sq = _get_indices_distances_from_sparse_matrix(
            d_sq, n_neighbors
        )
    else:
        assert isinstance(distances, np.ndarray)
        d_sq = np.power(distances, 2)
        indices, distances_sq = _get_indices_distances_from_dense_matrix(
            d_sq, n_neighbors
        )

    # exclude the first point, the 0th neighbor
    indices = indices[:, 1:]
    distances_sq = distances_sq[:, 1:]

    # choose sigma, the heuristic here doesn't seem to make much of a difference,
    # but is used to reproduce the figures of Haghverdi et al. (2016)
    if isinstance(distances, CSRBase):
        # as the distances are not sorted
        # we have decay within the n_neighbors first neighbors
        sigmas_sq = np.median(distances_sq, axis=1)
    else:
        # the last item is already in its sorted position through argpartition
        # we have decay beyond the n_neighbors neighbors
        sigmas_sq = distances_sq[:, -1] / 4
    sigmas = np.sqrt(sigmas_sq)

    # compute the symmetric weight matrix
    if not isinstance(distances, CSRBase):
        num = 2 * np.multiply.outer(sigmas, sigmas)
        den = np.add.outer(sigmas_sq, sigmas_sq)
        w = np.sqrt(num / den) * np.exp(-d_sq / den)
        # make the weight matrix sparse
        if not knn:
            mask = w > 1e-14
            w[~mask] = 0
        else:
            # restrict number of neighbors to ~k
            # build a symmetric mask
            mask = np.zeros(d_sq.shape, dtype=bool)
            for i, row in enumerate(indices):
                mask[i, row] = True
                for j in row:
                    if i not in set(indices[j]):
                        w[j, i] = w[i, j]
                        mask[j, i] = True
            # set all entries that are not nearest neighbors to zero
            w[~mask] = 0
    else:
        assert isinstance(d_sq, CSRBase)
        # need to copy the distance matrix here; what follows is inplace
        w = d_sq.copy()
        for i in range(len(d_sq.indptr[:-1])):
            row = d_sq.indices[d_sq.indptr[i] : d_sq.indptr[i + 1]]
            num = 2 * sigmas[i] * sigmas[row]
            den = sigmas_sq[i] + sigmas_sq[row]
            w.data[d_sq.indptr[i] : d_sq.indptr[i + 1]] = np.sqrt(num / den) * np.exp(
                -d_sq.data[d_sq.indptr[i] : d_sq.indptr[i + 1]] / den
            )
        w = w.tolil()
        for i, row in enumerate(indices):
            for j in row:
                if i not in set(indices[j]):
                    w[j, i] = w[i, j]
        w = w.tocsr()

    return w


# `_smooth_knn_dist`, `_membership_strengths` and the symmetrisation in `umap` below are ported from
# umap-learn 0.6, `umap/umap_.py` (`smooth_knn_dist`, `compute_membership_strengths` and
# `fuzzy_simplicial_set`), Copyright (c) 2017, Leland McInnes, BSD 3-Clause License
# (https://github.com/lmcinnes/umap/blob/master/LICENSE.txt).
# They keep its numba options so results stay bit-identical; umap-learn's versions are not
# disk-cached and are recompiled in every session.
_UMAP_FASTMATH = {"nsz", "arcp", "contract", "afn", "reassoc"}
_FLOATMAX = np.finfo(np.float32).max


@numba.njit(  # noqa: TID251 (fau's njit takes no `fastmath`/`locals`, needed to match umap-learn bit for bit)
    locals={
        "psum": numba.types.float32,
        "lo": numba.types.float32,
        "mid": numba.types.float32,
        "hi": numba.types.float32,
    },
    parallel=True,
    fastmath=_UMAP_FASTMATH,
    cache=True,
)
def _smooth_knn_dist(distances, k, local_connectivity):  # noqa: PLR0912
    n_iter = 64
    target = np.log2(k)
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    result = np.zeros(distances.shape[0], dtype=np.float32)
    flat_distances = distances.ravel()
    finite_distances = flat_distances[np.isfinite(flat_distances)]
    mean_distances = np.mean(finite_distances) if finite_distances.shape[0] > 0 else 0.0
    for i in numba.prange(distances.shape[0]):
        lo = 0.0
        hi = _FLOATMAX
        mid = 1.0
        ith_distances = distances[i]
        finite_ith_distances = ith_distances[np.isfinite(ith_distances)]
        non_zero_dists = finite_ith_distances[finite_ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > 1e-5:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)
        for _ in range(n_iter):
            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0
            if np.fabs(psum - target) < 1e-5:
                break
            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi >= _FLOATMAX:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0
        result[i] = mid
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(finite_ith_distances)
            result[i] = max(result[i], 1e-3 * mean_ith_distances)
        elif result[i] < 1e-3 * mean_distances:
            result[i] = 1e-3 * mean_distances
    return result, rho


@numba.njit(  # noqa: TID251 (fau's njit takes no `fastmath`/`locals`, needed to match umap-learn bit for bit)
    locals={
        "knn_dists": numba.types.float32[:, ::1],
        "sigmas": numba.types.float32[::1],
        "rhos": numba.types.float32[::1],
        "val": numba.types.float32,
    },
    parallel=True,
    fastmath=_UMAP_FASTMATH,
    cache=True,
)
def _membership_strengths(knn_indices, knn_dists, sigmas, rhos):
    """Membership strength of each kNN edge; 0 for self and for missing (-1) neighbours."""
    n_samples, n_neighbors = knn_indices.shape
    vals = np.zeros(knn_indices.shape, dtype=np.float32)
    for i in numba.prange(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1 or knn_indices[i, j] == i:
                continue
            if knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))
            vals[i, j] = val
    return vals


def umap(
    knn_indices: NDArray[np.int32 | np.int64],
    knn_dists: NDArray[np.float32 | np.float64],
    *,
    n_obs: int,
    n_neighbors: int,
    set_op_mix_ratio: float = 1.0,
    local_connectivity: float = 1.0,
) -> CSRBase:
    """Compute UMAP connectivities, as `umap.fuzzy_simplicial_set` :cite:p:`McInnes2018`.

    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.

    Re-implemented with a disk-cacheable kernel: umap-learn's is recompiled in every session.
    """
    knn_indices = np.ascontiguousarray(knn_indices)
    knn_dists = np.ascontiguousarray(knn_dists, dtype=np.float32)
    sigmas, rhos = _smooth_knn_dist(
        knn_dists, float(n_neighbors), float(local_connectivity)
    )
    vals = _membership_strengths(knn_indices, knn_dists, sigmas, rhos)
    k = knn_indices.shape[1]
    graph = sparse.csr_matrix(  # noqa: TID251 (umap-learn builds this class; `+`/`multiply` below rely on its semantics)
        (
            vals.ravel(),
            np.maximum(knn_indices, 0).ravel(),
            np.arange(0, n_obs * k + 1, k),
        ),
        shape=(n_obs, n_obs),
    )
    graph.sum_duplicates()
    graph.eliminate_zeros()
    transpose = graph.T.tocsr()
    prod = graph.multiply(transpose)
    if set_op_mix_ratio == 1.0:
        result = graph + transpose - prod
    else:
        result = (
            set_op_mix_ratio * (graph + transpose - prod)
            + (1.0 - set_op_mix_ratio) * prod
        )
    result = result.tocsr()
    result.eliminate_zeros()
    return result


def jaccard(
    knn_indices: NDArray[np.int32 | np.int64],
    *,
    n_obs: int,
    n_neighbors: int,
) -> CSRBase:
    """Derive Jaccard connectivities between data points from kNN indices.

    Re-implements the weighting method from Phenograph, :cite:p:`Levine2015`.

    Parameters
    ----------
    knn_indices
        The input matrix of nearest neighbor indices for each cell.
    n_obs
        Number of cells in the data-set.
    n_neighbors
        The number of nearest neighbors to consider.

    """
    # Construct unweighted kNN adjacency matrix (self excluded, as in PhenoGraph)
    adjacency = _get_sparse_matrix_from_indices_distances(
        knn_indices, np.ones_like(knn_indices), keep_self=False
    )

    # Compute |N(i) ∩ N(j)|
    i_idx = np.repeat(np.arange(n_obs), n_neighbors - 1)
    j_idx = knn_indices[:, 1:].ravel()
    rows_i = adjacency[i_idx, :]
    rows_j = adjacency[j_idx, :]
    shared = np.asarray(rows_i.multiply(rows_j).sum(axis=1)).ravel()

    # Jaccard index
    jaccard = shared / (2 * (n_neighbors - 1) - shared)

    # Build connectivity matrix, filtering out zeros
    mask = jaccard != 0
    connectivities = sparse.csr_matrix(  # noqa: TID251
        (jaccard[mask], (i_idx[mask], j_idx[mask])),
        shape=(n_obs, n_obs),
    )

    # Symmetrize by averaging (as default in PhenoGraph)
    connectivities = (connectivities + connectivities.T) / 2

    return connectivities
