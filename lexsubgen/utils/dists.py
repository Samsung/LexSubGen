from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse import csr_matrix, vstack


def parallel_argsort(matrix: np.ndarray, n_jobs: int = 4) -> np.ndarray:
    """
    Parallel argsort, i.e. batched processing of matrix where each batch
    is processed in parallel.

    Args:
        matrix: matrix to sort along last axis.
        n_jobs: number of workers

    Returns:
        indexes of elements in a sorted array that they have in original one
    """

    def task(batch):
        return np.argsort(batch, axis=-1)[:, ::-1]

    sorted_ids = np.zeros(matrix.shape, dtype=int)
    batch_size = int(np.ceil(matrix.shape[0] / n_jobs))
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        res_iter = pool.map(
            task, [matrix[i * batch_size : (i + 1) * batch_size] for i in range(n_jobs)]
        )
    for i, ids in enumerate(res_iter):
        s, e = i * batch_size, (i + 1) * batch_size
        sorted_ids[s:e] = ids
    return sorted_ids


def fast_np_sparse_batch_combine_two_dists(
    batch_fwd_dist: np.ndarray, batch_bwd_dist: np.ndarray
) -> np.ndarray:
    """
    Performs parallel combination of two distributions coming from backward and forward passes.
    Used to combine forward and backward passes of recurrent neural networks.

    Args:
        batch_fwd_dist: distribution coming from the forward pass.
        batch_bwd_dist: distribution coming from the backward pass.

    Returns:
        `numpy.ndarray` - combination of distributions.
    """
    vs = batch_fwd_dist.shape[-1]
    q_sparse = csr_matrix(
        (np.logspace(0.1, 100, num=vs, base=1.057)[::-1], (range(vs), range(vs))),
        shape=(vs, vs),
    )
    fwd_sorted_ids = parallel_argsort(batch_fwd_dist, n_jobs=20)
    bwd_sorted_ids = parallel_argsort(batch_bwd_dist, n_jobs=20)
    matrices = []
    for sample_num, (fwd_ids, bwd_ids) in enumerate(
        zip(fwd_sorted_ids, bwd_sorted_ids)
    ):
        rows = np.hstack([fwd_ids, bwd_ids])
        cols = np.hstack([np.arange(vs), np.arange(vs)])
        sparse_matrix = csr_matrix(
            (np.ones(2 * vs, dtype=bool), (rows, cols)), shape=(vs, vs)
        ).astype(np.int8)
        matrices.append(sparse_matrix)
    big_sparse_matrix = vstack(matrices)
    batch_logits = (big_sparse_matrix * q_sparse).max(axis=-1).toarray()
    return batch_logits.reshape(batch_fwd_dist.shape)
