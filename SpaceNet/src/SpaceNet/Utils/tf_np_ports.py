import tensorflow as tf
import numpy as np


def _generalized_eigh(
    A: tf.Tensor,
    B: tf.Tensor,
    eigvals_high_to_low: bool = False,
) -> tuple[tf.Tensor, tf.Tensor]:
    # https://discuss.ai.google.dev/t/compute-generalised-eigenvectors/22994/3
    # see NR section 11.0.5
    # L is a lower triangular matrix from the Cholesky decomposition of B
    L = tf.linalg.cholesky(B)
    # solve Y * L^T = A by a workaround
    # if https://github.com/tensorflow/tensorflow/issues/55371 is solved then this can be simplified
    Y = tf.transpose(tf.linalg.triangular_solve(L, tf.transpose(A)))
    # solve L * C = Y
    C = tf.linalg.triangular_solve(L, Y)
    # solve the equivalent eigenvalue problem
    e, v = tf.linalg.eigh(C)
    # reverse the sorting order to non-increasing
    e_rev = tf.reverse(e, axis=[-1])
    v_rev = tf.reverse(v, axis=[-1])
    eigvals_high_to_low = tf.constant(eigvals_high_to_low)
    e = tf.cond(eigvals_high_to_low, lambda: e_rev, lambda: e)
    v = tf.cond(eigvals_high_to_low, lambda: v_rev, lambda: v)
    # solve L^T * x = v, where x is the eigenvectors of the original problem
    v = tf.linalg.triangular_solve(tf.transpose(L), v, lower=False)
    # # normalize the eigenvectors
    # v = tf.math.l2_normalize(v, axis=0)
    return e, v


def eigh(
    a: tf.Tensor,
    b: tf.Tensor | None = None,
    subset_by_index: tuple[int, int] | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """To match a subset of 'eigh' of scipy package
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html
    """
    eigvals, eigvecs = _generalized_eigh(a, b) if b is not None else tf.linalg.eigh(a)
    if subset_by_index is not None:
        idx_0, idx_1 = subset_by_index
        eigvals = eigvals[idx_0 : idx_1 + 1]
        eigvecs = eigvecs[:, idx_0 : idx_1 + 1]
    return eigvals, eigvecs


def gradient(y, x, numpy: bool = False):
    # forward difference at start
    start = (y[1] - y[0]) / (x[1] - x[0])

    # central differences
    middle = (y[2:] - y[:-2]) / (x[2:] - x[:-2])

    # backward difference at end
    end = (y[-1] - y[-2]) / (x[-1] - x[-2])

    # hrrr.. what a journye
    if not numpy:
        return tf.concat([[start], middle, [end]], axis=0)
    else:
        return np.concatenate([[start], middle, [end]], axis=0)
