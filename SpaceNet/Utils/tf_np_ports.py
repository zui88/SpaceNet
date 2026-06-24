import tensorflow as tf
import numpy as np


def _generalized_eigh_scipy_compatible(A, B, eps=1e-12):
    """
    TensorFlow replacement for scipy.linalg.eigh(A, B)
    with post-processing to match SciPy behavior.

    Returns
    -------
    eigenvalues : (N,)
    eigenvectors : (N, N)  (columns = eigenvectors)
    """

    # ---- Hermitian enforcement ----
    A = 0.5 * (A + tf.linalg.adjoint(A))
    B = 0.5 * (B + tf.linalg.adjoint(B))

    # ---- eig(B) ----
    eigvals_B, eigvecs_B = tf.linalg.eigh(B)
    eigvals_B = tf.maximum(eigvals_B, eps)

    # ---- B^{-1/2} ----
    inv_sqrt = tf.linalg.diag(1.0 / tf.sqrt(eigvals_B))
    B_inv_sqrt = eigvecs_B @ inv_sqrt @ tf.linalg.adjoint(eigvecs_B)

    # ---- whitened problem ----
    A_tilde = B_inv_sqrt @ A @ B_inv_sqrt

    eigvals, eigvecs = tf.linalg.eigh(A_tilde)

    # ---- back-transform ----
    V = B_inv_sqrt @ eigvecs   # columns = eigenvectors

    # ============================================================
    # POST-PROCESSING (match SciPy)
    # ============================================================

    # ---- 1. B-normalization ----
    # v^H B v = 1
    BV = B @ V
    norms = tf.sqrt(tf.reduce_sum(tf.math.conj(V) * BV, axis=0))
    V = V / norms

    # ---- 2. deterministic sign / phase ----
    # make largest-magnitude entry real and positive
    abs_V = tf.abs(V)
    max_idx = tf.argmax(abs_V, axis=0)  # per column

    def fix_phase(i):
        v = V[:, i]
        idx = max_idx[i]
        pivot = v[idx]

        # phase = pivot / |pivot|
        phase = pivot / tf.cast(tf.abs(pivot) + 1e-16, pivot.dtype)

        return v / phase

    V = tf.stack([fix_phase(i) for i in range(V.shape[1])], axis=1)

    return eigvals, V


@tf.function
def _fix_phase_scipy(V: tf.Tensor, tol: float = 1e-12) -> tf.Tensor:
    abs_V = tf.abs(V)
    mask = abs_V > tol

    first_nonzero_idx = tf.argmax(
        tf.cast(mask, tf.int32),
        axis=0,
        output_type=tf.int32,
    )

    n_cols = tf.shape(V)[1]
    col_idx = tf.range(n_cols, dtype=tf.int32)

    pivots = tf.gather_nd(
        V,
        tf.stack([first_nonzero_idx, col_idx], axis=1),
    )

    tol_real = tf.cast(tol, V.dtype.real_dtype)
    phases = pivots / (tf.abs(pivots) + tol_real)

    # For all-zero columns, leave the phase factor as 1.
    phases = tf.where(
        tf.abs(pivots) > tol_real,
        phases,
        tf.ones_like(phases),
    )

    return V / phases[tf.newaxis, :]


def eigh(a, b, subset_by_index=None):
    if not tf.is_tensor(a):
        a = tf.cast(a, tf.float32)
    if not tf.is_tensor(b):
        b = tf.cast(b, tf.float32)

    eigvals, eigvecs = _generalized_eigh_scipy_compatible(a, b)
    eigvecs = _fix_phase_scipy(eigvecs)

    if subset_by_index is not None:
        i0, i1 = subset_by_index
        eigvals = eigvals[i0:i1 + 1]
        eigvecs = eigvecs[:, i0:i1 + 1]

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
        return tf.concat([
            [start],
            middle,
            [end]
        ], axis=0)
    else:
        return np.concatenate([
            [start],
            middle,
            [end]
        ], axis=0)
