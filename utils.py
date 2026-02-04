import tensorflow as tf
import numpy as np


def save_print_history(history):
    import matplotlib.pyplot as plt

    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("training_history.png", dpi=300)


def generalized_eigh_scipy_compatible(A, B, eps=1e-12):
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


def fix_phase_scipy(V, tol=1e-12):
    """
    Match SciPy/LAPACK phase convention:
    first non-zero element is real and positive
    """
    V_out = []

    for i in range(V.shape[1]):
        v = V[:, i]

        # find first non-zero index deterministically
        abs_v = tf.abs(v)
        mask = abs_v > tol

        # convert to python index safely
        idx = int(tf.argmax(tf.cast(mask, tf.int32)))

        pivot = v[idx]

        phase = pivot / (tf.abs(pivot) + tol)
        v_fixed = v / phase

        V_out.append(v_fixed)

    return tf.stack(V_out, axis=1)


def eigh(a, b, subset_by_index=None):
    if not tf.is_tensor(a):
        a = tf.cast(a, tf.float32)
    if not tf.is_tensor(b):
        b = tf.cast(b, tf.float32)
        
    eigvals, eigvecs = generalized_eigh_scipy_compatible(a, b)
    eigvecs          = fix_phase_scipy(eigvecs)

    if subset_by_index is not None:
        i0, i1  = subset_by_index
        eigvals = eigvals[i0:i1+1]
        eigvecs = eigvecs[:, i0:i1+1]

    return eigvals, eigvecs


def gradient(y, x, numpy : bool = False):
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
        
