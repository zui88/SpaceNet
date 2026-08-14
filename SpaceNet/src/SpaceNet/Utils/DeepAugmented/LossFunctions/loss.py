from abc import ABC, abstractmethod
from typing import Generator
import itertools

from tensorflow import keras
import tensorflow as tf
import numpy as np


def create_permutation_from_estimated_sources(predictions: np.ndarray) -> Generator:
    """
    Produce a permutational set of the predictions from k estimated sources.

    PARAMETERS
    ----------
    predictions
        size (k_est)
    """
    k_est = predictions.shape[0]
    perms = list(itertools.permutations(range(int(k_est))))

    for perm in perms:
        # don't know how many K because of faculty!
        yield tf.gather(predictions, perm)


class PermutatedLoss(keras.losses.Loss, ABC):
    # todo: remove the hard coded k_est
    def __init__(self, d_sources: int = 4, **kwargs):
        super().__init__(**kwargs)

        self.d_sources = d_sources
        self._permutations = tf.constant(
            list(itertools.permutations(range(d_sources))), dtype=tf.int32
        )

    @abstractmethod
    def compute_error(
        self,
        ground_truth: tf.Tensor,
        predictions: tf.Tensor,
    ) -> tf.Tensor:
        raise RuntimeError("Loss function not implemented")

    def loss(
        self, y_true: np.ndarray, y_pred: np.ndarray, verbose: bool = False
    ) -> np.ndarray:
        """
        This function is meant to be called by the user as to get the error.

        Parameters
        ----------
        y_true (batch, k)
        y_pred (batch, k)
        """
        loss_min_batched = []
        for ground_truth, predictions in zip(y_true, y_pred):
            ground_truth = np.expand_dims(ground_truth, axis=(0,1))
            if verbose:
                print(
                    f"truth: {ground_truth}, prediction: {predictions}, error: {self.compute_error(ground_truth, predictions)}"
                )

            loss_permuted = []
            for perm_pred in create_permutation_from_estimated_sources(predictions):
                error = self.compute_error(ground_truth, perm_pred)
                error = np.sqrt(1 / predictions.shape[0]) * np.linalg.norm(error)
                loss_permuted.append(error)
            loss_min_batched.append(np.min(np.stack(loss_permuted)))

        return np.stack(loss_min_batched)

    @tf.function
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        This function is meant to be called by the training pipeline.  It is optimized to work inside graph computation.

        Parameters
        ----------
        y_true (batch, k)
        y_pred (batch, k)
        """

        # (batch, n_perm, k)
        predictions_permuted = tf.gather(y_pred, self._permutations, axis=1)

        # (batch, 1, k)
        ground_truth = y_true[:, None]

        # (batch, n_perm, k)
        error = self.compute_error(ground_truth, predictions_permuted)

        # (batch, n_perm)
        losses = tf.sqrt(
            1.0 / tf.cast(self.d_sources, tf.float32)
        ) * tf.keras.ops.linalg.norm(error, axis=-1)

        # (batch,)
        min_loss = tf.reduce_min(losses, axis=1)

        return min_loss
