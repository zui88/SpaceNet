import itertools
from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow import keras


class PermutatedLoss(keras.losses.Loss, ABC):

    #todo remove the hard coded k_est
    def __init__(self, d_sources: int = 4, **kwargs):
        super().__init__(**kwargs)

        self.d_sources = d_sources

        self._permutations = tf.constant(
            list(itertools.permutations(range(d_sources))),
            dtype=tf.int32,
        )


    @abstractmethod
    def compute_error(
        self,
        ground_truth: tf.Tensor,
        predictions: tf.Tensor,
    ) -> tf.Tensor:
        raise RuntimeError(
            "Loss function not implemented"
        )


    @tf.function
    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Parameters
        ----------
        y_true.shape = (batch, k, ...)
        y_pred.shape = (batch, k, ...)
        """

        # (batch, n_perm, k, ...)
        predictions_permuted = tf.gather(y_pred, self._permutations, axis=1)

        # (batch, 1, k, ...)
        ground_truth = y_true[:, None]

        # (batch, n_perm, k, ...)
        error = self.compute_error(ground_truth, predictions_permuted)

        # flatten all dimensions except
        # batch and permutation
        error = tf.reshape(error, (tf.shape(error)[0], tf.shape(error)[1], -1))

        # (batch, n_perm)
        losses = tf.sqrt(1.0 / tf.cast(self.d_sources, tf.float32)) * tf.norm(error, axis=-1)

        # (batch,)
        return tf.reduce_min(losses, axis=1)