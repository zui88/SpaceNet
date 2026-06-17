from typing import Generator
import itertools
from abc import abstractmethod, ABC
import tensorflow as tf
from tensorflow import keras


def create_permutation_from_estimated_sources(predictions : tf.Tensor) -> Generator:
    """
    Produce a permutational set of the predictions from k estimated sources.

    PARAMETERS
    ----------
    predictions
        size (k_est x parameters)
    """
    batch_size = tf.shape(predictions)[0]
    perms      = list(itertools.permutations(range(batch_size)))

    for perm in perms:
        # don't know how many K because of faculty!
        yield tf.gather(predictions, perm)


class PermutatedLoss(keras.losses.Loss, ABC):


    @abstractmethod
    def compute_error(self, ground_truth, predictions):
        raise RuntimeError(
            "Loss function not implemented"
        )


    def call(self, y_true, y_pred, verbose: bool = False):
        ground_truth_batched = y_true
        predictions_batched  = y_pred
        loss_min_batched     = []

        for ground_truth, predictions in zip(ground_truth_batched, predictions_batched):
            if verbose: print(f"truth: {ground_truth}, prediction: {predictions}, error: {self.compute_error(ground_truth, predictions)}")
            loss_permuted = []
            for perm_pred in create_permutation_from_estimated_sources(predictions):
                error = self.compute_error(ground_truth, perm_pred)
                error = tf.math.sqrt(1 / predictions.shape[0]) * tf.linalg.norm(error)
                loss_permuted.append(error)
            loss_min_batched.append(tf.keras.ops.min(tf.stack(loss_permuted)))

        return tf.stack(loss_min_batched)