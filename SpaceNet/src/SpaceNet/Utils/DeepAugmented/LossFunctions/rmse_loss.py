from SpaceNet.Utils.DeepAugmented.LossFunctions.loss import PermutatedLoss

import tensorflow as tf


class RMSELoss(PermutatedLoss):
    """
    Root Means Square Error
    """

    def __init__(self, gain: float = 1, **kwargs):
        super().__init__(**kwargs)
        self.gain = gain

    def compute_error(
        self, ground_truth: tf.Tensor, predictions: tf.Tensor
    ) -> tf.Tensor:
        """

        Parameters
        ----------
        ground_truth: tf.Tensor (batch,1,2,d)
        predictions: tf.Tensor (batch,candidates,d)

        Returns
        -------

        """
        DELAY = 0
        ground_truth_delay = ground_truth[:, :, DELAY, :]  # (batch,1,d)
        delay = predictions
        delta_delay = (ground_truth_delay - delay) * self.gain
        return delta_delay
