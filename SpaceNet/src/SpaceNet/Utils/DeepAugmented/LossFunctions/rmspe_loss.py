from SpaceNet.Utils.DeepAugmented.LossFunctions.loss import PermutatedLoss

import numpy as np


class RMSPELoss(PermutatedLoss):
    """
    Root Means Square Phase Error
    """

    def __init__(self, d_source: int = 4, *args, **kwargs):
        super().__init__(d_source, *args, **kwargs)

    def compute_error(self, ground_truth, predictions):
        true_doa = ground_truth
        pred_doa = predictions

        error = (((true_doa - pred_doa) + (np.pi / 2)) % np.pi) - np.pi / 2

        return error
