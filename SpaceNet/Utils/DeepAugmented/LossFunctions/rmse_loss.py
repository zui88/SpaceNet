from SpaceNet.Utils.DeepAugmented.LossFunctions.loss import PermutatedLoss


class RMSELoss(PermutatedLoss):
    """
    Root Means Square Error
    """


    def __init__(self, gain: float = 1, **kwargs):
        super().__init__(**kwargs)
        self.gain = gain


    def compute_error(self, ground_truth, predictions):
        gt_delay = ground_truth[:, 0]
        delay = predictions
        delta_delay = (gt_delay - delay) * self.gain
        return delta_delay