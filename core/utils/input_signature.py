class InputSignature:
    """
    Represents the input tensor signature of a model.
    Automatically infers semantic meaning based on tensor rank.
    """

    def __init__(self, shape):
        """
        shape: list of integers, e.g. [1, 3, 512, 512]
        """
        self.shape = shape
        self.rank = len(shape)

    @property
    def is_asr(self):
        """Return True if tensor looks like ASR input [B, T, F]."""
        return self.rank == 3

    @property
    def is_cv(self):
        """Return True if tensor looks like CV input [B, C, H, W]."""
        return self.rank == 4

    @property
    def batch(self):
        return self.shape[0]

    def as_dict(self):
        """
        Return semantic dictionary based on tensor rank.
        """
        if self.is_asr:
            return {
                "type": "ASR",
                "batch": self.shape[0],
                "time": self.shape[1],
                "feature": self.shape[2]
            }

        elif self.is_cv:
            return {
                "type": "CV",
                "batch": self.shape[0],
                "channel": self.shape[1],
                "height": self.shape[2],
                "width": self.shape[3]
            }

        else:
            return {
                "type": "UNKNOWN",
                "shape": self.shape
            }
