# Copyright (c) 2026 PotterWhite
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
