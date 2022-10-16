import torch

from hw_asr.augmentations.base import AugmentationBase

class FrequencyMasking(AugmentationBase):
    def __init__(self, p, max_width):
        self.p = p
        self.max_width = max_width

    def __call__(self, spec):
        print(spec.shape)
        return spec

    def __repr__(self):
        return 'FrequencyMasking({:.3f}, {})'.format(self.p, self.max_width)
