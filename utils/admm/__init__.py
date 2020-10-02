from .abstract_admm import *
import numpy as np
from models._modules.npu_structured_pruner import get_npu_structured_sparsity_mask


# ADMM-Percentage-Pruner-Scheduler
class AdmmPercentagePrunerScheduler(AbstractAdmmScheduler):
    def __init__(self, model, percentage, prune_linear=True, prune_first_layer=True):
        super().__init__(model)
        self.percentage = percentage
        self.prune_linear = prune_linear
        self.prune_first_layer = prune_first_layer

    def custom_Z_regulation(self, z):
        # todo: first layer BN.weight..
        if len(z.shape) == 2 and self.prune_linear:  # linear
            pcen = np.percentile(abs(z), 100 * self.percentage)
            under_threshold = abs(z) < pcen
            z.data[under_threshold] = 0
            return z
        elif len(z.shape) == 4 and z.shape[1] == 3 and self.prune_first_layer:  # first layer
            pcen = np.percentile(abs(z), 100 * self.percentage)
            under_threshold = abs(z) < pcen
            z.data[under_threshold] = 0
            return z
        elif len(z.shape) == 4 and z.shape[1] > 3:  # conv and not first layer
            pcen = np.percentile(abs(z), 100 * self.percentage)
            under_threshold = abs(z) < pcen
            z.data[under_threshold] = 0
            return z
        return z


# ADMM-NPU-Scheduler
class AdmmNpuScheduler(AbstractAdmmScheduler):
    def __init__(self, model, non_zero_num):
        super().__init__(model)
        self.non_zero_num = non_zero_num

    def custom_Z_regulation(self, z):
        if len(z.shape) != 4 or z.shape[1] <= self.non_zero_num:
            # fc or first layer not include
            return z
        mask = get_npu_structured_sparsity_mask(z, self.non_zero_num)
        return z * mask
