import torch.nn as nn
import functools
from bisect import bisect_right


def set_bit_width(nbits, m):
    classname = m.__class__.__name__
    if classname.find('DNQ') != -1:
        m.set_dnq(nbits)
    return


def set_bit_width_wrapper(nbits=None):
    return functools.partial(set_bit_width, nbits)


class _DNQScheduler(object):
    def __init__(self, model, last_epoch=-1, nbits=4):
        if not isinstance(model, nn.Module):
            raise TypeError('{} is not an Module'.format(
                type(model).__name__))
        self.model = model
        self.base_nbits = nbits
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_nbits(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.model.apply(set_bit_width_wrapper(self.get_nbits()))


class MultiStepDNQ(_DNQScheduler):
    """Set the bit-width of each layer to the initial lr decayed
    by gamma once the number of epoch reaches one of the milestones. When
    last_epoch=-1, sets initial bitwidth  as bitwidth.
    Args:
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses nbits = 4 for all groups
        >>> # nbits = 4     if epoch < 100
        >>> # nbits = 8     if epoch>= 100
        >>> scheduler = MultiStepDNQ(model, milestones=[30,80], gamma=2)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, model, milestones, gamma=2, last_epoch=-1, nbits=4):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        super(MultiStepDNQ, self).__init__(model, last_epoch, nbits=nbits)

    def get_nbits(self):
        return self.base_nbits * self.gamma ** bisect_right(self.milestones, self.last_epoch)
