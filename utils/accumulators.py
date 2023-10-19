import torch
import torch.utils._pytree as pytree
from copy import deepcopy

class Mean:
    """
    Running average of the values that are 'add'ed
    """
    def __init__(self, update_weight=1):
        """
        :param update_weight: 1 for normal, 2 for t-average
        """
        self.average = None
        self.counter = 0
        self.update_weight = update_weight

    def add(self, value, weight=1):
        """Add a value to the accumulator"""
        self.counter += weight
        if self.average is None:
            self.average = deepcopy(value)
        else:
            delta = value - self.average
            self.average += delta * self.update_weight * weight / (self.counter + self.update_weight - 1)
            if isinstance(self.average, torch.Tensor):
                self.average.detach()

    def value(self):
        """Access the current running average"""
        return self.average

class EMA:
    """
    Exponential moving average
    """
    def __init__(self, alpha=0.995):
        """
        :param update_weight: 1 for normal, 2 for t-average
        """
        self.average = None
        self.cum_weight = None
        self.alpha=alpha

    def add(self, value):
        """Add a value to the accumulator"""
        if self.average is None:
            self.cum_weight = (1 - self.alpha)
            self.average = pytree.tree_map(lambda x: x.detach() * (1 - self.alpha), value)
        else:
            assert self.cum_weight is not None
            self.cum_weight = self.cum_weight * self.alpha + 1. * (1 - self.alpha)
            average_tree, tree_spec = pytree.tree_flatten(self.average)
            value_tree, _ = pytree.tree_flatten(value)
            self.average = pytree.tree_unflatten([a * self.alpha + b.detach() * (1-self.alpha) for (a, b) in zip(average_tree, value_tree)], tree_spec)

    def value(self):
        """Access the current running average"""
        return pytree.tree_map(lambda x: x / self.cum_weight, self.average)


class Max:
    """
    Keeps track of the max of all the values that are 'add'ed
    """
    def __init__(self):
        self.max = None

    def add(self, value):
        """
        Add a value to the accumulator.
        :return: `true` if the provided value became the new max
        """
        if self.max is None or value > self.max:
            self.max = deepcopy(value)
            return True
        else:
            return False

    def value(self):
        """Access the current running average"""
        return self.max
