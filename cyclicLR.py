import math
from bisect import bisect_right, bisect_left

import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

class CyclicCosAnnealingLR(_LRScheduler):
    r"""
    Implements reset on milestones inspired from CosineAnnealingLR pytorch
    
    Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch > last set milestone, lr is automatically set to \eta_{min}

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list of ints): List of epoch indices. Must be increasing.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer,milestones, eta_min=0, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.eta_min = eta_min
        self.milestones=milestones
        super(CyclicCosAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        
        if self.last_epoch >= self.milestones[-1]:
            return [self.eta_min for base_lr in self.base_lrs]

        idx = bisect_right(self.milestones,self.last_epoch)
        
        left_barrier = 0 if idx==0 else self.milestones[idx-1]
        right_barrier = self.milestones[idx]

        width = right_barrier - left_barrier
        curr_pos = self.last_epoch- left_barrier 
    
        return [self.eta_min + (base_lr - self.eta_min) *
               (1 + math.cos(math.pi * curr_pos/ width)) / 2
                for base_lr in self.base_lrs]


class CyclicLinearLR(_LRScheduler):
    r"""
    Implements reset on milestones inspired from Linear learning rate decay
    
    Set the learning rate of each parameter group using a linear decay
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart:

    .. math::

        \eta_t = \eta_{min} + (\eta_{max} - \eta_{min})(1 -\frac{T_{cur}}{T_{max}})

    When last_epoch > last set milestone, lr is automatically set to \eta_{min}
  
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list of ints): List of epoch indices. Must be increasing.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer,milestones, eta_min=0, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.eta_min = eta_min
        self.milestones=milestones
        super(CyclicLinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        
        if self.last_epoch >= self.milestones[-1]:
            return [self.eta_min for base_lr in self.base_lrs]

        idx = bisect_right(self.milestones,self.last_epoch)
        
        left_barrier = 0 if idx==0 else self.milestones[idx-1]
        right_barrier = self.milestones[idx]

        width = right_barrier - left_barrier
        curr_pos = self.last_epoch- left_barrier 
    
        return [self.eta_min + (base_lr - self.eta_min) *
               (1. - 1.0*curr_pos/ width)
                for base_lr in self.base_lrs]

##warm up
class LearningRateWarmUP(object):
    def __init__(self, optimizer, target_iteration, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.target_iteration = target_iteration
        self.target_lr = target_lr
        self.num_iterations = 0
        self.after_scheduler = after_scheduler

    def warmup_learning_rate(self, cur_iteration):
        warmup_lr = self.target_lr*float(cur_iteration)/float(self.target_iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self, cur_iteration):
        if cur_iteration <= self.target_iteration:
            self.warmup_learning_rate(cur_iteration)
        else:
            self.after_scheduler.step(cur_iteration)
'''
#################################
# TEST FOR SCHEDULER
#################################
'''
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch.nn as nn
    import torch.optim as optim

    net = nn.Sequential(nn.Linear(2,2))
    milestones = [5+x*40 for x in range(5)]
    print(milestones)
    optimizer = optim.SGD(net.parameters(),lr=0.005,momentum=0.9,weight_decay=0.0005,nesterov=True)
    scheduler_c = CyclicCosAnnealingLR(optimizer, milestones=milestones, eta_min=1e-6)
    # scheduler_c = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    scheduler = LearningRateWarmUP(optimizer=optimizer, target_iteration=5, target_lr=0.005,
                                   after_scheduler=scheduler_c)

    lr_log = []

    for i in range(100):
        # optimizer.step()
        scheduler.step(i)
        for param_group in optimizer.param_groups:
            lr_log.append(param_group['lr'])

    plt.plot(lr_log)
    plt.show()

