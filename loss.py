import torch
import torch.nn.functional as F
import torch.nn as nn


class WeightedFocalLoss2d(nn.Module):
    def __init__(self, gamma=2, power=1):
        super(WeightedFocalLoss2d, self).__init__()
        self.gamma = gamma
        self.power = power

    def crop(self, w, h, target):
        nt, ht, wt = target.size()
        offset_w, offset_h = (wt - w) // 2, (ht - h) // 2
        if offset_w > 0 and offset_h > 0:
            target = target[:, offset_h:-offset_h, offset_w:-offset_w]

        return target

    def to_one_hot(self, target, size):
        n, c, h, w = size

        ymask = torch.FloatTensor(size).zero_()
        new_target = torch.LongTensor(n, 1, h, w)
        if target.is_cuda:
            ymask = ymask.cuda(target.get_device())
            new_target = new_target.cuda(target.get_device())

        new_target[:, 0, :, :] = torch.clamp(target.detach(), 0, c - 1)
        ymask.scatter_(1, new_target, 1.0)

        return torch.autograd.Variable(ymask)

    def forward(self, input, target, weight=None):
        target = torch.squeeze(target)

        n, c, h, w = input.size()
        log_p = F.log_softmax(input, dim=1)

        target = self.crop(w, h, target)
        ymask = self.to_one_hot(target, log_p.size())

        if weight is not None:
            weight = torch.squeeze(weight)
            weight = self.crop(w, h, weight)
            for classes in range(c):
                ymask[:, classes, :, :] = ymask[:, classes, :, :] * (weight ** self.power)

        dweight = (1 - F.softmax(input, dim=1)) ** self.gamma
        logpy = (log_p * ymask * dweight).sum(1)
        loss = -(logpy).mean()

        return loss


def _weighted_cross_entropy_loss(preds, edges):
    """ Calculate sum of weighted cross entropy loss. """
    # Reference:
    #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
    #   https://github.com/s9xie/hed/issues/7
    mask = (edges > 0.5).float()
    b, c, h, w = mask.shape
    num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [b,].
    num_neg = c * h * w - num_pos                     # Shape: [b,].
    weight = torch.zeros_like(mask)
    weight[edges > 0.5] = num_neg / (num_pos + num_neg)
    weight[edges <= 0.5] = num_pos / (num_pos + num_neg)
    # Calculate loss.
    losses = F.binary_cross_entropy_with_logits(
        preds.float(), edges.float(), weight=weight, reduction='none')
    loss = torch.sum(losses) / b
    return loss


def weighted_cross_entropy_loss(preds, edges):
    """ Calculate sum of weighted cross entropy loss. """
    # Reference:
    #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
    #   https://github.com/s9xie/hed/issues/7
    mask = (edges > 0.5).float()
    b, c, h, w = mask.shape
    num_pos = torch.sum(mask, dim=[1, 2, 3], keepdim=True).float()  # Shape: [b,].
    num_neg = c * h * w - num_pos                     # Shape: [b,].
    weight = torch.zeros_like(mask)
    #weight[edges > 0.5]  = num_neg / (num_pos + num_neg)
    #weight[edges <= 0.5] = num_pos / (num_pos + num_neg)
    weight.masked_scatter_(edges > 0.5,
        torch.ones_like(edges) * num_neg / (num_pos + num_neg))
    weight.masked_scatter_(edges <= 0.5,
        torch.ones_like(edges) * num_pos / (num_pos + num_neg))
    # Calculate loss.
    # preds=torch.sigmoid(preds)
    # criterion = WeightedFocalLoss2d()
    # edges = torch.squeeze(edges)
    # losses = criterion(preds.float(), edges.float(), weight=weight)
    losses = F.binary_cross_entropy_with_logits(
        preds.float(), edges.float(), weight=weight, reduction='none')
    loss = torch.sum(losses) / b
    return loss


def bdcn_loss(inputs, targets, l_weight=1.1):
    # clip edge pixels
    inputs = inputs[:, :, 16:-16, 16:-16]
    targets = targets[:, :, 16:-16, 16:-16]

    mask = (targets > 0.).float()
    b, c, h, w = mask.shape
    pos = torch.sum(mask, dim=[1, 2, 3], keepdim=True).float()
    weight = torch.zeros_like(mask)  # Shape: [b,].
    neg = c * h * w - pos
    beta = neg * 1. / (pos + neg)
    weight.masked_scatter_(targets > 0.,
                           torch.ones_like(targets) * beta)
    weight.masked_scatter_(targets <= 0.,
                           torch.ones_like(targets) * (1.1 * (1 - beta)))
    # weights[i, t == 1] = neg * 1. / valid
    # weights[i, t == 0] = pos * balance / valid
    # weights = torch.Tensor(weights)

    # ### label smoothing
    # targets = torch.where(targets == 1., 0.95, 0.05)
    inputs = torch.sigmoid(inputs)
    # loss = nn.BCELoss(weight, size_average=False)(inputs, targets)
    loss = torch.nn.BCELoss(weight, reduction='sum')(inputs, targets)
    # loss = F.binary_cross_entropy(inputs, targets,weight)
    # loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=weight)
    return l_weight*loss


def bdcn_lossV2(inputs, targets, wm=None, l_weight=1.1):
    # clip edge pixels
    inputs = inputs[:, :, 16:-16, 16:-16]
    targets = targets[:, :, 16:-16, 16:-16]

    mask = (targets > 0.).float()
    b, c, h, w = mask.shape
    pos = torch.sum(mask, dim=[1, 2, 3], keepdim=True).float()
    weight = torch.zeros_like(mask)  # Shape: [b,].
    neg = c * h * w - pos
    beta = neg * 1. / (pos + neg)
    weight.masked_scatter_(targets > 0.,
                           torch.ones_like(targets) * beta)
    weight.masked_scatter_(targets <= 0.,
                           torch.ones_like(targets) * (11 * (1 - beta)))

    ### label smoothing
    targets = torch.where(targets == 1., 0.95, 0.05)

    inputs = torch.sigmoid(inputs)
    if wm is not None:
        loss = torch.nn.BCELoss(weight, reduction='none')(inputs, targets)
        wm = wm[..., 16:-16, 16:-16]
        loss = loss * wm
        loss = loss.mean()
    else:
        loss = torch.nn.BCELoss(weight, reduction='mean')(inputs, targets)
    return l_weight*loss


def bdcn_lossV3(inputs, targets, wm=None, l_weight=1.1):
    mask = (targets > 0.).float()
    b, c, h, w = mask.shape
    pos = torch.sum(mask, dim=[1, 2, 3], keepdim=True).float()
    weight = torch.zeros_like(mask)  # Shape: [b,].
    neg = c * h * w - pos
    beta = neg * 1. / (pos + neg)
    weight.masked_scatter_(targets > 0.,
                           torch.ones_like(targets) * beta)
    weight.masked_scatter_(targets <= 0.,
                           torch.ones_like(targets) * (11 * (1 - beta)))

    ### label smoothing
    targets = torch.where(targets == 1., 0.95, 0.05)

    inputs = torch.sigmoid(inputs)
    if wm is not None:
        loss = torch.nn.BCELoss(weight, reduction='none')(inputs, targets)
        loss = loss * wm
        loss = loss.mean()
    else:
        loss = torch.nn.BCELoss(weight, reduction='mean')(inputs, targets)
    return l_weight*loss


def bdcn_lossV4(inputs, targets, wm=None, l_weight=1.1):
    mask = (targets > 0.).float()
    b, c, h, w = mask.shape
    pos = torch.sum(mask, dim=[1, 2, 3], keepdim=True).float()
    weight = torch.zeros_like(mask)  # Shape: [b,].
    neg = c * h * w - pos
    beta = neg * 1. / (pos + neg)
    weight.masked_scatter_(targets > 0.,
                           torch.ones_like(targets) * beta)
    weight.masked_scatter_(targets <= 0.,
                           torch.ones_like(targets) * (11 * (1 - beta)))

    ### label smoothing
    targets = torch.where(targets == 1., 0.95, 0.05)

    inputs = torch.sigmoid(inputs)
    if wm is not None:
        loss = torch.nn.BCELoss(weight, reduction='none')(inputs, targets)
        loss = loss * wm
        loss = loss.mean()
    else:
        loss = torch.nn.BCELoss(weight, reduction='mean')(inputs, targets)
    return l_weight*(loss + torch.mean((inputs-targets)**2) * 0.5)

# PyTorch
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


def MultiLoss(inputs, targets, wm=None, l_weight=1.1):
    awl = AutomaticWeightedLoss(3)
    criterion1 = bdcn_lossV3
    criterion2 = IoULoss()
    criterion3 = F.smooth_l1_loss

    # clip edge pixels
    # inputs = inputs[:, :, 16:-16, 16:-16]
    # targets = targets[:, :, 16:-16, 16:-16]
    # if wm is not None:
    #     wm = wm[..., 16:-16, 16:-16]

    loss1 = criterion1(inputs, targets, wm, l_weight)
    loss2 = criterion2(inputs, targets)
    loss3 = criterion3(torch.sigmoid(inputs), targets)
    loss = awl(loss1, loss2, loss3)
    return loss





class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


if __name__ == '__main__':
    import numpy as np
    # predict = torch.randn((4, 1, 256, 256))
    # target = torch.randn((4, 1, 256, 256))
    # criterion = IoULoss()
    # loss = criterion(predict, target)
    awl = AutomaticWeightedLoss(9)
    print(awl.params.detach().numpy())