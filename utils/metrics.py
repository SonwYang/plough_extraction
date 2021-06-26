"""Metrics for segmentation.
"""

import torch
import math
import numpy as np
import torchvision


class Metrics:
    """Tracking mean metrics
    """

    def __init__(self, labels):
        """Creates an new `Metrics` instance.

        Args:
          labels: the labels for all classes.
        """

        self.labels = labels

        self.tn = 0
        self.fn = 0
        self.fp = 0
        self.tp = 0

    def add(self, actual, predicted):
        """Adds an observation to the tracker.

        Args:
          actual: the ground truth labels.
          predicted: the predicted labels.
        """
        predicted = torch.squeeze(predicted)
        # a = 1
        # b = 0
        # transform = torchvision.transforms.ToTensor()
        # a, b = transform(a), transform(b)
        # masks = torch.where(predicted > 0.5, torch.tensor(a).to(predicted.device), torch.tensor(b).to(predicted.device))
        masks = torch.argmax(predicted, 0)
        confusion = masks.view(-1).float() / actual.view(-1).float()

        self.tn += torch.sum(torch.isnan(confusion)).item()
        self.fn += torch.sum(confusion == float("inf")).item()
        self.fp += torch.sum(confusion == 0).item()
        self.tp += torch.sum(confusion == 1).item()

    def get_precision(self):
        try:
            precision = self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            precision = float("Inf")

        return precision

    def get_false_alarm(self):
        try:
            falseAlarm = self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            falseAlarm = float("Inf")

        return falseAlarm

    def get_missing_alarm(self):
        try:
            missing_alarm = self.fn / (self.tp + self.fn)
        except ZeroDivisionError:
            missing_alarm = float("Inf")

        return missing_alarm

    def get_miou(self):
        """Retrieves the mean Intersection over Union score.

        Returns:
          The mean Intersection over Union score for all observations seen so far.
        """
        return np.nanmean([self.tn / (self.tn + self.fn + self.fp), self.tp / (self.tp + self.fn + self.fp + 0.00000001)])

    def get_fg_iou(self):
        """Retrieves the foreground Intersection over Union score.

        Returns:
          The foreground Intersection over Union score for all observations seen so far.
        """

        try:
            iou = self.tp / (self.tp + self.fn + self.fp)
        except ZeroDivisionError:
            iou = float("Inf")

        return iou

    def get_mcc(self):
        """Retrieves the Matthew's Coefficient Correlation score.

        Returns:
          The Matthew's Coefficient Correlation score for all observations seen so far.
        """

        try:
            mcc = (self.tp * self.tn - self.fp * self.fn) / math.sqrt(
                (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)
            )
        except ZeroDivisionError:
            mcc = float("Inf")

        return mcc


# Todo:
# - Rewrite mIoU to handle N classes (and not only binary SemSeg)
