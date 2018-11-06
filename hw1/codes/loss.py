from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        return ((target - input) ** 2).mean(axis=0).sum() / 2.

    def backward(self, input, target):
        '''Your codes here'''
        return target - input
