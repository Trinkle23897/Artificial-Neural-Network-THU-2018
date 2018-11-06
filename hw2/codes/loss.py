from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        return ((target - input) ** 2).mean(axis=0).sum() / 2.

    def backward(self, input, target):
        return target - input


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target): # B * C
        B, C = input.shape
        shin = input - np.max(input, axis=1, keepdims=True)
        loss = -np.sum((shin - np.log(np.sum(np.exp(shin), axis=1, keepdims=True))) * target) / B
        return loss

    def backward(self, input, target):
        B, C = input.shape
        shin = input - np.max(input, axis=1, keepdims=True)
        prob = np.exp(shin - np.log(np.sum(np.exp(shin), axis=1, keepdims=True)))
        return (prob - target) / B

