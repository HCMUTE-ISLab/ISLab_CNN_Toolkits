import numpy as np

class SegmentMetrics(object):
    def __init__(self, truth, pred):
        """
        truh : np_array (W , H) values in classify form 0,1,...
        pred : np_array (W , H) values in classify form 0,1,...

        """
        self.truth = truth
        self.pred = pred
        self.smooth = 0.001
    def intersection(self):
        return np.sum(np.logical_and(self.pred, self.truth))
    def union(self):
        return np.sum(np.logical_or(self.pred, self.truth))
    def IOU(self):
        return (self.intersection() + self.smooth) / (self.union() + self.smooth)
    def mIOU(self):
        return np.mean(self.IOU())
        