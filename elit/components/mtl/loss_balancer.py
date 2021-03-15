# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-01-14 13:14

from collections import defaultdict, deque

from elit.common.structure import ConfigTracker


class MovingAverage(object):
    def __init__(self, maxlen=5) -> None:
        self._queue = defaultdict(lambda: deque(maxlen=maxlen))

    def append(self, key, value: float):
        self._queue[key].append(value)

    def average(self, key) -> float:
        queue = self._queue[key]
        return sum(queue) / len(queue)


class MovingAverageBalancer(MovingAverage, ConfigTracker):

    def __init__(self, maxlen=5, intrinsic_weighting=True) -> None:
        super().__init__(maxlen)
        ConfigTracker.__init__(self, locals())
        self.intrinsic_weighting = intrinsic_weighting

    def weight(self, task) -> float:
        avg_losses = dict((k, self.average(k)) for k in self._queue)
        weight = sum(avg_losses.values()) / avg_losses[task]
        if self.intrinsic_weighting:
            cur_loss = self._queue[task][-1]
            weight *= cur_loss / avg_losses[task]

        return weight
