import numpy as np
from metrics.segment import SegmentMetrics

truth = np.array(
    [
        [1,1,0],
        [1,1,0],
        [0,0,0]
    ]
)

pred = np.array(
    [
        [1,1,0],
        [1,1,0],
        [0,0,1]
    ]
)


m = SegmentMetrics(truth,pred)

print(m.mIOU())