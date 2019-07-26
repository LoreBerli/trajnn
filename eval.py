import numpy as np

def homography_transform(x, v):
    q = v * [x; np.ones(1, x.shape(1))]
    p = q(3,:)
    y = [q(1,:)./p; q(2,:)./p]
    return y