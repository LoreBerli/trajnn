import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
import random
import math
import matplotlib.cm as cm
from random import shuffle
import io
import PIL.Image
import time


def rando(p):
    u = np.random.random()
    i = 0
    s = p[0]
    while (u > s) and (i < len(p)):
        i += 1
        s = s + p[i]
    index = i
    return index


def chain(n):
    mu = np.array([1, 0, 0])
    P = np.array([[0, 1, 0],
                  [.01, .98, .01],
                  [0, 1, 0]])
    x = np.zeros((n + 1, 1), np.int32)
    x[0] = rando(mu)
    for i in range(n):
        x[i + 1] = rando(P[x[i, 0], :])
    return x


def genTrack(tStep, X, theta, vel, squaredSigmaInno):
    inno = 0 + np.sqrt(squaredSigmaInno) * np.random.normal(size=(tStep, 1))
    u = chain(tStep) + 1
    Xtraj = np.zeros((tStep, 2))
    for t in range(tStep):
        theta = theta + (u[t] - 2) / 70 + inno[t]
        dir_ = vel * np.array([np.cos(theta), np.sin(theta)])
        X += dir_.squeeze()
        Xtraj[t, :] = X.T
    return Xtraj

class FakeTrajMapDataset():
    def __init__(self, past_len, fut_len, n_roads=5, scale=220, pad=320, radius=5, transform=None):
        self.past_len = past_len
        self.fut_len = fut_len
        self.traj_len = past_len + fut_len
        self.transform = transform
        self.n_roads = n_roads
        self.scale = scale
        self.pad = pad
        self.radius = radius

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        m = np.zeros((self.scale + self.pad, self.scale + self.pad,4))
        for _ in range(self.n_roads):
            g = genTrack(self.traj_len + 10, np.array([-1, 0], np.float64), 0.1, 0.03, 0.03)
            scaled_track = g * self.scale / 2 + self.scale / 2 + self.pad / 2
            past = scaled_track[10:self.past_len + 10]
            future = scaled_track[10 + self.past_len:]
            scaled_track_int = scaled_track.copy().astype(np.int)

        for i in range(len(scaled_track_int)):
                m[1,scaled_track_int[i, 1] - self.radius:scaled_track_int[i, 1] + self.radius,
                    scaled_track_int[i, 0] - self.radius:scaled_track_int[i, 0] + self.radius] = 255
        m[2]=ndimage.binary_erosion(marc,structure=np.ones([4,4])).astype(mp.dtype)

        #m = np.expand_dims(m, 0)
        sample = {'past': past,
                  'future': future,
                  'map': m}
        if self.transform is not None:
            sample = self.transform(sample)
#return self.index[idx], self.istances[idx], self.labels[idx], self.presents[idx], self.video_track[idx], self.vehicles[idx], self.number_vec[idx], self.scene[idx], self.scene_one_hot[idx]

        return 0,past,future,[0,0],0,0,1,sample,sample