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




class TrackDataset():
    def __init__(self, past_len, fut_len, n_roads=5, scale=80, pad=480, radius=9, transform=None):

        self.past_len = past_len
        self.fut_len = fut_len
        self.traj_len = past_len + fut_len
        self.transform = transform
        self.n_roads = n_roads
        self.scale = scale
        self.pad = pad
        self.radius = radius
        self.fil=np.ones([6,6])
        self.count=16
        self.map=np.zeros((self.scale + self.pad, self.scale + self.pad))

    def __len__(self):
        return self.count

    def rando(self,p):
        u = np.random.random()
        i = 0
        s = p[0]
        while (u > s) and (i < len(p)):
            i += 1
            s = s + p[i]
        index = i
        return index

    def chain(self,n):
        mu = np.array([1, 0, 0])
        P = np.array([[0, 1, 0],
                      [.01, .98, .01],
                      [0, 1, 0]])
        x = np.zeros((n + 1, 1), np.int32)
        x[0] = self.rando(mu)
        for i in range(n):
            x[i + 1] = self.rando(P[x[i, 0], :])
        return x

    def genTrack(self,tStep, X, theta, vel, squaredSigmaInno):
        inno = 0 + np.sqrt(squaredSigmaInno) * np.random.normal(size=(tStep, 1))
        u = self.chain(tStep) + 1
        Xtraj = np.zeros((tStep, 2))
        for t in range(tStep):
            theta = theta + (u[t] - 2) / 70 + inno[t]
            dir_ = vel * np.array([np.cos(theta), np.sin(theta)])
            X += dir_.squeeze()
            Xtraj[t, :] = X.T
        return Xtraj

    def paint(self,ms,scaled_track_int):
        for i in range(len(scaled_track_int)):
                ms[scaled_track_int[i, 1] - self.radius:scaled_track_int[i, 1] + self.radius,
                    scaled_track_int[i, 0] - self.radius:scaled_track_int[i, 0] + self.radius] = 1
        return ms

    def gen(self):
        old=20
        m = np.zeros((self.scale + self.pad, self.scale + self.pad))
        for _ in range(self.n_roads):
            g = self.genTrack(self.traj_len + old, np.array([-1, 0], np.float64), 0.01, 0.05, 0.05)
            scaled_track = g * self.scale / 2 + self.scale / 2 + self.pad / 2
            past = scaled_track[old:self.past_len + old]
            future = scaled_track[old + self.past_len:]
            scaled_track_int = scaled_track.astype(np.int)
            m = self.paint(m, scaled_track_int)


        last_pt = past[-1]
        size = 160
        last_pt = last_pt.astype(np.int)
        m = m[last_pt[1] - size:last_pt[1] + size, last_pt[0] - size:last_pt[0] + size]


        m+=m*2-np.array(ndimage.binary_erosion(m,structure=self.fil).astype(np.int))


        past = np.squeeze(past - last_pt).astype(np.float32)/2.0
        future = np.squeeze(future - last_pt).astype(np.float32)/2.0

        m = np.expand_dims(m, 0)
        mapp=np.array(m)
        self.count-=1
        return 0,past,future,[0,0],0,0,1,mapp,mapp



    def get(self):
        return self.gen()


