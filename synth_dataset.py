import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from scipy.signal import medfilt, medfilt2d
import scipy.ndimage.filters as scifilt
from scipy.signal import savgol_filter
plt.ion()   # interactive mode
import random
import math
import matplotlib.cm as cm
from random import shuffle
import io
import PIL.Image
import time




class TrackDataset():
    def __init__(self, past_len,fut_len, queue,cfg,n_roads=3, scale=60, pad=1800, radius=9, old=10, extra=60, size=160, transform=None):

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
        self.extra = extra
        self.old = old
        self.size = size
        self.q=queue
        self.cfg=cfg


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

    def genTrackVect(self, X, theta, vel, squaredSigmaInno):
        tStep = self.past_len + self.fut_len + self.old + self.extra
        #tt = time.time()
        inno = 0 + np.sqrt(squaredSigmaInno) * np.random.normal(size=(tStep, 1))
        u = self.chain(tStep) + 1
        # accelerate = np.random.randint(-1, 1)  # -1 dec, 0 uniform, 1 acc
        # if accelerate == -1:
        # accl = (1.0+(np.random.random()-0.5)/30.0)
        # accl = (1.0 + (1 - 0.5) / 30.0)
        # accl = 0.08
        accl = np.clip(np.random.normal(scale=0.02), -0.08, 0.08)

        thetas_ = (u[:tStep] - 2) / 70 + inno[:tStep]
        thetas = np.cumsum([theta] + thetas_[:, 0].tolist())[1:]
        angles = np.array([np.cos(thetas), np.sin(thetas)])
        # rnd_steps = vel * np.cumprod(np.tile(accl, (1, tStep)))
        # vels = np.maximum(vel + np.cumsum(np.tile(accl, (1, tStep))), 0)
        acc_vels = np.maximum(vel + np.cumsum(np.tile(accl, (1, self.past_len + self.fut_len + self.extra))), 0)
        const_vels = np.tile(vel, (self.old))
        vels = np.concatenate((const_vels, acc_vels))
        # vels[:self.old] = vel
        vels[-self.extra:] = vel
        dirs_ = vels * angles
        Xtraj = X + np.cumsum(dirs_, 1).T
        #print('gen track time: {}'.format(time.time() - tt))
        return Xtraj

    def paint(self,ms,scaled_track_int):
        for i in range(len(scaled_track_int)):
                ms[scaled_track_int[i, 1] - self.radius:scaled_track_int[i, 1] + self.radius,
                    scaled_track_int[i, 0] - self.radius:scaled_track_int[i, 0] + self.radius] = 1
        return ms

    def gen(self):
        m = np.zeros((self.scale + self.pad, self.scale + self.pad))
        gDD = np.array([600, 600], np.float64)
        pasts = []
        futs = []

        #print('---getitem time: {}'.format(time.time() - ttt))
        for _ in range(self.n_roads):
            iidx = random.randint(20, 60)
            start_vel = np.clip(np.random.normal(loc=40 / 18, scale=0.4), 0, 50 / 18)  # vel_km/h = vel pixel/frame * 18
            start_angle = np.random.rand() * np.pi * 2
            sigma = np.random.random() / 200 + 0.0005
            # g = self.genTrackVect(gDD, start_angle, start_vel, 0.001)
            g = self.genTrackVect(gDD, start_angle, start_vel, sigma)
            gDD = g[iidx]
            scaled_track = g  # * self.scale + self.scale + self.pad
            pst = scaled_track[self.old:self.past_len + self.old]
            pasts.append(pst)
            fut = scaled_track[self.old + self.past_len:-self.extra]
            futs.append(fut)
            scaled_track_int = scaled_track.astype(np.int)
            m = self.paint(m, scaled_track_int)

            #print('---getitem time: {}'.format(time.time() - ttt))

        #print('---getitem time: {}'.format(time.time() - ttt))
        tra = random.randint(0, len(pasts) - 1)
        past = pasts[tra]
        future = futs[tra]
        last_pt = past[-1]
        last_pt = last_pt.astype(np.int)

        # m = scifilt.gaussian_filter(m, sigma=4, truncate=1)
        # m = scifilt.generic_filter(m, np.max, (2, 2))
        # m = sess.run(pooled, feed_dict={in_map: np.expand_dims(np.expand_dims(m, 0), -1)}).squeeze()
        #print('filtering: {}'.format(time.time() - tt))
        m = m[last_pt[1] - self.size:last_pt[1] + self.size, last_pt[0] - self.size:last_pt[0] + self.size]
        m += m * 2 - np.array(ndimage.binary_erosion(m, structure=self.fil).astype(np.int))
        m = medfilt2d(m, kernel_size=5)


        #print('---getitem time: {}'.format(time.time() - ttt))
        m = np.expand_dims(m, 0)


        mapp=np.array(m)
        self.count-=1
        past = np.squeeze(past - last_pt).astype(np.float32)/2.0
        future = np.squeeze(future - last_pt).astype(np.float32)/2.0

        return 0,past,future,[0,0],0,0,1,mapp,mapp

    def populate_queue(self):
        while(True):
            X = np.zeros([self.cfg['batch'], self.cfg['prev_leng'], self.cfg['dims']])
            gt = np.zeros([self.cfg['batch'], self.cfg['fut_leng'], self.cfg['dims']])
            imgs = []
            imgas=[]
            for b in range(0, self.cfg['batch']):
                '''
                istances --> past       (batch_size, past_len, 2)
                labels ----> future     (batch_size, future_len, 2)
                scene -----> map        (batch_size, 360, 360)
                '''

                _, istances, labels, _, _, _, _, scene, _ = self.gen()
                X[b, :] = istances
                gt[b, :] = labels
                sc=scene.transpose([1, 2, 0])
                m = np.eye(4)[np.array(sc, dtype=np.int32)]
                m = np.squeeze(m)
                imgs.append(m)
                imgas.append(sc)
                # info.append([(index,video_track,number_vec)])
                b += 1
            toX = np.array(X)
            toGT = np.array(gt)
            self.q.put([toX, toGT, " ", imgs,imgas])

    def get(self):
        return self.gen()


