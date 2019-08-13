import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
# Ignore warnings
import warnings
import cv2
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

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)



class TrackDataset():
    def __init__(self, past_len,fut_len, queue,cfg,n_roads=3, scale=60, pad=2000, radius=9, old=10, extra=60, size=160, transform=None):

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
        inno = 0 + np.sqrt(squaredSigmaInno) * np.random.normal(size=(tStep, 1))
        u = self.chain(tStep) + 1

        accl = np.clip(np.random.normal(scale=0.02), -0.08, 0.08)
        thetas_ = (u[:tStep] - 2) / 70 + inno[:tStep]
        thetas = np.cumsum([theta] + thetas_[:, 0].tolist())[1:]

        if np.random.rand() > 0.5:
            # pick a random point and make a pi/2 turn in a random direction
            turn_idx = np.random.randint(0, len(thetas))
            thetas[turn_idx:] = thetas[turn_idx:] + np.random.choice([-1, 1]) * np.pi/2

        angles = np.array([np.cos(thetas), np.sin(thetas)])
        acc_vels = np.maximum(vel + np.cumsum(np.tile(accl, (1, self.past_len + self.fut_len + self.extra))), 0)
        const_vels = np.tile(vel, self.old)
        vels = np.concatenate((const_vels, acc_vels))
        vels[-self.extra:] = vel

        vels = np.minimum(vels, 70/18)  # limit velocity to 70 Km/h to avoid tracks going out from the 320x320 final map
        # the present is centered in (0,0) so the maximum allowed distance for the future trajectory is 160px = 80m
        # 80m/4s ---> 20m/s ---> 70Km/h

        dirs_ = vels * angles
        Xtraj = X + np.cumsum(dirs_, 1).T
        return Xtraj, thetas, vels

    def paint(self, ms, scaled_track_int):
        if np.random.rand() > 0.5:
            shifted_lane = self.shift_track(scaled_track_int, 18).astype(np.int32)
            for i in range(len(shifted_lane)):
                ms[shifted_lane[i, 1] - self.radius:shifted_lane[i, 1] + self.radius,
                shifted_lane[i, 0] - self.radius:shifted_lane[i, 0] + self.radius] = 1

        for i in range(len(scaled_track_int)):
            ms[scaled_track_int[i, 1] - self.radius:scaled_track_int[i, 1] + self.radius,
                scaled_track_int[i, 0] - self.radius:scaled_track_int[i, 0] + self.radius] = 1
        return ms

    @staticmethod
    def shift_track(traj, offset, thetas=None, vels=None):
        if thetas is None or vels is None:
            diffs = np.diff(traj, axis=0)
            diffs = np.concatenate((diffs, np.expand_dims(diffs[-1], 0)))
            vels, thetas = cart2pol(diffs[:, 0], diffs[:, 1])

        # hortogonal_thetas = thetas - np.pi / 2
        # shifted_traj = traj + offset * np.array([np.cos(hortogonal_thetas), np.sin(hortogonal_thetas)]).T

        hortogonal_theta = thetas[0] - np.pi / 2
        x_offset = np.expand_dims(traj[0, :] + offset * np.array([np.cos(hortogonal_theta), np.sin(hortogonal_theta)]),
                                  1)
        angles = np.array([np.cos(thetas), np.sin(thetas)])
        newtraj = (x_offset + np.cumsum(vels * angles, 1)).T
        return newtraj

    def gen(self):
        m = np.zeros((self.scale + self.pad, self.scale + self.pad))
        gDD = np.array([self.scale/2+self.pad/2, self.scale/2+self.pad/2], np.float64)
        all_tracks = []
        all_thetas = []
        all_vels = []

        for _ in range(self.n_roads):
            iidx = random.randint(20, 60)
            start_vel = np.clip(np.random.normal(loc=40 / 18, scale=0.4), 0, 50 / 18)  # vel_km/h = vel pixel/frame * 18
            start_angle = np.random.rand() * np.pi * 2
            sigma = np.random.random() / 200 + 0.0005
            g, thetas, vels = self.genTrackVect(gDD, start_angle, start_vel, sigma)
            gDD = g[iidx]
            scaled_track = g
            all_tracks.append(scaled_track)
            all_thetas.append(thetas)
            all_vels.append(vels)
            scaled_track_int = scaled_track.astype(np.int)
            m = self.paint(m, scaled_track_int)

        tra = random.randint(0, len(all_tracks) - 1)

        selected_traj = all_tracks[tra]
        shift_offset = np.random.randint(-5, 1)

        shifted_traj = self.shift_track(selected_traj, shift_offset, all_thetas[tra], all_vels[tra])
        smoothed_traj = savgol_filter(shifted_traj, 19, 3, axis=0)

        past = smoothed_traj[self.old:self.past_len + self.old]
        future = smoothed_traj[self.old + self.past_len:-self.extra]
        last_pt = past[-1]
        scaled_last=last_pt
        scaled_last=scaled_last.astype(np.int)
        last_pt = last_pt.astype(np.int)

        crop_border = 5
        m = m[scaled_last[1]- self.size - crop_border:scaled_last[1] + self.size + crop_border,
            scaled_last[0] - self.size - crop_border:scaled_last[0] + self.size + crop_border]
        m += m * 2 - np.array(ndimage.binary_erosion(m, structure=self.fil)).astype(np.int)
        m = medfilt2d(m, kernel_size=np.random.choice([3, 5, 7]))  # noisy sidewalks
        m = m[crop_border:-crop_border, crop_border:-crop_border]

        #print('---getitem time: {}'.format(time.time() - ttt))
        #m=cv2.resize(m,(self.size*2,self.size*2),interpolation=cv2.INTER_NEAREST)
        m = np.expand_dims(m, 0)


        mapp=np.array(m)
        #print(mapp.shape,self.pad,last_pt,self.scale)
        self.count-=1
        past = np.squeeze(past - last_pt).astype(np.float32)
        future = np.squeeze(future - last_pt).astype(np.float32)

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
