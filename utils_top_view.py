import random
import os
import multiprocessing as mp
from PIL import Image
import numpy as np
import json
import glob
import itertools
import cv2
import dataset
import utils

# video_XXXXXX
#  - frame_X
#       - object_X
#            - past
#            - future
#            - present
#            - class
#            - box
test_set = ["../kitti_rev2/training/" + d for d in os.listdir("../kitti_rev2/training/")]

def to_offsets(points):
    pts=points
    pts=np.diff(pts, 1, -2,prepend=0)
    #pts=np.concatenate(([[0,0]],offs),-2)
    return pts

def from_offsets(points):
    return np.cumsum(points,-2)

class Loader():
    def get_top_paths(self, pt):
        tops = []
        for top_d in os.listdir(pt):
            for video in os.listdir(pt + "/" + top_d):
                tops.append(pt + "/" + top_d + "/" + video)
        return tops

    def __init__(self, config):
        self.cfg = config

        tracks = json.load(open("multiple-futures/world_traj_kitti.json"))
        print("TRACKS_init")
        dim_clip = self.cfg['dim_clip']

        past_len = self.cfg['prev_leng']
        future_len = self.cfg['fut_leng']
        self.q = mp.Queue(self.cfg['batch']*self.cfg['splits']*4)
        self.q_test = mp.Queue(self.cfg['batch']*10)

        self.data_train = dataset.TrackDataset(tracks, num_istances=past_len, num_labels=future_len,
                                          train=True, dim_clip=dim_clip,rot=True,shf=True)
        print("LOADER0_init")
        self.data_test = dataset.TrackDataset(tracks, num_istances=past_len, num_labels=future_len,
                                         train=False, dim_clip=dim_clip,rot=False,shf=False)
        print("LOADER1_init")
        self.data_rand_test = dataset.TrackDataset(tracks, num_istances=past_len, num_labels=future_len,
                                         train=False, dim_clip=dim_clip,rot=False,shf=True)

        self.iter_train = itertools.cycle(self.data_train)
        self.trains=[]
        for k in range(0,2):
            self.trains.append(itertools.cycle(dataset.TrackDataset(tracks, num_istances=past_len, num_labels=future_len,
                                          train=True, dim_clip=dim_clip,rot=True,shf=True)))

        self.iter_test = itertools.cycle(self.data_test)
        self.iters=[]
        for i in range(0,2):
            self.iters.append(itertools.cycle(dataset.TrackDataset(tracks, num_istances=past_len, num_labels=future_len,
                                         train=False, dim_clip=dim_clip,rot=False,shf=True)))
        #self.iter_rand_test = itertools.cycle(self.data_rand_test)




    def serve(self):

        X = np.zeros([self.cfg['batch'], self.cfg['prev_leng'], self.cfg['dims']])
        gt = np.zeros([self.cfg['batch'], self.cfg['fut_leng'], self.cfg['dims']])
        info = []
        imgs = []
        for b in range(0,self.cfg['batch']):
            '''
            istances --> past       (batch_size, past_len, 2)
            labels ----> future     (batch_size, future_len, 2)
            scene -----> map        (batch_size, 360, 360)
            '''

            index, istances, labels, presents, video_track, vehicles, number_vec, scene, scene_one_hot = next(self.trains[b%2])
            X[b,:]=istances
            gt[b,:]=labels
            imgs.append(scene.transpose([1,2,0]))
            info.append([(index,video_track,number_vec)])
            b+=1
        return np.array(X), np.array(gt), info, imgs

    def serve_test(self):

        X = np.zeros([self.cfg['batch'], self.cfg['prev_leng'], self.cfg['dims']])
        gt = np.zeros([self.cfg['batch'], self.cfg['fut_leng'], self.cfg['dims']])
        info = []
        imgs = []
        for b in range(0,self.cfg['batch']):
            '''
            istances --> past       (batch_size, past_len, 2)
            labels ----> future     (batch_size, future_len, 2)
            scene -----> map        (batch_size, 360, 360)
            '''

            index, istances, labels, presents, video_track, vehicles, number_vec, scene, scene_one_hot = next(self.iter_test)
            X[b,:]=istances
            gt[b,:]=labels
            imgs.append(scene.transpose([1,2,0]))
            info.append([(index,video_track,number_vec)])
            b+=1
        return np.array(X), np.array(gt), info, imgs

    def serve_random_test(self):

        X = np.zeros([self.cfg['batch'], self.cfg['prev_leng'], self.cfg['dims']])
        gt = np.zeros([self.cfg['batch'], self.cfg['fut_leng'], self.cfg['dims']])
        info = []
        imgs = []
        for b in range(0,self.cfg['batch']):
            '''
            istances --> past       (batch_size, past_len, 2)
            labels ----> future     (batch_size, future_len, 2)
            scene -----> map        (batch_size, 360, 360)
            '''

            index, istances, labels, presents, video_track, vehicles, number_vec, scene, scene_one_hot = next(self.iters[b%2])
            X[b,:]=istances
            gt[b,:]=labels
            imgs.append(scene.transpose([1,2,0]))
            info.append([(index,video_track,number_vec)])
            b+=1
        return np.array(X), np.array(gt), info, imgs

    def serve_stupid(self):
        X = np.zeros([self.cfg['batch'], self.cfg['prev_leng'], self.cfg['dims']])
        gt = np.zeros([self.cfg['batch'], self.cfg['fut_leng'], self.cfg['dims']])
        info = []
        imgs = []
        for b in range(0, self.cfg['batch']):
            '''
            istances --> past       (batch_size, past_len, 2)
            labels ----> future     (batch_size, future_len, 2)
            scene -----> map        (batch_size, 360, 360)
            '''

            index, istances, labels, presents, video_track, vehicles, number_vec, scene, scene_one_hot = next(
                self.iters[b % 2])
            X[b, :,0] = np.arange(-20,0)
            X[b, :, 1] = np.arange(-20, 0)
            gt[b, :,0] = np.arange(0, 40)
            gt[b, :, 1] = np.arange(0, 40)
            imgs.append(scene.transpose([1, 2, 0]))
            info.append([(index, video_track, number_vec)])
            b += 1
        return np.array(X), np.array(gt), info, imgs


