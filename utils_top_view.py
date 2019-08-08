import random
import os
import multiprocessing as mp
from PIL import Image
import numpy as np
import json
import glob
import itertools
import synth_dataset
import cv2
import time
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
#test_set = ["../kitti_rev2/training/" + d for d in os.listdir("../kitti_rev2/training/")]

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
        self.train_q=mp.Queue(256)
        self.test_q = mp.Queue(32)
        past_len = self.cfg['prev_leng']
        future_len = self.cfg['fut_leng']
        # self.q = mp.Queue(self.cfg['batch']*self.cfg['splits']*4)
        # self.q_test = mp.Queue(self.cfg['batch']*10)

        # self.data_train = dataset.TrackDataset(tracks, num_istances=past_len, num_labels=future_len,
        #                                   train=True, dim_clip=dim_clip,rot=True,shf=True)
        # print("LOADER0_init")
        # self.data_test = dataset.TrackDataset(tracks, num_istances=past_len, num_labels=future_len,
        #                                  train=False, dim_clip=dim_clip,rot=False,shf=False)
        # print("LOADER1_init")
        # self.data_rand_test = dataset.TrackDataset(tracks, num_istances=past_len, num_labels=future_len,
        #                                  train=False, dim_clip=dim_clip,rot=False,shf=True)

        #self.iter_train = itertools.cycle(self.data_train)
        if(config['test']):
            self.data_test = dataset.TrackDataset(tracks, num_istances=past_len, num_labels=future_len, train=False, dim_clip=dim_clip,rot=False,shf=False)

        else:
            train_thrds = []
            # self.data_train = synth_dataset.TrackDataset(20, 40, old=100, extra=100, scale=1, n_roads=3,self.q)
            for h in range(0, 4):
                tr = dataset.TrackDataset(h,self.train_q,self.cfg,tracks, num_istances=past_len, num_labels=future_len,
                                              train=True, dim_clip=dim_clip,rot=True,shf=True)
                thread = mp.Process(target=tr.populate_queue, name="thr" + str(h))
                thread.daemon = True
                print("train_loader_thread" + thread.name + "_init")
                train_thrds.append(thread)
            for t in train_thrds:
                t.start()
                # thread.join()
                print("Synth_loader_thread" + t.name + "_start")

            test_thrds = []
            # self.data_train = synth_dataset.TrackDataset(20, 40, old=100, extra=100, scale=1, n_roads=3,self.q)
            for h in range(0, 4):
                tr = dataset.TrackDataset(h,self.test_q,self.cfg,tracks, num_istances=past_len, num_labels=future_len,
                                              train=False, dim_clip=dim_clip,rot=False,shf=True)
                thread = mp.Process(target=tr.populate_queue, name="thr" + str(h))
                thread.daemon = True
                print("Test_loader_thread" + thread.name + "_init")
                test_thrds.append(thread)
            for t in test_thrds:
                t.start()
                # thread.join()
                print("Test_loader_thread" + t.name + "_start")

            # for k in range(0,2):
            #     self.trains.append(itertools.cycle(dataset.TrackDataset(tracks, num_istances=past_len, num_labels=future_len,
            #                                   train=True, dim_clip=dim_clip,rot=True,shf=True)))
            #
            #
            # self.iters=[]
            # for i in range(0,2):
            #     self.iters.append(itertools.cycle(dataset.TrackDataset(tracks, num_istances=past_len, num_labels=future_len,
            #                                  train=False, dim_clip=dim_clip,rot=False,shf=True)))

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

    def serve_multiprocess_train(self):
        toX,toGT,info,imgs,imga= self.train_q.get()
        return toX,toGT,info,imgs,imga

    def serve_multiprocess_test(self):
        toX,toGT,info,imgs,imga= self.test_q.get()
        return toX,toGT,info,imgs,imga

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

class Loader_synth():

    def get_top_paths(self, pt):
        tops = []
        for top_d in os.listdir(pt):
            for video in os.listdir(pt + "/" + top_d):
                tops.append(pt + "/" + top_d + "/" + video)
        return tops

    def __init__(self, config):
        self.cfg = config
        self.q = mp.Queue(maxsize=256)

        print("Synth_loader_init")

        past_len = self.cfg['prev_leng']
        future_len = self.cfg['fut_leng']

        thrds=[]
        #self.data_train = synth_dataset.TrackDataset(20, 40, old=100, extra=100, scale=1, n_roads=3,self.q)
        for h in range(0,16):
            tr=synth_dataset.TrackDataset(20, 40,queue=self.q,cfg=self.cfg, old=50, extra=50, scale=1, n_roads=3)
            thread=mp.Process(target=tr.populate_queue,name="thr"+str(h))
            thread.daemon = True
            print("Synth_loader_thread" + thread.name + "_init")
            thrds.append(thread)
        for t in thrds:
            t.start()
            #thread.join()
            print("Synth_loader_thread" + t.name + "_init")
        #self.iter_train = self.data_train



    def serve(self):
        X = np.zeros([self.cfg['batch'], self.cfg['prev_leng'], self.cfg['dims']])
        gt = np.zeros([self.cfg['batch'], self.cfg['fut_leng'], self.cfg['dims']])
        imgs = []
        for b in range(0,self.cfg['batch']):
            '''
            istances --> past       (batch_size, past_len, 2)
            labels ----> future     (batch_size, future_len, 2)
            scene -----> map        (batch_size, 360, 360)
            '''

            _, istances, labels, _, _, _, _, scene, _= self.iter_train.get()
            X[b,:]=istances
            gt[b,:]=labels
            imgs.append(scene.transpose([1,2,0]))
            #info.append([(index,video_track,number_vec)])
            b+=1
        toX=np.array(X)
        toGT=np.array(gt)

        return toX,toGT, None, imgs

    def serve_multiprocess(self):
        toX,toGT,info,imgs,imga= self.q.get()
        return toX,toGT,info,imgs,imga

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

            index, istances, labels, presents, video_track, vehicles, number_vec, scene, scene_one_hot = self.iter_train.get()
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
