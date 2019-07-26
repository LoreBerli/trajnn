import random
import os
import multiprocessing as mp
from PIL import Image
import numpy as np
import json
import glob
import cv2
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



class Loader():
    def get_top_paths(self, pt):
        tops = []
        for top_d in os.listdir(pt):
            for video in os.listdir(pt + "/" + top_d):
                tops.append(pt + "/" + top_d + "/" + video)
        return tops

    def __init__(self, config):
        self.cfg = config
        self.q = mp.Queue(self.cfg['batch']*self.cfg['splits']*4)
        self.q_test = mp.Queue(self.cfg['batch']*10)

        self.path = self.cfg['json_path']
        splits = self.cfg['splits']

        self.videos = self.get_top_paths(self.path)

        f_videos = [v for v in self.videos if v in test_set]
        t_videos = [v for v in self.videos if v not in test_set]
        if(self.cfg['fine']):
            real_t=[]
            for t in t_videos:
                nm=t.split("/")[-1]
                if not (nm.startswith("AMBA")) and not(nm.startswith("stuttgart")):
                    print(nm)
                    real_t.append(t)
            t_videos=real_t
        print("there are {0} test videos".format(str(len(f_videos))))
        print("there are {0} train videos".format(str(len(t_videos))))
        self.videos = t_videos
        self.total_data = 0
        step = len(self.videos) // splits
        Feeder(self.q_test, f_videos, 0, 0, self.cfg)
        video_splits = {i: [] for i in range(splits)}
        for v in range(len(self.videos)):
            video_splits[v % splits].append(self.videos[v])

        for f in video_splits.keys():
            tmp = Feeder(self.q, video_splits[f], f, f, self.cfg)
            self.total_data += tmp.tot
        print("there are {0} total trajs".format(self.total_data))

    def serve(self):
        X = np.zeros([self.cfg['batch'], self.cfg['prev_leng'], self.cfg['dims']])
        gt = np.zeros([self.cfg['batch'], self.cfg['fut_leng'], self.cfg['dims']])
        feat = np.zeros([self.cfg['batch'], 128])
        info = []
        boxes = []
        imgs = []
        flip = False
        if np.random.rand() > 0.5:
            flip = False

        for b in range(self.cfg['batch']):
            x, g, box, i, img = self.q.get()
            if flip:
                x[:, 0] = - x[:, 0]
                g[:, 0] = - g[:, 0]
                box[0] = - box[0]
                box[2] = - box[2]
                img = np.flip(img, 1)
            info.append(i)
            boxes.append(box)
            X[b, :] = x
            gt[b, :] = g
            imgs.append(img)

        return X, gt, feat, boxes, info, imgs

    def serve_test(self):
        X = np.zeros([self.cfg['batch'], self.cfg['prev_leng'], self.cfg['dims']])
        gt = np.zeros([self.cfg['batch'], self.cfg['fut_leng'], self.cfg['dims']])
        feat = np.zeros([self.cfg['batch'], 128])
        info = []
        boxes = []
        imgs = []

        for b in range(self.cfg['batch']):
            x, g, box, i, img = self.q_test.get()
            imgs.append(img)
            boxes.append(box)
            info.append(i)
            X[b, :] = x
            gt[b, :] = g
        return X, gt, feat, boxes, info, imgs

class Feeder():

    def loader(self, jpath):
        jsons = {}
        total = 0
        files = []
        video_frames = []
        for v in jpath:
            files.append(v)
            fm = json.load(open(v + "/trajectories.json"))
            trajs = 0
            for frm in fm.keys():
                trajs += len(fm[frm].keys())
            total += trajs

            for frm in fm.keys():
                jsons[v] = fm
                video_frames.append([v, frm])
        return jsons, total, files, video_frames

    def __init__(self, q, jpath, num, fil, cfg):
        self.total = jpath
        self.cfg = cfg
        self.q = q
        self.p = jpath
        self.data, self.tot, self.filenames, self.vid_frm = self.loader(self.total)
        # self.fs=np.load(self.cfg['feat_path'] + "/" + self.p.replace("video_", "") + ".npy")
        # self.get_image()
        print("started" + str(self.p))
        if (self.cfg['dims'] == 3):
            self.thread = mp.Process(target=self.feed3d)
        else:
            self.thread = mp.Process(target=self._feed)
        self.thread.daemon = True
        self.thread.start()

    def feat_loader(self, vid, frame):
        pass

    def get_segm(self, filenames, frame):
        img_path = filenames  # .replace(self.cfg['json_path'], "../image_02")
        epoc = img_path  # img_path.split("/")[0:-1]
        # epoc.append(img_path.split("/")[-1].zfill(4))
        # img_path = "/".join(epoc)
        # if (not os.path.exists(img_path  + "/deeplab_cache_small")):
        #     os.mkdir(img_path  + "/deeplab_cache_small")
        num = frame
        num = num.replace("frame_", "")
        num = num.zfill(8)
        # if (not os.path.exists(img_path + "/deeplab_cache_small/" + str(num) + ".npz")):
        #     img = np.load(img_path + "/deeplab_cache/" + str(num) + ".npz")
        #     gg = img.f.seg_map
        #     sx, sy = gg.shape[1], gg.shape[0]
        #     dims = np.array([sx, sy])
        #     gg = gg.astype('float32')
        #     gg = cv2.resize(gg, (256, 128))
        #     seg_map=gg
        #     with open(img_path + "/deeplab_cache_small/" + str(num) + ".npz", 'wb+') as f:
        #         np.savez(f, seg_map=seg_map,dims=dims)
        # else:
        img = np.load(img_path + "/deeplab_cache_small/" + str(num) + ".npz")
        gg = img.f.seg_map
        gg = gg.astype('float32')
        sx = img.f.dims[0]
        sy=img.f.dims[1]


        # channels = [0, 1, 12, 13, 18]
        # mat = np.eye(19)[np.array(gg, dtype=np.int32)]
        # mat_clean = np.zeros([128, 256, len(channels)])
        # for i, j in enumerate(channels):
        #     mat_clean[:, :, i] = mat[:, :, j]
        mat_clean=gg
        return mat_clean, sx, sy

    def smooth(self, y, N=4):

        y_padded = np.pad(y, ((N // 2, N - 1 - N // 2), (0, 0)), mode='edge')
        y[:, 0] = np.convolve(y_padded[:, 0], np.ones((N,)) / N, mode='valid')
        y[:, 1] = np.convolve(y_padded[:, 1], np.ones((N,)) / N, mode='valid')
        # y[:, 2] = np.convolve(y_padded[:,2], np.ones((N,)) / N, mode='valid')
        # y_smooth=np.concatenate([np.expand_dims()y_smooth_x,y_smooth_y],-1)

        return y

    def transform(self, zs, depth):
        # zs=0.54 * 721.0 / (1242*zs)
        focalLength = 721.0
        centerX = 1242 / 2.0
        centerY = 374 / 2.0
        scalingFactor = 1.0
        zs[:, 0] = np.clip(zs[:, 0], 0, 1241)
        zs[:, 1] = np.clip(zs[:, 1], 0, 373)
        points = []

        for uv in zs:

            Z = min(depth[uv[0], uv[1]] / scalingFactor, 80)

            if Z == 0: continue
            X = (uv[0] - centerX) * Z / focalLength
            Y = (uv[1] - centerY) * Z / focalLength
            points.append([X, Y, Z])
        return np.array(points)

    def back_transform(self, tred_points):
        focalLength = 721.0
        centerX = 1242 / 2.0
        centerY = 374 / 2.0
        twod_points = []

        for pt in tred_points:
            x = -centerX + (pt[0] * focalLength) / pt[2]
            y = -centerY + (pt[1] * focalLength) / pt[2]
            twod_points.append([x, y])
        return np.array(twod_points)



    def _feed(self):
        while True:
            random.shuffle(self.vid_frm)
            for frm in self.vid_frm:
                pt = frm[0]
                frame = frm[1]
                img, sx, sy = self.get_segm(pt, frm[1])
                self.d = self.data[pt]

                for object in self.d[frame]:
                    cls = self.d[frame][object]["track_class_name"]
                    bbox_ = self.d[frame][object]["box"]
                    bbox=[0,0,0,0]
                    if(self.cfg['center']):
                        bbox[0] = (bbox_[0] / (sx / 2.0))-1.0
                        bbox[1] = (bbox_[1] / float(sy))-0.5
                        bbox[2] = (bbox_[2] / (sx / 2.0))-1.0
                        bbox[3] = (bbox_[3] / float(sy))-0.5

                    else:
                        bbox[0] = bbox[0] / (sx/2.0)
                        bbox[1] = bbox[1] / sy
                        bbox[2] = bbox[2] / (sx/2.0)
                        bbox[3] = bbox[3] / sy


                    if (len(self.d[frame][object]["future"]) >= self.cfg['fut_leng']) and (
                            len(self.d[frame][object]["past"]) >= self.cfg['prev_leng'] - 1):
                        gt = np.clip(np.array(self.d[frame][object]["future"][0:self.cfg['fut_leng']]), -1000, 3000)
                        past = np.clip(np.array(self.d[frame][object]["past"][-self.cfg['prev_leng'] + 1:]), -1000,3000)
                        #############
                        if (np.sqrt(np.sum(np.square(gt[-1] - past[0]))) > 40):
                            pres = np.array(self.d[frame][object]["present"])
                            X = np.concatenate((past, np.expand_dims(pres, 0)), 0)
                            conc=np.concatenate((X, gt),0)/np.array(((sx/2.0),float(sy)),dtype=np.float)
                            if(self.cfg['center']):
                                conc=conc-np.array([1.0,0.5])
                            tot = self.smooth(conc)
                            self.q.put([tot[0:self.cfg['prev_leng']], tot[self.cfg['prev_leng']:], bbox,
                                        [pt, frame, object, sx, sy, cls], img])

    def feed3d(self):
        # TODO
        # PRENDERE LE COORDINATE DAL FILE .npy in world_cor_02
        while True:
            for i, d in enumerate(self.data):
                self.d = d

                f_keys = self.d.keys()
                random.shuffle(f_keys)
                for frame in f_keys:
                    depth = np.load(
                        self.cfg['depth_path'] + "/" + self.p[i].split("/")[-2] + "/" + self.p[i].split("/")[-1].zfill(
                            4) + "/" + frame.replace("frame_", "") + ".npy")
                    depth = cv2.resize(depth, (374, 1242))
                    for object in self.d[frame]:

                        bbox = self.d[frame][object]["box"]
                        bbox[0] = bbox[0] / self.cfg['out_size_y']
                        bbox[1] = bbox[1] / (self.cfg['out_size_x'] / 3.0)
                        bbox[2] = bbox[2] / self.cfg['out_size_y']
                        bbox[3] = bbox[3] / (self.cfg['out_size_x'] / 3.0)
                        # clas=self.d[frame][object]["class"]
                        #####CHECK PAST AND FUTURE
                        if (len(self.d[frame][object]["future"]) >= self.cfg['fut_leng']) and (
                                len(self.d[frame][object]["past"]) >= self.cfg['prev_leng'] - 1):
                            # QUIIIIIIIIIIIIIIIIIIIIIIIIIIIII
                            ############
                            gt = np.array(self.d[frame][object]["future"][0:self.cfg['fut_leng']])
                            past = np.array(self.d[frame][object]["past"][-self.cfg['prev_leng'] + 1:])
                            #############
                            if (np.sqrt(np.sum(np.square(gt[-1] - past[0]))) > 80):
                                pres = np.array(self.d[frame][object]["present"])
                                # feats = self.fs[int(frame.replace("frame_",""))][0]
                                feats = np.zeros(shape=128)
                                X = np.concatenate((past, np.expand_dims(pres, 0)), 0)
                                tot = np.concatenate((X, gt), 0)
                                tot = self.smooth(tot)
                                tot = self.transform(tot, depth)
                                # X=np.flip(X,-2)
                                self.q.put([tot[0:self.cfg['prev_leng']], tot[self.cfg['prev_leng']:], bbox, feats,
                                            [self.p[i], frame, object]])
