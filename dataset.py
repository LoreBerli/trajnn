import numpy as np
# from keras.utils import to_categorical
#from tensorflow.keras.utils import to_categorical
import csv
import matplotlib.pyplot as plt
import os
import random
import pdb
from matplotlib.colors import LinearSegmentedColormap
import cv2
import matplotlib as mpl
import itertools

# def to_categorical(y, num_classes=None, dtype='float32'):
#     """Converts a class vector (integers) to binary class matrix.
#     E.g. for use with categorical_crossentropy.
#     # Arguments
#         y: class vector to be converted into a matrix
#             (integers from 0 to num_classes).
#         num_classes: total number of classes.
#         dtype: The data type expected by the input, as a string
#             (`float32`, `float64`, `int32`...)
#     # Returns
#         A binary matrix representation of the input. The classes axis
#         is placed last.
#     # Example
#     ```python
#     # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
#     > labels
#     array([0, 2, 1, 2, 0])
#     # `to_categorical` converts this into a matrix with as many
#     # columns as there are classes. The number of rows
#     # stays the same.
#     > to_categorical(labels)
#     array([[ 1.,  0.,  0.],
#            [ 0.,  0.,  1.],
#            [ 0.,  1.,  0.],
#            [ 0.,  0.,  1.],
#            [ 1.,  0.,  0.]], dtype=float32)
#     ```
#     """
#
#     y = np.array(y, dtype='int')
#     input_shape = y.shape
#     if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
#         input_shape = tuple(input_shape[:-1])
#     y = y.ravel()
#     if not num_classes:
#         num_classes = np.max(y) + 1
#     n = y.shape[0]
#     categorical = np.zeros((n, num_classes), dtype=dtype)
#     categorical[np.arange(n), y] = 1
#     output_shape = input_shape + (num_classes,)
#     categorical = np.reshape(categorical, output_shape)
#     return categorical


# scene! 0:background 1:street 2:sidewalk, 3:structure 4: vegetation
#colors = [(0, 0, 0), (0.87, 0.87, 0.87), (0.54, 0.54, 0.54), (0.49, 0.33, 0.16), (0.29, 0.57, 0.25)]
# scene! 0:background 1:street 2:sidewalk, 3: vegetation
colors = [(0, 0, 0), (0.87, 0.87, 0.87), (0.54, 0.54, 0.54),  (0.29, 0.57, 0.25)]
cmap_name = 'scene_list'
cm = LinearSegmentedColormap.from_list(
    cmap_name, colors, N=4)


class TrackDataset:
    def __init__(self,h,q,cfg, tracks, num_istances, num_labels, train, dim_clip,rot,shf):

        self.tracks = tracks
        self.cfg=cfg
        self.dim_clip = dim_clip
        self.q=q
        self.video_track = []   # '0001'
        self.vehicles = []      # 'Car'
        self.number_vec = []    # '4'
        self.index = []         # '50'
        self.istances = []      # [num_istances,2]
        self.labels = []        # [num_labels,2]
        self.presents = []      # position in complete scene
        self.scene = []         # [dim_clip,dim_clip,1]
        self.scene_one_hot = [] # [dim_clip,dim_clip,4]
        self.rot=rot
        num_total = num_istances + num_labels
        self.video_split = self.get_desire_track_files(train)

        tot_vids=len(self.video_split)
        if(not self.cfg['test']):
            if(train):
                step=1
            else:
                step=1
            print("pre ",len(self.video_split))
            self.video_split=self.video_split[h*(tot_vids/step):(h+1)*(tot_vids/step)]
            print(len(self.video_split))
        random.shuffle(self.video_split)

        for video in self.video_split:
            vehicles = self.tracks[video].keys()

            video_id = video[-9:-5]
            path_scene = 'multiple-futures/maps/2011_09_26__2011_09_26_drive_' + video_id + '_sync_map.png'
            scene_track = cv2.imread(path_scene, 0)-1

            scene_track[np.where(scene_track == 3)] = 0
            scene_track[np.where(scene_track == 2)] = 3
            scene_track[np.where(scene_track == 1)] = 2
            scene_track[np.where(scene_track == 4)] = 1


            random.shuffle(vehicles)
            for vec in vehicles:
                class_vec = tracks[video][vec]['cls']
                num_vec = vec.split('_')[1]
                points = np.array(tracks[video][vec]['trajectory']).T
                len_track = len(points)
                indexes=[i for i in range(0,len_track)]

                if(shf):
                    random.shuffle(indexes)



                for count in indexes:

                    if len_track - count > num_total:

                        temp_istance = points[count:count + num_istances].copy()
                        temp_label = points[count + num_istances:count + num_total].copy()

                        origin = temp_istance[-1]
                        # if np.var(temp_istance[:,0]) < 0.1 and np.var(temp_istance[:,1]) < 0.1:
                        #     st = np.zeros((20,2))
                        # else:
                        st = temp_istance - origin
                        st=st*2.0
                        #
                        # if np.var(temp_istance[:,0]) < 0.1 and np.var(temp_istance[:,1]) < 0.1:
                        #     fu = np.zeros((40,2))
                        # else:
                        fu = temp_label - origin
                        fu=fu*2.0
                        scene = scene_track[int(origin[1]) * 2 - self.dim_clip:int(origin[1]) * 2 + self.dim_clip,
                                           int(origin[0]) * 2 - self.dim_clip:int(origin[0]) * 2 + self.dim_clip]



                        scene_one_hot = scene
                        angles=[0,90,180,360]
                        ##############
                        if(self.rot):

                            # angle = angles[random.randint(0, 3)]
                            angle = random.randint(0, 360)
                            matRot_track = cv2.getRotationMatrix2D((0, 0), angle, 1)
                            matRot_scene = cv2.getRotationMatrix2D((self.dim_clip, self.dim_clip), angle, 1)
                            st = cv2.transform(st.reshape(-1, 1, 2), matRot_track).squeeze()
                            fu = cv2.transform(fu.reshape(-1, 1, 2), matRot_track).squeeze()
                            scene=          cv2.warpAffine(scene,         matRot_scene, (scene.shape[0], scene.shape[1]), borderValue=0, flags=cv2.INTER_NEAREST)
                            scene_one_hot = cv2.warpAffine(scene_one_hot, matRot_scene,(scene.shape[0], scene.shape[1]),borderValue=(1, 0, 0, 0), flags=cv2.INTER_NEAREST)

                        ##############
                        scene = np.expand_dims(scene, 0)


                        self.index.append(count)
                        self.istances.append(st)
                        self.labels.append(fu)
                        self.presents.append(origin)
                        self.video_track.append(video_id)
                        self.vehicles.append(class_vec)
                        self.number_vec.append(num_vec)
                        self.scene.append(scene)
                        self.scene_one_hot.append(scene_one_hot)

        self.index = np.array(self.index)
        self.istances = self.istances
        self.labels = self.labels
        self.presents = self.presents
        self.video_track = np.array(self.video_track)
        self.vehicles = np.array(self.vehicles)
        self.number_vec = np.array(self.number_vec)
        self.scene = np.array(self.scene)

        self.scene_one_hot = np.array(self.scene_one_hot)

        # self.scene_one_hot = to_categorical(self.scene_one_hot)

    def save_scenes_with_tracks(self,folder_save):

        for video in self.video_split:
            fig = plt.figure()
            video_id = video[-9:-5]
            im = plt.imread('multiple-futures/maps/2011_09_26__2011_09_26_drive_' + video_id + '_sync_map.png')
            implot = plt.imshow(im, cmap=cm)
            for t in self.tracks[video].keys():
                points = np.array(self.tracks[video][t]['trajectory']).T
                if (len(points.shape) > 1):
                    plt.plot(points[:, 0] * 2, points[:, 1] * 2)
            plt.savefig(folder_save + video_id + '.png')
            plt.close(fig)

    def save_dataset(self,folder_save):

        for i in range(len(self.istances)):
            video = self.video_track[i]
            vehicle = self.vehicles[i]
            number = self.number_vec[i]
            story = self.istances[i]
            future = self.labels[i]
            scene_track = self.scene[i]

            saving_list = ['only_tracks','only_scenes','tracks_on_scene']

            for sav in saving_list:
                folder_save_type = folder_save + sav + '/'
                if not os.path.exists(folder_save_type + video):
                    os.makedirs(folder_save_type + video)
                video_path = folder_save_type + video + '/'
                if not os.path.exists(video_path + vehicle + number):
                     os.makedirs(video_path + vehicle + number)
                vehicle_path = video_path + '/' + vehicle + number + '/'
                if sav == 'only_tracks':
                    self.draw_track(story, future, index_tracklet=self.index[i], path=vehicle_path)
                if sav == 'only_scenes':
                    self.draw_scene(scene_track, index_tracklet=self.index[i], path=vehicle_path)
                if sav == 'tracks_on_scene':
                    self.draw_scene_with_track(story, scene_track, index_tracklet=self.index[i], future=future, path=vehicle_path)

    def draw_track(self,story,future,index_tracklet,path):
        story = story.cpu().numpy()
        plt.plot(story[:, 0], -story[:, 1], c='blue', marker='o', markersize=1)
        if future is not None:
            future = future.cpu().numpy()
            plt.plot(future[:, 0], -future[:, 1], c='green', marker='o', markersize=1)
        plt.axis('equal')
        plt.savefig(path + str(index_tracklet) + '.png')
        plt.close()

    def draw_scene(self, scene_track, index_tracklet, path):
        cv2.imwrite(path + str(index_tracklet) + '.png', scene_track)

    def draw_scene_with_track(self, story, scene_track, index_tracklet, future=None, path=''):
        plt.imshow(scene_track, cmap=cm)
        story = story.cpu().numpy()
        plt.plot(story[:, 0]*2+self.dim_clip, story[:, 1]*2+self.dim_clip, c='blue', marker='o', markersize=1)
        if future is not None:
            future = future.cpu().numpy()
            plt.plot(future[:, 0]*2+self.dim_clip, future[:, 1]*2+self.dim_clip, c='green', marker='o', markersize=1)
        plt.savefig(path + str(index_tracklet) + '.png')
        plt.close()

    def get_desire_track_files(self,train):
        ''' Get videos only from the splits defined in DESIRE: https://arxiv.org/abs/1704.04394
        Splits obtained by the authors:
        all: [1, 2, 5, 9, 11, 13, 14, 15, 17, 18, 27, 28, 29, 32, 48, 51, 52, 56, 57, 59, 60, 70, 84, 91]
        train: [5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91]
        test: [1, 2, 15, 18, 29, 32, 52, 70]
        '''

        if train:
            #desire_ids = [5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91]
            desire_ids = [5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91]
        else:
            desire_ids = [1, 2, 15, 18, 29, 32, 52, 70]

        tracklet_files = ['video_2011_09_26__2011_09_26_drive_' + str(x).zfill(4) + '_sync'
                          for x in desire_ids]
        return tracklet_files

    def __getitem__(self, idx):
        return self.index[idx], self.istances[idx], self.labels[idx], self.presents[idx], self.video_track[idx], self.vehicles[idx], self.number_vec[idx], self.scene[idx], self.scene_one_hot[idx]

    def populate_queue(self):
        i=0
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

                _, istances, labels, _, _, _, _, scene, _ = self[i%self.__len__()]
                X[b, :] = istances
                gt[b, :] = labels
                sc=scene.transpose([1, 2, 0])
                m = np.eye(4)[np.array(sc, dtype=np.int32)]
                m = np.squeeze(m)
                imgs.append(m)
                imgas.append(sc)
                # info.append([(index,video_track,number_vec)])
                b += 1
                i += 1

            toX = np.array(X)
            toGT = np.array(gt)
            self.q.put([toX, toGT, " ", imgs,imgas])

    def __len__(self):
        return len(self.istances)

class TrackDataset_infer:
    def __init__(self, tracks, num_istances, num_labels, train, dim_clip,rot,shf):

        self.tracks = tracks
        self.dim_clip = dim_clip

        self.video_track = []   # '0001'
        self.vehicles = []      # 'Car'
        self.number_vec = []    # '4'
        self.index = []         # '50'
        self.istances = []      # [num_istances,2]
        self.labels = []        # [num_labels,2]
        self.presents = []      # position in complete scene
        self.scene = []         # [dim_clip,dim_clip,1]
        self.scene_one_hot = [] # [dim_clip,dim_clip,4]
        self.rot=rot
        num_total = num_istances + num_labels
        self.video_split,ids = self.get_desire_track_files(train)

        for video in self.video_split:
            vehicles = self.tracks[video].keys()

            video_id = video[-9:-5]
            path_scene = 'multiple-futures/maps/2011_09_26__2011_09_26_drive_' + video_id + '_sync_map.png'
            scene_track = cv2.imread(path_scene, 0)-1

            scene_track[np.where(scene_track == 3)] = 0
            scene_track[np.where(scene_track == 2)] = 3
            scene_track[np.where(scene_track == 1)] = 2
            scene_track[np.where(scene_track == 4)] = 1


            random.shuffle(vehicles)
            for vec in vehicles:
                class_vec = tracks[video][vec]['cls']
                num_vec = vec.split('_')[1]
                points = np.array(tracks[video][vec]['trajectory']).T
                len_track = len(points)
                indexes=[i for i in range(0,len_track)]

                if(shf):
                    random.shuffle(indexes)



                for count in indexes:

                    if len_track - count > num_total:

                        temp_istance = points[count:count + num_istances].copy()
                        temp_label = points[count + num_istances:count + num_total].copy()

                        origin = temp_istance[-1]
                        # if np.var(temp_istance[:,0]) < 0.1 and np.var(temp_istance[:,1]) < 0.1:
                        #     st = np.zeros((20,2))
                        # else:
                        st = temp_istance - origin
                        st=st*2.0
                        #
                        # if np.var(temp_istance[:,0]) < 0.1 and np.var(temp_istance[:,1]) < 0.1:
                        #     fu = np.zeros((40,2))
                        # else:
                        fu = temp_label - origin
                        fu=fu*2.0
                        scene = scene_track[int(origin[1]) * 2 - self.dim_clip:int(origin[1]) * 2 + self.dim_clip,
                                           int(origin[0]) * 2 - self.dim_clip:int(origin[0]) * 2 + self.dim_clip]



                        scene_one_hot = scene
                        angles=[0,90,180,360]
                        ##############
                        if(self.rot):

                            # angle = angles[random.randint(0, 3)]
                            angle = random.randint(0, 360)
                            matRot_track = cv2.getRotationMatrix2D((0, 0), angle, 1)
                            matRot_scene = cv2.getRotationMatrix2D((self.dim_clip, self.dim_clip), angle, 1)
                            st = cv2.transform(st.reshape(-1, 1, 2), matRot_track).squeeze()
                            fu = cv2.transform(fu.reshape(-1, 1, 2), matRot_track).squeeze()
                            scene= cv2.warpAffine(scene, matRot_scene, (scene.shape[0], scene.shape[1]), borderValue=0, flags=cv2.INTER_NEAREST)
                            #scene_one_hot = cv2.warpAffine(scene_one_hot, matRot_scene,(scene.shape[0], scene.shape[1]),borderValue=(1, 0, 0, 0), flags=cv2.INTER_NEAREST)

                        ##############
                        scene = np.expand_dims(scene, 0)


                        self.index.append(count)
                        self.istances.append(st)
                        self.labels.append(fu)
                        self.presents.append(origin)
                        self.video_track.append(video_id)
                        self.vehicles.append(class_vec)
                        self.number_vec.append(num_vec)
                        self.scene.append(scene)
                        #self.scene_one_hot.append(scene_one_hot)

        self.index = np.array(self.index)
        self.istances = self.istances
        self.labels = self.labels
        self.presents = self.presents
        self.video_track = np.array(self.video_track)
        self.vehicles = np.array(self.vehicles)
        self.number_vec = np.array(self.number_vec)
        self.scene = np.array(self.scene)

        self.scene_one_hot = self.scene
        # self.scene_one_hot = to_categorical(self.scene_one_hot)

    def save_scenes_with_tracks(self,folder_save):

        for video in self.video_split:
            fig = plt.figure()
            video_id = video[-9:-5]
            im = plt.imread('multiple-futures/maps/2011_09_26__2011_09_26_drive_' + video_id + '_sync_map.png')
            implot = plt.imshow(im, cmap=cm)
            for t in self.tracks[video].keys():
                points = np.array(self.tracks[video][t]['trajectory']).T
                if (len(points.shape) > 1):
                    plt.plot(points[:, 0] * 2, points[:, 1] * 2)
            plt.savefig(folder_save + video_id + '.png')
            plt.close(fig)

    def save_dataset(self,folder_save):

        for i in range(len(self.istances)):
            video = self.video_track[i]
            vehicle = self.vehicles[i]
            number = self.number_vec[i]
            story = self.istances[i]
            future = self.labels[i]
            scene_track = self.scene[i]

            saving_list = ['only_tracks','only_scenes','tracks_on_scene']

            for sav in saving_list:
                folder_save_type = folder_save + sav + '/'
                if not os.path.exists(folder_save_type + video):
                    os.makedirs(folder_save_type + video)
                video_path = folder_save_type + video + '/'
                if not os.path.exists(video_path + vehicle + number):
                     os.makedirs(video_path + vehicle + number)
                vehicle_path = video_path + '/' + vehicle + number + '/'
                if sav == 'only_tracks':
                    self.draw_track(story, future, index_tracklet=self.index[i], path=vehicle_path)
                if sav == 'only_scenes':
                    self.draw_scene(scene_track, index_tracklet=self.index[i], path=vehicle_path)
                if sav == 'tracks_on_scene':
                    self.draw_scene_with_track(story, scene_track, index_tracklet=self.index[i], future=future, path=vehicle_path)

    def draw_track(self,story,future,index_tracklet,path):
        story = story.cpu().numpy()
        plt.plot(story[:, 0], -story[:, 1], c='blue', marker='o', markersize=1)
        if future is not None:
            future = future.cpu().numpy()
            plt.plot(future[:, 0], -future[:, 1], c='green', marker='o', markersize=1)
        plt.axis('equal')
        plt.savefig(path + str(index_tracklet) + '.png')
        plt.close()

    def draw_scene(self, scene_track, index_tracklet, path):
        cv2.imwrite(path + str(index_tracklet) + '.png', scene_track)

    def draw_scene_with_track(self, story, scene_track, index_tracklet, future=None, path=''):
        plt.imshow(scene_track, cmap=cm)
        story = story.cpu().numpy()
        plt.plot(story[:, 0]*2+self.dim_clip, story[:, 1]*2+self.dim_clip, c='blue', marker='o', markersize=1)
        if future is not None:
            future = future.cpu().numpy()
            plt.plot(future[:, 0]*2+self.dim_clip, future[:, 1]*2+self.dim_clip, c='green', marker='o', markersize=1)
        plt.savefig(path + str(index_tracklet) + '.png')
        plt.close()

    def get_desire_track_files(self,train):
        ''' Get videos only from the splits defined in DESIRE: https://arxiv.org/abs/1704.04394
        Splits obtained by the authors:
        all: [1, 2, 5, 9, 11, 13, 14, 15, 17, 18, 27, 28, 29, 32, 48, 51, 52, 56, 57, 59, 60, 70, 84, 91]
        train: [5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91]
        test: [1, 2, 15, 18, 29, 32, 52, 70]
        '''

        if train:
            #desire_ids = [5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91]
            desire_ids = [5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91]
        else:
            desire_ids = [1, 2, 15, 18, 29, 32, 52, 70]

        tracklet_files = ['video_2011_09_26__2011_09_26_drive_' + str(x).zfill(4) + '_sync'
                          for x in desire_ids]
        return tracklet_files

    def get_infer_track_files(self,train):
        ''' Get videos only from the splits defined in DESIRE: https://arxiv.org/abs/1704.04394
        Splits obtained by the authors:
        all: [1, 2, 5, 9, 11, 13, 14, 15, 17, 18, 27, 28, 29, 32, 48, 51, 52, 56, 57, 59, 60, 70, 84, 91]
        train: [5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91]
        test: [1, 2, 15, 18, 29, 32, 52, 70]
        '''
        if(train):
            _path="trajRNN/multiple-futures/train.csv"
        else:
            _path = "trajRNN/multiple-futures/val.csv"

        tracks_csv=open(_path,'r')
        tracks_reader=csv.reader(tracks_csv)
        ids = []
        videos=[]
        for l in tracks_reader:
            videos.append(int(l[0]))
            ids.append((int(l[0]), int(l[1])))

        tracklet_files = ['video_2011_09_26__2011_09_26_drive_' + str(x).zfill(4) + '_sync'
                          for x in videos]
        return tracklet_files,ids

    def __getitem__(self, idx):
        return self.index[idx], self.istances[idx], self.labels[idx], self.presents[idx], self.video_track[idx], self.vehicles[idx], self.number_vec[idx], self.scene[idx], self.scene_one_hot[idx]

    def __len__(self):
        return len(self.istances)



