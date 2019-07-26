import numpy as np

def feed(self):
    # TODO
    # PRENDERE LE COORDINATE DAL FILE .npy in world_cor_02
    for i, d in enumerate(self.data):
            self.d = d
            # fil_nam=self.filename
            f_keys = self.d.keys()

            for frame in f_keys:
                # depth=np.load(self.cfg['depth_path']+"/"+d.split("/")[-2]+"/"+d.split("/")[-1].zfill(4)+"/"+frame+".npy")
                img, sx, sy = self.get_segm(self.p[i], frame)
                # depth=0.54*721.0/(1242.0*depth)
                # depth=cv2.resize(depth,(img.size))

                # depth=np.pad(depth,((12,12),(12,12)),mode="edge")
                for object in self.d[frame]:
                    cls = self.d[frame][object]["track_class_name"]
                    bbox = self.d[frame][object]["box"]
                    bbox[0] = bbox[0] / sy
                    bbox[1] = bbox[1] / (sx / 4.0)
                    bbox[2] = bbox[2] / sy
                    bbox[3] = bbox[3] / (sx / 4.0)
                    # clas=self.d[frame][object]["class"]
                    #####CHECK PAST AND FUTURE
                    if (len(self.d[frame][object]["future"]) >= self.cfg['fut_leng']) and (
                            len(self.d[frame][object]["past"]) >= self.cfg['prev_leng'] - 1):
                        # QUIIIIIIIIIIIIIIIIIIIIIIIIIIIII
                        ############
                        gt = np.clip(np.array(self.d[frame][object]["future"][0:self.cfg['fut_leng']]), -1000, 3000)
                        past = np.clip(np.array(self.d[frame][object]["past"][-self.cfg['prev_leng'] + 1:]), -1000,
                                       3000)
                        #############
                        if (np.sqrt(np.sum(np.square(gt[-1] - past[0]))) > 80):
                            pres = np.array(self.d[frame][object]["present"])
                            # feats = self.fs[int(frame.replace("frame_",""))][0]
                            feats = np.zeros(shape=128)
                            X = np.concatenate((past, np.expand_dims(pres, 0)), 0)
                            tot = self.smooth(np.concatenate((X, gt), 0))

                            # x=np.arange(0,1,1.0/tot.shape[0])*600
                            # y=(1.0+np.sin(np.arange(0,4,4.0/tot.shape[0])))*150
                            # tot=np.concatenate([np.expand_dims(x,1),np.expand_dims(y,1)],-1)

                            # X=np.flip(X,-2)
                            self.q.put([tot[0:self.cfg['prev_leng']], tot[self.cfg['prev_leng']:], bbox, feats,
                                        [self.p[i], frame, object, sx, sy, cls], img])