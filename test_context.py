import shutil
import numpy as np
import tensorflow as tf
import model
import model_double
import yaml
import utils
import random
import os
import time
import cv2
import json
import scipy.io as sio
import drawer
import name_generator

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def test():
    cfg = get_config()
    filtered=True
    cfg['batch']=1
    inpts = tf.placeholder(tf.float32, [cfg['batch'], cfg['prev_leng'], cfg['dims']])
    outs = tf.placeholder(tf.float32, [cfg['batch'], cfg['pred_ext'] + cfg['fut_leng'], cfg['dims']])
    mod = model.rec_model(cfg)

    # loader = utils.Loader(cfg)
    optimizer = tf.train.AdamOptimizer(cfg['lr'])

    # mini=opti.minimize(mod.loss)
    gvs = optimizer.compute_gradients(mod.loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    mini = optimizer.apply_gradients(capped_gvs)
    init = tf.initializers.global_variables()
    saver = tf.train.Saver()

    newp = str(time.time()).split(".")[0][-4:] + "_test_" + cfg["load_path"].split("/")[-3]
    if (filtered):
        newp+="_FILTERED"
    os.mkdir(newp)
    os.mkdir(newp + "/model")
    os.mkdir(newp + "/data")
    total = []
    test_paths=[d for d in os.listdir("../kitti_rev2/training/")]
    with tf.Session() as sess:
        if (cfg['load'] == "True"):
            saver.restore(sess, cfg['load_path'])
        else:
            "TESTING must have load=true"
            exit()

        print("OK")
        jsons, tot, file_names = loader(test_paths)

        for i, d in enumerate(jsons):
            print(i)
            df={}
            f_keys = d.keys()
            for frame in f_keys:
                df[frame]={}
                img, sx, sy = get_segm("../kitti_rev2/training/"+file_names[i], frame)

                for object in d[frame]:
                    df[frame][object]={}
                    cls = d[frame][object]["track_class_name"]

                    if (len(d[frame][object]["future"]) >= cfg['fut_leng']) and (
                            len(d[frame][object]["past"]) >= cfg['prev_leng'] - 1):

                        gt = np.clip(np.array(d[frame][object]["future"][0:cfg['fut_leng']]), -1000, 3000)
                        past = np.clip(np.array(d[frame][object]["past"][-cfg['prev_leng'] + 1:]), -1000, 3000)

                        if (np.sqrt(np.sum(np.square(gt[-1]-past[0])))>80) or not filtered:
                            pres = np.array(d[frame][object]["present"])
                            X = np.concatenate((past, np.expand_dims(pres, 0)), 0)
                            tot = smooth(np.concatenate((X, gt), 0))
                            xo=tot[0:cfg['prev_leng']]
                            x = xo / (sx / 4.0, sy)
                            gto=tot[cfg['prev_leng']:]
                            gt = gto / (sx / 4.0, sy)
                            x=np.expand_dims(x,0)
                            gt = np.expand_dims(gt, 0)
                            tot = np.concatenate([x[:, -cfg['pred_ext']:], gt], -2)

                            imx=np.expand_dims(img,0)

                            o = sess.run(mod.out,
                                         feed_dict={inpts: x, mod.image: imx, outs: tot, mod.target: tot,
                                                    mod.inputs: x})
                            o_scaled=scale_up(o[0],sx,sy)
                            gt_scaled=gto
                            df[frame][object]["past"] = xo.tolist()
                            df[frame][object]["gt"] = gt_scaled.tolist()
                            df[frame][object]["pred"] = o_scaled.tolist()
                            df[frame][object]["class"] = cls
            with open(newp + "/data/"+file_names[i]+".json","w") as out_f:
                json.dump(df,out_f,sort_keys=True)
        f = open(newp + "/dat", "w")
        f.write(str(set(total)))
        f.close()

def test_gan():
    cfg = get_config()
    filtered = True
    cfg['batch'] = 1
    inpts = tf.placeholder(tf.float32, [cfg['batch'], cfg['prev_leng'], cfg['dims']])
    outs = tf.placeholder(tf.float32, [cfg['batch'], cfg['pred_ext'] + cfg['fut_leng'], cfg['dims']])
    mod = model.rec_model(cfg)

    # loader = utils.Loader(cfg)
    optimizer = tf.train.AdamOptimizer(cfg['lr'])

    # mini=opti.minimize(mod.loss)
    gvs = optimizer.compute_gradients(mod.loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    mini = optimizer.apply_gradients(capped_gvs)
    init = tf.initializers.global_variables()
    saver = tf.train.Saver()

    newp = str(time.time()).split(".")[0][-4:] + "_test_" + cfg["load_path"].split("/")[-3]
    os.mkdir(newp)
    os.mkdir(newp + "/model")
    os.mkdir(newp + "/data")
    total = []
    test_paths=[d for d in os.listdir("../kitti_rev2/training/")]
    with tf.Session() as sess:
        if (cfg['load'] == "True"):
            saver.restore(sess, cfg['load_path'])
        else:
            "TESTING must have load=true"
            exit()

        print("OK")
        jsons, tot, file_names = loader(test_paths)

        for i, d in enumerate(jsons):
            df={}
            f_keys = d.keys()
            for frame in f_keys:
                df[frame]={}
                img, sx, sy = get_segm("../kitti_rev2/training/"+file_names[i], frame)

                for object in d[frame]:
                    df[frame][object]={}
                    cls = d[frame][object]["track_class_name"]

                    if (len(d[frame][object]["future"]) >= cfg['fut_leng']) and (
                            len(d[frame][object]["past"]) >= cfg['prev_leng'] - 1):

                        gt = np.clip(np.array(d[frame][object]["future"][0:cfg['fut_leng']]), -1000, 3000)
                        past = np.clip(np.array(d[frame][object]["past"][-cfg['prev_leng'] + 1:]), -1000, 3000)

                        if (np.sqrt(np.sum(np.square(gt[-1]-past[0])))>80) or not filtered:
                            pres = np.array(d[frame][object]["present"])
                            X = np.concatenate((past, np.expand_dims(pres, 0)), 0)
                            tot = smooth(np.concatenate((X, gt), 0))
                            xo=tot[0:cfg['prev_leng']]
                            x = xo / (sx / 4.0, sy)
                            gto=tot[cfg['prev_leng']:]
                            gt = gto / (sx / 4.0, sy)
                            x=np.expand_dims(x,0)
                            gt = np.expand_dims(gt, 0)
                            tot = np.concatenate([x[:, -cfg['pred_ext']:], gt], -2)

                            imx=np.expand_dims(img,0)

                            o = sess.run(mod.out,
                                         feed_dict={inpts: x, mod.image: imx, outs: tot, mod.target: tot,
                                                    mod.inputs: x})
                            o_scaled=scale_up(o[0],sx,sy)
                            gt_scaled=gto
                            df[frame][object]["past"] = xo.tolist()
                            df[frame][object]["gt"] = gt_scaled.tolist()
                            df[frame][object]["pred"] = o_scaled.tolist()
                            df[frame][object]["class"] = cls
            with open(newp + "/data/"+file_names[i]+".json","w") as out_f:
                json.dump(df,out_f,sort_keys=True)
        f = open(newp + "/dat", "w")
        f.write(str(set(total)))
        f.close()

def smooth(y,N=3):

    y_padded = np.pad(y, ((N // 2, N - 1 - N // 2),(0,0)), mode='edge')
    y[:,0] = np.convolve(y_padded[:,0], np.ones((N,)) / N, mode='valid')
    y[:, 1] = np.convolve(y_padded[:,1], np.ones((N,)) / N, mode='valid')
    #y[:, 2] = np.convolve(y_padded[:,2], np.ones((N,)) / N, mode='valid')
    # y_smooth=np.concatenate([np.expand_dims()y_smooth_x,y_smooth_y],-1)

    return y

def loader(jpath):
    jsons=[]
    total=0
    files=[]
    for v in jpath:
        files.append(v)
        fm=json.load(open("../kitti_rev2/training/"+v+"/trajectories.json"))
        trajs=0
        for frm in fm.keys():
            trajs+=len(fm[frm].keys())
        total+=trajs
        jsons.append(fm)
    return jsons,total,files

def get_segm(filenames,frame):
    #print(filenames)
    img_path = filenames#.replace(self.cfg['json_path'], "../image_02")
    epoc = img_path#img_path.split("/")[0:-1]
    #epoc.append(img_path.split("/")[-1].zfill(4))
    #img_path = "/".join(epoc)
    num=frame
    num=num.replace("frame_","")
    num=num.zfill(8)
    img=np.load(img_path+"/deeplab_cache/"+str(num)+".npz")
    gg=img.f.seg_map
    sx,sy=gg.shape[1],gg.shape[0]
    gg=gg.astype('float32')
    gg=cv2.resize(gg,(256,128))
    return gg,sx,sy




def scale_up(gts, sx, sy):
    poins = np.asarray(gts)
    poins = poins * (sx / 4.0, sy)
    return np.array(poins, dtype=np.int32)


def get_config():
    with open("config.yaml") as raw_cfg:
        cfg = yaml.load(raw_cfg)
    return cfg


def main():
    cfg=get_config()
    if cfg['type']==2:
        test_gan()
    else:
        test()


main()
