import shutil
import numpy as np
import tensorflow as tf
import model
import model_double
import yaml
import utils
import sys
import random
import os
import time
import cv2
import json
import scipy.io as sio
import drawer
import name_generator
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file



def test(cf):
    cfg = cf
    print("TEST_")
    filtered=True
    cfg['batch']=1
    cfg['old']=False
    cfg['inverted']=False
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"#str(cfg['GPU'])
    inpts = tf.placeholder(tf.float32, [cfg['batch'], cfg['prev_leng'], cfg['dims']])
    outs = tf.placeholder(tf.float32, [cfg['batch'], cfg['pred_ext'] + cfg['fut_leng'], cfg['dims']])
    mod = model.rec_model(cfg)
    saver = tf.train.Saver()
    newp = str(time.time()).split(".")[0][-4:] + "_test_" + cfg["load_path"].split("/")[-3]
    if (filtered):
        newp+="_FILTERED"
    os.mkdir(newp)
    os.mkdir(newp + "/model")
    os.mkdir(newp + "/data")
    os.mkdir(newp + "/imgs")
    total = []
    test_paths=[d for d in os.listdir("../kitti_rev2/training/")]
    for t in test_paths:
        os.mkdir(newp+"/imgs/"+t)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        print(cfg['load'],cfg['load_path'] )
        if (cfg['load'] == True):
            saver = tf.train.import_meta_graph("/".join(cfg['load_path'].split("/")[0:-2]) + "/model/model.ckpt.meta")

            graph = tf.get_default_graph()
            print_tensors_in_checkpoint_file(cfg['load_path'], all_tensors=False, tensor_name='DEC/DEC/gru_cell/candidate/bias',all_tensor_names=False)
            #saver.restore(sess, cfg['load_path'])
            for v in graph.get_all_collection_keys():
                vars=graph.get_collection(v)
                for v in vars:
                    print(v)
            saver.restore(sess, cfg['load_path'])

            print("MODEL LOADED")
        else:
            print("TESTING must have load=True")
            exit()

        jsons, tot, file_names,vmf = loader(test_paths)

        for i, d in enumerate(jsons):
            print(i)
            df={}
            f_keys = d.keys()
            for frame in f_keys:
                df[frame]={}
                img, sx, sy = get_segm_new("../kitti_rev2/training/"+str(file_names[i])+"/deeplab_cache", frame)

                for object in d[frame]:
                    df[frame][object]={}
                    cls = d[frame][object]["track_class_name"]

                    if (len(d[frame][object]["future"]) >= cfg['fut_leng']) and (
                            len(d[frame][object]["past"]) >= cfg['prev_leng'] - 1):

                        gt = np.clip(np.array(d[frame][object]["future"][0:cfg['fut_leng']]), -1000, 3000)
                        past = np.clip(np.array(d[frame][object]["past"][-cfg['prev_leng'] + 1:]), -1000, 3000)

                        if (np.sqrt(np.sum(np.square(gt[-1]-past[0])))>80) or not filtered:
                            pres = np.array(d[frame][object]["present"])
                            bbox = d[frame][object]["box"]
                            bbox[0] = (bbox[0] / (sx / 2.0)) - 1.0
                            bbox[1] = (bbox[1] / float(sy)) - 0.5
                            bbox[2] = (bbox[2] / (sx / 2.0)) - 1.0
                            bbox[3] = (bbox[3] / float(sy)) - 0.5
                            bbox=np.array(bbox)
                            o_bbox=bbox
                            bbox=np.expand_dims(bbox,0)
                            X = np.concatenate((past, np.expand_dims(pres, 0)), 0)
                            conc = np.concatenate((X, gt), 0) / np.array(((sx / 2.0), float(sy)))
                            conc = conc - np.array([1.0, 0.5])
                            tot = smooth(conc)
                            xo=tot[0:cfg['prev_leng']]
                            x = xo
                            gto=tot[cfg['prev_leng']:]
                            old=gto
                            old_x=x
                            gt = gto
                            x=np.expand_dims(x,0)
                            gt = np.expand_dims(gt, 0)
                            tot = np.concatenate([x[:, -cfg['pred_ext']:], gt], -2)

                            imx=np.expand_dims(img,0)

                            o = sess.run(mod.out,
                                         feed_dict={inpts: x, mod.image: imx, outs: tot, mod.target: tot,mod.box:bbox,
                                                    mod.inputs: x,mod.drop:1.0})
                            poins = o[0] + np.array([1.0, 0.5])
                            o_scaled = poins * (sx / 2.0, float(sy))
                            gto = gto + np.array([1.0, 0.5])
                            gt_scaled=gto* (sx / 2.0, float(sy))
                            xo = xo + np.array([1.0, 0.5])
                            xo=xo* (sx / 2.0, sy)
                            df[frame][object]["past"] = xo.tolist()
                            df[frame][object]["gt"] = gt_scaled.tolist()
                            df[frame][object]["pred"] = o_scaled.tolist()
                            df[frame][object]["class"] = cls

                            im = drawer.draw_points(o[0], old_x, old, cfg,["../kitti_rev2/training/"+str(file_names[i]),frame,"0000",sx,sy,"test"],o_bbox)
                            im.save(newp+"/imgs/"+file_names[i]+"/"+frame+".png")

            with open(newp + "/data/"+file_names[i]+".json","w") as out_f:
                json.dump(df,out_f,sort_keys=True)
        f = open(newp + "/dat", "w")
        f.write(str(set(total)))
        f.close()

def _test(cf):
    cfg = cf
    print("TEST_")
    filtered=True
    cfg['batch']=1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['GPU'])
    inpts = tf.placeholder(tf.float32, [cfg['batch'], cfg['prev_leng'], cfg['dims']])
    outs = tf.placeholder(tf.float32, [cfg['batch'], cfg['pred_ext'] + cfg['fut_leng'], cfg['dims']])
    mod = model.rec_model(cfg)
    saver = tf.train.Saver()
    newp = str(time.time()).split(".")[0][-4:] + "_test_" + cfg["load_path"].split("/")[-3]
    if (filtered):
        newp+="_FILTERED"
    os.mkdir(newp)
    os.mkdir(newp + "/model")
    os.mkdir(newp + "/data")
    os.mkdir(newp + "/imgs")
    total = []
    test_paths=[d for d in os.listdir("../kitti_rev2/training/")]
    for t in test_paths:
        os.mkdir(newp+"/imgs/"+t)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        print(cfg['load'],cfg['load_path'] )
        if (cfg['load'] == True):
            saver.restore(sess, cfg['load_path'])
            print("MODEL LOADED")
        else:
            print("TESTING must have load=True")
            exit()

        jsons, tot, file_names = loader(test_paths)

        for i, d in enumerate(jsons):
            print(i)
            df={}
            f_keys = d.keys()
            for frame in f_keys:
                df[frame]={}
                img, sx, sy = get_segm_new("../kitti_rev2/training/"+str(file_names[i])+"/deeplab_cache", frame)

                for object in d[frame]:
                    df[frame][object]={}
                    cls = d[frame][object]["track_class_name"]

                    if (len(d[frame][object]["future"]) >= cfg['fut_leng']) and (
                            len(d[frame][object]["past"]) >= cfg['prev_leng'] - 1):

                        gt = np.clip(np.array(d[frame][object]["future"][0:cfg['fut_leng']]), -1000, 3000)
                        past = np.clip(np.array(d[frame][object]["past"][-cfg['prev_leng'] + 1:]), -1000, 3000)

                        if (np.sqrt(np.sum(np.square(gt[-1]-past[0])))>80) or not filtered:
                            pres = np.array(d[frame][object]["present"])
                            bbox = d[frame][object]["box"]
                            bbox[0] = bbox[0] / float(sy)
                            bbox[1] = bbox[1] / float(sy)
                            bbox[2] = bbox[2] / float(sy)
                            bbox[3] = bbox[3] / float(sy)
                            bbox=np.array(bbox)
                            bbox=np.expand_dims(bbox,0)
                            X = np.concatenate((past, np.expand_dims(pres, 0)), 0)
                            tot = smooth(np.concatenate((X, gt), 0))
                            xo=tot[0:cfg['prev_leng']]
                            x = xo /(float(sy),float(sy))
                            gto=tot[cfg['prev_leng']:]
                            gt = gto /(float(sy),float(sy))
                            x=np.expand_dims(x,0)
                            gt = np.expand_dims(gt, 0)
                            tot = np.concatenate([x[:, -cfg['pred_ext']:], gt], -2)

                            imx=np.expand_dims(img,0)

                            o = sess.run(mod.out,
                                         feed_dict={inpts: x, mod.image: imx, outs: tot, mod.target: tot,mod.box:bbox,
                                                    mod.inputs: x})
                            o_scaled=scale_up(o[0],sy,sy)
                            gt_scaled=gto
                            df[frame][object]["past"] = xo.tolist()
                            df[frame][object]["gt"] = gt_scaled.tolist()
                            df[frame][object]["pred"] = o_scaled.tolist()
                            df[frame][object]["class"] = cls
                            ########
                            ot=(newp+"/imgs/"+file_names[i].split("/"[0:-1]))
                            #[pt, frame, object, sx, sy, cls]
                            im = drawer.draw_points(o_scaled, xo, gt_scaled, cfg,[ot,file_names[i][-1],"0000",sx,sy,"test"],bbox)
                            im.save(newp+"/imgs/"+file_names[i])
                            ########
            with open(newp + "/data/"+file_names[i]+".json","w") as out_f:
                json.dump(df,out_f,sort_keys=True)
        f = open(newp + "/dat", "w")
        f.write(str(set(total)))
        f.close()



def test_broken(cf):
    cfg = cf
    print("TEST_")
    filtered = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    inpts = tf.placeholder(tf.float32, [cfg['batch'], cfg['prev_leng'], cfg['dims']])
    outs = tf.placeholder(tf.float32, [cfg['batch'], cfg['pred_ext'] + cfg['fut_leng'], cfg['dims']])
    #mod = model.rec_model(cfg)
    cfg['old']=True
    newp = str(time.time()).split(".")[0][-4:] + "_test_" + cfg["load_path"].split("/")[-3]
    if (filtered):
        newp += "_FILTERED"
    os.mkdir(newp)
    os.mkdir(newp + "/model")
    os.mkdir(newp + "/data")
    os.mkdir(newp + "/imgs")
    cfg['center']=False
    total = []
    test_paths = [d for d in os.listdir("../kitti_rev2/training/")]
    for t in test_paths:
        os.mkdir(newp + "/imgs/" + t)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        print(cfg['load'], cfg['load_path'])
        if (cfg['load'] == True):
            saver = tf.train.import_meta_graph("/".join(cfg['load_path'].split("/")[0:-2])+"/model/model.ckpt.meta")
            saver.restore(sess, tf.train.latest_checkpoint("/".join(cfg['load_path'].split("/")[0:-2])+"/model/"))
            graph = tf.get_default_graph()
            graph_def = sess.graph.as_graph_def(add_shapes=True)
            coll=graph.get_all_collection_keys()
            mod=graph

            c=0
            tensors_per_node = [node.values() for node in graph.get_operations()]
            tensor_names = [(tensor.name, tensor) for tensors in tensors_per_node for tensor in tensors if tensor.name.startswith("GEN/DEC/dense")]
            # (u'Placeholder:0', < tf.Tensor 'Placeholder:0' shape=(1, 10, 2)dtype = float32 >),
            # (u'Placeholder_1:0', < tf.Tensor 'Placeholder_1:0' shape=(1, 31, 2)dtype = float32 >),
            # (u'Placeholder_2:0', < tf.Tensor 'Placeholder_2:0' shape=(32, 10, 2)dtype = float32 >),
            # (u'Placeholder_1_1:0', < tf.Tensor 'Placeholder_1_1:0' shape=(32, 31, 2)dtype = float32 >),
            # (u'Placeholder_2_1:0', < tf.Tensor 'Placeholder_2_1:0' shape=(32, 10, 2)dtype = float32 >),
            # (u'Placeholder_3:0', < tf.Tensor 'Placeholder_3:0' shape=(32, 31, 2)dtype = float32 >),
            # (u'Placeholder_4:0', < tf.Tensor 'Placeholder_4:0' shape=(32, 4)dtype = float32 >),
            # (u'Placeholder_5:0', < tf.Tensor 'Placeholder_5:0' shape=(32, 128)dtype = float32 >),
            # (u'Placeholder_6:0', < tf.Tensor 'Placeholder_6:0' shape=(32,) dtype = float32 >),
            # (u'Placeholder_7:0', < tf.Tensor 'Placeholder_7:0' shape=(32,)dtype = float32 >),
            # (u'Placeholder_8:0', < tf.Tensor 'Placeholder_8:0' shape=(32,)dtype = float32 >),
            # (u'Placeholder_9:0', < tf.Tensor 'Placeholder_9:0' shape=(32, 128, 256)dtype = float32 >)
            for t in tensor_names:
                print(t)
            out=graph.get_tensor_by_name("GEN/DEC/dense/Tensordot:0")
            b_out=graph.get_tensor_by_name("GEN/DEC/dense/BiasAdd:0")
            inpts=graph.get_tensor_by_name("Placeholder:0")
            targ=graph.get_tensor_by_name("Placeholder_2_1:0")
            images = graph.get_tensor_by_name("Placeholder_9:0")
            outs=graph.get_tensor_by_name("Placeholder_1:0")
            boc=graph.get_tensor_by_name("Placeholder_4:0")

        else:
            print("TESTING must have load=True")
            exit()

        jsons, tot, file_names, vmf = loader(test_paths)

        for i, d in enumerate(jsons):
            print(i)
            df = {}
            f_keys = d.keys()
            for frame in f_keys:
                df[frame] = {}
                img, sx, sy = get_segm("../kitti_rev2/training/" + str(file_names[i]), frame)

                for object in d[frame]:
                    df[frame][object] = {}
                    cls = d[frame][object]["track_class_name"]

                    if (len(d[frame][object]["future"]) >= cfg['fut_leng']) and (
                            len(d[frame][object]["past"]) >= cfg['prev_leng'] - 1):

                        gt = np.clip(np.array(d[frame][object]["future"][0:cfg['fut_leng']]), -1000, 3000)
                        past = np.clip(np.array(d[frame][object]["past"][-cfg['prev_leng'] + 1:]), -1000, 3000)
                        fact=4.0

                        if (np.sqrt(np.sum(np.square(gt[-1] - past[0]))) > 60) or not filtered:
                            pres = np.array(d[frame][object]["present"])
                            bbox = d[frame][object]["box"]

                            bbox[0] = (bbox[0] / (sx /fact))
                            bbox[1] = (bbox[1] / float(sy))
                            bbox[2] = (bbox[2] / (sx / fact))
                            bbox[3] = (bbox[3] / float(sy))
                            bbox = np.array(bbox)
                            o_bbox = bbox
                            bbox = np.expand_dims(bbox, 0)
                            X = np.concatenate((past, np.expand_dims(pres, 0)), 0)
                            #conc = np.concatenate((X, gt), 0) / np.array(((float(sy)), float(sy)))
                            conc = np.concatenate((X, gt), 0) / np.array((sx/4.0,sy))
                            tot = smooth(conc)
                            xo = tot[0:cfg['prev_leng']]
                            x = xo
                            gto = tot[cfg['prev_leng']:]
                            old = gto
                            old_x = x
                            gt = gto
                            x = np.expand_dims(x, 0)
                            gt = np.expand_dims(gt, 0)
                            tot = np.concatenate([x[:, -cfg['pred_ext']:], gt], -2)

                            imx = np.expand_dims(img, 0)

                            # o = sess.run(out,
                            #              feed_dict={inpts: x, mod.image: imx, outs: tot, mod.target: tot, mod.box: bbox,
                            #                         mod.inputs: x})
                            ##########################
                            #HORRIBLE
                            x=np.tile(x,[cfg['batch'],1,1])
                            imx = np.tile(imx, [cfg['batch'], 1, 1])
                            tot= np.tile(tot, [cfg['batch'], 1, 1])
                            bbox=np.tile(bbox, [cfg['batch'], 1])

                            ##########################
                            o,bias = sess.run([out,b_out],feed_dict={inpts: x,images:imx,targ:x, outs:tot,boc:bbox})
                            o=bias
                            poins = o[0]
                            o_scaled = poins * (sx / fact, float(sy))
                            gto = gto
                            gt_scaled = gto * (sx / fact, float(sy))
                            xo = xo
                            xo = xo * (sx / fact, float(sy))
                            df[frame][object]["past"] = xo.tolist()
                            df[frame][object]["gt"] = gt_scaled.tolist()
                            df[frame][object]["pred"] = o_scaled.tolist()
                            df[frame][object]["class"] = cls

                            im = drawer.draw_points(o[0], old_x, old, cfg,
                                                    ["../kitti_rev2/training/" + str(file_names[i]), frame, "0000", sx,
                                                     sy, "test"], o_bbox)
                            im.save(newp + "/imgs/" + file_names[i] + "/" + frame + ".png")

            with open(newp + "/data/" + file_names[i] + ".json", "w") as out_f:
                json.dump(df, out_f, sort_keys=True)
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
    os.mkdir(newp+"/imgs")
    total = []
    test_paths=[d for d in os.listdir("../kitti_rev2/training/")]
    for t in test_paths:
        os.mkdir(newp+"/imgs/"+t)
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
                            bbox = d[frame][object]["box"]
                            bbox[0] = bbox[0] / float(sy)
                            bbox[1] = bbox[1] / float(sy)
                            bbox[2] = bbox[2] / float(sy)
                            bbox[3] = bbox[3] / float(sy)
                            X = np.concatenate((past, np.expand_dims(pres, 0)), 0)
                            tot = smooth(np.concatenate((X, gt), 0))
                            xo=tot[0:cfg['prev_leng']]
                            x = xo /(float(sy),float(sy))
                            gto=tot[cfg['prev_leng']:]
                            gt = gto /(float(sy),float(sy))
                            x=np.expand_dims(x,0)
                            gt = np.expand_dims(gt, 0)
                            tot = np.concatenate([x[:, -cfg['pred_ext']:], gt], -2)

                            imx=np.expand_dims(img,0)

                            o = sess.run(mod.out,
                                         feed_dict={inpts: x, mod.image: imx,
                                                    outs: tot, mod.target: tot,  mod.box: bbox,
                                                    mod.inputs: x})
                            o_scaled=scale_up(o,sy,sy)
                            gt_scaled=gto
                            df[frame][object]["past"] = xo.tolist()
                            df[frame][object]["gt"] = gt_scaled.tolist()
                            df[frame][object]["pred"] = o_scaled.tolist()
                            df[frame][object]["class"] = cls
                            ###############
                            ot=(newp+"/imgs/"+file_names[i].split("/"[0:-1]))
                            #[pt, frame, object, sx, sy, cls]
                            im = drawer.draw_points(o_scaled, xo, gt_scaled, cfg,[ot,file_names[i][-1],"0000",sx,sy,"test"],bbox)
                            im.save(newp+"/imgs/"+file_names[i])
                            #################

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
    video_frames = []
    for v in jpath:
        files.append(v)
        fm=json.load(open("../kitti_rev2/training/"+v+"/trajectories.json"))
        trajs=0
        for frm in fm.keys():
            trajs+=len(fm[frm].keys())
        total+=trajs
        jsons.append(fm)
        # for frm in fm.keys():
        #     jsons[v] = fm
        #     video_frames.append([v, frm])
    return jsons,total,files,video_frames

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


def get_segm_new(filenames, frame):
    # print(filenames)
    img_path = filenames  # .replace(self.cfg['json_path'], "../image_02")
    epoc = img_path  # img_path.split("/")[0:-1]
    # epoc.append(img_path.split("/")[-1].zfill(4))
    # img_path = "/".join(epoc)
    num = frame
    num = num.replace("frame_", "")
    num = num.zfill(8)
    img = np.load(img_path +"/"+ num+".npz")
    gg = img.f.seg_map
    sx, sy = gg.shape[1], gg.shape[0]
    gg = gg.astype('float32')
    gg = cv2.resize(gg, (256, 128))

    channels = [0, 1, 12, 13, 18]
    mat = np.eye(19)[np.array(gg, dtype=np.int32)]
    mat_clean = np.zeros([128, 256, len(channels)])
    for i, j in enumerate(channels):
        mat_clean[:, :, i] = mat[:, :, j]
    return mat_clean, sx, sy

def scale_up(gts, sx, sy):
    poins = np.asarray(gts)
    poins = poins * sy
    return np.array(poins, dtype=np.int32)


def get_config(pt):
    with open(pt+"/data/config.yaml") as raw_cfg:
        cfg = yaml.load(raw_cfg)
        cfg['load']=True
        cfg['load_path']=pt+"/model/model.ckpt"
    return cfg


def main():
    "34600_AUTOREGRESSIVE_l2_inverted_RNN10_30_64_512_clean_salmon"
    print(sys.argv[1])
    cfg=get_config(sys.argv[1])

    if cfg['type']==2:
        test_gan()
    else:
        test(cfg)
        #test_broken(cfg)


main()
