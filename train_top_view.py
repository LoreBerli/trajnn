import shutil
import numpy as np
import time
import dataset
import tensorflow as tf
import datetime
import model_top_view as model
import model_double
import yaml
import utils_top_view as utils
import random
import gc
import json
import os
import drawer
import name_generator

# os.environ["CUDA_VISIBLE_DEVICES"]="6"
rnd_names = ["Artful", "Aardvark", "Zesty", "Zapus", "Precise", "Pangolin", "Lucid", "Lynx", "Utopic", "Unicorn"]
nms = ["RNN", "CTX", "GAN", "LINEAR", "OLD", "MULTIPLE"]


def train():
    cfg = get_config()
    inpts = tf.placeholder(tf.float32, [cfg['batch'], cfg['prev_leng'], cfg['dims']])
    outs = tf.placeholder(tf.float32, [cfg['batch'], cfg['fut_leng'], cfg['dims']])
    # mod=model.rec_model(cfg)
    mod = model.rec_model(cfg)
    # opti= tf.train.RMSPropOptimizer(cfg['lr'], decay=0.9, momentum=0.5)
    opti = tf.train.AdamOptimizer(cfg['lr'])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['GPU'])
    if (cfg['clipping']):
        gvs = opti.compute_gradients(mod.loss)
        capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in gvs]
        mini = opti.apply_gradients(capped_gvs)
    else:
        mini = opti.minimize(mod.loss)

    loader = utils.Loader(cfg)

    init = tf.initializers.global_variables()
    saver = tf.train.Saver()

    if (cfg['type'] != 3):
        newp = str(time.time()).split(".")[0][-5:] + "_" + cfg['prefix'] + "_" + "-".join(
            nms[i] for i in range(cfg['type'] + 1)) + str(cfg['prev_leng']) + "_" + str(cfg['fut_leng']) + "_" + str(
            cfg['units']) + "_" + str(cfg['lat_size']) + "_" + "_".join(name_generator.get_combo())
    else:
        newp = str(time.time()).split(".")[0][-5:] + "_" + cfg['prefix'] + "_" + nms[cfg['type']] + str(
            cfg['prev_leng']) + "_" + str(cfg['fut_leng']) + "_" + str(cfg['units']) + "_" + str(
            cfg['lat_size']) + "_" + "_".join(name_generator.get_combo())

    os.mkdir(newp)
    os.mkdir(newp + "/model")
    os.mkdir(newp + "/data")
    shutil.copy("config.yaml", newp + "/data/" + "config.yaml")
    tf.summary.scalar("loss", mod.loss)
    # tf.summary.scalar("leng_loss", mod.leng_loss)
    # tf.summary.scalar("dirs_loss", mod.dirs_loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    merge = tf.summary.merge_all()

    with tf.Session(config=config) as sess:
        if (cfg['load'] == "True"):
            saver.restore(sess, cfg['load_path'])
            print("LOADED MODEL at " + cfg['load_path'])
        else:
            sess.run(init)

        train_writer = tf.summary.FileWriter("logs/" + newp, sess.graph)
        # test_writer = tf.summary.FileWriter(newp + "/data", sess.graph)
        print(newp)
        for e in range(cfg['epochs']):
            for i in range(0, 1000):

                x, gt, info, img = loader.serve()
                noiz = np.random.randn(cfg['batch'], 32)
                img = np.eye(4)[np.array(img, dtype=np.int32)]
                img = np.squeeze(img)

                summary, ls, o, _ = sess.run([merge, mod.loss, mod.out, mini],
                                             feed_dict={inpts: utils.to_offsets(x), mod.noiz: noiz,
                                                        outs: utils.to_offsets(gt), mod.target: utils.to_offsets(gt),
                                                        mod.inputs: utils.to_offsets(x), mod.drop: 0.9, mod.image: img,
                                                        mod.factor: [1.0]})
                if (i % 500 == 0):
                    print("TRAIN ", ls)
                if (i == 1):
                    summ = 0
                    all_errors_at_t = []
                    all_ade_at_t = []
                    for tst in range(0, 20):
                        x, gt, info, imga = loader.serve_test()
                        futures = []
                        noiz = np.random.randn(cfg['batch'], 32)
                        img = np.eye(4)[np.array(imga, dtype=np.int32)]
                        img = np.squeeze(img)
                        stds = np.arange(1.2, 1.0, -0.2 / 3.0)
                        for fs in range(0, len(stds)):
                            imc, summary, ls, o = sess.run([mod.crops, merge, mod.loss, mod.out],
                                                           feed_dict={inpts: utils.to_offsets(x),
                                                                      outs: utils.to_offsets(gt),
                                                                      mod.target: utils.to_offsets(gt),
                                                                      mod.inputs: utils.to_offsets(x),
                                                                      mod.drop: 1.0, mod.noiz: noiz, mod.image: img,
                                                                      mod.factor: [stds[fs]]})

                            summ += ls
                            futures.append(o)
                        futures = np.array(futures)
                        print("FUTURES", futures.shape)
                        futures = np.transpose(futures, [1, 0, 2, 3])

                        for k in range(cfg['batch']):
                            preds = futures
                            all_errors = np.sqrt(np.sum(((preds[k] - gt[k]) ** 2), -1))
                            best_future_id = np.argmin(np.mean(all_errors, -1))
                            best_future_errors_at_t = all_errors[best_future_id]
                            best_future_ade_at_t = np.divide(np.cumsum(best_future_errors_at_t),
                                                             np.arange(1, cfg['fut_leng'] + 1))
                            drawer.draw_scenes(imga[k], str(e) + "-" + str(tst) + "-" + str(k), newp,
                                               futures=futures[k, :], past=x[k],
                                               gt=gt[k], dim=cfg['dim_clip'], text=str(best_future_errors_at_t),
                                               special=best_future_id)

                            drawer.draw_crops(imc[k], str(k), newp)
                            all_errors_at_t.append(best_future_errors_at_t)
                            all_ade_at_t.append(best_future_ade_at_t)
                    # print((all_errors_at_t))
                    mean_errors_at_t = np.mean(all_errors_at_t, 0)
                    mean_ade_at_t = np.mean(all_ade_at_t, 0)
                    # print(str(mean_errors_at_t) + " mean err at t  "+str(mean_ade_at_t)+" mean ade at t")

                    print(str(mean_errors_at_t[-1]) + " 4s err " + str(mean_ade_at_t[-1]) + " 4s ade")

                    print(str(np.mean(summ)) + " iteration " + str(i) + "of " + str(1000) + " ,at epoch " + str(
                        e) + " of " + str(cfg['epochs']))

            if (e % 3 == 0):
                print("SAVING " + newp)
                saver.save(sess, newp + "/model/model.ckpt")

            # if(e%3==0):
            #     x, gt, f, box,info,img = loader.serve_test()
            #
            #     tot = np.concatenate([x[:,-cfg['pred_ext']:], gt], -2)
            #     summary, ls, o = sess.run([merge, mod.loss, mod.out],
            #                               feed_dict={inpts: x,mod.image:img, outs: tot, mod.target: tot, mod.inputs: x, mod.feats: f,mod.box:box})
            #     print(info[0])
            #     for k in range(min(16,len(info))):
            #         im=drawer.draw_points(o[k],x[k],gt[k],cfg,info[k])
            #
            #         im.save(newp+"/"+str(e)+"_"+str(k)+".png")#"#str("_".join(info[i]))+".png")
            # print("SAVING "+newp)
            # saver.save(sess,newp+"/model/model.ckpt")


def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)


def test_multiple(cfg):
    # cfg = get_config()
    tf.reset_default_graph()
    cfg['batch'] = 1
    inpts = tf.placeholder(tf.float32, [cfg['batch'], cfg['prev_leng'], cfg['dims']])
    outs = tf.placeholder(tf.float32, [cfg['batch'], cfg['fut_leng'], cfg['dims']])

    # mod=model.rec_model(cfg)
    mod = model.rec_model(cfg)
    # opti= tf.train.RMSPropOptimizer(cfg['lr'], decay=0.9, momentum=0.5)
    opti = tf.train.AdamOptimizer(cfg['lr'])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['GPU'])
    if (cfg['clipping']):
        gvs = opti.compute_gradients(mod.loss)
        capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in gvs]
        mini = opti.apply_gradients(capped_gvs)
    else:
        mini = opti.minimize(mod.loss)

    loader = utils.Loader(cfg,True)

    init = tf.initializers.global_variables()
    saver = tf.train.Saver()
    err_per_im={}
    # if (cfg['type'] != 3):
    #     newp = str(time.time()).split(".")[0][-5:] + "_" + cfg['prefix'] + "_" + "-".join(
    #         nms[i] for i in range(cfg['type'] + 1)) + str(cfg['prev_leng']) + "_" + str(cfg['fut_leng']) + "_" + str(
    #         cfg['units']) + "_" + str(cfg['lat_size']) + "_" + "_".join(name_generator.get_combo())
    # else:
    newp = str(time.time()).split(".")[0][-5:] + "_test_"+cfg['load_path'].split("/")[-3]

    os.mkdir(newp)
    os.mkdir(newp + "/model")
    os.mkdir(newp + "/data")
    shutil.copy("config.yaml", newp + "/data/" + "config.yaml")
    tf.summary.scalar("loss", mod.loss)
    # tf.summary.scalar("leng_loss", mod.leng_loss)
    # tf.summary.scalar("dirs_loss", mod.dirs_loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    merge = tf.summary.merge_all()

    with tf.Session(config=config) as sess:

        if (cfg['load'] == True):
            saver.restore(sess, cfg['load_path'])
            print("LOADED MODEL at " + cfg['load_path'])
        else:
            exit()
            # sess.run(init)

        train_writer = tf.summary.FileWriter("logs/" + newp, sess.graph)

        print(newp)

        summ = 0
        all_errors_at_t = []
        all_ade_at_t = []
        for i in range(0, len(loader.data_test)):

            index, x, gt, presents, video_track, vehicles, number_vec, scene, scene_one_hot = loader.data_test[i]
            info = [(index, video_track, number_vec)]
            imga = [scene.transpose([1, 2, 0])]

            noiz = np.random.randn(cfg['batch'], 8)
            img = np.eye(4)[np.array(imga, dtype=np.int32)]
            img = np.squeeze(img)
            img = [img]
            x = np.expand_dims(x, 0)
            gt = np.expand_dims(gt, 0)

            summary, ls, o, dspec = sess.run([ merge, mod.loss, mod.out, mod.d_spec],
                                                  feed_dict={inpts: utils.to_offsets(x),
                                                             outs: utils.to_offsets(gt),
                                                             mod.target: utils.to_offsets(gt),
                                                             mod.inputs: utils.to_offsets(x),
                                                             mod.drop: 1.0, mod.noiz: noiz, mod.image: img,
                                                             mod.factor: [1.0]})

            summ += ls

            futures = np.array(o)
            futures = np.transpose(futures, [1, 0, 2, 3])

            for k in range(cfg['batch']):
                preds = futures
                gt_s = gt[k] / 2.0
                pre_s = preds[k] / 2.0
                all_errors = np.sqrt(np.sum(((pre_s - gt_s) ** 2), -1))

                best_future_id = np.argmin(np.mean(all_errors, -1))
                best_future_errors_at_t = all_errors[best_future_id]
                best_future_ade_at_t = np.divide(np.cumsum(best_future_errors_at_t),
                                                 np.arange(1, cfg['fut_leng'] + 1))

                err_per_im[str(index)]=[best_future_errors_at_t[-1]]
                drawer.draw_simple_scenes(imga[k],
                                   str(index), newp, futures=futures[k, :],
                                   past=x[k],
                                   gt=gt[k], dim=cfg['dim_clip'], weird=dspec[k],
                                   text=str(best_future_errors_at_t), special=best_future_id)
                all_errors_at_t.append(best_future_errors_at_t)
                all_ade_at_t.append(best_future_ade_at_t)

        with open(newp + "/data/errs", "w+") as fl2:
            json.dump(err_per_im,fl2)
        fl2.close()

        mean_errors_at_t = np.mean(all_errors_at_t, 0)
        mean_ade_at_t = np.mean(all_ade_at_t, 0)
        print("_____________")
        print(mean_errors_at_t)
        print(np.mean(mean_errors_at_t, 0))
        print("_____________")
        print(mean_ade_at_t)
        print(np.mean(mean_ade_at_t, 0))
        with open(newp + "/data/data", "w+") as fl:
            fl.write(newp + " \n")
            fl.write("ERRORS \n" + str(mean_errors_at_t))
            fl.write("\n ADES \n" + str(mean_ade_at_t))
        fl.close()


def train_multiple(cfg):
    # cfg = get_config(cf)
    tf.reset_default_graph()
    inpts = tf.placeholder(tf.float32, [cfg['batch'], cfg['prev_leng'], cfg['dims']])
    outs = tf.placeholder(tf.float32, [cfg['batch'], cfg['fut_leng'], cfg['dims']])

    # mod=model.rec_model(cfg)
    mod = model.rec_model(cfg)
    # opti= tf.train.RMSPropOptimizer(cfg['lr'], decay=0.9, momentum=0.5)
    opti = tf.train.AdamOptimizer(cfg['lr'])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['GPU'])
    if (cfg['clipping']):
        gvs = opti.compute_gradients(mod.loss)
        capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in gvs]
        mini = opti.apply_gradients(capped_gvs)
    else:
        mini = opti.minimize(mod.loss)

    #if(cfg['real_data']):
    #    loader = utils.Loader(cfg,False)
    #if(cfg['synth_data']):
    loader_s = utils.Loader_synth(cfg)
    loader = utils.Loader(cfg,False)

    init = tf.initializers.global_variables()
    saver = tf.train.Saver()

    # if(cfg['type']!= 3):
    #     newp=str(time.time()).split(".")[0][-5:]+"_"+cfg['prefix']+"_"+"-".join(nms[i] for i in range(cfg['type']+1))+str(cfg['prev_leng'])+"_"+str(cfg['fut_leng'])+"_"+str(cfg['units'])+"_"+str(cfg['lat_size'])+"_"+"_".join(name_generator.get_combo())
    # else:
    newp = str(time.time()).split(".")[0][-5:] + "_" + cfg['prefix'] + "_" + nms[cfg['type']] + str(
        cfg['prev_leng']) + "_" + str(cfg['fut_leng']) + "_" + str(cfg['units']) + "_" + str(
        cfg['lat_size']) + "_" + "_".join(name_generator.get_combo())

    os.mkdir(newp)
    os.mkdir(newp + "/model")
    os.mkdir(newp + "/data")
    shutil.copy("config.yaml", newp + "/data/" + "config.yaml")
    tf.summary.scalar("loss", mod.loss)
    # tf.summary.scalar("leng_loss", mod.leng_loss)
    # tf.summary.scalar("dirs_loss", mod.dirs_loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    merge = tf.summary.merge_all()

    with tf.Session(config=config) as sess:

        if (cfg['load'] == True):
            saver.restore(sess, cfg['load_path'])
            print("LOADED MODEL at " + cfg['load_path'])
        else:
            sess.run(init)

        train_writer = tf.summary.FileWriter("logs/" + newp, sess.graph)
        # test_writer = tf.summary.FileWriter(newp + "/data", sess.graph)
        print(newp)
        for e in range(cfg['epochs']):
            t_l = 0
            tt=time.time()
            for i in range(0, 1000):
                if not cfg['combined']:
                    if(cfg['real_data'] and not cfg['synth_data']):
                        x, gt, _, img,_ = loader.serve_multiprocess_train()
                    if(cfg['synth_data'] and not cfg['real_data']):
                        x, gt, _, img,_ = loader_s.serve_multiprocess()

                    elif(cfg['synth_data'] and cfg['real_data'] and e<cfg['pretrain']):
                        x, gt, _, img,_ = loader_s.serve_multiprocess()
                    else:
                        x, gt, _, img,_ = loader.serve_multiprocess_train()
                elif(i%2==0):
                    x, gt, _, img,_ = loader.serve_multiprocess_train()
                    x2, gt2, _, img2,_ = loader_s.serve_multiprocess()
                    x[0:cfg['batch']/2]=x2[0:cfg['batch']/2]
                    gt[0:cfg['batch']/2]=gt2[0:cfg['batch']/2]
                    img[0:cfg['batch']/2]=img2[0:cfg['batch']/2]

                else:
                    x, gt, _, img,_ = loader_s.serve_multiprocess()
                    x2, gt2, _, img2,_ = loader.serve_multiprocess_train()
                    x[0:cfg['batch']/2]=x2[cfg['batch']/2:]
                    gt[0:cfg['batch']/2]=gt2[cfg['batch']/2:]
                    img[0:cfg['batch']/2]=img2[cfg['batch']/2:]



                #
                # if (e < 6 and i % 2 == 0):
                #     x, gt, _, img = loader_s.serve()
                # else:
                #     x, gt, _, img = loader.serve()
                noiz = np.random.randn(cfg['batch'], 8)
                # img = np.eye(4)[np.array(img, dtype=np.int32)]
                # img = np.squeeze(img)

                ls, _ = sess.run([mod.loss, mini],
                                 feed_dict={inpts: utils.to_offsets(x), mod.noiz: noiz, outs: utils.to_offsets(gt),
                                            mod.target: utils.to_offsets(gt), mod.inputs: utils.to_offsets(x),
                                            mod.drop: 0.9, mod.image: img, mod.factor: [1.0]})
                t_l += ls

                if (i % 100 == 0):
                    print(time.time()-tt)
                    tt=time.time()
                    print("TRAIN ", t_l / 100.0)
                    t_l = 0.0
                if (i ==1):
                    summ = 0
                    all_errors_at_t = []
                    all_ade_at_t = []
                    for tst in range(0, 10):
                        if (tst == 1 and cfg['synth_data']):
                            x, gt, info, imgd,imga = loader_s.serve_multiprocess()

                        else:
                            x, gt, info, imgd,imga = loader.serve_multiprocess_test()
                            # imgd = np.eye(4)[np.array(imga, dtype=np.int32)]
                            # imgd = np.squeeze(imgd)


                        noiz = np.random.randn(cfg['batch'], 8)
                        #imgd=imga
                        # imgd = np.eye(4)[np.array(imga, dtype=np.int32)]
                        # imgd = np.squeeze(imgd)

                        old_c, imc, summary, ls, o, dspec = sess.run(
                            [mod.coo, mod.crops, merge, mod.loss, mod.out, mod.d_spec],
                            feed_dict={inpts: utils.to_offsets(x), outs: utils.to_offsets(gt),
                                       mod.target: utils.to_offsets(gt), mod.inputs: utils.to_offsets(x),
                                       mod.drop: 1.0, mod.noiz: noiz, mod.image: imgd, mod.factor: [1.0]})

                        summ += ls

                        futures = np.array(o)
                        futures = np.transpose(futures, [1, 0, 2, 3])
                        # print(gt.shape)
                        # print (futures.shape)
                        for k in range(cfg['batch']):

                            # futures[k][0] = gt[k]
                            # futures[k][0][-1][0] = futures[k][0][-1][0] + 1.0
                            preds = futures
                            gt_s=gt[k]/2.0
                            pre_s=preds[k]/2.0


                            all_errors = np.sqrt(np.sum(((pre_s - gt_s) ** 2), -1))
                            best_future_id = np.argmin(np.mean(all_errors, -1))
                            best_future_errors_at_t = all_errors[best_future_id]
                            best_future_ade_at_t = np.divide(np.cumsum(best_future_errors_at_t),
                                                             np.arange(1, cfg['fut_leng'] + 1))

                            # drawer.draw_scenes(old_c[k][-1], imc[k][-1, :], imga[k],
                            #                    str(e) + "-" + str(tst) + "-" + str(k), newp, futures=futures[k, :],
                            #                    past=x[k],
                            #                    gt=gt[k], dim=cfg['dim_clip'], weird=dspec[k],
                            #                    text=str(best_future_errors_at_t), special=best_future_id)
                            # drawer.draw_crops(imc[k], str(k), newp, dspec[k])
                            all_errors_at_t.append(best_future_errors_at_t)
                            all_ade_at_t.append(best_future_ade_at_t)

                    mean_errors_at_t = np.mean(all_errors_at_t, 0)
                    mean_ade_at_t = np.mean(all_ade_at_t, 0)

                    print(str(mean_errors_at_t[-1]) + " 4s err " + str(mean_ade_at_t[-1]) + " 4s ade")
                    print(str(np.mean(summ)) + " iteration " + str(i) + "of " + str(1000) + " ,at epoch " + str(
                        e) + " of " + str(cfg['epochs']))

            if (e % 3 == 0):
                print("SAVING " + newp)
                saver.save(sess, newp + "/model/model_at_ep_"+str(e)+".ckpt")
                saver.save(sess, newp + "/model/model_last.ckpt")

        return newp


def scale_up(gts, sx, sy):
    poins = np.asarray(gts)
    poins = poins * (sx / 4.0, sy)
    return np.array(poins, dtype=np.int32)


def get_config(cf):
    with open(cf) as raw_cfg:
        cfg = yaml.load(raw_cfg)
    return cfg


def main(cf="config.yaml"):
    cfg = get_config(cf)
    if (cfg['test'] == True):
        print("TEST")
        test_multiple(cfg)

    # elif cfg['type']==2:
    #     print("TRAIN")
    #     train_GAN()

    elif cfg['type'] == 5:
        print("TRAIN")
        train_multiple(cfg)





    else:
        print("TRAIN")
        train()
    # train_GAN()


main()
