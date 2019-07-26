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
import drawer
import name_generator


#os.environ["CUDA_VISIBLE_DEVICES"]="6"
rnd_names=["Artful","Aardvark","Zesty","Zapus","Precise","Pangolin","Lucid","Lynx","Utopic","Unicorn"]
nms=["RNN","CTX","GAN","LINEAR"]
def train():
    cfg = get_config()
    inpts = tf.placeholder(tf.float32,[cfg['batch'],cfg['prev_leng'],cfg['dims']])
    outs = tf.placeholder(tf.float32, [cfg['batch'], cfg['pred_ext']+cfg['fut_leng'], cfg['dims']])

    #mod=model.rec_model(cfg)
    mod=model.rec_model(cfg)
    # opti= tf.train.RMSPropOptimizer(cfg['lr'], decay=0.9, momentum=0.5)
    opti= tf.train.AdamOptimizer(cfg['lr'])
    os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg['GPU'])
    if(cfg['clipping']):
        gvs = opti.compute_gradients(mod.loss)
        capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in gvs]
        mini=opti.apply_gradients(capped_gvs)
    else:
        mini=opti.minimize(mod.loss)


    loader=utils.Loader(cfg)

    init=tf.initializers.global_variables()
    saver=tf.train.Saver()

    if(cfg['type']!= 3):
        newp=str(time.time()).split(".")[0][-5:]+"_"+cfg['prefix']+"_"+"-".join(nms[i] for i in range(cfg['type']+1))+str(cfg['prev_leng'])+"_"+str(cfg['fut_leng'])+"_"+str(cfg['units'])+"_"+str(cfg['lat_size'])+"_"+"_".join(name_generator.get_combo())
    else:
        newp=str(time.time()).split(".")[0][-5:]+"_"+cfg['prefix']+"_"+nms[cfg['type']]+str(cfg['prev_leng'])+"_"+str(cfg['fut_leng'])+"_"+str(cfg['units'])+"_"+str(cfg['lat_size'])+"_"+"_".join(name_generator.get_combo())

    os.mkdir(newp)
    os.mkdir(newp+"/model")
    os.mkdir(newp+"/data")
    shutil.copy("config.yaml",newp+"/data/"+"config.yaml")
    tf.summary.scalar("loss",mod.loss)
    tf.summary.scalar("leng_loss", mod.leng_loss)
    tf.summary.scalar("dirs_loss", mod.dirs_loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    merge = tf.summary.merge_all()
    with tf.Session(config=config) as sess:
        if(cfg['load']=="True"):
            saver.restore(sess,cfg['load_path'])
            print("LOADED MODEL at " +cfg['load_path'])
        else:
            sess.run(init)

        train_writer=tf.summary.FileWriter("logs/"+newp,sess.graph)
        #test_writer = tf.summary.FileWriter(newp + "/data", sess.graph)
        print(newp)
        for e in range(cfg['epochs']):
            x, gt, f, box,info,img = loader.serve_test()
            tot = np.concatenate([x[:,-cfg['pred_ext']:], gt], -2)
            summary, ls, o = sess.run([merge, mod.loss, mod.out],
                                      feed_dict={inpts: x, outs: tot, mod.target: tot, mod.inputs: x, mod.drop:1.0,mod.feats: f,mod.box:box,mod.image:img})
            print(info[0])

            # for k in range(min(16,len(info))):
            #     im=drawer.draw_points(o[k],x[k],gt[k],cfg,info[k])
            #     im.save(newp+"/"+str(e)+"_"+str(k)+".png")
            for i in range(0,loader.total_data//cfg['batch']):
                # print(str(e)+" _ "+str(i))
                x, gt,f,box,info,img = loader.serve()
                tot = np.concatenate([x[:,-cfg['pred_ext']:], gt], -2)
                ls,_,o=sess.run([mod.loss,mini,mod.out],feed_dict={inpts:x,outs:tot,mod.target:tot,mod.image:img,mod.inputs:x,mod.box:box,mod.feats:f,mod.drop:0.7})
                if(i%400==0):
                    print("TRAIN ",ls)
                if(i%400==0):
                    summ=0
                    for tst in range(0,20):
                        x, gt,f,box,info,img = loader.serve_test()

                        tot=np.concatenate([x[:,-cfg['pred_ext']:],gt],-2)
                        summary,ls,o = sess.run([merge,mod.loss,mod.out], feed_dict={inpts: x,mod.image:img,mod.drop:1.0, outs: tot, mod.target: tot,mod.box:box, mod.inputs: x,mod.feats:f})
                        summ+=ls
                        for k in range(4):
                            im = drawer.draw_points(o[k], x[k], gt[k], cfg, info[k],box[k])
                            im.save(newp + "/" + str(e) + "_" + str(tst)+"_"+str(k) + ".png")
                        # print(x[0])
                    train_writer.add_summary(summary, (loader.total_data*e)+i)
                    print(str(summ/20.0)+" iteration "+str(i)+"of "+ str(loader.total_data//cfg['batch'])+" ,at epoch "+str(e)+" of "+str(cfg['epochs']))
                    # x, gt, f, box, info, img = loader.serve_test()
                    #
                    # tot = np.concatenate([x[:, -cfg['pred_ext']:], gt], -2)
                    # summary, ls, o = sess.run([merge, mod.loss, mod.out],
                    #                           feed_dict={inpts: x, mod.image: img, outs: tot, mod.target: tot,
                    #                                      mod.inputs: x, mod.feats: f, mod.box: box})
                    # print(info[0])

                        #drawer.points_alone(o[k],x[k],gt[k],k,newp)
                    if (i % 4000 == 0):
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
            print("SAVING "+newp)
            saver.save(sess,newp+"/model/model.ckpt")


def train_GAN():
    cfg = get_config()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg['GPU'])
    inpts = tf.placeholder(tf.float32, [cfg['batch'], cfg['prev_leng'], cfg['dims']])
    outs = tf.placeholder(tf.float32, [cfg['batch'], cfg['pred_ext'] + cfg['fut_leng'], cfg['dims']])
    real_imgs=tf.placeholder(tf.float32,shape=[cfg['batch'],128,256,5])

    # mod=model.rec_model(cfg)
    mod = model.rec_model(cfg)
    optimizer = tf.train.AdamOptimizer(cfg['lr'])

    r_logits=  mod.discrim(inpts,outs,real_imgs)
    f_logits = mod.discrim(inpts,mod.out, real_imgs,reuse=True)

    r_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits)))
    f_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
    disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits)*0.9) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
    d_opti=tf.train.AdamOptimizer(cfg['d_lr'])
    dim_opti=tf.train.AdamOptimizer(cfg['im_d_lr'])
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)*0.9))
    wei=cfg['wei']
    alpha=tf.placeholder(dtype=tf.float32)
    d_step=d_opti.minimize(disc_loss,var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='DISCR'))
    dim_step=dim_opti.minimize(disc_loss,var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='DIM'))
    gvs = optimizer.compute_gradients(mod.loss+gen_loss*alpha,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='GEN'))
    print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='GEN'))
    print("============================================================")
    print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='DISCR'))
    capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in gvs]
    mini = optimizer.apply_gradients(capped_gvs)
    loader = utils.Loader(cfg)
    init = tf.initializers.global_variables()
    saver = tf.train.Saver()
    init=tf.initializers.global_variables()
    saver=tf.train.Saver()

    newp=str(time.time()).split(".")[0][-4:]+"_"+cfg['prefix']+"_"+"-".join(nms[i] for i in range(cfg['type']+1))+str(cfg['prev_leng'])+"_"+str(cfg['fut_leng'])+"_"+"_".join(name_generator.get_combo())

    os.mkdir(newp)
    os.mkdir(newp + "/model")
    os.mkdir(newp + "/data")
    shutil.copy("config.yaml", newp + "/data/" + "config.yaml")
    tf.summary.scalar("loss", mod.loss)
    tf.summary.scalar("leng_loss", mod.leng_loss)
    tf.summary.scalar("dirs_loss", mod.dirs_loss)
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
        print("OK")
        for e in range(cfg['epochs']):
            wei=max(wei+0.1,1.0)
            # x, gt, f, box, info, img = loader.serve_test()
            # tot = np.concatenate([x[:, -cfg['pred_ext']:], gt], -2)
            # summary, ls, o = sess.run([merge, mod.loss, mod.out],
            #                           feed_dict={inpts: x, outs: tot, mod.target: tot, mod.inputs: x, mod.feats: f,
            #                                      mod.box: box, mod.image: img})
            # print(info[0])
            #
            # for k in range(min(16, len(info))):
            #     im = drawer.draw_points(o[k], x[k], gt[k], cfg, info[k])
            #     im.save(newp + "/" + str(e) + "_" + str(k) + ".png")
            for i in range(0, loader.total_data // cfg['batch']):
                # print(str(e)+" _ "+str(i))

                for k in range(cfg["disc_runs_per_gen_runs"]):
                    x, gt, f, box, info, img = loader.serve()
                    tot = np.concatenate([x[:, -cfg['pred_ext']:], gt], -2)
                    sess.run([d_step,dim_step],feed_dict={inpts: x, outs: tot, mod.target: tot, mod.image: img, mod.inputs: x,
                                                   mod.box: box, mod.feats: f,real_imgs:img})
                ls, _, o = sess.run([mod.loss, mini, mod.out],
                                    feed_dict={inpts: x, outs: tot, mod.target: tot, mod.image: img, mod.inputs: x,
                                               mod.box: box, mod.feats: f,real_imgs:img,alpha:wei})
                if (i % 200 == 0):
                    summ = 0
                    d_summ=0
                    fake_loss=0
                    real_loss=0
                    g_l=0
                    for tst in range(0, 20):
                        x, gt, f, box, info, img = loader.serve_test()
                        tot = np.concatenate([x[:, -cfg['pred_ext']:], gt], -2)
                        summary, ls, o,gl = sess.run([merge, mod.loss, mod.out,gen_loss],
                                                  feed_dict={inpts: x, mod.image: img, outs: tot, mod.target: tot,
                                                             mod.box: box, mod.inputs: x, mod.feats: f,real_imgs:img})
                        rl,fl,dls,r_l,f_l= sess.run([r_logits,f_logits,disc_loss,r_loss,f_loss], feed_dict={inpts: x, outs: tot, mod.target: tot, mod.image: img, mod.inputs: x,
                                                    mod.box: box, mod.feats: f, real_imgs: img})

                        summ += ls+gl
                        g_l+=gl
                        d_summ+=dls
                        fake_loss+=f_l
                        real_loss+=r_l
                        # print(x[0])
                    train_writer.add_summary(summary, (loader.total_data * e) + i)
                    print("fake: "+str(fake_loss/20.0)+" real: "+str(real_loss/20.0))
                    print("GEN_TOTAL: "+str(summ / 20.0) + " DISC: "+str(d_summ/20.0)+" GEN_ADVERSARIAL:"+str(gl/20.0)+" iteration " + str(i) + "of " + str(
                        loader.total_data // cfg['batch']) + " ,at epoch " + str(e) + " of " + str(cfg['epochs']))
                if (i % 200 == 0):
                    x, gt, f, box, info, img = loader.serve_test()

                    tot = np.concatenate([x[:, -cfg['pred_ext']:], gt], -2)
                    summary, ls, o = sess.run([merge, mod.loss, mod.out],
                                              feed_dict={inpts: x, mod.image: img, outs: tot, mod.target: tot,
                                                         mod.inputs: x, mod.feats: f, mod.box: box,real_imgs:img})
                    for k in range(min(16, len(info))):
                        im = drawer.draw_points(o[k], x[k], gt[k], cfg, info[k])
                        im.save(newp + "/" + str(e) + "_" + str(k) + ".png")  # "#str("_".join(info[i]))+".png")
                if(i%2000==0):
                    saver.save(sess, newp + "/model/model.ckpt")

def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)

def test():
    cfg = get_config()
    inpts = tf.placeholder(tf.float32,[cfg['batch'],cfg['prev_leng'],cfg['dims']])
    outs = tf.placeholder(tf.float32, [cfg['batch'], cfg['pred_ext']+cfg['fut_leng'], cfg['dims']])
    #mod=model.rec_model(cfg)
    mod=model.rec_model(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['GPU'])
    loader = utils.Loader(cfg)
    optimizer = tf.train.AdamOptimizer(cfg['lr'])

    # mini=opti.minimize(mod.loss)
    gvs = optimizer.compute_gradients(mod.loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    mini = optimizer.apply_gradients(capped_gvs)
    init = tf.initializers.global_variables()
    saver = tf.train.Saver()
    newp = str(time.time())[-4:]+"_test_" + cfg["load_path"].split("/")[-3]
    os.mkdir(newp)
    os.mkdir(newp + "/model")
    os.mkdir(newp + "/data")
    total=[]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if (cfg['load'] == "True"):
            saver.restore(sess, cfg['load_path'])
        else:
            "TESTING must have load=true"

        print("OK")
        for i in range(0, 200):
                x, gt, f, box, info, img = loader.serve_test()

                tot = np.concatenate([x[:, -cfg['pred_ext']:], gt], -2)
                o = sess.run(mod.out,
                                          feed_dict={inpts: x, mod.image: img, outs: tot, mod.target: tot, mod.box: box,
                                                     mod.inputs: x, mod.feats: f})
                for k in range(0,len(info)):
                    total.extend(" ".join(str(info[k])))
                    with open(newp + "/data/test" + frame + names[l] + ".json", "w") as out_f:
                        json.dump(df, out_f, sort_keys=True)
                    #im = drawer.draw_points(x[k], o[k], gt[k], cfg, info[k])

        f=open(newp+"/dat","w")
        f.write(str(set(total)))
        f.close()




def scale_up(gts,sx,sy):
    poins=np.asarray(gts)
    poins = poins* (sx / 4.0, sy)
    return np.array(poins, dtype=np.int32)

def get_config():
    with open("config.yaml") as raw_cfg:
        cfg = yaml.load(raw_cfg)
    return cfg

def main():
    cfg = get_config()
    if cfg['type']==2:
        train_GAN()
    else:
        #test()
        train()
    #train_GAN()
main()
