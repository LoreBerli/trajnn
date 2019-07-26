import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as tfn
import utils
import sketcher
import time
import os
from PIL import Image
import io
#from tensorflow.contrib.legacy_seq2seq import basic_rnn_seq2seq



leng=100
EPOCHS=400

def build_paths(lr,enc,lat,activ):
    if(activ==None):
        a="NONE"
    else:
        a=str(activ.__name__)[0:8]

    dr = str(time.time())+"_"+str(lr)+"_"+str(enc)+"_"+str(lat)+"_"+a
    os.mkdir("out/" + dr)
    os.mkdir("out/" + dr + "/imgs")
    os.mkdir("out/" + dr + "/gene")
    return dr


class vae:
    def __init__(self,latent_size=128,batch_size=256,enc_size=64,learn_rate=0.001,eps=16,accel=0.1,activ=None):
        self.batch_size = batch_size
        self.leng = leng
        self.epochs=eps
        self.act=activ
        self.accel=accel
        self.target=tf.placeholder(tf.float32,[self.batch_size,self.leng,5])
        self.category=tf.placeholder(tf.float32,[self.batch_size,7])
        self.input=tf.placeholder(tf.float32,[self.batch_size,self.leng,5])
        self.input_size=tf.placeholder(tf.int32,[self.batch_size])
        self.alpha=tf.placeholder(tf.float32)
        self.latent_size=latent_size
        self.enc_size=enc_size
        self.initia=tf.initializers.glorot_uniform()
        self.dr=build_paths(leng,enc_size,latent_size,activ)
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
        self.cell_enc=tfn.LSTMCell
        self.cell_dec = tfn.LSTMCell
        self.dropout=tf.placeholder(tf.float32)
        self.out_cat=tf.placeholder(tf.float32,[self.batch_size,7])
        self.samples_c=tf.placeholder(tf.float32,[self.batch_size,self.latent_size])
        self.sampled_z_c=tf.placeholder(tf.float32,[self.batch_size,self.latent_size])
        self.latent_h=tf.placeholder(tf.float32,[self.batch_size,self.latent_size])
        self.cr = tf.placeholder(tf.float32, [self.batch_size])
        self.last = self.build_GRU()

        self.lat_loss=self.latent_loss()
        self.rec_loss=self.reconstruction_loss()
        self.class_l=self.class_loss()
        self.cross_l=self.cross_loss()
        l2_loss = tf.losses.get_regularization_loss()
        self.loss=self.alpha*self.lat_loss+self.rec_loss+self.cross_l+self.class_l
        tf.summary.scalar("class_loss",self.class_l)
        tf.summary.scalar("cross_loss",self.cross_l)
        tf.summary.scalar("reconstruction_loss", self.rec_loss)
        tf.summary.scalar("latent_loss",self.lat_loss)
        tf.summary.scalar("total_loss",self.loss)


        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        #self.optimizer=tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        #self.optimizer=tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate)


        #self.optimizer=tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,decay=0.9)
        gvs = self.optimizer.compute_gradients(self.loss)
        gvs = [(grad,var) for grad,var in gvs if grad is not None]
        #capped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gvs]
        for grad, vars in gvs:
            print(vars, vars.name)
            tf.summary.histogram(vars.name, vars)
        self.minimize = self.optimizer.minimize(self.loss)
        #self.minimize=self.optimizer.apply_gradients(capped_gvs)
        self.merged = tf.summary.merge_all()


    def latent_loss(self):
        #lat = 0.5*tf.reduce_sum(tf.square(self.z_mean_c) + tf.square(self.z_std_c) - tf.log(tf.square(self.z_std_c)) - 1, 1)
        lat=-0.5*tf.reduce_sum(1 + self.z_std_c - tf.square(self.z_mean_c) - tf.exp(self.z_std_c), axis=-1)/self.latent_size
        lat = tf.reduce_mean(lat,axis=0)
        return lat
    def reconstruction_loss(self):

            # flat_targ=tf.reshape(self.target[:,:,2:5],shape=[self.batch_size,-1])
            # flat_last=tf.reshape(self.last[:,:,2:5],shape=[self.batch_size,-1])
            mask_t = (tf.ones([self.batch_size,leng])-self.target[:,:,-1]*0.9)

            good=tf.reduce_sum(mask_t,axis=-1)
            sqrd_diff=tf.reduce_sum(tf.square(self.target[:,:,0:2]-self.last[:,:,0:2],name="sqr"),axis=-1,name="reduce_sum")*mask_t
            #sqrd_diff=tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target[:,:,0:2],logits=self.last[:,:,0:2]),axis=-1)
            #sqrd_diff = tf.square(self.target[:, :, 0] - self.last[:, :, 0], name="abs")+tf.square(self.target[:, :, 1] - self.last[:, :, 1], name="abs2")
            print(sqrd_diff)
            sqrd_diff = tf.reduce_sum(sqrd_diff, axis=-1,name="reduce_sum")/good
            print(sqrd_diff)
            # pts=tf.reduce_sum(sqrd_diff,axis=-1,name="fckme")
            # print(pts)
            sqr_loss=tf.reduce_mean(sqrd_diff,axis=0)

            return sqr_loss
            #return tf.losses.mean_squared_error(self.target[:,:,0:2],self.last[:,:,0:2])+0.5*tf.losses.softmax_cross_entropy(self.target[:,:,2:5],self.last[:,:,2:5])

    def cross_loss(self):
        mask2 = (tf.ones([self.batch_size, leng]) + self.target[:, :, -1]*0.9)
        good = tf.reduce_sum(mask2, axis=-1)
        #mask_t = tf.ones([self.batch_size, leng]) - self.target[:, :, -1]
        cross_loss = tf.losses.softmax_cross_entropy(self.target[:, :, 2:], self.last[:, :, 2:], weights=mask2,
                                                     reduction=tf.losses.Reduction.NONE)
        self.cr = mask2
        #cross_loss = cross_loss * mask2

        cross_loss = tf.reduce_sum(cross_loss, axis=-1)/good
        print(cross_loss)
        cross_loss = tf.reduce_mean(cross_loss, axis=0)
        return cross_loss

    def class_loss(self):
        class_loss = tf.losses.softmax_cross_entropy(self.category, self.out_cat, reduction=tf.losses.Reduction.NONE)
        class_loss = tf.reduce_mean(class_loss, axis=-1)
        return  class_loss


    def make_image(self,tensor):
        """
        Convert an numpy representation image to Image protobuf.
        Copied from https://github.com/lanpa/tensorboard-pytorch/
        """
        height, width, channel = tensor.shape

        image = Image.fromarray(tensor)
        import io
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                                width=width,
                                colorspace=channel,
                                encoded_image_string=image_string)



    def build(self):
        with tf.variable_scope("vae_model", reuse=tf.AUTO_REUSE):
            x = self.input
            cat=tf.layers.dense(self.category,16)
            big_cat=tf.tile(tf.expand_dims(self.category,axis=1),(1,self.leng,1))
            print("pre",x)
            #x_t = tf.tile(x, (1, 1, leng))
            x_t=tf.concat([x,big_cat],axis=-1)
            print("post",x_t)

            cell_fw =tfn.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tfn.LSTMCell(self.enc_size,name="fow_cell0",initializer=self.initia),output_keep_prob=self.dropout),tf.nn.rnn_cell.DropoutWrapper(tfn.LSTMCell(self.enc_size,name="fow_cell1",initializer=self.initia),output_keep_prob=self.dropout),tf.nn.rnn_cell.DropoutWrapper(tfn.LSTMCell(self.enc_size,name="fow_cell2",initializer=self.initia,activation=tf.nn.tanh),output_keep_prob=self.dropout)])#(self.enc_size, name="fow_cell",activation=self.act,reuse=tf.AUTO_REUSE,initializer=self.initia,forget_bias=0.9)
            cell_bw =tfn.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tfn.LSTMCell(self.enc_size,name="bow_cell0",initializer=self.initia),output_keep_prob=self.dropout),tf.nn.rnn_cell.DropoutWrapper(tfn.LSTMCell(self.enc_size,name="bow_cell1",initializer=self.initia),output_keep_prob=self.dropout),tf.nn.rnn_cell.DropoutWrapper(tfn.LSTMCell(self.enc_size,name="bow_cell2",initializer=self.initia,activation=tf.nn.tanh),output_keep_prob=self.dropout)])#(self.enc_size, name="fow_cell",activation=self.act,reuse=tf.AUTO_REUSE,initializer=self.initia,forget_bias=0.9)

            #cell_fw =tf.nn.rnn_cell.DropoutWrapper(tfn.LSTMCell(self.enc_size,name="fow_cell0",initializer=self.initia),output_keep_prob=self.dropout)
            #cell_fw =self.cell_enc(self.enc_size, name="baw_cell",activation=self.act)
            #outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw, inputs=x_t, dtype=tf.float32,sequence_length=self.input_size, time_major=False, scope="encoder")
            #outputs,state=tf.nn.dynamic_rnn(cell_fw,inputs=x, dtype=tf.float32,sequence_length=self.input_size, time_major=False, scope="encoder")
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw, inputs=x, dtype=tf.float32, sequence_length=self.input_size,
                                               time_major=False, scope="encoder")


            if(self.cell_enc==tfn.LSTMCell):
                #latent_h=tf.concat([tf.concat([state[1].c,state[2].c],axis=-1)+state[0].c,cat],axis=-1,name="latent_concat")
                ##self.latent_h =tf.layers.dense(tf.concat([state[0].c,state[1].c,state[2].c], axis=-1,name="latent_concat"),self.latent_size)
                self.latent_h = tf.layers.dense(
                    tf.concat([state[0][0].c + state[1][0].c, state[0][1].c + state[1][1].c, state[0][2].c+state[0][2].c], axis=-1, name="latent_concat"),self.latent_size,kernel_initializer=self.initia)

                #self.latent_h =tf.layers.dense(state[0].c+state[1].c+state[2].c,self.latent_size)
                #self.latent_h=tf.layers.dense(state.c,self.latent_size)
                #post=tf.concat([self.latent_h,cat],axis=-1)
                #latent_h=tf.reshape(tf.concat([outputs[0],outputs[1]],axis=-1),shape=[self.batch_size,-1])
                #latent_c = tf.concat([state.c, state.h], axis=-1)
                #print("LATENT",latent_c)
            else:
                self.latent=tf.concat([state[0],state[1]],axis=-1)
                print("ELSE LATENT ",self.latent)




            self.z_mean_c=tf.layers.dense(self.latent_h,self.latent_size,kernel_initializer=self.initia,name="MEAN")
            self.z_std_c=tf.layers.dense(self.latent_h,self.latent_size,kernel_initializer=self.initia,name="STD")

            mu_c=self.z_mean_c
            sigma_c=self.z_std_c
            self.samples_c = tf.random_normal([self.batch_size,self.latent_size], mu_c,sigma_c, dtype=tf.float32)

            self.sampled_z_c = mu_c + tf.exp(sigma_c/2)* self.samples_c
            next_state=tf.concat([self.sampled_z_c,cat],axis=-1)
        with tf.variable_scope("dec", reuse=False):
            print("SAMPLED ",self.sampled_z_c)
            if (self.cell_dec == tfn.LSTMCell):
                latent_state=tfn.LSTMStateTuple(next_state,tf.zeros_like(next_state))#,tfn.LSTMStateTuple(mu,sigma))
            else:
                latent_state=self.sampled_z_c



            res=tf.zeros_like(x)
            second=tfn.OutputProjectionWrapper(tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.latent_size+16,initializer=self.initia),output_keep_prob=self.dropout),5,activation=tf.nn.tanh)

            print("RES",res)
            coord_outs, dec_state = tf.nn.dynamic_rnn(
                #self.cell_dec(self.latent_size, name="dec",initializer=self.initia),
                second,
                #tfn.OutputProjectionWrapper(tfn.MultiRNNCell([self.cell_enc(self.latent_size, name="decc", use_peepholes=True),self.cell_dec(self.latent_size, name="decc2", use_peepholes=True)]),2),
                res,
                initial_state=latent_state,
                sequence_length=self.input_size,
                time_major=False,
                dtype=tf.float32,
                scope='RNN_cord')


            print("OUT : ",coord_outs)
            self.out_cat=tf.layers.dense(tf.layers.flatten(coord_outs),17)
            #flat_out=tf.reshape(coord_outs,[self.batch_size,self.leng*self.latent_size])
            #out=tf.layers.dense(coord_outs, self.leng*5)
            return coord_outs

    def build_GRU(self):
        with tf.variable_scope("vae_model", reuse=tf.AUTO_REUSE):
            x = self.input
            acti=None

            cat=tf.layers.dense(self.category,16)
            big_cat=tf.tile(tf.expand_dims(cat,axis=1),(1,self.leng,1))
            print("pre",x)            #x_t = tf.tile(x, (1, 1, leng))
            x_t=tf.concat([x,big_cat],axis=-1)
            print("post",x_t)

            cell_fw =tfn.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tfn.GRUCell(self.enc_size,name="fow_cell0",activation=acti,kernel_initializer=self.initia),output_keep_prob=self.dropout),tf.nn.rnn_cell.DropoutWrapper(tfn.GRUCell(self.enc_size,name="fow_cell1",activation=acti,kernel_initializer=self.initia),output_keep_prob=self.dropout)])#(self.enc_size, name="fow_cell",activation=self.act,reuse=tf.AUTO_REUSE,initializer=self.initia,forget_bias=0.9)
            cell_bw =tfn.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tfn.GRUCell(self.enc_size,name="bow_cell0",activation=acti,kernel_initializer=self.initia),output_keep_prob=self.dropout),tf.nn.rnn_cell.DropoutWrapper(tfn.GRUCell(self.enc_size,name="bow_cell1",activation=acti,kernel_initializer=self.initia),output_keep_prob=self.dropout)])#(self.enc_size, name="fow_cell",activation=self.act,reuse=tf.AUTO_REUSE,initializer=self.initia,forget_bias=0.9)
            #cell_fw =tf.nn.rnn_cell.DropoutWrapper(tfn.GRUCell(self.enc_size,name="fow_cell0"),output_keep_prob=self.dropout)
            #cell_bw =tf.nn.rnn_cell.DropoutWrapper(tfn.GRUCell(self.enc_size,name="bow_cell0"),output_keep_prob=self.dropout)

            #cell_fw =tf.nn.rnn_cell.DropoutWrapper(tfn.LSTMCell(self.enc_size,name="fow_cell0",initializer=self.initia),output_keep_prob=self.dropout)
            #cell_fw =self.cell_enc(self.enc_size, name="baw_cell",activation=self.act)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw, inputs=x_t, dtype=tf.float32,sequence_length=self.input_size, time_major=False, scope="encoder")
            #outputs,state=tf.nn.dynamic_rnn(cell_fw,inputs=x, dtype=tf.float32,sequence_length=self.input_size, time_major=False, scope="encoder")
            #outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw, inputs=x_t, dtype=tf.float32, sequence_length=self.input_size,time_major=False, scope="encoder")

            self.latent_h=tf.layers.dense(tf.concat([state[0][0] , state[1][0], state[0][1] ,state[1][1],cat],axis=-1, name="latent_concat"), self.latent_size,kernel_initializer=self.initia)





            self.z_mean_c=tf.layers.dense(self.latent_h,self.latent_size,kernel_initializer=self.initia,name="MEAN")
            self.z_std_c=tf.layers.dense(self.latent_h,self.latent_size,kernel_initializer=self.initia,name="STD")
            mu_c=self.z_mean_c
            sigma_c=self.z_std_c
            self.samples_c = tf.random_normal([self.batch_size,self.latent_size], 0.0,1.0, dtype=tf.float32)

            self.sampled_z_c = mu_c + tf.exp(sigma_c/2)* self.samples_c
            self.sampled_z_c=tf.nn.tanh(self.sampled_z_c)
            #self.sampled_z_c = mu_c + sigma_c * self.samples_c
            next_state=tf.concat([self.sampled_z_c,cat],axis=-1)
            latent_state = tf.layers.dense(next_state,512)
        with tf.variable_scope("dec", reuse=False):
            print("SAMPLED ",self.sampled_z_c)


            res=tf.zeros_like(x)
            second=tfn.OutputProjectionWrapper(tf.nn.rnn_cell.DropoutWrapper(tfn.GRUCell(512,kernel_initializer=self.initia),output_keep_prob=self.dropout),2,activation=tf.nn.tanh)

            print("RES",res)
            coord_outs, dec_state = tf.nn.dynamic_rnn(
                second,
                res,
                initial_state=latent_state,
                sequence_length=self.input_size,
                time_major=False,
                dtype=tf.float32,
                scope='RNN_cord')

            print("x",latent_state)
            state_outs, _ = tf.nn.dynamic_rnn(
                #self.cell_dec(self.latent_size, name="dec",initializer=self.initia),
                tfn.OutputProjectionWrapper(tf.nn.rnn_cell.DropoutWrapper(tfn.GRUCell(512 ,kernel_initializer=self.initia), output_keep_prob=self.dropout),3),
                coord_outs,
                initial_state=latent_state,
                sequence_length=self.input_size,
                time_major=False,
                dtype=tf.float32,
                scope='RNN_stat')


            print("OUT : ",coord_outs)
            print("OUT : ", state_outs)
            coord_outs=tf.concat([coord_outs,state_outs],axis=-1)
            self.out_cat=tf.layers.dense(tf.layers.flatten(coord_outs),7)
            #flat_out=tf.reshape(coord_outs,[self.batch_size,self.leng*self.latent_size])
            #out=tf.layers.dense(coord_outs, self.leng*5)
            return coord_outs




    def train(self,sess):
        saver = tf.train.Saver()
        saver.restore(sess, "./out/1548108293.8154037_100_256_128_NONE/model.ckpt")

        tiled_leng=np.tile(self.leng,self.batch_size)
        self.writer = tf.summary.FileWriter("./test/"+str(self.dr), sess.graph)
        alpha=0.1
        al=alpha
        lrs=[0.14]
        iters=int((2*70000)/self.batch_size)
        m_iter=iters
        train_data, test_data=utils.get_train_test_gens(self.batch_size,leng)
        show_test = next(test_data)
        x_show, cat_x_show = show_test
        means = np.random.normal(0, 1.0, [self.batch_size, self.latent_size])
        y_show = x_show
        lr=0.00001
        ot = sketcher.save_batch_diff_z_axis(np.zeros_like(y_show[0:64]), y_show[0:64], "out/" + str(self.dr) + "/imgs",str(0) + "_" + str(0))
        # gen = sketcher.save_batch_diff_z_axis(np.zeros_like(y_show[0:64]), np.zeros_like(y_show[0:64]), "out/" + str(self.dr) + "/gene",
        #                                       str(0) + "_" + str(0) + "generated")
        #
        # ot_t=tf.placeholder(dtype=tf.uint8)
        # gen_t = tf.placeholder(dtype=tf.uint8)
        # self.writer.add_summary(tf.summary.image("ot", ot_t)).eval()
        # self.writer.add_summary(tf.summary.image("gent", gen_t)).eval()


        for e in range(0,self.epochs):

            # if(e%2==0):
            #     al=min(1.0,alpha)
            #     alpha = al* self.accel

            if(e%5==2):
                al=al+0.05
            for i in range(0,iters):
                both= next(train_data)
                x_,cat_x =both
                y_=x_

                ls,_=sess.run([self.loss,self.minimize],feed_dict={self.dropout:0.7,self.input:x_,self.target:y_,self.category:cat_x,self.input_size:tiled_leng,self.alpha:al,self.learning_rate:lr})



                if(i%1000==0):
                    saver.save(sess,"out/" + str(self.dr)+"/model.ckpt")


                if (i % 50 == 0):

                    both_test = next(test_data)
                    x_test, cat_x_test = both_test
                    y_test = x_test
                    #print("LOSS:", ls,"at iter:",str(i),"of ",str(m_iter), " EPOCH:", str(e), " ALPHA:", str(al),"LR: ",str(lr))
                    summary,cr,tst_ls=sess.run([self.merged,self.cr,self.loss],
                             feed_dict={self.dropout:1.0,self.input: x_test, self.category:cat_x_test,self.target: y_test, self.input_size: tiled_leng,self.alpha:al,self.learning_rate:lr})
                    self.writer.add_summary(summary, i+(e*m_iter))
                    print("TEST_LOSS:",tst_ls ," LOSS:", ls, "at iter:", str(i), "of ", str(m_iter), " EPOCH:", str(e), " ALPHA:", str(al),
                          "LR: ", str(lr))
                if(i%200==0):
                    self.tot,s= sess.run([self.last,self.latent_h], feed_dict={self.dropout:1.0,self.input: x_show,self.category:cat_x_show, self.input_size: tiled_leng,self.alpha:al,self.learning_rate:lr})
                    self.test = sess.run(self.last, feed_dict={self.dropout: 1.0, self.sampled_z_c: means,
                                                               self.category: cat_x_show, self.input_size: tiled_leng,
                                                               self.alpha: al, self.learning_rate: lr})
                    for f in range(4):
                        ot = sketcher.save_batch_diff_z_axis(self.tot[f*16:(f+1)*16],  y_show[f*16:(f+1)*16],"out/" + str(self.dr) + "/imgs",str(e)+"_"+str(i)+"_"+str(f))

                        gen=sketcher.save_batch_diff_z_axis(self.test[f*16:(f+1)*16], self.test[f*16:(f+1)*16], "out/" + str(self.dr) + "/gene",
                                                             str(e) + "_" + str(i)+"generated"+"_"+str(f))
                        self.tensor_image = self.make_image(np.asarray(ot))
                        self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='test_'+str(f), image=self.tensor_image)]))
                        self.tensor_image = self.make_image(np.asarray(gen))
                        self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='gen_'+str(f), image=self.tensor_image)]))


                #tot = sketcher.save_batch_diff_z_axis(list((1 + tt[0:16]) * 128), list((1 + y_draw[0:16]) * 128),"out/" + str(self.dr) + "/imgs", str(e))





class mlp_ae:
    def __init__(self,latent,BATCH):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.BATCH=BATCH
        self.latent=latent
        self.dr = build_paths()
        self.input = tf.placeholder(dtype=tf.float32, shape=[BATCH, leng* 5])
        self.output=self.build()
        self.loss=self.ae_loss(self.output,self.input)
        self.min=self.optimizer.minimize(self.loss)
        #?????


    def build(self):
        #crude autoencoder
        x=self.input
        input_size=leng*5
        acti=tf.nn.relu
        print(x)
        x = tf.layers.dense(x,512,activation=acti)

        x = tf.layers.dense(x,128,activation=tf.nn.sigmoid)
        middle = tf.layers.dense(x,self.latent, activation=acti)
        x = tf.layers.dense(middle,128,activation=acti)
        x = tf.layers.dense(x,input_size,activation=tf.nn.tanh)

        #NONONONONONONO
        self.last= x
        self.center,self.std=tf.nn.moments(middle,axes=[0,1])


        self.sampler=middle
        return x

    def sample(self,sess,md,std):

        seeds = np.random.normal(md,std,[BATCH,self.latent])
        out = sess.run(self.last, feed_dict={self.sampler: seeds})
        tt = np.reshape(out, [BATCH, leng, 5])
        tot = sketcher.save_batch_diff_z_axis(list((1 + tt[0:16]) * 128), list((1 + tt[0:16]) * 128),
                                              "out/" + str(self.dr) + "/imgs", "SAMPLING")

    def ae_loss(self,y_pred,y_true):
        return tf.losses.mean_squared_error(y_true,y_pred)

    def train(self,sess):


        for e in range(0,EPOCHS):
            g = utils.get_coord_drawings_z_axis(BATCH, leng)
            for i in range(0,1000):
                x_,y_ = next(g)
                x_=np.reshape(x_,[BATCH,-1])
                y_=np.reshape(y_,[BATCH,-1])
                ls=sess.run([self.loss,self.min],feed_dict={self.input:x_})

            out,m,avg,std=sess.run([self.output,self.sampler,self.center,self.std],feed_dict={self.input:x_})

            print(avg,std)
            tt=np.reshape(out,[BATCH,leng,5])
            y_draw=np.reshape(y_,[BATCH,leng,5])
            self.sample(sess,avg,std)
            tot = sketcher.save_batch_diff_z_axis(list((1 + tt[0:16]) * 128), list((1 + y_draw[0:16]) * 128),"out/" + str(self.dr) + "/imgs", str(e))
