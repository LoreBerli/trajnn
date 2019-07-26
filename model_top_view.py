import tensorflow as tf
import tensorflow.contrib.rnn as tfn
import numpy as np
import spatial_transformer

tf.nn.seq2seq = tf.contrib.legacy_seq2seq


class rec_model():
    def __init__(self, config):
        self.dims = config['dims']
        self.state_size = config['units']
        #self.debug_points=tf.placeholder(tf.float32, [self.config['batch'], self.config['pred_ext'] + self.config['fut_leng'], 2])
        self.latent_size = config['lat_size']
        self.inputs = tf.placeholder(tf.float32, shape=[config['batch'], config['prev_leng'], self.dims])
        self.target = tf.placeholder(tf.float32,
                                     shape=[config['batch'], config['pred_ext'] + config['fut_leng'], self.dims])
        self.crops=tf.placeholder(tf.float32,
                                     shape=[config['batch'],config['pred_ext'] + config['fut_leng'], 16,16,4])
        self.clas = tf.placeholder(tf.float32, shape=[config['batch']])
        #self.dirs_loss = tf.placeholder(tf.float32, shape=[config['batch']])
        self.leng_loss = tf.placeholder(tf.float32, shape=[config['batch']])
        self.noiz = tf.placeholder(tf.float32, shape=[config['batch'], 8])
        self.image = tf.placeholder(tf.float32,
                                    shape=[config['batch'], config['dim_clip'] * 2, config['dim_clip'] * 2, 4])
        self.config = config
        self.factor = tf.placeholder(tf.float32, shape=[1])
        self.drop = tf.placeholder(tf.float32)
        self.enc = self.enc_cell(self.state_size)
        self.dec = self.dec_cell(self.state_size)
        self.enc_i=tf.placeholder(tf.float32, shape=[config['batch'], config['prev_leng'], self.dims])

        if config['type'] == 0:
            self.out = self.rnn_alone()
            self.loss = self.m_loss()
        elif config['type'] == 3:
            self.out = self.linear()
            self.loss = self.m_loss()
        elif config['type'] == 5:
            self.out = self.rnn_with_ctx_multiple()
            self.loss = self.multiple_loss()
        elif config['old']:
            self.out = self.build_recurrent()
            self.loss = self.m_loss()
        else:
            self.out = self.rnn_with_ctx_test()
            self.loss = self.m_loss()



    def enc_cell(self, num):
        # return tfn.MultiRNNCell([tfn.GRUCell(num,name="enc_cell"+str(i)) for i in range(self.config['num'])])
        return tfn.DropoutWrapper(tfn.GRUCell(num, name="enc_cell"), state_keep_prob=self.drop)

    def dec_cell(self, num):
        # return tfn.MultiRNNCell([tfn.GRUCell(num,name="dec_cell"+str(i)) for i in range(self.config['num'])])
        return tfn.DropoutWrapper(tfn.GRUCell(num, name="dec_cell"), state_keep_prob=self.drop)

    def rnn_with_ctx(self):
        with tf.variable_scope("GEN"):
            if (self.image.shape > 3):
                img = self.image
            else:
                img = tf.expand_dims(self.image, -1)

            # img = tf.layers.conv2d(img, 16, [3,3], padding='same', dilation_rate=3, activation=tf.nn.leaky_relu)
            # img = tf.layers.max_pooling2d(img, [2, 2], strides=[2, 2], padding='same')
            # img = tf.layers.conv2d(img, 32, [3, 3], padding='same', dilation_rate=3, activation=tf.nn.leaky_relu)
            # img = tf.layers.max_pooling2d(img, [2, 2], strides=[2, 2], padding='same')
            # img = tf.layers.conv2d(img, 64, [3, 3], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            # img = tf.layers.conv2d(img, 128, [3, 3], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            img = tf.layers.conv2d(img, 16, [5, 5], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            img = tf.layers.conv2d(img, 32, [5, 5], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            #print(img)
            img = tf.layers.flatten(img)
            img_embed = tf.layers.dense(img, 256)
            img_embed = tf.layers.dropout(img_embed, rate=self.drop)
            # img_embed=tf.contrib.layers.layer_norm(img_embed)
            if (self.config['past_img_inputs']):
                mini_embed = tf.layers.dense(img_embed, 8, activation=tf.nn.tanh)
                past_inputs = tf.concat(
                    [self.inputs, tf.tile(tf.expand_dims(mini_embed, 1), [1, self.config['prev_leng'], 1])], -1)
            else:
                past_inputs = self.inputs

            e_outs, e_state = tf.nn.dynamic_rnn(self.enc, past_inputs, dtype=tf.float32, scope="ENC")



            if self.config['fut_img_inputs']:
                img_embed_ = tf.layers.dense(img_embed, self.config["img_embed"])
                seq_embed = tf.layers.dense(e_state, self.latent_size - self.config["img_embed"])
                embed = tf.concat([img_embed_, seq_embed], -1)
            else:
                seq_embed = tf.layers.dense(e_state, self.latent_size)
                embed = seq_embed
            ############################
            #NOISE
            embed=tf.concat([embed,self.noiz],-1)
            ############################
            if (self.config['vae']):
                self.z_mean_c = tf.layers.dense(self.noiz, self.latent_size, name="MEAN")
                self.z_std_c = tf.layers.dense(self.noiz, self.latent_size, name="STD")

                mu_c = self.z_mean_c
                sigma_c = self.z_std_c

                self.samples_c = tf.random_normal([self.config['batch'], self.latent_size], 0.0, 1.0, dtype=tf.float32)

                self.sampled_z_c = mu_c + self.factor[0] *tf.exp(sigma_c) * self.samples_c
                #embed = self.sampled_z_c
                tf.concat([embed, self.sampled_z_c], -1)
            ############################
            Xin = tf.layers.dense(embed, self.state_size)
            #e_ts = tf.layers.dense(embed, (self.config['fut_leng'] + self.config['pred_ext']) * (self.state_size))
            e_ts=tf.tile(embed,[1,self.config['fut_leng'] + self.config['pred_ext'],1])
            e_ts = tf.reshape(e_ts, [self.config['batch'], self.config['fut_leng'] + self.config['pred_ext'],
                                     self.state_size])

            with tf.variable_scope("DEC"):
                if (self.config['autoregressive']):
                    # d_outs, d_state = self.self_feeding_rnn(tfn.GRUCell(self.state_size),self.config['fut_leng'] + self.config['pred_ext'], e_ts, Xin)
                    d_outs, d_state = self.self_feeding_rnn(tfn.GRUCell(self.state_size),
                                                            self.config['fut_leng'] + self.config['pred_ext']
                                                            , e_ts, Xin)
                else:
                    d_outs, d_state = tf.nn.dynamic_rnn(tfn.GRUCell(self.state_size), e_ts, initial_state=Xin,
                                                        dtype=tf.float32, scope="DEC")
                    d_outs = tf.layers.dense(d_outs, 2)

                print("d_outs", d_outs)

            bias = self.inputs[:, -1, :]
            bias = tf.expand_dims(bias, 1)
            my_b = d_outs[:, 0, :]
            my_b = tf.expand_dims(my_b, 1)
            diff = my_b - bias
            d_outs = d_outs - diff
            return d_outs

    def rnn_with_ctx_multiple(self):
        with tf.variable_scope("GEN"):

            img = self.image
            # img = tf.layers.conv2d(img, 16, [3,3], padding='same', dilation_rate=3, activation=tf.nn.leaky_relu)
            # img = tf.layers.max_pooling2d(img, [2, 2], strides=[2, 2], padding='same')
            # img = tf.layers.conv2d(img, 32, [3, 3], padding='same', dilation_rate=3, activation=tf.nn.leaky_relu)
            # img = tf.layers.max_pooling2d(img, [2, 2], strides=[2, 2], padding='same')
            # img = tf.layers.conv2d(img, 64, [3, 3], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            # img = tf.layers.conv2d(img, 128, [3, 3], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            # img = tf.layers.conv2d(img, 8, [5, 5], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            #img = tf.layers.conv2d(img, 8, [5, 5], strides=[1,1], padding='same', activation=tf.nn.leaky_relu)
            #img=tf.layers.max_pooling2d(img,[2,2],[2,2])
            print("IMG_SHAPE",img.shape)
            map_in = img
            #print(img)
            img = tf.layers.flatten(img)
            img_embed = tf.layers.dense(img, 256)
            img_embed = tf.layers.dropout(img_embed, rate=self.drop)
            self.embed = img_embed
            # img_embed=tf.contrib.layers.layer_norm(img_embed)
            if (self.config['past_img_inputs']):
                mini_embed = tf.layers.dense(img_embed, 8, activation=tf.nn.tanh)
                past_inputs = tf.concat(
                    [self.inputs, tf.tile(tf.expand_dims(mini_embed, 1), [1, self.config['prev_leng'], 1])], -1)
            else:
                #past_inputs = tf.reverse(self.inputs,[-2])
                past_inputs = self.inputs

            e_outs, e_state = tf.nn.dynamic_rnn(self.enc, past_inputs, dtype=tf.float32, scope="ENC")

            if self.config['fut_img_inputs']:
                img_embed_ = tf.layers.dense(img_embed, self.config["img_embed"])
                seq_embed = tf.layers.dense(e_state, self.latent_size - self.config["img_embed"])
                embed = tf.concat([img_embed_, seq_embed], -1)
            else:
                seq_embed = tf.layers.dense(e_state, self.latent_size)
                embed = seq_embed
            ############################
            embed_f = tf.concat([embed, self.noiz], -1)
            ############################
            ############################

            with tf.variable_scope("MULTIPLE",reuse=tf.AUTO_REUSE):
                embs = []
                outs = []
                for k in range(0, 5):
                    if (self.config['vae']):
                        self.z_mean_c = tf.layers.dense(embed_f, self.latent_size, name="MEAN")
                        self.z_std_c = tf.layers.dense(embed_f, self.latent_size, name="STD")

                        mu_c = self.z_mean_c
                        sigma_c = self.z_std_c

                        self.samples_c = tf.random_normal([self.config['batch'], self.latent_size], 0.0, 1.0, dtype=tf.float32,name="rnd")
                        #print(self.samples_c.shape)
                        self.sampled_z_c = mu_c + tf.exp(self.factor[0] * sigma_c) * self.samples_c
                        embed = self.sampled_z_c



                ############################
                    Xin = tf.layers.dense(embed, 32,name="XIN")
                    BXIN= tf.layers.dense(embed, self.state_size,name="BXIN")
                    e_ts = tf.layers.dense(embed, (self.config['fut_leng'] + self.config['pred_ext']) * (32),name="e_TS")
                    e_ts = tf.reshape(e_ts, [self.config['batch'], self.config['fut_leng'] + self.config['pred_ext'],
                                             32],name="e_TS_reshaped")
                    self.enc_i,sts = self.self_feeding_rnn(tfn.GRUCell(32,name="first2"),
                                                                        self.config['prev_leng'], e_ts, Xin)
                    with tf.variable_scope("DEC",reuse=tf.AUTO_REUSE):

                            if (self.config['autoregressive']):
                                # d_ou, d_state = self.self_feeding_rnn(tfn.GRUCell(32,name="first"),
                                #                                         self.config['fut_leng'] + self.config['pred_ext']
                                #                                         , e_ts, Xin)
                                d_ou, d_state = tf.nn.dynamic_rnn(tfn.GRUCell(self.state_size), e_ts,
                                                                    initial_state=Xin, dtype=tf.float32, scope="DEC")
                                d_ou = tf.layers.dense(d_ou, 2)
                                d_spec = d_ou
                                self.d_spec=tf.cumsum(d_spec, -2,name="cumsum")
                                inps=self.d_spec

                                for iter in range(0,1):
                                    d_outs, BXIN= self.small_map_self_feeding_rnn(tfn.GRUCell(self.state_size,name="second"),self.config['fut_leng'] + self.config['pred_ext'],inps, BXIN, map_in)
                                    inps=d_outs
                            else:
                                d_outs, d_state = tf.nn.dynamic_rnn(tfn.GRUCell(self.state_size), e_ts, initial_state=Xin,dtype=tf.float32, scope="DEC")
                                d_outs = tf.layers.dense(d_outs, 2)
                                self.d_spec = tf.cumsum(d_outs, -2, name="cumsum")
                                print("d_outs", d_outs)

                            d_final=inps
                            outs.append(d_final)

                return outs

    def rnn_with_ctx_test(self):
        with tf.variable_scope("GEN"):
            # if (self.image.shape > 3):
            #     img = self.image
            # else:
            #     img = tf.expand_dims(self.image, -1)
            img = self.image
            # img = tf.layers.conv2d(img, 16, [3,3], padding='same', dilation_rate=3, activation=tf.nn.leaky_relu)
            # img = tf.layers.max_pooling2d(img, [2, 2], strides=[2, 2], padding='same')
            # img = tf.layers.conv2d(img, 32, [3, 3], padding='same', dilation_rate=3, activation=tf.nn.leaky_relu)
            # img = tf.layers.max_pooling2d(img, [2, 2], strides=[2, 2], padding='same')
            # img = tf.layers.conv2d(img, 64, [3, 3], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            # img = tf.layers.conv2d(img, 128, [3, 3], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            # img = tf.layers.conv2d(img, 8, [5, 5], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            # img = tf.layers.conv2d(img, 16, [5, 5], strides=[1,1], padding='same', activation=tf.nn.leaky_relu)
            img=tf.layers.max_pooling2d(img,[2,2],[2,2])
            print("IMG_SHAPE",img.shape)
            map_in = img

            img = tf.layers.flatten(img)
            img_embed = tf.layers.dense(img, 256)
            img_embed = tf.layers.dropout(img_embed, rate=self.drop)
            self.embed = img_embed
            # img_embed=tf.contrib.layers.layer_norm(img_embed)
            if (self.config['past_img_inputs']):
                mini_embed = tf.layers.dense(img_embed, 8, activation=tf.nn.tanh)
                past_inputs = tf.concat(
                    [self.inputs, tf.tile(tf.expand_dims(mini_embed, 1), [1, self.config['prev_leng'], 1])], -1)
            else:
                past_inputs = self.inputs

            e_outs, e_state = tf.nn.dynamic_rnn(self.enc, past_inputs, dtype=tf.float32, scope="ENC")

            # e_state = tf.concat([e_state[-1], e_state[-2]], -1)

            if self.config['fut_img_inputs']:
                img_embed_ = tf.layers.dense(img_embed, self.config["img_embed"])
                seq_embed = tf.layers.dense(e_state, self.latent_size - self.config["img_embed"])
                embed = tf.concat([img_embed_, seq_embed], -1)
            else:
                seq_embed = tf.layers.dense(e_state, self.latent_size)
                embed = seq_embed
            ############################
            embed = tf.concat([embed, self.noiz], -1)
            ############################
            ############################
            if (self.config['vae']):
                self.z_mean_c = tf.layers.dense(embed, self.latent_size, name="MEAN")
                self.z_std_c = tf.layers.dense(embed, self.latent_size, name="STD")

                mu_c = self.z_mean_c
                sigma_c = self.z_std_c

                self.samples_c = tf.random_normal([self.config['batch'], self.latent_size], 0.0, 1.0, dtype=tf.float32)

                self.sampled_z_c = mu_c + tf.exp(self.factor[0] * sigma_c) * self.samples_c
                embed = self.sampled_z_c

            ############################
            Xin = tf.layers.dense(embed, self.state_size)
            e_ts=embed

            e_ts=tf.expand_dims(embed,1)

            #e_ts = tf.tile(e_ts, [1, self.config['fut_leng'] + self.config['pred_ext'], 1])
            e_ts = tf.layers.dense(embed, (self.config['fut_leng'] + self.config['pred_ext']) * (self.state_size))

            e_ts = tf.reshape(e_ts, [self.config['batch'], self.config['fut_leng'] + self.config['pred_ext'],
                                     self.state_size])

            with tf.variable_scope("DEC"):
                if (self.config['autoregressive']):
                    d_ou, d_state = self.self_feeding_rnn(tfn.GRUCell(self.state_size),
                                                            self.config['fut_leng'] + self.config['pred_ext']
                                                            , e_ts, Xin)
                    # d_ou, d_state = self.small_map_self_feeding_rnn(tfn.GRUCell(self.state_size),
                    #                                                   self.config['fut_leng'] + self.config['pred_ext'],
                    #                                                   e_ts,Xin, map_in)
                    # bias = self.inputs[:, -1, :]
                    # bias = tf.expand_dims(bias, 1)
                    # my_b = d_ou[:, 0, :]
                    # my_b = tf.expand_dims(my_b, 1)
                    # diff = my_b - bias
                    # d_ou = d_ou - diff
                    # print("DOU,",d_ou)
                    # print("D_ST",d_state)
                    #d_outs, d_state = self.small_map_self_feeding_rnn(tfn.GRUCell(self.state_size),self.config['fut_leng'] + self.config['pred_ext'], e_ts, Xin,map_in)
                    d_spec=d_ou#tf.cumsum(d_ou,-2)

                    # emb = tf.layers.dense(d_st, 32)
                    d_outs, d_state = self.small_map_self_feeding_rnn(tfn.GRUCell(self.state_size,name="last_cell"),self.config['fut_leng'] + self.config['pred_ext'],self.d_spec, tf.zeros([8,2]), map_in)
                else:
                    d_outs, d_state = tf.nn.dynamic_rnn(tfn.GRUCell(self.state_size), e_ts, initial_state=Xin,dtype=tf.float32, scope="DEC")
                    d_outs = tf.layers.dense(d_outs, 2)

                print("d_outs", d_outs)
            d_spec=self.d_spec+d_outs
            d_spec=tf.cumsum(d_spec, -2)
            # bias = self.inputs[:, -1, :]
            # bias = tf.expand_dims(bias, 1)
            # my_b = d_spec[:, 0, :]
            # my_b = tf.expand_dims(my_b, 1)
            # diff = my_b - bias
            # d_outs = d_outs - diff

            return d_spec

    def self_feeding_rnn(self, cell, seqlen, Hin, Xin, processing=tf.identity):
        with tf.variable_scope("rnn_points",reuse=tf.AUTO_REUSE):
            buffer = tf.TensorArray(dtype=tf.float32, size=seqlen,name="buff")

            inputs = tf.transpose(Hin, [1, 0, 2],name="inp_transposed")

            inputs_ta = tf.TensorArray(dtype=tf.float32, size=seqlen, clear_after_read=False,name="inputs_ta")
            inputs_ta = inputs_ta.unstack(inputs,name="inpt_unstacl")

            in_first = inputs[0]
            initial_state = (0, Xin, Xin, buffer, inputs_ta)
            condition = lambda i, *_: i < seqlen

            def do_time_step(i, state, xo, ta, inp_ta):
                st = inp_ta.read(i)
                s1 = st
                #xd = tf.concat([s1, xo], -1,name="xd_concat")
                #xd = tf.layers.dense(xd, self.state_size,name="self_xd")
                wut = cell(s1, state)
                Yt, Ht = wut
                next = Ht
                Yro = tf.layers.dense(Yt, 2,name="out_rnn")
                return (1 + i, next, Yt, ta.write(i, Yro), inp_ta)

            _, Hout, yout, final_stack, _ = tf.while_loop(condition, do_time_step, initial_state)

            ta_stack = final_stack.stack()

            # Yo=ta_stack
            Yo = tf.transpose(ta_stack, perm=[1, 0,
                                              2],name="out_tran")  # tf.reshape(ta_stack, shape=((self.config['batch'],seqlen, self.config['dims'])))
            return Yo, Hout

    def map_self_feeding_rnn(self, cell, seqlen, Hin, Xin, MapIn, processing=tf.identity):
        with tf.variable_scope("map_rnn",reuse=tf.AUTO_REUSE):
            buffer = tf.TensorArray(dtype=tf.float32, size=seqlen,name="map_buf")
            crops = tf.TensorArray(dtype=tf.float32, size=seqlen,name="map_crops")
            #zer = tf.ones([8, 2])*80.0
            zer = tf.layers.dense(Xin,2,name="map_zer")
            inputs = tf.transpose(Hin, [1, 0, 2],name="map_transpo")

            inputs_ta = tf.TensorArray(dtype=tf.float32, size=seqlen, clear_after_read=False,name="map_inpts")
            inputs_ta = inputs_ta.unstack(inputs)

            in_first = inputs[0]
            initial_state = (0, in_first, Xin, buffer, inputs_ta, zer, crops)
            condition = lambda i, *_: i < seqlen
            #print("MAPIN", MapIn.shape)

            def do_time_step(i, state, xo, ta, inp_ta, cords, crps):
                st = inp_ta.read(i)
                st = tf.concat([st, xo, cords], -1)
                # old_cords=cords
                cord = tf.layers.dense(st, 2,name="map_cords")
                crop = tf.image.extract_glimpse(MapIn, tf.constant([32,32]), cord, centered=False, normalized=True,name="map_glimps")
                crop = tf.layers.conv2d(crop, 1, [3, 3], padding='same', activation=tf.nn.leaky_relu,name="map_conv")

                cropt = tf.layers.dense(tf.layers.flatten(crop), 32, activation=tf.nn.leaky_relu,name="map_cropt")
                #print("CROP", crop)
                xd = tf.concat([st, cropt], -1)
                # xd = tf.layers.dense(xd, 32)
                xd = tf.layers.dense(xd, self.state_size,name="map_xd")
                # print("xo", xd)
                # print("state", state)

                wut = cell(xd, state)
                # print(wut)
                Yt, Ht = wut
                next = Ht
                #
                # print("YT", Yt)
                # print("HT", Ht)
                # Ht=tf.layers.dense(tf.concat([Ht,initi],-1),initi.shape[-1])

                # Yro=tf.concat([x_real,Yt],-1)
                Yro = tf.layers.dense(Yt, 2,name="map_outs")
                # print("yro", Yro)

                return (1 + i, next, Yt, ta.write(i, Yro), inp_ta, Yro, crps.write(i, crop))

            _, Hout, yout, final_stack, _, _, crp = tf.while_loop(condition, do_time_step, initial_state)

            self.crops = crp.stack()
            self.crops = tf.transpose(self.crops, [1, 0, 2, 3, 4])
            ta_stack = final_stack.stack()

            # Yo=ta_stack
            Yo = tf.transpose(ta_stack, perm=[1, 0,
                                              2])  # tf.reshape(ta_stack, shape=((self.config['batch'],seqlen, self.config['dims'])))


            return Yo, Hout

    def small_map_self_feeding_rnn(self, cell, seqlen, Hin, Xin, MapIn, processing=tf.identity):
        with tf.variable_scope("map_rnn", reuse=tf.AUTO_REUSE):
            buffer = tf.TensorArray(dtype=tf.float32, size=seqlen,name="bufff")
            crops=tf.TensorArray(dtype=tf.float32, size=seqlen,name="crops")

            inputs = tf.transpose(Hin, [1, 0, 2])

            inputs_ta = tf.TensorArray(dtype=tf.float32, size=seqlen, clear_after_read=False,name="inta")
            inputs_ta = inputs_ta.unstack(inputs)

            in_first = inputs[0]
            zer = inputs[0]

            initial_state = (0, Xin, Xin, buffer, inputs_ta,zer,crops)
            condition = lambda i, *_: i < seqlen
            #print("MAPIN",MapIn.shape)

            def do_time_step(i, state, xo, ta, inp_ta,cords,crps):
                st = inp_ta.read(i)
                cord = st*2.0#tf.layers.dense(cords,2,activation=tf.nn.tanh)
                print("CORD",cord)

                cord=tf.reverse(cord,[-1])
                crop = tf.image.extract_glimpse(MapIn, tf.constant([10,10]), cord,centered=True,normalized=False)
                crop = tf.layers.conv2d(crop,4, [3,3],strides=[2,2], padding='same',name="conv")
                cropt = tf.layers.dense(tf.layers.flatten(crop), 32,name="denscrop")

                xd = tf.concat([xo,st,cropt], -1)
                #xd = tf.layers.dense(xd, self.state_size,name="tostatesize")
                #print("xo", xd)
                #print("state", state)

                wut = cell(xd, state)
                #print(wut)
                Yt, Ht = wut
                next = Ht

                Yro = st+tf.layers.dense(Yt, 2,name="out")

                #print("yro", Yro)

                return (1 + i, next, Yt, ta.write(i, Yro), inp_ta,Yro,crps.write(i,crop))

            _, Hout, yout, final_stack, _,_ ,crp= tf.while_loop(condition, do_time_step, initial_state)

            self.crops=crp.stack()
            self.crops=tf.transpose(self.crops,[1,0,2,3,4])
            ta_stack = final_stack.stack()

            # Yo=ta_stack
            Yo = tf.transpose(ta_stack, perm=[1, 0,
                                              2])  # tf.reshape(ta_stack, shape=((self.config['batch'],seqlen, self.config['dims'])))

            return Yo, Hout

    def rnn_alone(self):
        # outputs,state=tf.nn.dynamic_rnn(cell_fw,inputs=x, dtype=tf.float32,sequence_length=self.input_size, time_major=False, scope="encoder")
        e_outs, e_state = tf.nn.dynamic_rnn(self.enc, self.inputs, dtype=tf.float32, scope="ENC")

        e_state = e_state[-1] + e_state[-2]
        e_state = tf.layers.flatten(e_state)

        latent = tf.layers.dense(e_state, self.latent_size)
        # latent=tf.concat([latent,ft],-1)
        # latent=tf.concat([tf.layers.dense(latent,self.state_size/2),ft],-1)

        latent = tf.layers.dense(latent, self.state_size)
        e_ts = latent
        # e_ts=latent_s
        Hin = (e_ts, e_ts)
        Xin = tf.layers.dense(latent, self.latent_size)

        # latent_s = tf.layers.dense(latent, self.latent_size, activation=tf.nn.leaky_relu)
        #
        e_ts = tf.layers.dense(latent, (self.config['fut_leng'] + self.config['pred_ext']) * self.state_size)
        e_ts = tf.reshape(e_ts,
                          [self.config['batch'], self.config['fut_leng'] + self.config['pred_ext'], self.state_size])
        with tf.variable_scope("DEC"):
            # print("SELF")
            # d_outs, d_state = tf.nn.dynamic_rnn(tfn.GRUCell(self.state_size), e_ts, initial_state=latent_s,
            #                                     dtype=tf.float32, scope="DEC")
            d_outs, d_state = self.self_feeding_rnn(tfn.GRUCell(self.state_size),
                                                    self.config['fut_leng'] + self.config['pred_ext']
                                                    , e_ts, latent)
            # d_outs = self.better_self_ffed(tfn.GRUCell(self.state_size),
            #                                self.config['fut_leng'] + self.config['pred_ext'],
            #                                self.config['batch'], self.state_size, e_ts)
            print("d_outs", d_outs)
            bias = self.inputs[:, -self.config['pred_ext']:-self.config['pred_ext'] + 1, :]
            my_b = d_outs[:, :1, :]
            diff = my_b - bias
            d_outs = d_outs - diff

        return d_outs

    def discrim(self, trjs, targets, ims, reuse=False):
        with tf.variable_scope("DIM", reuse=reuse):
            img = ims
            # img = tf.expand_dims(ims, -1)
            img = tf.layers.conv2d(img, 16, [3, 3], strides=[2, 2], padding='same')
            img = tf.nn.leaky_relu(img)
            img = tf.layers.max_pooling2d(img, [2, 2], strides=[1, 1], padding='same')
            img = tf.layers.conv2d(img, 32, [3, 3], strides=[2, 2], padding='same')
            img = tf.nn.leaky_relu(img)
            img = tf.layers.conv2d(img, 64, [3, 3], strides=[2, 2], padding='same')
            img = tf.nn.leaky_relu(img)
            img = tf.layers.flatten(img)
            img = tf.layers.dense(img, 1024)
            img = tf.nn.leaky_relu(img)
            img = tf.layers.dense(img, 256)

        with tf.variable_scope("DISCR", reuse=reuse):
            trgts = targets[:, self.config['pred_ext']:, :]
            total = tf.concat([trjs, trgts], -2)

            e_outs, e_state = tf.nn.dynamic_rnn(self.enc_cell(256), total, dtype=tf.float32)
            # e_state=tf.layers.conv2d(total,16,[3,2])

            fet_i = tf.layers.dense(img, 128)
            fet_state = tf.layers.dense(e_state[0] + e_state[1], 256)
            fet = tf.concat([fet_i, fet_state], -1)
            fet = tf.nn.leaky_relu(fet)

            fet = tf.layers.dense(fet, 64)
            fet = tf.nn.leaky_relu(fet)

            fet = tf.layers.dense(fet, 32)
            fet = tf.layers.dense(fet, 1)

            return fet

    def _discrim(self, trjs, targets, ims, reuse=False):
        with tf.variable_scope("DIM", reuse=reuse):
            img = ims
            # img = tf.expand_dims(ims, -1)
            img = tf.layers.conv2d(img, 16, [3, 3], strides=[2, 2], padding='same')
            img = tf.nn.leaky_relu(img)
            img = tf.layers.max_pooling2d(img, [2, 2], strides=[1, 1], padding='same')
            img = tf.layers.conv2d(img, 32, [3, 3], strides=[2, 2], padding='same')
            img = tf.nn.leaky_relu(img)
            img = tf.layers.conv2d(img, 64, [3, 3], strides=[2, 2], padding='same')
            img = tf.nn.leaky_relu(img)
            img = tf.layers.flatten(img)
            img = tf.layers.dense(img, 1024)
            img = tf.nn.leaky_relu(img)
            img = tf.layers.dense(img, 256)

        with tf.variable_scope("DISCR", reuse=reuse):
            trgts = targets[:, self.config['pred_ext']:, :]
            total = tf.concat([trjs, trgts], -2)
            print(total)
            e_outs, e_state = tf.nn.dynamic_rnn(self.enc_cell(256), total, dtype=tf.float32)
            # e_state=tf.layers.conv2d(total,16,[3,2])

            fet_i = tf.layers.dense(img, 128)
            fet_state = tf.layers.dense(e_state[0] + e_state[1], 128)
            fet = tf.concat([fet_i, fet_state], -1)
            fet = tf.nn.leaky_relu(fet)

            fet = tf.layers.dense(fet, 64)
            fet = tf.nn.leaky_relu(fet)

            fet = tf.layers.dense(fet, 32)
            fet = tf.layers.dense(fet, 1)

            return fet

    def build_recurrent(self):
        # outputs,state=tf.nn.dynamic_rnn(cell_fw,inputs=x, dtype=tf.float32,sequence_length=self.input_size, time_major=False, scope="encoder")
        # ft = tf.layers.dense(self.feats, self.state_size)
        ft = tf.layers.dense(tf.layers.flatten(self.inputs), self.state_size)
        # ft = tf.layers.dense(tf.concat([ft, inps], -1), self.state_size)
        # p_inpts = tf.tile(tf.expand_dims(ft, 1), [1,self.config['prev_leng'], 1])
        p_inpts = tuple([ft] * 2)
        # ft = tf.layers.dense(ft, 32,activation=tf.tanh)
        print("FT", ft)
        e_outs, e_state = tf.nn.dynamic_rnn(self.enc, self.inputs, initial_state=p_inpts, dtype=tf.float32, scope="ENC")
        print("ESTATE", e_state)
        e_state = tf.stack([e for e in e_state], -1)
        e_state = tf.layers.flatten(e_state)
        # e_state=tf.concat([e_state,e_state,e_state],-1)
        print("ESTATE", e_state)
        latent = tf.layers.dense(e_state, self.latent_size)
        # latent=tf.concat([latent,ft],-1)
        # latent=tf.concat([tf.layers.dense(latent,self.state_size/2),ft],-1)

        latent = tf.layers.dense(latent, self.state_size)
        print(latent)
        latent_s = latent
        print("LATETNS", latent)
        # latent_s = tuple([latent] * 2)
        # latent = tf.concat([latent, ft], -1)
        # min_ft=tf.layers.dense(self.feats,16)
        # ft_inpts=tf.layers.flatten(e_outs)
        # future_inpts=tf.layers.dense(ft_inpts,self.config['fut_leng']*8)
        # print(future_inpts)

        e_ts = tf.layers.dense(e_state, (self.config['fut_leng'] + self.config['prev_leng']) * self.state_size / 2)
        e_ts = tf.reshape(e_ts, [self.config['batch'], self.config['fut_leng'] + self.config['prev_leng'],
                                 self.state_size / 2])
        future_inpts = tf.tile(tf.expand_dims(ft, 1), [1, self.config['fut_leng'] + self.config['prev_leng'], 1])
        fut_inpts = tf.concat([e_ts, future_inpts], -1)
        # future_inpts=tf.layers.dense(fut_inpts,8)
        # future_inpts=tf.concat([e_ts,future_inpts],-1)
        # future_inpts=tf.concat([last_past,future_inpts],-1)
        print("FUTUREINPS", future_inpts)
        inpts = tf.tile(tf.expand_dims(ft, 1), [1, self.config['prev_leng'] + self.config['fut_leng'], 1])
        # d_outs,d_state= tf.nn.dynamic_rnn(self.dec,inpts,initial_state=latent,dtype=tf.float32,scope="DEC")
        with tf.variable_scope("DEC"):
            # d_outs,_=self.self_feeding_rnn(self.dec,self.config['fut_leng'],future_inpts[:,0],latent)
            d_outs, d_state = tf.nn.dynamic_rnn(tfn.GRUCell(self.state_size), fut_inpts, initial_state=latent_s,
                                                dtype=tf.float32, scope="DEC")
            print("d_outs", d_outs)
            projection = tf.layers.dense(d_outs, 2)
            print("PROJECT", projection)
        # return d_outs
        # mask = tf.zeros(shape=[self.config['batch'],self.config['prev_leng']+self.config['fut_leng']-1,2])
        # mask=tf.concat([self.inputs[:,0:1,:],mask],axis=1)
        #
        # projection=mask+projection
        return projection

    def latent_loss(self):
        # lat = 0.5*tf.reduce_sum(tf.square(self.z_mean_c) + tf.square(self.z_std_c) - tf.log(tf.square(self.z_std_c)) - 1, 1)
        lat = -0.5 * tf.reduce_sum(1 + self.z_std_c - tf.square(self.z_mean_c) - tf.exp(self.z_std_c), axis=-1)
        lat = tf.reduce_mean(lat, axis=0)
        return lat

    def cross_loss(self):
        # self.target #[b,s,2]
        # self.out    #[b,s,2]
        # self.image  #[b,x,y,4]
        xy_targ=tf.cumsum(self.target,-2)
        xy_out = self.out#tf.cumsum(self.out, -2)
        xy_targ = tf.reshape(tf.cast(xy_targ*2.0+160.0, tf.int32),
                             shape=[self.config['batch'], self.config['pred_ext'] + self.config['fut_leng'], 2])
        xy_out = tf.reshape(tf.cast(xy_out*2.0+160.0, tf.int32),
                            shape=[self.config['batch'], self.config['pred_ext'] + self.config['fut_leng'], 2])

        pts_out = tf.unstack(xy_out, axis=0)
        pts_targ = tf.unstack(xy_targ, axis=0)
        images= tf.unstack(self.image,axis=0)
        cls_ts=[]
        cls_os=[]
        for i in range(self.config['batch']):
            cl_targ = tf.gather_nd(images[i], pts_targ[i])
            cls_ts.append(cl_targ)
            cl_out = tf.gather_nd(images[i], pts_out[i])
            cls_os.append(cl_out)
        cls_ts=tf.stack(cls_ts)
        cls_os = tf.stack(cls_os)
        return tf.losses.softmax_cross_entropy(cls_ts,cls_os)

    def cross_loss_per_traj(self,traj):
        # self.target #[b,s,2]
        # self.out    #[b,s,2]
        # self.image  #[b,x,y,4]
        #xy_targ=tf.cumsum(self.target,-2)
        xy_out = traj#self.out#tf.cumsum(self.out, -2)
        xy_targ = tf.reshape(tf.cast(self.target*2.0+160.0,tf.int32),
                             shape=[self.config['batch'], self.config['pred_ext'] + self.config['fut_leng'], 2])
        xy_out = tf.reshape(tf.cast(xy_out*2.0+160.0, tf.int32),
                            shape=[self.config['batch'], self.config['pred_ext'] + self.config['fut_leng'], 2])

        pts_out = tf.unstack(xy_out, axis=0)
        pts_targ = tf.unstack(xy_targ, axis=0)
        images= tf.unstack(self.image,axis=0)
        cls_ts=[]
        cls_os=[]
        for i in range(self.config['batch']):

            cl_targ = tf.gather_nd(images[i], pts_targ[i])
            cls_ts.append(cl_targ)

            cl_out = tf.gather_nd(images[i], pts_out[i])

            cls_os.append(cl_out)
        cls_ts=tf.stack(cls_ts)
        cls_os = tf.stack(cls_os)
        return tf.losses.softmax_cross_entropy(cls_ts,cls_os)

    def enc_loss(self):
        sqrd_diff = tf.reduce_sum(tf.squared_difference(self.inputs, self.enc_i), -1)
        meaned=tf.reduce_mean(sqrd_diff)
        return meaned


    def multiple_loss(self):
        scores=[]
        stck=tf.stack(self.out)

        to_list=tf.unstack(stck)
        bg = tf.cumsum(self.target, -2)
        for i,o in enumerate(to_list):
            sqrd_diff = tf.reduce_sum(tf.squared_difference(o, bg), -1)
            scores.append(sqrd_diff)
        scores=tf.stack(scores)
        #print("SCORES",scores)
        meaned=tf.reduce_mean(scores,-1)
        #print("MEANED",meaned)
        mini= tf.reduce_min(meaned,0)

        return tf.reduce_mean(mini)

    def m_loss(self):
        if (self.config['old']):
            return 0.0
        mask = np.ones(shape=[self.config['batch'], self.config['pred_ext'] + self.config['fut_leng']])

        #print("MASK", mask.shape)
        if (self.config['inverted']):
            scales = np.arange(1.0, 10.0, (9.0 / float(self.config['pred_ext'] + self.config['fut_leng'])))
        else:
            scales = np.arange(10.0, 1.0, -(9.0 / float(self.config['pred_ext'] + self.config['fut_leng'])))
        #print("SCALES", scales.shape)
        mask = mask * scales
        xy_targ=tf.cumsum(self.target,-2)
        #xy_out = tf.cumsum(self.out, -2)
        sqrd_diff=tf.reduce_sum(tf.squared_difference(self.out,xy_targ),-1)
        # print(sqrd_diff)

        # sqrd_diff=tf.reduce_sum(tf.square(self.out-self.target),-1)
        #sqrd_diff = tf.losses.mean_squared_error(self.target, self.out)

        avg_dir_out = - self.out[:, 0, :] + self.out[:, 20, :]
        avg_dir_tar = -self.target[:, 0, :] + self.target[:, 20, :]
        avg_dir_tar = avg_dir_tar / tf.norm(avg_dir_tar)
        avg_dir_out = avg_dir_out / tf.norm(avg_dir_out)
        self.dirs = tf.losses.cosine_distance(avg_dir_tar, avg_dir_out, -1)

        avg_dir_out = - self.out[:, 10, :] + self.out[:, 30, :]
        avg_dir_tar = -self.target[:, 10, :] + self.target[:, 30, :]
        avg_dir_tar = avg_dir_tar / tf.norm(avg_dir_tar)
        avg_dir_out = avg_dir_out / tf.norm(avg_dir_out)
        self.dirs += tf.losses.cosine_distance(avg_dir_tar, avg_dir_out, -1)

        avg_dir_out = - self.out[:, 20, :] + self.out[:, -1, :]
        avg_dir_tar = -self.target[:, 20, :] + self.target[:, -1, :]
        avg_dir_tar = avg_dir_tar / tf.norm(avg_dir_tar)
        avg_dir_out = avg_dir_out / tf.norm(avg_dir_out)
        self.dirs += tf.losses.cosine_distance(avg_dir_tar, avg_dir_out, -1)

        dist_out = tf.abs(self.out[:, 1:, :] - self.out[:, :-1, :])
        dist_tar = tf.abs(self.target[:, 1:, :] - self.target[:, :-1, :])
        leng = tf.square(tf.reduce_sum(dist_out, -1) - tf.reduce_sum(dist_tar, -1))

        #sqrd_diff=sqrd_diff*mask

        sqrd_diff=tf.reduce_mean(sqrd_diff)
        # sqrd_diff=tf.reduce_sum(sqrd_diff,-1)
        # sqrd_diff=tf.nn.l2_loss(self.out-self.target)
        self.dirs_loss = tf.reduce_mean(self.dirs)
        self.leng_loss = tf.reduce_mean(leng)
        total = 0.0
        if ('l2' in self.config['rec_losses']):
            total += tf.reduce_mean(sqrd_diff)
        if ('dir' in self.config['rec_losses']):
            total += 0.5 * self.dirs_loss
        if ('leng' in self.config['rec_losses']):
            total += self.leng_loss
        if (self.config['vae']):
            total += self.latent_loss()
        total += self.cross_loss()

        return total
