import tensorflow as tf
import tensorflow.contrib.rnn as tfn
import numpy as np
import spatial_transformer
tf.nn.seq2seq = tf.contrib.legacy_seq2seq


class rec_model():
    def __init__(self,config):
        self.dims=config['dims']
        self.state_size=config['units']
        self.latent_size=config['lat_size']
        self.inputs=tf.placeholder(tf.float32,shape=[config['batch'],config['prev_leng'],self.dims])
        self.target=tf.placeholder(tf.float32,shape=[config['batch'],config['pred_ext']+config['fut_leng'],self.dims])
        self.box=tf.placeholder(tf.float32,shape=[config['batch'],4])
        self.feats=tf.placeholder(tf.float32,shape=[config['batch'],128])
        self.clas = tf.placeholder(tf.float32, shape=[config['batch']])
        self.dirs_loss= tf.placeholder(tf.float32, shape=[config['batch']])
        self.leng_loss = tf.placeholder(tf.float32, shape=[config['batch']])
        self.noiz= tf.placeholder(tf.float32, shape=[config['batch'],100])
        self.image=tf.placeholder(tf.float32,shape=[config['batch'],128,256])
        self.config=config
        self.drop=tf.placeholder(tf.float32)
        self.enc = self.enc_cell(self.state_size)
        self.dec= self.dec_cell(self.state_size)
        
        if config['type']==0:
            self.out = self.rnn_alone()
        elif config['type']==3:
            self.out = self.linear()
        elif config['old']:
            self.out=self.build_recurrent()
        else:
            self.out=self.rnn_with_ctx()

        self.loss = self.m_loss()
    def enc_cell(self,num):
        #return tfn.MultiRNNCell([tfn.GRUCell(num,name="enc_cell"+str(i)) for i in range(self.config['num'])])
        return tfn.DropoutWrapper(tfn.GRUCell(num,name="enc_cell"),state_keep_prob=self.drop)

    def dec_cell(self,num):
        #return tfn.MultiRNNCell([tfn.GRUCell(num,name="dec_cell"+str(i)) for i in range(self.config['num'])])
        return tfn.DropoutWrapper(tfn.GRUCell(num, name="dec_cell"),state_keep_prob=self.drop)

    def state_dec(self,st):
        # return tf.concat([st for i in range(3)],-1)
        return (st for i in range(3))

    def build_autoregressive_cell(self,num,cell):
        pass


    def linear(self):
        print("LINEAR")

        with tf.variable_scope("GEN"):

            flat=tf.layers.flatten(self.inputs)
            #e_state= tf.layers.dense(flat,self.state_size,activation=tf.nn.leaky_relu)

            if(self.image.shape>3):
                img=self.image
            else:
                img=tf.expand_dims(self.image,-1)

            img=tf.expand_dims(self.image,-1)
            img=tf.layers.conv2d(img,16,[4,4],padding='same',dilation_rate=3)

            img=tf.layers.max_pooling2d(img,[2,2],strides=[1,1],padding='same')
            img=tf.layers.conv2d(img,32,[4,4],padding='same',dilation_rate=2)
            img = tf.layers.max_pooling2d(img, [2, 2], strides=[1, 1], padding='same')
            img=tf.layers.conv2d(img,64,[3,3],strides=[2,2],padding='same')
            img=tf.layers.flatten(img)
            img=tf.layers.dense(img,128,activation=tf.nn.leaky_relu)
            emb = tf.layers.dense(flat, 128)
            latent=tf.concat([img,emb],-1)
            e_ts = tf.layers.dense(latent, self.latent_size)
            Hin = (e_ts, e_ts)

            print("EMB",emb)
            Xin = tf.layers.dense(tf.concat([img,emb],-1),(self.config['fut_leng'] + self.config['pred_ext'])*self.latent_size)
            print(Xin)
            Xin=tf.reshape(Xin,[self.config['batch'],self.config['fut_leng'] + self.config['pred_ext'],self.latent_size])
            print(Xin)
            with tf.variable_scope("DEC"):

                # d_outs, d_state = self.self_feeding_rnn(self.dec_cell(self.latent_size),self.config['fut_leng'] + self.config['pred_ext']
                #                                         , Hin, Xin)
                d_outs,d_state=tf.nn.dynamic_rnn(self.dec_cell(self.latent_size),Xin,initial_state=Hin)
                d_outs=tf.layers.dense(d_outs,2)
                print("d_outs", d_outs)

            return d_outs

    def __rnn_with_ctx(self):
        with tf.variable_scope("GEN"):


            if (self.image.shape > 3):
                img = self.image
            else:
                img = tf.expand_dims(self.image, -1)

            if(self.config["stn"]):
                identity = np.array([[1., 0., 0.],
                                     [0., 1., 0.]])
                identity = identity.flatten()
                identity=np.tile(np.expand_dims(identity,0),[self.config['batch'],1])

                theta = tf.Variable(initial_value=identity)
                img = spatial_transformer.transformer(img, theta, [256,128])
                print("-------",img)

            img = tf.layers.conv2d(img, 16, [5, 5], padding='same', dilation_rate=2, activation=tf.nn.leaky_relu)
            img = tf.layers.max_pooling2d(img, [2, 2], strides=[2, 2], padding='same')
            img = tf.layers.conv2d(img, 32, [3, 3], padding='same', dilation_rate=2, activation=tf.nn.leaky_relu)
            img = tf.layers.max_pooling2d(img, [2, 2], strides=[2, 2], padding='same')
            img = tf.layers.conv2d(img, 64, [3, 3], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            img = tf.layers.conv2d(img, 64, [3, 3], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            print(img)
            img = tf.layers.flatten(img)

            img_embed = tf.layers.dense(img, 512)

            if(self.config['past_box_inputs'] and self.config['past_img_inputs']):
                mini_embed = tf.layers.dense(img_embed, 8, activation=tf.nn.tanh)
                past_inputs = tf.concat(
                    [self.inputs, tf.tile(tf.expand_dims(self.box, 1), [1, self.config['prev_leng'], 1]),
                     tf.tile(tf.expand_dims(mini_embed, 1), [1, self.config['prev_leng'], 1])], -1)
            elif self.config['past_box_inputs'] :
                past_inputs = tf.concat(
                    [self.inputs, tf.tile(tf.expand_dims(self.box, 1), [1, self.config['prev_leng'], 1])], -1)
            else:
                past_inputs = self.inputs

            e_outs, e_state = tf.nn.dynamic_rnn(self.enc, past_inputs, dtype=tf.float32, scope="ENC")
            e_state = tf.concat([e_state[-1], e_state[-2]], -1)


            if self.config['fut_img_inputs'] :
                img_embed_ = tf.layers.dense(img_embed, self.config["img_embed"])
                seq_embed = tf.layers.dense(e_state, self.latent_size -self.config["img_embed"])
                embed = tf.concat([img_embed_, seq_embed, self.box], -1)
            else:
                seq_embed = tf.layers.dense(e_state, self.latent_size)
                embed = tf.concat([seq_embed, self.box], -1)

            bias = self.inputs[:, -self.config['pred_ext'], :]
            bias = tf.expand_dims(bias, 1)
            Xin = tf.layers.dense(embed, self.state_size)
            e_ts = tf.layers.dense(embed, (self.config['fut_leng'] + self.config['pred_ext']) * (self.state_size))
            e_ts = tf.reshape(e_ts, [self.config['batch'], self.config['fut_leng'] + self.config['pred_ext'],
                                     self.state_size])
            with tf.variable_scope("DEC"):
                if(self.config['autoregressive']):
                    #d_outs, d_state = self.self_feeding_rnn(tfn.GRUCell(self.state_size),self.config['fut_leng'] + self.config['pred_ext'], e_ts, Xin)
                    d_outs, d_state = self.self_feeding_rnn(tfn.GRUCell(self.state_size),
                                                        self.config['fut_leng'] + self.config['pred_ext']
                                                        , e_ts, Xin)
                else:
                    d_outs, d_state = tf.nn.dynamic_rnn(tfn.GRUCell(self.state_size), e_ts, initial_state=Xin,dtype=tf.float32, scope="DEC")
                    d_outs=tf.layers.dense(d_outs,2)


                print("d_outs", d_outs)


            my_b = d_outs[:, :1, :]
            diff = my_b - bias
            d_outs = d_outs - diff
            return d_outs

    def rnn_with_ctx(self):
        with tf.variable_scope("GEN"):
            if (self.image.shape > 3):
                img = self.image
            else:
                img = tf.expand_dims(self.image, -1)

            img = tf.expand_dims(self.image, -1)

            img = tf.layers.conv2d(img, 16, [3,3], padding='same', dilation_rate=2, activation=tf.nn.leaky_relu)
            img = tf.layers.max_pooling2d(img, [2, 2], strides=[2, 2], padding='same')
            img = tf.layers.conv2d(img, 32, [3, 3], padding='same', dilation_rate=2, activation=tf.nn.leaky_relu)
            img = tf.layers.max_pooling2d(img, [2, 2], strides=[2, 2], padding='same')
            img = tf.layers.conv2d(img, 64, [3, 3], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            img = tf.layers.conv2d(img, 128, [3, 3], strides=[2, 2], padding='same', activation=tf.nn.leaky_relu)
            print(img)
            img = tf.layers.flatten(img)
            img_embed = tf.layers.dense(img, 256)
            img_embed=tf.layers.dropout(img_embed,rate=self.drop)
            #img_embed=tf.contrib.layers.layer_norm(img_embed)
            if(self.config['past_box_inputs'] and self.config['past_img_inputs']):
                mini_embed = tf.layers.dense(img_embed, 8, activation=tf.nn.tanh)
                past_inputs = tf.concat(
                    [self.inputs, tf.tile(tf.expand_dims(self.box, 1), [1, self.config['prev_leng'], 1]),
                     tf.tile(tf.expand_dims(mini_embed, 1), [1, self.config['prev_leng'], 1])], -1)
            elif self.config['past_box_inputs'] :
                past_inputs = tf.concat(
                    [self.inputs, tf.tile(tf.expand_dims(self.box, 1), [1, self.config['prev_leng'], 1])], -1)
            else:
                past_inputs = self.inputs

            e_outs, e_state = tf.nn.dynamic_rnn(self.enc, past_inputs, dtype=tf.float32, scope="ENC")
            #e_state = tf.concat([e_state[-1], e_state[-2]], -1)


            if self.config['fut_img_inputs'] :
                img_embed_ = tf.layers.dense(img_embed, self.config["img_embed"])
                seq_embed = tf.layers.dense(e_state, self.latent_size -self.config["img_embed"])
                embed = tf.concat([img_embed_, seq_embed, self.box], -1)
            else:
                seq_embed = tf.layers.dense(e_state, self.latent_size)
                embed = tf.concat([seq_embed, self.box], -1)

            bias = self.inputs[:, -1, :]
            bias = tf.expand_dims(bias, 1)
            Xin = tf.layers.dense(embed, self.state_size)
            e_ts = tf.layers.dense(embed, (self.config['fut_leng'] + self.config['pred_ext']) * (self.state_size))

            e_ts = tf.reshape(e_ts, [self.config['batch'], self.config['fut_leng'] + self.config['pred_ext'],
                                     self.state_size])

            with tf.variable_scope("DEC"):
                if(self.config['autoregressive']):
                    #d_outs, d_state = self.self_feeding_rnn(tfn.GRUCell(self.state_size),self.config['fut_leng'] + self.config['pred_ext'], e_ts, Xin)
                    d_outs, d_state = self.self_feeding_rnn(tfn.GRUCell(self.state_size),
                                                        self.config['fut_leng'] + self.config['pred_ext']
                                                        , e_ts, Xin)
                else:
                    d_outs, d_state = tf.nn.dynamic_rnn(tfn.GRUCell(self.state_size), e_ts, initial_state=Xin,dtype=tf.float32, scope="DEC")
                    d_outs=tf.layers.dense(d_outs,2)


                print("d_outs", d_outs)

            my_b = d_outs[:, self.config['pred_ext'], :]
            my_b = tf.expand_dims(my_b, 1)
            diff = my_b - bias
            d_outs = d_outs - diff
            return d_outs

    def _rnn_with_ctx(self):
        with tf.variable_scope("GEN"):
            if(self.image.shape>3):
                img=self.image
            else:
                img=tf.expand_dims(self.image,-1)
            print(img)
            img=tf.layers.conv2d(img,16,[5,5],padding='same',dilation_rate=2,activation=tf.nn.leaky_relu)
            img=tf.layers.max_pooling2d(img,[2,2],strides=[2,2],padding='same')
            img=tf.layers.conv2d(img,32,[3,3],padding='same',dilation_rate=2,activation=tf.nn.leaky_relu)
            img = tf.layers.max_pooling2d(img, [2, 2], strides=[2,2], padding='same')
            img=tf.layers.conv2d(img,64,[3,3],strides=[2,2],padding='same',activation=tf.nn.leaky_relu)
            img = tf.layers.conv2d(img, 64, [3, 3], strides=[2, 2], padding='same',activation=tf.nn.leaky_relu)
            print(img)
            img=tf.layers.flatten(img)

            img_embed=tf.layers.dense(img,512)
            mini_embed = tf.layers.dense(img_embed, 8, activation=tf.nn.tanh)
            #box_inputs=tf.concat([self.inputs,tf.tile(tf.expand_dims(self.box,1),[1,self.config['prev_leng'],1])],-1)

            box_inputs=tf.concat([self.inputs,tf.tile(tf.expand_dims(self.box,1),[1,self.config['prev_leng'],1]),tf.tile(tf.expand_dims(mini_embed,1),[1,self.config['prev_leng'],1])],-1)
            e_outs,e_state= tf.nn.dynamic_rnn(self.enc,box_inputs,dtype=tf.float32,scope="ENC")
            print("ESTATE",e_state)
            e_state=tf.concat([e_state[-1],e_state[-2]],-1)
            #e_state=tf.layers.flatten(e_state)
            print("ESTATE", e_state)





            img_embed_=tf.layers.dense(img_embed,self.latent_size/2)

            seq_embed=tf.layers.dense(e_state,self.latent_size/2)
            #embed=seq_embed
            embed=tf.concat([img_embed_,seq_embed,self.box],-1)


            #latent_s=tf.layers.dense(embed,self.latent_size,activation=tf.nn.tanh)

            # e_ts = tf.layers.dense(latent, self.state_size,activation=tf.nn.leaky_relu)
            # e_ts=tf.expand_dims(e_ts,1)
            # e_ts=tf.tile(e_ts,[1, self.config['fut_leng'] + self.config['pred_ext'],1])
            # print("ETS",e_ts)

            bias = self.inputs[:, -self.config['pred_ext'], :]
            bias=tf.expand_dims(bias,1)


            Xin=tf.layers.dense(embed,self.state_size)
            e_ts = tf.layers.dense(embed, (self.config['fut_leng'] + self.config['pred_ext']) * (self.state_size))
            e_ts = tf.reshape(e_ts,[self.config['batch'], self.config['fut_leng'] + self.config['pred_ext'],self.state_size])
            with tf.variable_scope("DEC"):
                d_outs, d_state = self.self_feeding_rnn(tfn.GRUCell(self.state_size),self.config['fut_leng'] + self.config['pred_ext']
                                                        , e_ts, Xin)
                # d_outs = self.better_self_ffed(tfn.GRUCell(self.state_size),self.config['fut_leng'] + self.config['pred_ext'],
                #                                self.config['batch'],self.state_size,e_ts)
                print("d_outs", d_outs)
                # e = tf.reshape(d_outs, [self.config['batch'],-1, self.state_size])
                # print("e", d_outs)
                #d_outs=tf.layers.dense(d_outs,self.dims)
                #projection=tf.reshape(projection,shape=[self.config['batch'],self.config['fut_leng'] + self.config['pred_ext'],2])
                #print("PROJECT",projection)
            print("INPUTS",self.inputs)
            # bias=tf.tile(self.inputs[:,-self.config['pred_ext']:-self.config['pred_ext']+1,:],[1,self.config['fut_leng'] + self.config['pred_ext'],1])
            # my_b=tf.tile(d_outs[:,:1,:],[1,self.config['fut_leng'] + self.config['pred_ext'],1])
            #bias=self.inputs[:,-self.config['pred_ext']:-self.config['pred_ext']+1,:]
            my_b=d_outs[:,:1,:]
            diff=my_b-bias
            d_outs=d_outs- diff
            return d_outs



    def self_feeding_rnn(self,cell, seqlen, Hin, Xin, processing=tf.identity):
        buffer = tf.TensorArray(dtype=tf.float32,size=seqlen)
        inputs = tf.transpose(Hin, [1, 0, 2])

        inputs_ta = tf.TensorArray(dtype=tf.float32, size=seqlen, clear_after_read=False)
        inputs_ta = inputs_ta.unstack(inputs)

        in_first=inputs[0]
        initial_state = (0, in_first,Xin,buffer,inputs_ta)
        condition = lambda i, *_: i < seqlen

        def do_time_step(i, state, xo,ta,inp_ta):
            st = inp_ta.read(i)
            s1 = st
            xd=tf.concat([s1,xo],-1)
            xd= tf.layers.dense(xd,self.state_size,activation=tf.nn.tanh)
            print("xo",xd)
            print("state",state)
            wut= cell(xd, state)
            print(wut)
            Yt, Ht=wut
            next=Ht

            print("YT",Yt)
            print("HT",Ht)
            #Ht=tf.layers.dense(tf.concat([Ht,initi],-1),initi.shape[-1])

            #Yro=tf.concat([x_real,Yt],-1)
            Yro =tf.layers.dense(Yt,2)
            print("yro",Yro)

            return (1 + i, next, Yt,ta.write(i,Yro),inp_ta)


        _, Hout,yout,final_stack ,_= tf.while_loop(condition, do_time_step, initial_state)

        ta_stack = final_stack.stack()
        print("STACK",ta_stack.shape)
        #Yo=ta_stack
        Yo = tf.transpose(ta_stack,perm=[1,0,2])#tf.reshape(ta_stack, shape=((self.config['batch'],seqlen, self.config['dims'])))
        print("YO",Yo)
        print("YOUT",yout)
        print("HOUT",Hout)
        return Yo, Hout

    def better_self_ffed(self,cell, max_time,batch,size,inpts):
        inputs = tf.transpose(inpts,[1,0,2])
        sequence_length = max_time#tf.placeholder(shape=(batch,), dtype=tf.int32)
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time,clear_after_read = False)
        inputs_ta = inputs_ta.unstack(inputs)

        def loop_fn(time, cell_output, cell_state, loop_state):
            emit_output = cell_output  # == None for time == 0
            if cell_output is None:  # time == 0
                next_cell_state = inputs_ta.read(time)
            else:
                next_cell_state = cell_state

            inputs_ta.write((time+1),next_cell_state)
            elements_finished = (time >= sequence_length)
            finished = tf.reduce_all(elements_finished)
            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch, size], dtype=tf.float32),
                lambda: inputs_ta.read(time))
            next_loop_state = None
            return (elements_finished, next_input, next_cell_state,
                    emit_output, next_loop_state)

        outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
        outputs = outputs_ta.stack()
        outputs=tf.transpose(outputs,[1,0,2])
        print("OUTPUT",outputs)
        return outputs

    def rnn_alone(self):
        # outputs,state=tf.nn.dynamic_rnn(cell_fw,inputs=x, dtype=tf.float32,sequence_length=self.input_size, time_major=False, scope="encoder")
        e_outs, e_state = tf.nn.dynamic_rnn(self.enc, self.inputs, dtype=tf.float32, scope="ENC")

        e_state = e_state[-1] + e_state[-2]
        e_state = tf.layers.flatten(e_state)

        latent=tf.layers.dense(e_state,self.latent_size)
        #latent=tf.concat([latent,ft],-1)
        #latent=tf.concat([tf.layers.dense(latent,self.state_size/2),ft],-1)

        latent=tf.layers.dense(latent,self.state_size)
        e_ts=latent
        #e_ts=latent_s
        Hin=(e_ts,e_ts)
        Xin=tf.layers.dense(latent,self.latent_size)

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
                                                    , e_ts,latent)
            # d_outs = self.better_self_ffed(tfn.GRUCell(self.state_size),
            #                                self.config['fut_leng'] + self.config['pred_ext'],
            #                                self.config['batch'], self.state_size, e_ts)
            print("d_outs", d_outs)
            bias = self.inputs[:, -self.config['pred_ext']:-self.config['pred_ext'] + 1, :]
            my_b = d_outs[:, :1, :]
            diff = my_b - bias
            d_outs = d_outs - diff

        return d_outs

    def discrim(self,trjs,targets,ims,reuse=False):
        with tf.variable_scope("DIM",reuse=reuse):
            img=ims
            #img = tf.expand_dims(ims, -1)
            img = tf.layers.conv2d(img, 16, [3, 3], strides=[2, 2], padding='same')
            img= tf.nn.leaky_relu(img)
            img = tf.layers.max_pooling2d(img, [2, 2], strides=[1, 1], padding='same')
            img = tf.layers.conv2d(img, 32, [3, 3], strides=[2, 2], padding='same')
            img= tf.nn.leaky_relu(img)
            img = tf.layers.conv2d(img, 64, [3, 3], strides=[2, 2], padding='same')
            img= tf.nn.leaky_relu(img)
            img = tf.layers.flatten(img)
            img = tf.layers.dense(img, 1024)
            img= tf.nn.leaky_relu(img)
            img=tf.layers.dense(img,256)


        with tf.variable_scope("DISCR",reuse=reuse):
            trgts=targets[:,self.config['pred_ext']:,:]
            total=tf.concat([trjs,trgts],-2)
            print(total)
            e_outs,e_state= tf.nn.dynamic_rnn(self.enc_cell(256),total,dtype=tf.float32)
            #e_state=tf.layers.conv2d(total,16,[3,2])

            fet_i = tf.layers.dense(img,128)
            fet_state= tf.layers.dense(e_state[0]+e_state[1],256)
            fet=tf.concat([fet_i,fet_state],-1)
            fet = tf.nn.leaky_relu(fet)

            fet= tf.layers.dense(fet,64)
            fet = tf.nn.leaky_relu(fet)

            fet= tf.layers.dense(fet,32)
            fet=tf.layers.dense(fet,1)

            return fet

    def _discrim(self,trjs,targets,ims,reuse=False):
        with tf.variable_scope("DIM",reuse=reuse):
            img=ims
            #img = tf.expand_dims(ims, -1)
            img = tf.layers.conv2d(img, 16, [3, 3], strides=[2, 2], padding='same')
            img= tf.nn.leaky_relu(img)
            img = tf.layers.max_pooling2d(img, [2, 2], strides=[1, 1], padding='same')
            img = tf.layers.conv2d(img, 32, [3, 3], strides=[2, 2], padding='same')
            img= tf.nn.leaky_relu(img)
            img = tf.layers.conv2d(img, 64, [3, 3], strides=[2, 2], padding='same')
            img= tf.nn.leaky_relu(img)
            img = tf.layers.flatten(img)
            img = tf.layers.dense(img, 1024)
            img= tf.nn.leaky_relu(img)
            img=tf.layers.dense(img,256)


        with tf.variable_scope("DISCR",reuse=reuse):
            trgts=targets[:,self.config['pred_ext']:,:]
            total=tf.concat([trjs,trgts],-2)
            print(total)
            e_outs,e_state= tf.nn.dynamic_rnn(self.enc_cell(256),total,dtype=tf.float32)
            #e_state=tf.layers.conv2d(total,16,[3,2])

            fet_i = tf.layers.dense(img,128)
            fet_state= tf.layers.dense(e_state[0]+e_state[1],128)
            fet=tf.concat([fet_i,fet_state],-1)
            fet = tf.nn.leaky_relu(fet)

            fet= tf.layers.dense(fet,64)
            fet = tf.nn.leaky_relu(fet)

            fet= tf.layers.dense(fet,32)
            fet=tf.layers.dense(fet,1)

            return fet

    def build_recurrent(self):
        # outputs,state=tf.nn.dynamic_rnn(cell_fw,inputs=x, dtype=tf.float32,sequence_length=self.input_size, time_major=False, scope="encoder")
        #ft = tf.layers.dense(self.feats, self.state_size)
        ft = tf.layers.dense(tf.layers.flatten(self.inputs), self.state_size)
        #ft = tf.layers.dense(tf.concat([ft, inps], -1), self.state_size)
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
        latent_s=latent
        print("LATETNS", latent)
        #latent_s = tuple([latent] * 2)
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
            d_outs, d_state = tf.nn.dynamic_rnn(tfn.GRUCell(self.state_size), fut_inpts, initial_state=latent_s, dtype=tf.float32, scope="DEC")
            print("d_outs", d_outs)
            projection = tf.layers.dense(d_outs, 2)
            print("PROJECT", projection)
        # return d_outs
        # mask = tf.zeros(shape=[self.config['batch'],self.config['prev_leng']+self.config['fut_leng']-1,2])
        # mask=tf.concat([self.inputs[:,0:1,:],mask],axis=1)
        #
        # projection=mask+projection
        return projection

    def build_seq2seq(self):
        mode="train"
        sele=np.tile([self.config['prev_leng']],[self.config['batch']])
        sele=np.array(sele,dtype=np.int32)
        if mode == "train":
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.inputs,sequence_length=sele)
        # elif mode == "infer":
        #     helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        #         embedding=embedding,
        #         start_tokens=tf.tile([GO_SYMBOL], [batch_size]),
        #         end_token=END_SYMBOL)
        lat=tf.layers.dense(self.feats,self.state_size)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self.dec,
            helper=helper,
            initial_state=lat)#self.dec.zero_state(self.config['batch'], tf.float32))
        outputs, _ ,_= tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=20)
        print("OUTPUTSSS",outputs[0])
        projection=tf.layers.dense(outputs[0],2)
        print("PROJE",projection)
        return projection


    def m_loss(self):
        if(self.config['old']):
            return 0.0
        mask=np.ones(shape=[self.config['batch'],self.config['pred_ext']+self.config['fut_leng']])
        print("MASK",mask.shape)
        if(self.config['inverted']):
            scales=np.arange(1.0,10.0,(9.0/float(self.config['pred_ext']+self.config['fut_leng'])))
        else:
            scales = np.arange(10.0, 1.0, -(9.0 / float(self.config['pred_ext'] + self.config['fut_leng'])))
        print("SCALES",scales.shape)
        mask=mask*scales
        sqrd_diff=tf.reduce_sum(tf.squared_difference(self.out,self.target),-1)
        #print(sqrd_diff)


        #sqrd_diff=tf.reduce_sum(tf.square(self.out-self.target),-1)
        #sqrd_diff=tf.losses.mean_squared_error(self.target,self.out)

        dist_out=tf.abs(self.out[:,1:,:]-self.out[:,:-1,:])
        dist_tar=tf.abs(self.target[:,1:,:]-self.target[:,:-1,:])
        avg_dir_out=     - self.out[:,0,:] +    self.out[:, -1, :]
        avg_dir_tar = -self.target[:, 0, :] + self.target[:, -1, :]
        avg_dir_tar=avg_dir_tar/tf.norm(avg_dir_tar)
        avg_dir_out=avg_dir_out/tf.norm(avg_dir_out)
        self.dirs=tf.losses.cosine_distance(avg_dir_tar,avg_dir_out,-1)
        leng=tf.square(tf.reduce_sum(dist_out,-1)-tf.reduce_sum(dist_tar,-1))


        #sqrd_diff=sqrd_diff*mask

        #sqrd_diff=tf.reduce_mean(sq)
        sqrd_diff=tf.reduce_sum(sqrd_diff,-1)
        #sqrd_diff=tf.nn.l2_loss(self.out-self.target)
        self.dirs_loss=tf.reduce_mean(self.dirs)
        self.leng_loss=tf.reduce_mean(leng)
        total=0.0
        if('l2' in self.config['rec_losses']):
            total+=tf.reduce_mean(sqrd_diff)
        if('dir' in self.config['rec_losses']):
            total+=self.dirs_loss
        if('leng' in self.config['rec_losses']):
            total+=self.leng_loss

        return total

