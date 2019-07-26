import tensorflow as tf
import tensorflow.contrib.rnn as tfn
import numpy as np

tf.nn.seq2seq = tf.contrib.legacy_seq2seq
class rec_model():
    def __init__(self,config):

        self.state_size=config['units']
        self.latent_size=config['lat_size']
        self.inputs=tf.placeholder(tf.float32,shape=[config['batch'],config['prev_leng'],2])
        self.target=tf.placeholder(tf.float32,shape=[config['batch'],config['fut_leng'],2])
        self.box=tf.placeholder(tf.float32,shape=[config['batch'],4])
        self.feats=tf.placeholder(tf.float32,shape=[config['batch'],128])
        self.clas = tf.placeholder(tf.float32, shape=[config['batch']])
        self.config=config
        self.enc = self.enc_cell(self.state_size)
        self.dec= self.dec_cell(self.state_size)
        #self.out = self.build_recurrent()
        self.out1,self.out2=self.build_recurrent()
        self.out=[self.out1,self.out2]
        self.loss = self.m_loss()
    def enc_cell(self,num):
        return tfn.GRUCell(num,name="enc_cell")
    def dec_cell(self,num):
        return tfn.GRUCell(num,name="dec_cell")

    def build_autoregressive_cell(self,num,cell):
        pass

    def self_feeding_rnn(self,cell, seqlen, Hin, Xin, processing=tf.identity):
        '''Unroll cell by feeding output (hidden_state) of cell back into in as input.
           Outputs are passed through `processing`. It is up to the caller to ensure that the processed
           outputs have suitable shape to be input.'''

        veclen = tf.shape(Xin)[-1]
        # this will grow from [ BATCHSIZE, 0, VELCEN ] to [ BATCHSIZE, SEQLEN, VECLEN ]
        buffer = tf.TensorArray(dtype=tf.float32, size=seqlen)
        initi=Hin
        initial_state = (0, Hin, Xin, buffer)
        condition = lambda i, *_: i < seqlen
        print(initial_state)
        def do_time_step(i, state, xo, ta):

            # xd=tf.concat([x_real,xo],-1)
            #xd= tf.layers.dense(xo,64)
            Yt, Ht = cell(xo, state)
            print(Yt,Ht)
            #Ht=tf.layers.dense(tf.concat([Ht,initi],-1),initi.shape[-1])

            #Yro=tf.concat([x_real,Yt],-1)
            Yro =processing(Yt)
            print("yro",Yro)
            return (1 + i, Ht, Yro, ta.write(i, Yro))
        _, Hout, _, final_ta = tf.while_loop(condition, do_time_step, initial_state)

        ta_stack = final_ta.stack()

        Yo = tf.reshape(ta_stack, shape=((-1, seqlen, veclen)))
        print("YO",Yo)
        return Yo, Hout

    def build_recurrent(self):
        # outputs,state=tf.nn.dynamic_rnn(cell_fw,inputs=x, dtype=tf.float32,sequence_length=self.input_size, time_major=False, scope="encoder")
        ft=tf.layers.dense(self.feats,self.state_size,activation=tf.nn.tanh)
        print("FT", ft)
        e_outs,e_state= tf.nn.dynamic_rnn(self.enc,self.inputs,dtype=tf.float32,scope="ENC")

        latent=tf.layers.dense(e_state,self.latent_size)


        latent=tf.layers.dense(latent,self.state_size)

        future_inpts=tf.tile(tf.expand_dims(ft,1),[1,self.config['fut_leng'],1])
        print("FUTUREINPS",future_inpts)
        inpts=tf.tile(tf.expand_dims(ft,1),[1,self.config['fut_leng'],1])
        #d_outs,d_state= tf.nn.dynamic_rnn(self.dec,inpts,initial_state=latent,dtype=tf.float32,scope="DEC")
        with tf.variable_scope("DEC1"):
            #d_outs,_=self.self_feeding_rnn(self.dec,self.config['fut_leng'],future_inpts[:,0],latent)
            d_outs1, d_state1 = tf.nn.dynamic_rnn(self.dec, future_inpts, initial_state=latent, dtype=tf.float32, scope="DEC")
            print("d_outs", d_outs1)
            projection1=tf.layers.dense(d_outs1,2)
            print("PROJECT",projection1)
        with tf.variable_scope("DEC2"):
            #d_outs,_=self.self_feeding_rnn(self.dec,self.config['fut_leng'],future_inpts[:,0],latent)
            d_outs2, d_state2 = tf.nn.dynamic_rnn(self.dec, future_inpts, initial_state=latent, dtype=tf.float32, scope="DEC")
            print("d_outs", d_outs2)
            projection2=tf.layers.dense(d_outs2,2)
            print("PROJECT",projection2)
        #return d_outs
        return projection1,projection2



    def m_loss(self):
        sqrd_diff1=tf.reduce_sum(tf.abs(self.out1-self.target),-1)
        sqrd_diff2 = tf.reduce_sum(tf.abs(self.out2 - self.target), -1)
        #sqrd_diff=tf.reduce_mean(sq)
        sqrd_diff1=tf.reduce_sum(sqrd_diff1,-1)
        sqrd_diff2 = tf.reduce_sum(sqrd_diff2, -1)
        sqrd_diff1=tf.reduce_mean(sqrd_diff1)
        sqrd_diff2=tf.reduce_mean(sqrd_diff2)
        losses=tf.stack([sqrd_diff1,sqrd_diff2],-1)
        best=tf.argmin(losses,output_type=tf.int32)
        print(best)
        #sqrd_diff=tf.nn.l2_loss(self.out-self.target)
        return losses[best]

