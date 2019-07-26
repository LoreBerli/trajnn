import tensorflow as tf



def self_feeding_rnn(self,cell, seqlen, Hin, Xin, processing=tf.identity):
    '''Unroll cell by feeding output (hidden_state) of cell back into in as input.
       Outputs are passed through `processing`. It is up to the caller to ensure that the processed
       outputs have suitable shape to be input.'''
    veclen = tf.shape(Xin)[-1]
    # this will grow from [ BATCHSIZE, 0, VELCEN ] to [ BATCHSIZE, SEQLEN, VECLEN ]
    first=Hin
    buffer = tf.TensorArray(dtype=tf.float32, size=seqlen)
    initial_state = (0, Hin, Xin, buffer,first)
    condition = lambda i, *_: i < seqlen
    print(initial_state)

    def do_time_step(i, state, xo, ta,fr):
        #fir=tf.layers.dense(first,32)
        #state=tf.layers.dropout(tf.layers.dense(tf.concat([state,fir],-1),self.dec_size),self.dropout)
        Yt, Ht = cell(xo, state)
        print("YT",Ht.shape)
        Yro = tf.identity(Yt)#tf.layers.dense(Yt,3+self.gauss*6)
        return 1 + i, Ht, Yro, ta.write(i, Yro),fr

    _, Hout, _, final_ta,_ = tf.while_loop(condition, do_time_step, initial_state)

    ta_stack = final_ta.stack()
    Yo = tf.reshape(ta_stack, shape=((-1, seqlen, veclen)))
    return Yo, Hout
