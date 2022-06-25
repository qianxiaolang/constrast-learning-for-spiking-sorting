import tensorflow as tf
from myUtils import ln
import numpy as np
import math
def branch(data):

    with tf.variable_scope('branch',reuse=tf.AUTO_REUSE):
        # data=tf.layers.dense(data,1000,tf.nn.leaky_relu,kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        data=tf.layers.dense(data,500,tf.nn.leaky_relu,kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        data=tf.layers.dense(data,200,tf.nn.leaky_relu,kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        data=tf.layers.dense(data,50,tf.nn.leaky_relu,kernel_initializer=tf.random_normal_initializer(stddev=0.1))

        # data = tf.layers.dense(data, 60, tf.nn.leaky_relu,
        #                        kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        # data = tf.layers.dense(data, 70, tf.nn.leaky_relu, kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        # data = tf.layers.dense(data, 50, tf.nn.leaky_relu, kernel_initializer=tf.random_normal_initializer(stddev=0.1))

        # cell=tf.nn.rnn_cell.LSTMCell(num_units=50,initializer=tf.random_normal_initializer(stddev=0.1),)
        # zero_state=cell.zero_state(batch_size=batch_size,dtype=tf.float32)
        # outputs,hidden_state=tf.nn.dynamic_rnn(cell,data,initial_state=zero_state,dtype=tf.float32)
    return data

def positionEncoder():
    pe = np.zeros([10, 81], dtype=np.float32)
    position = np.reshape(np.arange(0, 10, dtype=np.float32), [-1, 1])
    # 偶
    div_term_even = np.reshape(np.exp(np.arange(0, 81, 2, dtype=np.float32) * (-math.log(10000.0) / 81)), [1, -1])

    # 奇
    div_term_odd = np.reshape(np.exp(np.arange(0, 80, 2, dtype=np.float32) * (-math.log(10000.0) / 81)), [1, -1])
    #
    pe[:, 0::2] = np.sin(position * div_term_even)
    pe[:, 1::2] = np.cos(position * div_term_odd)

    ret = tf.convert_to_tensor(pe, dtype=tf.float32)
    return tf.expand_dims(ret,axis=0)

def encoder(input_data,block_num=1,num_head=9):
    # 这一块加入位置编码

    # input_data = ln(input_data, scope='init')
    pe = positionEncoder()
    input_data = pe/10 + input_data
    input_data=tf.layers.dropout(input_data,rate=0.1,training=True)

    all_ret=[]
    for i in range(block_num):
        input_data,ret=sub_encoder(input_data,i,num_head)
        all_ret.append(ret)

        #下面我们进行全连接，我个人理解是是将维度特征更加的通用化
        # input_data=ffc(input_data,i,[81*3,81])
        input_data = ffc(input_data, i, [40, 81])

    return input_data,all_ret

def ffc(input_data,block_num,num_units):
    queries=input_data
    with tf.variable_scope('encoder_block_ffc'+str(block_num),reuse=tf.AUTO_REUSE):
        input_data=tf.layers.dense(input_data,num_units[0],tf.nn.leaky_relu,kernel_initializer=tf.random_normal_initializer(stddev=0.1))
        input_data=tf.layers.dense(input_data,num_units[1],tf.nn.leaky_relu,kernel_initializer=tf.random_normal_initializer(stddev=0.1))

        input_data+=queries

        input_data=ln(input_data)

        return input_data


def sub_encoder(input_data,block_num,num_head=9):
    with tf.variable_scope('encoder_block' + str(block_num),reuse=tf.AUTO_REUSE):
        # --------------------------------------------------------------------隔离线，原始的多头注意力------------------------------------------------------------------------------------------------
        Q_ = tf.layers.dense(input_data, 81, use_bias=True,kernel_initializer=tf.random_normal_initializer(stddev=0.1),activation=tf.nn.leaky_relu)
        K_ = tf.layers.dense(input_data, 81, use_bias=True,kernel_initializer=tf.random_normal_initializer(stddev=0.1),activation=tf.nn.leaky_relu)
        V_ = tf.layers.dense(input_data, 81, use_bias=True,kernel_initializer=tf.random_normal_initializer(stddev=0.1),activation=tf.nn.leaky_relu)
        # 下面我们进行多头注意力机制

        q = tf.concat(tf.split(Q_, num_or_size_splits=num_head, axis=-1), axis=0)
        k = tf.concat(tf.split(K_, num_or_size_splits=num_head, axis=-1), axis=0)
        v = tf.concat(tf.split(V_, num_or_size_splits=num_head, axis=-1), axis=0)

        # 下面我们开始进行点积操作
        d_k = q.get_shape().as_list()[-1]
        outputs = tf.matmul(q, tf.transpose(k, [0, 2, 1]))
        outputs /= d_k ** 0.5

        outputs = tf.nn.softmax(outputs)
        ret1=outputs
        # outputs = tf.transpose(outputs, [0, 2, 1])

        outputs = tf.matmul(outputs, v)


        outputs = tf.concat(tf.split(outputs, num_or_size_splits=num_head, axis=0), axis=2)

        outputs += input_data
        # outputs = ln(outputs)

        return -outputs,ret1