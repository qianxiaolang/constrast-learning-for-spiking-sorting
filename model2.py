import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from dataLoad import get_dataset
from modules import encoder
from modules import branch
from scipy.optimize import linear_sum_assignment
import pickle
import datetime
import seaborn as sns
from cvPre import cvTranspose
from myUtils import ln

def modelFuc(path='m',train_test=True):
    with tf.variable_scope(path,reuse=tf.AUTO_REUSE):
        if train_test:
            repeat_num = 15
        else:
            repeat_num = 1

        batch_size = 512
        dataset = get_dataset(repeat_num, batch_size, path)
        train = tf.data.Iterator.from_structure(output_types=dataset.output_types, output_shapes=dataset.output_shapes)
        train_init = train.make_initializer(dataset)
        data, label,index = train.get_next()

        # batch_size = 512
        # dataset = get_dataset(1, batch_size, path)
        # test = tf.data.Iterator.from_structure(output_types=dataset.output_types, output_shapes=dataset.output_shapes)
        # test_init = test.make_initializer(dataset)
        # data, label, index = test.get_next()


        data1 = data[:, 4, :, :]
        data2 = data[:, 0, :, :]
        data3 = data[:, 1, :, :]
        data4 = data[:, 2, :, :]
        data5 = data[:, 3, :, :]

        # 下面我们开始主体框架
        # 首先是原样本进行主题框架
        data1, ret_map = encoder(data1)
        data2, _ = encoder(data2)
        data3, _ = encoder(data3)
        data4, _ = encoder(data4)
        data5, _ = encoder(data5)

        # 下面开始构建分支头
        data1 = tf.reshape(data1, [-1, 32 * 81])

        data2 = tf.reshape(data2, [-1, 32 * 81])
        data3 = tf.reshape(data3, [-1, 32 * 81])
        data4 = tf.reshape(data4, [-1, 32 * 81])
        data5 = tf.reshape(data5, [-1, 32 * 81])

        # data1=tf.reduce_mean(data1,1)
        #
        # data2=tf.reduce_mean(data2,1)
        # data3=tf.reduce_mean(data3,1)
        # data4=tf.reduce_mean(data4,1)
        # data5=tf.reduce_mean(data5,1)

        aug_data = tf.concat([data2, data3, data4, data5], axis=0)

        row_data = branch(data1)
        aug_data = branch(aug_data)


        def iic_loss(px, qx):
            k = px.shape.as_list()[1]
            px = tf.tile(px, [4, 1])

            px = tf.transpose(px)

            p = px @ qx

            p = (p + tf.transpose(p)) / 2

            p = tf.clip_by_value(p, clip_value_min=1e-6, clip_value_max=tf.float32.max)

            p /= tf.reduce_sum(p)

            pc = tf.broadcast_to(tf.reshape(tf.reduce_sum(p, axis=1), [1, k]), [k, k])
            pr = tf.broadcast_to(tf.reshape(tf.reduce_sum(p, axis=0), [k, 1]), [k, k])

            loss = -tf.reduce_sum(p * (tf.math.log(p) - tf.math.log(pc) - tf.math.log(pr)))

            return loss

        def branch_2(data):
            with tf.variable_scope('branch_2', reuse=tf.AUTO_REUSE):
                data = tf.nn.softmax(
                    tf.layers.dense(data, 2, kernel_initializer=tf.random_normal_initializer(stddev=0.1)))

            return data

        def get_loss_B(row_data, aug_data):
            loss = 0

            px = branch_2(row_data)
            qx = branch_2(aug_data)
            temp = iic_loss(px, qx)
            loss += temp

            return loss

        def eveluate(row_data, label, sess):
            ret = []

            table = np.zeros([2, 20], dtype=np.int32)
            p = branch_2(row_data)
            p = tf.cast(tf.argmax(p, axis=1), dtype=tf.int32)
            label = tf.cast(label, dtype=tf.int32)
            p_eval, label_eval = sess.run([p, label])
            for j in range(len(label_eval)):
                table[p_eval[j]][label_eval[j]] += 1

            return table

        loss_B = get_loss_B(row_data, aug_data)
        op_B = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss_B)
        saver = tf.train.Saver(tf.trainable_variables())
        if train_test:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(train_init)

                b = 0

                while True:
                    try:
                        loss_eval, _ = sess.run([loss_B, op_B])
                        print('当前损失值为：', loss_eval)

                        if b % 50 == 0:
                            acc_table = eveluate(row_data, label, sess)
                            print(acc_table)

                        if b%50==0:
                            saver.save(sess,'model/'+path+'.ckpt')
                        b += 1
                    except:
                        break



            print("当前节点的训练已经结束")
        else:
            # t=tf.train.get_checkpoint_state('model/checkpoint')
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(train_init)
                saver.restore(sess,'model/'+path+'.ckpt')
                left=[]
                right=[]

                while True:
                    try:
                        result=branch_2(row_data)
                        result=tf.cast(tf.argmax(result, axis=1), dtype=tf.int32)
                        result_eval,index_eval=sess.run([result,index])
                        for i in range(len(result_eval)):
                            if result_eval[i]==0:
                                left.append(index_eval[i])
                            else:
                                right.append(index_eval[i])
                    except:
                        break

                left=np.array(left,dtype=np.int)
                right=np.array(right,dtype=np.int)

                #   其中True代表是叶子节点，False代表的是非叶子节点
                with open('middle/' + path + 'r.pkl', 'wb') as f:
                    pickle.dump(right, f)

                with open('middle/' + path + 'l.pkl', 'wb') as f:
                    pickle.dump(left, f)

                if len(left)<2500:
                    if len(right)<2500:
                        return True,True
                    else:

                        return True,False
                else:
                    if len(right)<2500:
                        return False,True
                    else:
                        return False,False


                return True,True


# modelFuc(path='m',train_test=False)