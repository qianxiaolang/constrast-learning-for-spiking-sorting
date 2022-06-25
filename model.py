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

from myUtils import ln

from contrastive import contrastive_aug

def iic_loss(px, qx):
    k = px.shape.as_list()[1]
    px = tf.tile(px, [5, 1])

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

def branch_4(data):
    with tf.variable_scope('branch_4', reuse=tf.AUTO_REUSE):
        data = tf.nn.softmax(
            tf.layers.dense(data, 4, kernel_initializer=tf.random_normal_initializer(stddev=0.1)))

    return data

def get_loss_A(row_data,aug_data):
    px=branch_4(row_data)
    qx=branch_4(aug_data)
    loss=iic_loss(px,qx)

    return loss

def get_loss_B(row_data, aug_data):
    px = branch_2(row_data)
    qx = branch_2(aug_data)
    loss = iic_loss(px, qx)

    return loss


# def eveluate(row_data,data, label, sess):
#     table = np.zeros([2, 20], dtype=np.int32)
#     p = branch_2(row_data)
#     p = tf.cast(tf.argmax(p, axis=1), dtype=tf.int32)
#     p_eval,label_eval = sess.run([p,label],feed_dict={})
#     for j in range(len(label_eval)):
#         table[p_eval[j]][label_eval[j]] += 1
#
#     return table

def body(data):
    data1 = data[:, 5, :, :]
    data2 = data[:, 0, :, :]
    data3 = data[:, 1, :, :]
    data4 = data[:, 2, :, :]
    data5 = data[:, 3, :, :]
    data6 = data[:, 4, :, :]

    # 下面我们开始主体框架
    # 首先是原样本进行主题框架
    data1, ret_map = encoder(data1)
    data2, _ = encoder(data2)
    data3, _ = encoder(data3)
    data4, _ = encoder(data4)
    data5, _ = encoder(data5)
    data6, _ = encoder(data6)

    # 下面开始构建分支头
    data1 = tf.reshape(data1, [-1, 10 * 81])

    data2 = tf.reshape(data2, [-1, 10 * 81])
    data3 = tf.reshape(data3, [-1, 10 * 81])
    data4 = tf.reshape(data4, [-1, 10 * 81])
    data5 = tf.reshape(data5, [-1, 10 * 81])
    data6 = tf.reshape(data6, [-1, 10 * 81])

    aug_data = tf.concat([data2, data3, data4, data5,data6], axis=0)

    row_data = branch(data1)
    aug_data = branch(aug_data)

    loss_A = get_loss_A(row_data, aug_data)
    with tf.variable_scope('loss_4'):
        op_A=tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss_A)

    loss_B = get_loss_B(row_data, aug_data)
    with tf.variable_scope('loss_2'):
        op_B = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss_B)

    return row_data,loss_A,op_A,loss_B,op_B

def modelFuc(sess,path='m',train_test=True):
    with tf.variable_scope(path,reuse=tf.AUTO_REUSE):
        alter=1
        if train_test:
            if len(path) == 1:
                alter=2
                repeat_num=50
                batch_size=1024
            else:
                repeat_num=40
                alter=2
                batch_size=1024
        else:
            repeat_num = 1


        if train_test:
            # 进行初始训练数据迭代器
            # batch_size = 1024
            data_holder=tf.placeholder(dtype=tf.float32,shape=(None,6,10,81))
            # label_holder=tf.placeholder(dtype=tf.int32,shape=(None))

            res,loss_4,op_4, loss, op = body(data_holder)

            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.trainable_variables())

            b = 0
            start = datetime.datetime.now()
            with open('middle/'+path+'.pkl','rb') as f:
                real_label=pickle.load(f)

            with open('middle/'+path+'d.pkl','rb') as f:
                real_data=pickle.load(f)

            if len(real_label)<=5000:
                repeat_num=40
                alter=10
                batch_size=512

            for i in range(repeat_num):
                index=np.arange(len(real_label))
                np.random.shuffle(index)
                real_data=real_data[index]
                real_label=real_label[index]

                for j in range(len(real_label)//batch_size+1):
                    data=contrastive_aug(real_data[j*batch_size:(j+1)*batch_size])
                    if alter==0:
                        loss_eval, _ = sess.run([loss, op], feed_dict={data_holder: data})
                        print('当前二分类损失值为：', loss_eval)
                    else:
                        if (b // alter) % 2 == 0:
                            loss_eval, _ = sess.run([loss, op], feed_dict={data_holder:data})
                            print('当前二分类损失值为：', loss_eval)
                        else:
                            loss_eval, _ = sess.run([loss_4, op_4], feed_dict={data_holder:data})
                            print('当前四分类损失值为：', loss_eval)

                    if b%50==0:
                        # acc_table = eveluate(res, data,real_label[j*batch_size:(j+1)*batch_size], sess)
                        label=real_label[j*batch_size:(j+1)*batch_size]
                        table = np.zeros([2, 20], dtype=np.int32)
                        p = branch_2(res)
                        p = tf.cast(tf.argmax(p, axis=1), dtype=tf.int32)
                        p_eval = sess.run(p, feed_dict={data_holder:data})
                        for j in range(len(label)):
                            table[p_eval[j]][label[j]] += 1

                        print(table)

                    b+=1


            end = datetime.datetime.now()
            print('程序运行时间：', end - start)
            print("当前节点的训练已经结束,训练了",b,'次')
            #     下面我们进行评价
            all_start=datetime.datetime.now()
            index = np.arange(len(real_label))
            np.random.shuffle(index)
            real_data = real_data[index]
            real_label = real_label[index]

            left_label=[]
            left_data=[]
            right_label=[]
            right_data=[]
            for j in range(len(real_label) // 1024 + 1):
                temp=real_data[j * 1024:(j + 1) * 1024]
                data = contrastive_aug(temp)
                # result, _, _, _, _ = body(data_holder)
                result = branch_2(res)
                result = tf.cast(tf.argmax(result, axis=1), dtype=tf.int32)
                result_eval=sess.run(result,feed_dict={data_holder:data})
                label=real_label[j*1024:(j+1)*1024]

                with tf.device('/cpu:0'):
                    start = datetime.datetime.now()
                    for k in range(len(result_eval)):
                        if result_eval[k]==0:
                            left_label.append(label[k])
                            left_data.append(temp[k])
                        else:
                            right_label.append(label[k])
                            right_data.append(temp[k])

                    end=datetime.datetime.now()
                    print("程序计算label的时间：",end-start)

            with tf.device('/cpu:0'):
                start=datetime.datetime.now()
                left_label = np.array(left_label, dtype=np.int)
                left_data=np.array(left_data)
                right_label = np.array(right_label, dtype=np.int)
                right_data=np.array(right_data)

                #   其中True代表是叶子节点，False代表的是非叶子节点
                with open('middle/' + path + 'r.pkl', 'wb') as f:
                    pickle.dump(right_label, f)

                with open('middle/'+path+'rd.pkl','wb') as f:
                    pickle.dump(right_data,f)

                with open('middle/' + path + 'l.pkl', 'wb') as f:
                    pickle.dump(left_label, f)

                with open('middle/'+path+'ld.pkl','wb') as f:
                    pickle.dump(left_data,f)

                end=datetime.datetime.now()
                print("程序保存时间：",end-start)

                all_end=datetime.datetime.now()
                print("程序整个保存时间为：",all_end-all_start)

                if len(left_label) < 2500:
                    if len(right_label) < 2500:
                        return True, True
                    else:

                        return True, False
                else:
                    if len(right_label) < 2500:
                        return False, True
                    else:
                        return False, False

                return True, True


        else:
            print(666)
