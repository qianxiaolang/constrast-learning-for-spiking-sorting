import os
import tensorflow as tf
import pickle
from contrastive import contrastive_aug
import numpy as np

def load_data(path):
    path=str(path, encoding = "utf-8")
    with open('createData/allData.pkl', 'rb') as f:
        all_data=pickle.load(f)

    with open('createData/allLabel.pkl', 'rb') as f:
        all_label=pickle.load(f)

    try:
        with open('middle/'+path+'.pkl','rb') as f:
            choice=pickle.load(f)
    except:
        choice=np.arange(len(all_label))
        with open('middle/'+path+'.pkl','wb') as f:
            pickle.dump(choice,f)


    for i in range(len(choice)):
        data=contrastive_aug(all_data[choice[i]])
        # t=np.array(data.append(all_data[i]))
        yield data,all_label[choice[i]],choice[i]

def get_dataset(repeat_num,batch_size,path):
    dataset=tf.data.Dataset.from_generator(load_data,(tf.float32,tf.float32,tf.float32),output_shapes=([6,10,81],(),()),args=[path])
    dataset=dataset.repeat(repeat_num)
    dataset=dataset.shuffle(15*batch_size)
    dataset=dataset.batch(batch_size)

    return dataset

# dataset=get_dataset(1,512,"r")
# itera=dataset.make_one_shot_iterator()
# d=itera.get_next()
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     count=0
#
#     while True:
#         sess.run(d)
#         count += 1
#         print(count)
# if __name__=='__main__':
#     dataset=get_dataset(10,128)
#     with tf.Session() as sess:
#         itera=dataset.make_one_shot_iterator()
#         data=itera.get_next()
#         print(sess.run(data))

# with open('data/mouseData.pkl','rb') as f:
#     all_data=pickle.load(f)
#
# with open('data/mouseLabel.pkl','rb') as f:
#     all_label=pickle.load(f)
#
# print(999)

# 下面我们这里构建新的数据集
if __name__=='__main__':
    with open('createData/allLabel.pkl', 'rb') as f:
        all_label=pickle.load(f)
    print(len(all_label))
    # newMouseData=[]
    # newMouseLabel=[]
    # # t=[7,8,9,11,22,23,24,25,26,27,28]
    # t=[7,8,27,28]
    # label=0
    # for name in t:
    #     with open('data/Unit_'+str(name)+'.npy','rb') as f:
    #         data=np.load(f)
    #         newMouseData.append(data)
    #         newMouseLabel.append(np.full(len(data),label,dtype=np.float))
    #         label+=1
    #
    #
    # newMouseData=np.concatenate(newMouseData,axis=0)
    # newMouseLabel=np.concatenate(newMouseLabel,axis=0)
    #
    # index=np.arange(len(newMouseData))
    # np.random.shuffle(index)
    #
    # newMouseData=newMouseData[index]
    # newMouseLabel=newMouseLabel[index]
    #
    # with open('data/fourMouseData.pkl', 'wb') as f:
    #     pickle.dump(newMouseData,f)
    #
    # with open('data/fourMouseLabel.pkl','wb') as f:
    #      pickle.dump(newMouseLabel,f)