import pickle
import numpy as np
from model import modelFuc
import tensorflow as tf
import datetime


class node(object):
    left=None
    right=None
    value=-1
    filename='m'

def trainTreeAttention():
    head = root = node()
    que = []
    que.append(root)

    count = 0
    start = datetime.datetime.now()
    with tf.Session() as sess:
        while len(que) != 0:
            root = que[0]
            del que[0]

            # with tf.device('/gpu'):

            l_path, r_path=modelFuc(sess,root.filename, True)

            # l_path, r_path = modelFuc(root.filename, False)

            l_node = node()
            l_node.filename = root.filename + 'l'

            r_node = node()
            r_node.filename = root.filename + 'r'

            if l_path == True:
                l_node.value = count
                count += 1

                if r_path == True:
                    r_node.value = count
                    count += 1
                else:
                    que.append(r_node)

            else:
                que.append(l_node)

                if r_path == True:
                    r_node.value = count
                    count += 1
                else:
                    que.append(r_node)

            root.left=l_node
            root.right=r_node

    end = datetime.datetime.now()
    print('程序运行时间：', end - start)
    seq(head)

def testTreeAttention():
    head=anti_seq()
    l=[head]
    leafNode=[]

    while len(l)!=0:
        head=l[0]
        del l[0]
        if head.left==None and head.right==None:
            leafNode.append(head)
        else:
            l.append(head.left)
            l.append(head.right)


    table=np.zeros([len(leafNode),len(leafNode)],dtype=np.int)

    # with open('middle/m.pkl', 'rb') as f:
    #     label=np.array(pickle.load(f),dtype=np.int)

    count=0
    for leaf in leafNode:
        with open('middle/'+leaf.filename+'.pkl','rb') as f:
            modelLabel=pickle.load(f)

        for ele in modelLabel:
            table[count][ele]+=1
        count+=1

    print(table)


# 下面我们进行序列化
def seq(head):

    with open('middle/seq.pkl', 'wb') as f:
        pickle.dump(head,f)


def anti_seq():
    with open('middle/seq.pkl', 'rb') as f:
        head=pickle.load(f)

    return head

if __name__=='__main__':
    # with tf.device('/cpu'):
    trainTreeAttention()
    testTreeAttention()
# 程序运行时间： 0:11:04.200161，当前是transformer，准确率98.4
# [[   4    0    0    0    1    1    2 1981    1    0    0]
#  [   0    0    2    0    6 1988    4    0    0    0    1]
#  [   0    0    3    1    1    0    4    3 1997    0    0]
#  [  30   26    5    5 1951    0    5    0    0    7    0]
#  [1965   11   26   26   16    5   36    2    1    4    6]
#  [   0   24    0    0   11    1    0    2    1    0 1993]
#  [   0 1939    0    0   14    1    0    4    0    0    0]
#  [   0    0    0 1956    0    0    0    2    0    6    0]
#  [   0    0    0    1    0    0    0    6    0 1983    0]
#  [   1    0   13   11    0    2 1948    0    0    0    0]
#  [   0    0 1951    0    0    2    1    0    0    0    0]]