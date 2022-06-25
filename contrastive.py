import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 以下的data都是二维数据
def contrastive_aug(data):

    # batch,channel,point=data.shape
    # data=np.reshape(data,[-1,point])
    # scaler=StandardScaler()
    # data=scaler.fit_transform(data)
    # data=np.reshape(data,[batch,channel,point])
    data=data/350.0
    ret=[]
    for i in range(len(data)):
        # 第一种判断扩展幅值
        data1=extend_amp(data[i])
        #第二种左移右移
        data2=leftAndRight(data[i])
        # 第三种是加随机噪声
        data3=randomNoise(data[i])
        # 第四种是进行峰值重叠
        data4=overLap(data[i])
        # 第五种是进行基线飘移
        data5=baseFloat(data[i])

        temp=np.array([data1,data2,data3,data4,data5,data[i]])
        ret.append(temp)

    return np.array(ret)


def overLap(data):
    # 这个是单通道的
    ret=data.copy()
    channel,point=data.shape
    index1=np.random.randint(0,channel,1)
    index2=np.random.randint(0,channel,1)
    # index2 = 1
    scale=np.random.uniform(0.6,0.9,1)

    temp = data[index2, :].copy() * scale
    if np.random.randint(0,2,1)==0:
        #向左移动
        left=np.random.randint(15,23,1)[0]
        temp[0:-left]=temp[left:]
    else:
        right=np.random.randint(25,35,1)[0]
        temp[right:]=temp[0:point-right]

    ret[index1,:]+=temp
    return ret

def baseFloat(data):
    t=np.random.uniform(-0.05,0.05,1)
    temp = data.copy()
    temp+=t
    return temp

def setZero(data):

    channel,point=data.shape
    temp=data.copy()
    choice=np.random.choice(channel,1,replace=False)
    amp = np.random.uniform(0.95, 1.05)
    temp[choice,:]=0
    temp+=np.random.normal(0, 0.01, temp.shape)

    return temp

def randomNoise(data):

    temp = data.copy()
    temp *= np.random.uniform(0.9, 1.1, temp.shape)

    return temp

def leftAndRight(data):
    left=np.random.randint(0,2)
    movement=np.random.randint(0,4)
    temp = data.copy()
    if movement!=0:
        if left==1:
            temp[:,0:-movement]=temp[:,movement:]
            temp[:,-movement:]=temp[:,-movement].copy().reshape([-1,1])
        else:
            temp[:,movement:]=temp[:,0:-movement]
            temp[:,0:movement]=temp[:,0:1]

    temp += np.random.normal(0, 0.05, temp.shape)

    return temp


def extend_amp(data):
    #0.8-0.9，0.9-1.1，1.1-1.2的概率分别为0.1，0.8，0.1
    odds=np.random.randint(10)
    if odds==0:
        amp=np.random.uniform(0.9,0.95)
    elif odds==1:
        amp=np.random.uniform(1.05,1.10)
    else:
        amp=np.random.uniform(0.95,1.05)

    temp=data*amp
    # temp+=np.random.normal(0,0.05,temp.shape)

    return temp

if __name__=='__main__':
    with open('createData/allData.pkl', 'rb') as f:
        data=pickle.load(f)
    t=contrastive_aug(data[0:10])
    for row in range(5):
        for col in range(10):
            plt.subplot(5,10,row*10+col+1)
            plt.ylim([-1,1])
            plt.plot(t[row][col])

    plt.show()