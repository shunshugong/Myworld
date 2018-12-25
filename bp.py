
import numpy as np

def sigmoid(x): #激活函数，将得到的值归一化在某一范围内
    return 1/(1+np.exp(-x))
input1 = np.array([[0.35],[0.9],[0.58],[0.78]])
w1 = np.random.rand(3,4)
print('w1:',w1)
w2 = np.random.rand(2,3)
print('w2:',w2)
real = np.array([[0.5],[0.7]])
for i in range(100):
    output1 = sigmoid(np.dot(w1,input1)) #中间层
    output2 = sigmoid(np.dot(w2,output1)) #输出层
    cost = np.square(real-output2)/2 #计算误差
    dalta2 = output2*(1-output2)*(real-output2) #梯度
    dalta1 = output1*(1-output1)*w2.T.dot(dalta2) #梯度
    w2 = w2+dalta2.dot(output1.T) #更新w2
    w1 = w1+dalta1.dot(input1.T) #更新w1
    print(output1)
    print(output2)
    print(cost)
