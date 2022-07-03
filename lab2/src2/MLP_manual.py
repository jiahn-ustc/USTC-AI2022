import torch
import numpy as np
import math
#from scipy.special import softmax
from matplotlib import pyplot as plt

def tanh(x):
    return np.tanh(x)

def softmax(x):
    max_value = np.max(x);
    return np.exp(x-max_value)/np.sum(np.exp(x-max_value))


class MLP:
    def __init__(self,layer=[10,10, 8, 8, 4],activation='tanh',lr=0.01):
        # 初始化所需参数   
        if activation == 'tanh':
            self.activation=tanh
        # 每一个layers也为行向量
        self.layers = [np.zeros((layer[0],1))]
        self.weights = []
        self.lr = lr
        self.biases = []
        self.softmax = softmax
        for i in range(1,len(layer)):
            self.weights.append(np.random.random((layer[i],layer[i-1])))
            self.layers.append(np.zeros((layer[i],1)))
            self.biases.append(np.random.random((layer[i],1)))
        #print('weights=',self.weights)


    def forward(self, x):
        # 前向传播
        # 每一层的layer均为列向量
        self.layers[0] = x.reshape(-1,1)

        for i in range(1,len(self.layers)-1):
            self.layers[i] = self.activation(np.dot(self.weights[i-1],self.layers[i-1]) + self.biases[i-1])
        
        k = len(self.layers)-1
        self.layers[k] = softmax(np.dot(self.weights[k-1],self.layers[k-1]) + self.biases[k-1])

    def backward(self,y): # 自行确定参数表
        # 反向传播
        y_predict = self.layers[-1]
        # pre_gradient_neurons为列向量
        pre_gradient_neurons = y_predict - y.reshape(-1,1)
        gradientWeightList = []
        gradientBiasList = []
        # 从最后一层到第一层进行遍历
        for i in range(len(self.layers)-1,0,-1):
            
            # gradient_bias为列向量
            gradient_bias = pre_gradient_neurons
            # (-1,1)为列向量，(1,-1)为行向量
            gradient_weight = np.dot(gradient_bias,self.layers[i-1].reshape(1,-1))
            gradientWeightList.append(gradient_weight)
            gradientBiasList.append(gradient_bias)
            
            if(i>1):
                s_ = 1 - np.power(self.layers[i-1],2)
                pre_gradient_neurons = np.dot(self.weights[i-1].T,pre_gradient_neurons ) * s_
        j=0    
        for i in range(len(self.layers)-1,0,-1):
            self.weights[i-1] -= self.lr * gradientWeightList[j]
            self.biases[i-1] -= self.lr * gradientBiasList[j]
            j += 1
        

class torchNetwork(torch.nn.Module):
    def __init__(self):
        super(torchNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.tanh = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(10, 8)
        self.linear3 = torch.nn.Linear(8, 8)
        self.linear4 = torch.nn.Linear(8, 4)
        #self.softmax = torch.nn.Softmax(dim=1)
    def forward(self,x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.tanh(x)
        x = self.linear3(x)
        x = self.tanh(x)
        x = self.linear4(x)
        #x = self.softmax(x)
        return x


def train(mlp: MLP, epochs,  inputs, labels):
    '''
        mlp: 传入实例化的MLP模型
        epochs: 训练轮数
        lr: 学习率，在mlp中已规定
        inputs: 生成的随机数据
        labels: 生成的one-hot标签
    '''
    
    MyMlpLoss = []
    Index = []
    for j in range(epochs):
        print('{}th training in mlp progress'.format(j+1))
        loss = 0
        for i in range(len(inputs)):
            #print('i=',i)
            mlp.forward(inputs[i])
            mlp.backward(labels[i])
          
        for i in range(len(inputs)):
            mlp.forward(inputs[i])
            t = labels[i].tolist().index(1)
            loss += -math.log((mlp.layers[-1].flatten())[t])
        loss /= len(inputs)
        Index.append(j+1)
        MyMlpLoss.append(loss)
        

    return MyMlpLoss,Index


if __name__ == '__main__':
    # 设置随机种子,保证结果的可复现性
    np.random.seed(1)
    learningRate = 0.001
    mlp = MLP(lr=learningRate)
    # 生成数据
    inputs = np.random.randn(100, 10)

    # 生成one-hot标签
    labels = np.eye(4)[np.random.randint(0, 4, size=(1, 100))].reshape(100, 4)
    label_pytorch = []
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if (labels[i][j] == 1):
                label_pytorch.append(j)
                break
    # 训练模型
    epochs=30000

    MyMlpLoss = []
    Index = []
    
    MyMlpLoss,Index =train(mlp, epochs,inputs, labels)

    # 打印mlp的weight和bias
    print('MLP weights and biases:')
    print('w=',mlp.weights)
    print('b=',mlp.biases)
    
    # 测试MLP
    correct = 0
    #print('开始预测')
    for i in range(len(inputs)):
        mlp.forward(inputs[i])
        predict_i=mlp.layers[-1].flatten()
       # print('the input is:',inputs[i])
        #print('the predict is:', predict_i)
        #print('the correct is:', labels[i])
        t = labels[i].tolist().index(1)
        if predict_i.tolist().index(np.max(predict_i))==t:
         #   print('predict is correct')
            correct += 1
        #else:
          # print('predict is wrong')
        #print('\n')
    print('MLP Accuracy:', correct/len(inputs))
    
    # 训练torch模型
    
    pytorchLoss = []
    torch_mlp = torchNetwork()
    
    optimizer = torch.optim.Adam(torch_mlp.parameters(), lr=learningRate)
    
    for i in range(epochs):
        print('{}th training in torch mlp progress'.format(i+1))        
        torch_inputs = torch.from_numpy(inputs).float()
        torch_labels = torch.from_numpy(np.array(label_pytorch)).long()
        predict = torch_mlp(torch_inputs)
        optimizer.zero_grad()
        loss = torch.nn.CrossEntropyLoss()(predict, torch_labels)
        loss.backward()
        optimizer.step()
        pytorchLoss.append(loss.item())
    
    # 测试torch模型
    print('torch Accuracy:', torch.sum(torch.argmax(predict, dim=1) == torch_labels)/len(inputs))

    print('torch weights and biases:')
    #打印torch的weight和bias
    for name,param in torch_mlp.named_parameters():
        if name in ['linear1.weight','linear1.bias']:
            print(name,param.data.numpy())
        elif name in ['linear2.weight','linear2.bias']:
            print(name,param.data.numpy())
        elif name in ['linear3.weight','linear3.bias']:
            print(name,param.data.numpy())
        elif name in ['linear4.weight','linear4.bias']:
            print(name,param.data.numpy())

    
    # 绘制loss曲线
    plt.plot(Index,MyMlpLoss,color='r',label="MLP")
    plt.plot(Index,pytorchLoss,color='b',label="torch")
    plt.legend()
    #plt.plot(pytorchLoss,color='b')
    #plt.axis([0,epochs,0,2])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    #plt.title('LOSS-EPOCHS')
    plt.show()
    
           
    
    
    
    