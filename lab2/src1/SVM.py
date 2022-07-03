import numpy as np
from cvxpy import *
from utils import *

class SupportVectorMachine:
    def __init__(self, C=1, kernel='Linear', epsilon=1e-4):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel

        # Hint: 你可以在训练后保存这些参数用于预测
        # SV即Support Vector，表示支持向量，SV_alpha为优化问题解出的alpha值，
        # SV_label表示支持向量样本的标签。
        # self.SV = None
        # self.SV_alpha = None
        self.w = None
        self.b = None
        # self.SV_label = None

    def KERNEL(self, x1, x2, d=2, sigma=1):
        #d for Poly, sigma for Gauss
        if self.kernel == 'Gauss':
            K = np.exp(-(np.sum((x1 - x2) ** 2)) / (2 * sigma ** 2))
        elif self.kernel == 'Linear':
            K = np.dot(x1,x2)
        elif self.kernel == 'Poly':
            K = (np.dot(x1,x2) + 1) ** d
        else:
            raise NotImplementedError()
        return K
    
    def fit(self, train_data, train_label):
        '''
        TODO：实现软间隔SVM训练算法
        train_data：训练数据，是(N, 7)的numpy二维数组，每一行为一个样本
        train_label：训练数据标签，是(N,)的numpy数组，和train_data按行对应
        '''
        numAttributes = len(train_data[0])
        numTrainData = len(train_data)
       # print('numTrainData=',numTrainData)
       # print('C=',self.C)
        # alpha为alpha1,alpha2,...
        alpha=Variable(numTrainData)
        
        constraints=[]
        for i in range(numTrainData):
            constraints += [alpha[i]>=0,alpha[i]<=self.C]
            #print(constraints)
        temp = 0
        for i in range(numTrainData):
            temp += alpha[i]*int(train_label[i])
        constraints += [temp==0]
        # print(constraints)
        # w为行向量
        self.w = np.zeros((1,numAttributes))
        
        
        # y为列向量
        y = train_label.reshape(-1,1)
        y_T = y.reshape(1,-1)
        # Yij即为yi*yj
        Y = np.dot(y,y_T)
        K = np.zeros((numTrainData,numTrainData))
        for i in range(numTrainData):
            #print('i=',i)
            x_i=train_data[i]
            for j in range(numTrainData):
                x_j=train_data[j]
                K[i][j] = self.KERNEL(x_i,x_j)
        #print('K=',K)
        
        const = 0.5*quad_form(alpha,Y*K)-sum(alpha)
        obj = Minimize(const)
        prob = Problem(obj,constraints)
        prob.solve(solver=ECOS)
        #print('status:',prob.status)
        #print('optimal value:',prob.value)
        #print('optimal alpha:',alpha.value)
        for i in range(numTrainData):
            self.w += alpha.value[i]*train_data[i]*int(train_label[i])
        #print('w=',self.w)
        result = 0
        for i in range(numTrainData):
            y_i = train_label[i]
            result += y_i
            for j in range(numTrainData):
                result -= alpha.value[j]*int(train_label[j])*K[i][j]
                
        #self.b=train_label[0]
        #for j in range(numTrainData):
        #    self.b -= alpha.value[j]*int(train_label[j])*K[0][j]
        #print('b=',self.b)
        self.b = result/numTrainData
        
    def predict(self, test_data):
        '''
        TODO：实现软间隔SVM预测算法
        train_data：测试数据，是(M, 7)的numpy二维数组，每一行为一个样本
        必须返回一个(M,)的numpy数组，对应每个输入预测的标签，取值为1或-1表示正负例
        '''
        result_labels=[]
        for i in range(len(test_data)):
            data = test_data[i]
            predict = np.sum(self.w * data) + self.b
            if(predict>=0):
                result_labels.append(1)
            else:
                result_labels.append(-1)
        result_labels=np.array(result_labels)
        return result_labels
if __name__ == '__main__':
    train_features, train_labels, test_features, test_labels=load_svm_dataset()
    model = SupportVectorMachine(1, 'Poly', 1e-4)
    model.fit(train_features,train_labels)
    result_labels = model.predict(test_features)
    correct = 0
    for i in range(len(result_labels)):
        if result_labels[i]==test_labels[i]:
            correct += 1
    print(correct/len(test_labels))