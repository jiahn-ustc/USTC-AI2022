from cvxpy import length
import numpy as np
import math
from utils import load_decisiontree_dataset



class Node(object):
    def __init__(self,item):
        # item表示节点类型：root，middle,leaf
        self.item = item
        self.children = []
        # branchValue为分支属性的值
        self.branchValue = []
        # branchIndex为分支属性，只有一个属性,范围为[0,len(train_features[0])-1]
        self.branchIndex = None
        # value表示叶节点的预测值
        self.value = None
    def __str__(self):
        return str(self.item)
    def add(self,item):
        node = Node(item)
        self.children.append(node)
        return node
    def node_print(self):
        # print('开始打印')
        if(self.item == 'middle'):
            print('分支属性:',self.branchIndex)
            for child in self.children:
                if (child.item == 'middle'):
                    print('子节点分支属性:',child.branchIndex)
                else:
                    print('子节点预测值:',child.value)
        for child in self.children:
            child.node_print()


class TrainSets(object):
    def __init__(self,train_features,train_labels):
        self.train_features = train_features
        self.train_labels = train_labels

def getI(a,b):
    if a==0 and b!=0:
        return -b*math.log(b,2)
    if a!=0 and b==0:
        return -a*math.log(a,2)
    return -a*math.log(a,2)-b*math.log(b,2)

def np_count(nparray, x):
    i = 0
    for n in nparray:
        if n == x:
            i += 1
    return i

def is_equal(D,A):
    temp = []
    for index in A:
        temp.append(D.train_features[0][index])
    for i in range(len(D.train_features)):
        t = []
        for index in A:
            t.append(D.train_features[i][index])
        if t!=temp:
            return False
    return True
        

def getOptimal(D,A):
    max_v = 0
    max_index = 0
    X_i= []
    P_i = []
    N_i = []
    p = np_count(D.train_labels,1)
    n = np_count(D.train_labels,0)
    I=getI(p/(p+n),n/(p+n))
    start = 0
    for index in range(len(A)):
        attr = A[index]
        v = I
        # a为attr属性的取值集合,取值范围为[0,10]
        a = []
        for i in range(len(D.train_features)):
            a.append(int(D.train_features[i][attr]))
        # x[i]表示取值为i的个数
        # Positives[i]表示在取值为i中正样本的个数
        # Negatives[i]表示在取值为i中负样本的个数
        x = [0]*11
        Positives = [0]*11
        Negatives = [0]*11
        for i in range(len(a)):
            x[a[i]] += 1
            if(D.train_labels[i]==1):
                Positives[a[i]] += 1
            else:
                Negatives[a[i]] += 1
        #print('index=',index)
        #print('x=',x)
        #print('Positives=',Positives)
        #print('Negatives=',Negatives)
        for i in range(len(x)):
            if x[i]!=0:
                v -= x[i]/(p+n)*getI(Positives[i]/(Positives[i]+Negatives[i]),Negatives[i]/(Positives[i]+Negatives[i]))  
        #print('v=',v)
        if start==0:
            max_v = v
            max_index = index
            start = 1
            X_i = x
            P_i = Positives
            N_i = Negatives
        else:
            if(max_v<v):
                max_v = v
                max_index = index
                X_i = x
                P_i = Positives
                N_i = Negatives
    return A[max_index],X_i,P_i,N_i

def TreeGenerate(D,A,T):
    node = Node("middle")
    result1 = np.asarray([0]*len(D.train_labels))
    result2 = np.asarray([1]*len(D.train_labels))
    if (D.train_labels==result1).all():
        node.item="leaf"
        node.value = 0
        return node
    if (D.train_labels==result2).all():
        node.item="leaf"
        node.value = 1
        return node
    if len(A)==0 or is_equal(D,A)==True:
        node.item="leaf"
        numN=np_count(D.train_labels,0)
        numP=np_count(D.train_labels,1)
        if numN>numP:
            node.value = 0
        else:
            node.value = 1
        return node
    
    
    # attr为最优划分属性，根据最大的信息增益，即为trainFeatures中的下标(即第几个)
    # x为取值范围，为0到10，x[i]表示取值为i的个数
    attr,x,P,N = getOptimal(D,A)
    #print('最优划分属性为:',attr)
    
    
    # 计算划分前的准确率
    numN=np_count(D.train_labels,0)
    numP=np_count(D.train_labels,1)
    if numN>numP:
        resultPredict = 0
    else:
        resultPredict = 1
    numDivide = len(T.train_labels)
    correctDivide = 0
    for i in range(len(T.train_labels)):
        if(T.train_labels[i]==resultPredict):
            correctDivide += 1
    if numDivide!=0:
        Acc1 = correctDivide/numDivide
        #print('划分前的准确率:',Acc1)
    
    # 计算划分后的正确率
    # Predictions[i]为attr为i时的预测值
    Predictions = []
    for i in range(len(x)):
        if(x[i]!=0):
            if(P[i]>N[i]):
                Predictions.append(1)
            else:
                Predictions.append(0)
        else:
            Predictions.append(resultPredict)
    correctNoDivide = 0
    for i in range(len(T.train_labels)):
        test_features = T.train_features[i]
        test_labels = T.train_labels[i]
        if(test_labels==Predictions[test_features[attr]]):
            correctNoDivide += 1
    if numDivide!=0:
        Acc2 = correctNoDivide/numDivide
        #print('划分前的准确率:',Acc2)

    # 若划分前正确率更高，则不划分
    # 第一层禁止剪枝
    if(numDivide!=0 and Acc1>Acc2 and len(A)<=8 ):
        #print('不进行划分,attr=',attr)
        node.item="leaf"
        node.value = resultPredict
        return node

    # 开始划分
    node.branchIndex = attr
    for i in range(len(x)):
        # value为该属性的取值
        value = i
        node.branchValue.append(value)

        newTrainFeatures = []
        newTrainLabels = []
        for j in range(len(D.train_labels)):
            if D.train_features[j][attr]==value:
                newTrainFeatures.append(D.train_features[j])
                newTrainLabels.append(D.train_labels[j])
        newTrainFeatures = np.asarray(newTrainFeatures)
        newTrainLabels = np.asarray(newTrainLabels)

        newTestFeatures = []
        newTestLabels = []
        for j in range(len(T.train_labels)):
            if T.train_features[j][attr]==value:
                newTestFeatures.append(T.train_features[j])
                newTestLabels.append(T.train_labels[j])
        newTestFeatures = np.asarray(newTestFeatures)
        newTestLabels = np.asarray(newTestLabels)

        newA = A.copy()
        newA.remove(attr)
        newD = TrainSets(newTrainFeatures,newTrainLabels)
        newT = TrainSets(newTestFeatures,newTestLabels)
        
        node.children.append(TreeGenerate(newD,newA,newT))
    return node



class DecisionTree:
    def __init__(self):
        self.root = None
    def fit(self, train_features, train_labels):
        '''
        TODO: 实现决策树学习算法.
        train_features是维度为(训练样本数,属性数)的numpy数组
        train_labels是维度为(训练样本数, )的numpy数组
        '''
        # 按照3:1把训练集分割为训练集和测试集
        length = len(train_features)
        index = int(3/4*length)
        train_features_train = train_features[:index]
        train_features_test = train_features[index:]
        train_labels_train = train_labels[:index]
        train_labels_test = train_labels[index:]
        A = list(range(len(train_features[0])))
        D = TrainSets(train_features_train, train_labels_train)
        T = TrainSets(train_features_test, train_labels_test)
        self.root = TreeGenerate(D,A,T)
        #print('训练结束')
        #self.root.node_print()
        
        
        

    def predict(self, test_features):
        '''
        TODO: 实现决策树预测.
        test_features是维度为(测试样本数,属性数)的numpy数组
        该函数需要返回预测标签，返回一个维度为(测试样本数, )的numpy数组
        '''
        result_labels = []
        for line in test_features:
            node = self.root
            while node.item!="leaf":
                index = node.branchIndex
                node = node.children[line[index]]
            result_labels.append(node.value)
        result_labels = np.array(result_labels)
        return result_labels
        

# treenode: [attr, feat[attr] == 1, feat[attr] == 2, feat[attr] == 3]



if __name__ == '__main__':
    '''
    tree = Tree()
    tree.root.add(1)
    temp_node=tree.root.add(2)
    tree.root.add(278)
    s=temp_node.add(3)
    s.add(11451)
    temp_node2=temp_node.add(4)
    temp_node2.add(5)
    tree.tree_print(tree.root)
    '''
    train_features, train_labels, test_features, test_labels = load_decisiontree_dataset()
    model = DecisionTree()
    model.fit(train_features, train_labels)
    result_labels = model.predict(test_features)
    num = 0
    correct = 0
    for i in range(len(result_labels)):
        num += 1
        if(result_labels[i]==test_labels[i]):
            correct += 1
    print('train accuracy:',correct/num)

