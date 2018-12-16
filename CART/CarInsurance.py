import pandas as pd
import numpy as np
import math
# from sklearn.feature_extraction import DictVectorizer
# from sklearn import preprocessing
import cart_reg#回归树函数
import matplotlib.pyplot as plt
import seaborn as sns

# 训练集与测试集的提取
app_train = pd.read_csv('train.csv')
app_test = pd.read_csv('test.csv')
app_true = pd.read_csv('submission.csv')

#数据相关性分析，注：观测后关闭，以免影响程序进程
data = app_train.corr()
sns.heatmap(data)
plt.show()

#风险指数
y_true = app_true['Score']

#对train集进行处理：提取输出y即score，删除0，1 列
train_labels = app_train['Score']
print(type(np.mat(train_labels)))
train_del=[0,1]#,4,5,9,10,12,13,16,18,19,21,23,25,26,28,29,30,33
app_train.drop(app_train.columns[train_del],axis=1,inplace=True)

#对test集进行处理：处理方法同上，提出ID，删除0列
submit = app_test[['Id']]
test_del = [0]#,3,4,8,9,11,12,15,17,18,20,22,24,25,27,28,29,32
app_test.drop(app_test.columns[test_del],axis=1,inplace=True)

#类别特征处理
# dict_vec = DictVectorizer(sparse=False)
# app_train = dict_vec.fit_transform(app_train.to_dict(orient='record'))
# app_test  = dict_vec.fit_transform(app_test.to_dict(orient='record'))
# print(app_train.shape,app_test.shape)

#将训练集的输入与输出合并，y放在最后一列
x_train = np.mat(app_train)
# print(x_train.shape,np.mat(train_labels).T.shape)
x_train = np.hstack((x_train,np.mat(train_labels).T))
print(x_train)

#将测试集与输出合并，同上
test = np.mat(app_test)
y_true = np.mat(y_true).T
print(test.shape,y_true.shape)
testmat = np.hstack((test,y_true))

#输出数据集维数
print(x_train.shape,test.shape,testmat.shape)

#RegressionTree
myTree = cart_reg.createTree(x_train,ops=(3, 80))#创建树模型

myFinalTree = cart_reg.prune(myTree, testmat)#根据测试集对树进行剪枝
#出现merge说明发生剪枝
score_pre = cart_reg.createForeCast(myFinalTree, test)#预测数据

#取整
score_pre = np.floor(score_pre)

#calculate the RSME：
def RSME(pre,ture):
	loss=pre-ture
	cost = np.sum(loss**2)/(len(loss))
	result = math.sqrt(cost)
	return result
RSME_result = RSME(np.array(score_pre),np.array(y_true))
print(RSME_result)

#export the final date to the file:
submit['Score'] = score_pre
out_filename = 'submit'+str(RSME_result)+'.csv'
submit.to_csv(out_filename, index = False)
print('Over')

#plt the image of risk 
plt.plot(y_true,color='g',label="true")
plt.plot(score_pre,color='r',label="predict")
plt.xlabel("Id_num")
plt.ylabel("Risk")
plt.legend(loc="best")
plt.show()