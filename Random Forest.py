from numpy import *
import csv

#数据处理 ，读取原训练集数据，将字符型数据转为ASCII码，返回一个矩阵
def readcsv(file,num):
    w=csv.reader(open(file)) #读取文件
    l=[];k=0
    for line in w:         #读取文件中指定数量的数据
        l.append(line)
        k+=1
        if k==num+1:      #读到指定数据量退出
            break
    l.remove(l[0])        #将第一行feature量剔除
    s1=[];s2=[]
    for line in l:       #遍历每一行
        for i in line:
            try:
                s2.append(int(i))
            except:
                s2.append(ord(i)) #当碰到字符时将字符转换为ASCII码
        s2.remove(s2[0])          #将每一行数据第一个index量剔除
        s1.append(s2);s2=[]
    return mat(s1)              #返回数据矩阵

#dataset=readcsv()

#根据最优划分将数据集划分为两部分，以feature下的value划分
def spiltdataset(dataset,feature,value):
    mat0=[];mat1=[]
    l=dataset[:,feature].T.tolist()[0]  #标签为feature的那一列，放入列表中
    for i in range(len(l)):        #遍历每一个value
        if l[i]>value:             #大于value的那一行放入mat0中
            mat0.append(dataset[i,:].tolist()[0])
        else:                      #小于value的那一行放入mat1中
            mat1.append(dataset[i,:].tolist()[0])
    return mat(mat0),mat(mat1)

#求均值
def regleaf(dataset):
    return mean(dataset[:,0])

#求方差
def regerr(dataset):
    return var(dataset[:,0])*dataset.shape[0]

#找到最好的划分特征与值，传入数据矩阵，求均值函数，和求方差函数
def choosebestspilt(dataset,leaftype,errtype):
    if len(set(dataset[:,0].T.tolist()[0]))==1: #判断数据集的第一行得分是否相同，相同则返回平均值
        return None,leaftype(dataset)
    m,n=shape(dataset)
    S=errtype(dataset)     #原数据集评分的方差，
    bests=inf;bestindex=0;bestvalue=0  #初始化最好的划分特征以及对应的值，划分两部分数据集得分方差
    for featindex in range(1,n):  #遍历数据集每一个特征
        for splitvalue in set(dataset[:,featindex].T.tolist()[0]): #遍历每一特征下的值（不重复）
            mat0,mat1=spiltdataset(dataset,featindex,splitvalue)   #以此划分为两部分，
            if mat0.shape[0]<20 or mat1.shape[0]<20:         #当两部分数据个数小于20无需划分
                continue
            news=errtype(mat0)+errtype(mat1)                #当前划分下的方差
            if news<S:                                     #比较此方差与上一个划分结果的方差，
                bestindex=featindex                        #若小的话则将其设为当前最优划分
                bestvalue=splitvalue
                bests=news
    mat0,mat1=spiltdataset(dataset,bestindex,bestvalue)
    if mat0.shape[0]<20 or mat1.shape[0]<20:               #决策树叶子的最大数据集个数为20
        return None,leaftype(dataset)
    return bestindex,bestvalue                            #返回最优的划分标签与划分值

#创建树
def creattree(dataset,leaftype,errtype):
    feature,value=choosebestspilt(dataset,leaftype,errtype)  #得到当前数据集下的最优划分与划分值
    if feature==None:                        #如果划分无划分特征，则设为叶子
        return value
    regtree={}
    regtree["spInd"]=feature
    regtree["spvalue"]=value
    le,ri=spiltdataset(dataset,feature,value)
    regtree["left"]=creattree(le,leaftype,errtype) #递归决策二叉树左子树
    regtree["right"]=creattree(ri,leaftype,errtype)#递归决策二叉树右子树
    return regtree


#计算树的层数
def gettreedepth(regtree):
    if isinstance(regtree,float):
        return 0
    else:
        i= gettreedepth(regtree["right"])
        j= gettreedepth(regtree["left"])
        if i<j:
            return j+1
        else:
            return i+1

#计算叶节点个数
def getnumleafs(regtree):
    if isinstance(regtree, float):
        return 1
    return getnumleafs(regtree["left"])+getnumleafs(regtree["right"])


#回归测试函数
def classify(regtree,testvec):
    if not isinstance(regtree, float):  #判断当前树节点是否为叶子，递归条件
        featureindex=regtree["spInd"]   #
        spvalue=regtree["spvalue"]
        left=regtree["left"]
        right=regtree["right"]
        if(testvec[featureindex])>spvalue:  #测试数据集当前值是否大于最优划分值
            classlabel=classify(left,testvec) #递归到左子树中去，最终返回叶子的值
        else:
            classlabel=classify(right,testvec)#递归到右子树中去，最终返回叶子的值
    else:
        classlabel=regtree             #递归结束
    return classlabel

#读取训练集数据、测试集数据
fileanswer="D:/Python/CARTreg/submission.csv"
filetest="D:/Python/CARTreg/test.csv"
filetrain="D:/Python/CARTreg/train.csv"
ans=readcsv(fileanswer,2000)#读取submission中数据
answer=ans.tolist()
datasettest=array(readcsv(filetest,2000))
datasettest=insert(datasettest, 0, values=0, axis=1)
datasettrain=array(readcsv(filetrain,32000))
regtree=[]
trainset=[]#每个决策树训练集
numtree=50 #每个森林决策树个数为，
sum=0

for i in range (numtree):#森林部分，每个森林训练了50个决策树
    a = [int(random.uniform(0, datasettrain.shape[0]-1)) for i in range(datasettrain.shape[0])]
    #产生一个0~32000重复随机数的列表
    for i in range(len(a)):
        trainset.append(datasettrain[a[i]])  #随机森林每一棵树的训练集有放回抽样
    regtree.append(creattree(mat(trainset),regleaf,regerr))#将每个决策树放入 regtree列表中
    trainset=[]
#测试
for i in range(2000):  #遍历每一个测试集数据
    sumtemp=0
    for j in range(50):
        testvector= datasettest[i]
        temp=classify(regtree[j], testvector) #测试集数据在每一棵树下的回归结果
        sumtemp+=temp
    err=(answer[i][0]-sumtemp/50) # 每一个数据得分与原数据之差
    err2=power(err,2)
    sum+=err2
err3=sum/2000
err4=power(err3,0.5)
print(err4)    #最终结果
