rm(list=ls())
#基本包
library(rJava)
library(xlsx)     #读表
library(reshape)  #修改变量名称
#未知
library(lattice)
library(NLP)
library(SnowballC)
library(mvtnorm)
library(hdrcde)
library(locfit)
library(ash)
library(KernSmooth)
library(misc3d)
library(rgl)
library(ks)
library(sp)
library(grid)
library(vcd)
library(topicmodels)
library(rFerns)
library(ranger)
library(Boruta) 
library(lattice)
library(caret)
library(slam)
library(Matrix)
library(foreach)
library(glmnet)
library(dplyr)   #截取
#画图表
library(plotrix)
library(igraph)
library(ggplot2)
#基本统计信息
library(car)      #拟合和评价回归模型
library(gvlma)
library(leaps)    #全子集回归
library(nnet)
library(caret)
library(pastecs)
#颜色
library(RColorBrewer)
library(rainbow)
#画地图
library(ggmap)
library(maps)
library(mapdata)
library(maptools)
#支持向量机
library(e1071)
library(SVMMatch)
library(kernlab)
#主成分
library(pcaPP)
#文本挖掘
library(tm)
library(Rwordseg)
library(wordcloud)
library(wordcloud2)
#随机森林
library(randomForest)
#s神经网络
library(Rcpp)#与RSNNS关联
library(RSNNS)#涉及到神经网络中的其它拓扑结构和网络模型
#Stuttgart Neural Network Simulator（SNNS）是德国斯图加特大学开发的优秀神经网络仿真软件
library(nnet)#提供了最常见的前馈反向传播神经网络算法
library(AMORE)#提供了更为丰富的控制参数，并可以增加多个隐藏层
library(neuralnet)#提供了弹性反向传播算法和更多的激活函数形式
library(autoencoder)
library(deepnet)#实现了一些Deep Learning结构和Neural Network相关算法，
#包括BP，RBM训练，Deep Belief Net，Deep Auto-Encoder
#聚类分析
library(cluster)
library(MASS)
library(stats)  #hclust,kmeans等函数
library(fpc)
library(amap)

rm(list=ls())
cat("\014")  



#-----------------------------大样本总体数据------------------------------------#


#----------------------#
#训练70%，预测30%的数据
#----------------------#
rm(list=ls())
cat("\014")             #清空控制台，快捷键control+L


whole<-read.xlsx("C:/Users/lenovo/Desktop/data/t1/hard.xlsx",3,encoding="UTF-8")
whole= whole[sample(1:nrow(whole),length(1:nrow(whole))),1:ncol(whole)]
wholeValues= whole[,1:8]
wholeTargets = decodeClassLabels(whole[,9])
whole= splitForTrainingAndTest(wholeValues, wholeTargets, ratio=0.3)#只预测30%的数据
whole= normTrainingAndTestSet(whole)
model = mlp(whole$inputsTrain, whole$targetsTrain, size=5, learnFunc="Quickprop", 
            learnFuncParams=c(0.1, 2.0, 0.0001, 0.1),
            maxit=1000, inputsTest=whole$inputsTest, targetsTest=whole$targetsTest) 
predictions = predict(model,whole$inputsTest)
confusionMatrix(whole$targetsTest,predictions)

print(model)
summary(model)


#----------------------------------------------#
#训练一个时间界面数据，预测下一个时间界面的数据
#----------------------------------------------#


























#-----------------------------小样本交叉数据------------------------------------#




#--------------------#
#交叉数据训练准备工作#
#--------------------#
rm(list=ls())

ae<-read.xlsx("C:/Users/lenovo/Desktop/data/fuzzy/ae.xlsx",1,encoding="UTF-8")
sp=c(rep(1,10))                 #这里变量名称是sp
ae=cbind(ae,sp)[,-9]
aees<-read.xlsx("c:/users/lenovo/desktop/data/fuzzy/aees.xlsx",1,encoding="UTF-8")
sp=c(rep(-1,18))                #这里变量名称如果是sp1的话，就会导致下面的组合不起来
aees=cbind(aees,sp)[,-9]
all=rbind(ae,aees)
#这里只用到all的数据，可以把其他数据去掉
remove(ae,aees,sp)

#-------------# 
#重命名变量名称
#-------------#
names(all)=c("price","huan","zhenfu","totalMV","circulationMV","DPE","TTM","PBV","species")

#-----------------# 
#第一步：打乱行顺序
#-----------------#
all = all[sample(1:nrow(all),length(1:nrow(all))),1:ncol(all)]

#-----------------# 
#第二步：输入和输出
#-----------------#
#定义网络输入 
allValues= all[,1:8]
#定义网络输出，并将数据进行格式转换 !!!
allTargets = decodeClassLabels(all[,9])
print(allTargets)

#------------------------------# 
#第三步：划分训练和检验样本比例
#------------------------------#
all = splitForTrainingAndTest(allValues, allTargets, ratio=0.15)

#-----------------# 
#第四步：数据标准化
#-----------------#
#数据标准化有多种方法(z-normalization, min-max scale, etc…)，
#采用min-max scale方法，可以将数据映射到[0，1]区间。 
all = normTrainingAndTestSet(all)

#------------------# 
#第五步：执行mlp命令
#------------------#
#利用mlp命令执行前馈反向传播神经网络算法 
model = mlp(all$inputsTrain, all$targetsTrain, size=5, learnFunc="Quickprop", 
            learnFuncParams=c(0.1, 2.0, 0.0001, 0.1),
            maxit=10000, inputsTest=all$inputsTest, targetsTest=all$targetsTest) 
print(model)
summary(model)

#------------------# 
#第六步：模型预测
#------------------#
#利用上面建立的模型进行预测 
predictions = predict(model,all$inputsTest)
#生成混淆矩阵，观察预测精度 
confusionMatrix(all$targetsTest,predictions)









#-----------------------------用硬科技去预测高科技------------------------------------#
rm(list=ls())
cat("\014")  
hard<-read.xlsx("C:/Users/lenovo/Desktop/data/t1/hard.xlsx",1,encoding="UTF-8")
high<-read.xlsx("c:/users/lenovo/desktop/data/t1/high.xlsx",1,encoding="UTF-8")
#把硬科技作为训练集，高科技作为测试集，结果都是
























#-----------------------------R自带样本数据------------------------------------#



#--------#
#demo训练#
#--------#

#目的：神经网络R练手，为autoencoder准备
data(iris)
#下面的sample 函数类似随机抽样，将数据的--行--顺序打乱（注意看括号）
iris = iris[sample(1:nrow(iris),length(1:nrow(iris))),1:ncol(iris)]
#定义网络输入 
irisValues= iris[,2:5]
#定义网络输出，并将数据进行格式转换 !!!
irisTargets = decodeClassLabels(iris[,1])
print(irisTargets)
#从中划分出训练样本和检验样本 
iris = splitForTrainingAndTest(irisValues, irisTargets, ratio=0.15)
#数据标准化 
iris = normTrainingAndTestSet(iris)
print(iris)
#利用mlp命令执行前馈反向传播神经网络算法 
model = mlp(iris$inputsTrain, iris$targetsTrain, size=5, learnFunc="Quickprop", 
            learnFuncParams=c(0.1, 2.0, 0.0001, 0.1),
            maxit=100, inputsTest=iris$inputsTest, targetsTest=iris$targetsTest) 
print(model)
summary(model)
#利用上面建立的模型进行预测 
predictions = predict(model,iris$inputsTest)
#生成混淆矩阵，观察预测精度 
confusionMatrix(iris$targetsTest,predictions)






