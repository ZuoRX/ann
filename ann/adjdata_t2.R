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

whole=read.xlsx("C:/Users/lenovo/Desktop/data/t2/adj_data_t2.xlsx",2,encoding="UTF-8",header = FALSE)
names(whole)[31]=c("species")
names(whole)
whole=as.data.frame(whole[,c(-1,-2)])




#-------------------------------第一步ANN--------------------------------------#

whole= whole[sample(1:nrow(whole),length(1:nrow(whole))),1:ncol(whole)]
wholeValues= whole[,-29]
wholeTargets = decodeClassLabels(whole[,29])
whole= splitForTrainingAndTest(wholeValues, wholeTargets, ratio=0.3)#只预测30%的数据
whole= normTrainingAndTestSet(whole)
model = mlp(whole$inputsTrain, whole$targetsTrain, size=5, learnFunc="Quickprop", 
            learnFuncParams=c(0.1, 2.0, 0.0001, 0.1),
            maxit=1000, inputsTest=whole$inputsTest, targetsTest=whole$targetsTest) 
predictions = predict(model,whole$inputsTest)
confusionMatrix(whole$targetsTest,predictions)

print(model)
summary(model)
print(predictions)
#预测结果
#     predictions
# targets  1  2
#   1      8 12
#   2      4 11

#只用了整体30%的数据预测
a_ann=16/(19+16)
print(a_ann)
#错误率0.4571429             原始是0.6170213



#-------------------------------第二步SVM---------------------------------------#
rm(list=ls())
cat("\014")  

whole=read.xlsx("C:/Users/lenovo/Desktop/data/t2/adj_data_t2.xlsx",2,encoding="UTF-8",header = FALSE)
names(whole)[31]=c("species")
names(whole)
whole=as.data.frame(whole[,c(-1,-2)])

x=whole[,-29]
y=whole[,29]#将species作为结果变量
svmt2=lssvm(species~.,data=whole)#下面lssvm是kernlab包的函数，可以自动估计核函数参数
print(svmt2)
pred=predict(svmt2,x)
table(pred,y)
#           y               Training error : 0.155172 
# pred   hard high
# hard   45    6
# high   12   53




#-----------------------------第三步autoencoder+SVM-----------------------------#
rm(list=ls())
cat("\014") 

auto<-read.xlsx("C:/Users/lenovo/Desktop/data/t2/autoencoder_adj1.xlsx",1,encoding="UTF-8")
auto=(auto)[,-1]

whole=read.xlsx("C:/Users/lenovo/Desktop/data/t2/adj_data_t2.xlsx",2,encoding="UTF-8",header = FALSE)
names(whole)[31]=c("species")

auto=cbind(auto,whole[31])

x=auto[,-29]
y=auto[,29]#将species作为结果变量
svm_auto_t2=lssvm(species~.,data=auto)#下面lssvm是kernlab包的函数，可以自动估计核函数参数
print(svm_auto_t2)
pred=predict(svm_auto_t2,x)
table(pred,y)
#           y            Training error : 0.206897 
# pred   hard high
# hard   43   10
# high   14   49





