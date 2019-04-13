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
library(psych)
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

hard<-read.xlsx("C:/Users/lenovo/Desktop/data/t2/hardt2.xlsx",1,encoding="UTF-8")
high<-read.xlsx("C:/Users/lenovo/Desktop/data/t2/hight2.xlsx",1,encoding="UTF-8")
whole=rbind(hard,high)
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
#   1      8 10
#   2     19 10

#只用了整体30%的数据预测
a_ann=29/(29+18)
print(a_ann)
#错误率0.6170213



#-------------------------------第二步SVM---------------------------------------#
rm(list=ls())
cat("\014")  

hard<-read.xlsx("C:/Users/lenovo/Desktop/data/t2/hardt2.xlsx",1,encoding="UTF-8")
high<-read.xlsx("C:/Users/lenovo/Desktop/data/t2/hight2.xlsx",1,encoding="UTF-8")
whole=rbind(hard,high)
names(whole)[31]=c("species")
names(whole)
whole=as.data.frame(whole[,c(-1,-2)])

x=whole[,-29]
y=whole[,29]#将species作为结果变量
svmt2=lssvm(species~.,data=whole)#下面lssvm是kernlab包的函数，可以自动估计核函数参数
print(svmt2)
pred=predict(svmt2,x)
table(pred,y)
#           y
# pred   hard high
# hard   56   15
# high   21   64

#-----------预测具体结果----------#
print(pred)
#--------挑选出预测错误的编号-----#
num=function(x,y){
  k=rep(0,156)
  for (i in 1:156){
    if(x[i]!=y[i])
      k[i]=i
  }
  return(k)
}
aaa=num(pred,whole[,29])
aaa
a=data.frame(aaa)
write.xlsx(a,"C:/Users/lenovo/Desktop/data/t2/prednum.xlsx","sheet 1")
#------计算正确和错误个数---------#
n=which(aaa!=0)
data.frame(n)
length(n)
#存在交叉部分19个，即预测错误36-19=17个，错误率17/156=
aa=17/156
print(aa)
#错误率为0.1089744


#-------------------------------第三步pca+SVM---------------------------------------#
rm(list=ls())
cat("\014")  

hard<-read.xlsx("C:/Users/lenovo/Desktop/data/t2/hardt2.xlsx",1,encoding="UTF-8")
high<-read.xlsx("C:/Users/lenovo/Desktop/data/t2/hight2.xlsx",1,encoding="UTF-8")
whole=rbind(hard,high)
names(whole)[31]=c("species")
whole=as.data.frame(whole[,c(-1,-2)])
#remove("hard","high")
#去除缺失值，之前标0
#whole=whole[c(-75,-76,-77,-153,-154,-155,-156),]
whole_a<-whole[-29]

#---------normalize-------------#
#标准化
#whole_b=scale(whole_a)
#归一化
center <- sweep(whole_a, 2, apply(whole_a, 2, min),"-") #在列的方向上减去最小值，不加‘-’也行
R <- apply(whole_a, 2, max) - apply(whole_a,2,min)      #算出极差，即列上的最大值-最小值
whole_c<- sweep(center, 2, R, "/")       #把减去均值后的矩阵在列的方向上除以极差向量

#------------pca----------#
pca <- prcomp(whole_c)
print(pca)

# pca$rotation#特征向量
# pca$sdev#贡献率就是指某个主成分的方差占全部方差的比重，
summary(pca)#选取的时候用累计贡献率
pca_data<-predict(pca)

#---------svm-------------#
x=pca_data[,1:5]
x<-as.matrix(x)
y=whole[,29]#将species作为结果变量
y<-as.character(y)
pca_svm<-cbind(x,y)
pca_svm<-as.data.frame(pca_svm)
svmt2=lssvm(y~.,data=pca_svm)#下面lssvm是kernlab包的函数，可以自动估计核函数参数
print(svmt2)

pred=predict(svmt2,pca_svm)
table(pred,y)
# y
# pred   hard high
# hard   30    2
# high   47   77

#           y
# pred   hard high
# hard   56   15
# high   21   64


#-----------预测具体结果----------#
print(pred)
#--------挑选出预测错误的编号-----#
num=function(x,y){
  k=rep(0,156)
  for (i in 1:156){
    if(x[i]!=y[i])
      k[i]=i
  }
  return(k)
}
aaa=num(pred,whole[,29])
aaa
a=data.frame(aaa)
write.xlsx(a,"C:/Users/lenovo/Desktop/data/t2/prednum_pca.xlsx","sheet 1")
#------计算正确和错误个数---------#
n=which(aaa!=0)
data.frame(n)
length(n)
#存在交叉部分20个，即预测错误49-20=29个，错误率29/156=
aa=29/156
print(aa)
#错误率为 0.1858974
























#----------------------#
#归一化
center <- sweep(whole_a, 2, apply(whole_a, 2, min),"-") #在列的方向上减去最小值，不加‘-’也行
R <- apply(whole_a, 2, max) - apply(whole_a,2,min)      #算出极差，即列上的最大值-最小值
whole_scale<- sweep(center, 2, R, "/")       #把减去均值后的矩阵在列的方向上除以极差向量

# fa<-fa.parallel(whole_a,fa="both",n.iter = 100,show.legend = FALSE,
#                 main = "Scree plot with parallel analysis") 
# pc<-principal(whole_b,nfactors = 1)
# summary(fa)
# print(fa)






#-----------------------------第三步autoencoder+SVM-----------------------------#
rm(list=ls())
cat("\014") 

auto<-read.xlsx("C:/Users/lenovo/Desktop/data/t2/autoencoder_1.xlsx",1,encoding="UTF-8")
auto=(auto)[,-1]

hard<-read.xlsx("C:/Users/lenovo/Desktop/data/t2/hardt2.xlsx",1,encoding="UTF-8")
high<-read.xlsx("C:/Users/lenovo/Desktop/data/t2/hight2.xlsx",1,encoding="UTF-8")
whole=rbind(hard,high)
names(whole)[31]=c("species")

auto=cbind(auto,whole[31])

x=auto[,-29]
y=auto[,29]#将species作为结果变量
svm_auto_t2=lssvm(species~.,data=auto)#下面lssvm是kernlab包的函数，可以自动估计核函数参数
print(svm_auto_t2)
pred=predict(svm_auto_t2,x)
table(pred,y)
#           y
# pred   hard high
# hard   53   26
# high   24   53

#-----------预测具体结果----------#
print(pred)
#--------挑选出预测错误的编号-----#
num=function(x,y){
  k=rep(0,156)
  for (i in 1:156){
    if(x[i]!=y[i])
      k[i]=i
  }
  return(k)
}
aaa=num(pred,auto[,29])
aaa
a=data.frame(aaa)
write.xlsx(a,"C:/Users/lenovo/Desktop/data/t2/prednum_auto.xlsx","sheet 1")
#------计算正确和错误个数---------#
n=which(aaa!=0)
data.frame(n)
length(n)
#存在交叉部分20个，即预测错误50-20=30个，错误率30/156=
aa=30/156
print(aa)
#错误率为0.1923077


#-----------------------------第四步去除重复特征autoencoder+SVM-----------------------------#
rm(list=ls())
cat("\014") 

auto<-read.xlsx("C:/Users/lenovo/Desktop/data/t2/autoencoder_1.xlsx",2,encoding="UTF-8")

hard<-read.xlsx("C:/Users/lenovo/Desktop/data/t2/hardt2.xlsx",1,encoding="UTF-8")
high<-read.xlsx("C:/Users/lenovo/Desktop/data/t2/hight2.xlsx",1,encoding="UTF-8")
whole=rbind(hard,high)
names(whole)[31]=c("species")
auto=cbind(auto,whole[31])

x=auto[,-5]
y=auto[,5]#将species作为结果变量
svm_auto_t2=lssvm(species~.,data=auto)#下面lssvm是kernlab包的函数，可以自动估计核函数参数
print(svm_auto_t2)
pred=predict(svm_auto_t2,x)
table(pred,y)
#           y
# pred   hard high
# hard   54   29
# high   23   50

#-----------预测具体结果----------#
print(pred)
#--------挑选出预测错误的编号-----#
num=function(x,y){
  k=rep(0,156)
  for (i in 1:156){
    if(x[i]!=y[i])
      k[i]=i
  }
  return(k)
}
aaa=num(pred,auto[,5])
aaa
a=data.frame(aaa)
write.xlsx(a,"C:/Users/lenovo/Desktop/data/t2/prednum_auto_5.xlsx","sheet 1")
#------计算正确和错误个数---------#
n=which(aaa!=0)
data.frame(n)
length(n)
#存在交叉部分20个，即预测错误52-20=32个，错误率32/156=
aa=32/156
print(aa)
#错误率为0.2051282












































