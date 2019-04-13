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
print("\014")
#control+enter 运行当前行


#核心是提取特征，达到降低维度的目的
#-----------------------------t2数据-----------------------------------#

hard<-read.xlsx("C:/Users/001/Desktop/data/t2/hardt2.xlsx",1,encoding="UTF-8")
high<-read.xlsx("C:/Users/001/Desktop/data/t2/hight2.xlsx",1,encoding="UTF-8")
whole=rbind(hard,high)

names(whole)[31]=c("species")
names(whole)
data1=as.matrix(whole[,c(-1,-2,-31)])


nl=3                          
unit.type = "logistic"      
Nx.patch=30                    ## 1.与特征相对应？
Ny.patch=30                    ## 2.？？？
N.input = Nx.patch*Ny.patch  
N.hidden = 30*30               ## 3.number of units in the hidden layer
lambda = 0.0002              
beta = 6                    
rho = 0.01                    
epsilon <- 0.001            
max.iterations = 2000        
autoencoder.object <- autoencode(X.train=data1,nl=nl,N.hidden=N.hidden,
                                 unit.type=unit.type,lambda=lambda,beta=beta,rho=rho,epsilon=epsilon,
                                 optim.method="BFGS",max.iterations=max.iterations,
                                 rescale.flag=TRUE,rescaling.offset=0.001)
summary(autoencoder.object)# 数据如何解读？
print(autoencoder.object)
## Report mean squared error for training and test sets:
cat("autoencode(): mean squared error for training set: ",
    round(autoencoder.object$mean.error.training.set,3),"\n")
## Visualize hidden units' learned features:
visualize.hidden.units(autoencoder.object,Nx.patch,Ny.patch)

X.output <- predict(autoencoder.object, X.input=data1, hidden.output=FALSE)$X.output

op <- par(no.readonly = TRUE)  
par(mfrow=c(3,2),mar=c(2,2,2,2))
for (n in c(1,2,3)){
  ## input image:           4.?????????????????
  image(matrix(data1[n,],nrow=Ny.patch,ncol=Nx.patch),axes=FALSE,main="Input image",
        col=gray((0:20)/20))
  ## output image:
  image(matrix(X.output[n,],nrow=Ny.patch,ncol=Nx.patch),axes=FALSE,main="Output image",
        col=gray((0:20)/20))
}
par(op)  ## restore plotting par's
X.output
write.xlsx(t(X.output),"C:/Users/001/Desktop/data/t2/autoencoder1.xlsx","sheet 1")















#-----------------------------R自带样本数据------------------------------------#

data('training_matrix_N=5e3_Ninput=100')  ## the matrix contains 5e3 image 


## Set up the autoencoder architecture:
nl=3                          ## number of layers (default is 3: input, hidden, output)
unit.type = "logistic"        ## specify the network unit type, i.e., the unit's 
## activation function ("logistic" or "tanh")
Nx.patch=10                   ## width of training image patches, in pixels
Ny.patch=10                   ## height of training image patches, in pixels
N.input = Nx.patch*Ny.patch   ## number of units (neurons) in the input layer (one unit per pixel)
N.hidden = 10*10              ## number of units in the hidden layer
lambda = 0.0002               ## weight decay parameter     
beta = 6                      ## weight of sparsity penalty term 
rho = 0.01                    ## desired sparsity parameter
epsilon <- 0.001              ## a small parameter for initialization of weights 
## as small gaussian random numbers sampled from N(0,epsilon^2)
max.iterations = 2000         ## number of iterations in optimizer

## Train the autoencoder on training.matrix using BFGS optimization method 
## (see help('optim') for details):
## Not run: 
autoencoder.object <- autoencode(X.train=training.matrix,nl=nl,N.hidden=N.hidden,
                                 unit.type=unit.type,lambda=lambda,beta=beta,rho=rho,epsilon=epsilon,
                                 optim.method="BFGS",max.iterations=max.iterations,
                                 rescale.flag=TRUE,rescaling.offset=0.001)

## End(Not run)
## N.B.: Training this autoencoder takes a long time, so in this example we do not run the above 
## autoencode function, but instead load the corresponding pre-trained autoencoder.object.

## Report mean squared error for training and test sets:
cat("autoencode(): mean squared error for training set: ",
    round(autoencoder.object$mean.error.training.set,3),"\n")

## Visualize hidden units' learned features:
visualize.hidden.units(autoencoder.object,Nx.patch,Ny.patch)

## Compare the output and input images (the autoencoder learns to approximate 
## inputs in its outputs using features learned by the hidden layer):

## Predict the output matrix corresponding to the training matrix 
## (rows are examples, columns are input channels, i.e., pixels)
X.output <- predict(autoencoder.object, X.input=training.matrix, hidden.output=FALSE)$X.output 

## Compare outputs and inputs for 3 image patches (patches 7,26,16 from 
## the training set) - outputs should be similar to inputs:
op <- par(no.readonly = TRUE)   ## save the whole list of settable par's.
par(mfrow=c(3,2),mar=c(2,2,2,2))
for (n in c(7,26,16)){
  ## input image:
  image(matrix(training.matrix[n,],nrow=Ny.patch,ncol=Nx.patch),axes=FALSE,main="Input image",
        col=gray((0:32)/32))
  ## output image:
  image(matrix(X.output[n,],nrow=Ny.patch,ncol=Nx.patch),axes=FALSE,main="Output image",
        col=gray((0:32)/32))
}
par(op)  ## restore plotting par's



Var1 <- c(rnorm(50, 1, 0.5), rnorm(50, -0.6, 0.2))
Var2 <- c(rnorm(50, -0.8, 0.2), rnorm(50, 2, 1))
x <- matrix(c(Var1, Var2), nrow = 100, ncol = 2)
y <- c(rep(1, 50), rep(0, 50))
dnn <- sae.dnn.train(x, y, hidden = c(5, 5))
summary(dnn)
print(dnn)
## predict by dnn
test_Var1 <- c(rnorm(50, 1, 0.5), rnorm(50, -0.6, 0.2))
test_Var2 <- c(rnorm(50, -0.8, 0.2), rnorm(50, 2, 1))
test_x <- matrix(c(test_Var1, test_Var2), nrow = 100, ncol = 2)
a=nn.test(dnn, test_x, y)
summary(a)
print(a)



#-----------------------------小样本交叉数据------------------------------------#


#去除了高科技和硬科技交叉的20条数据
whole<-read.xlsx("C:/Users/lenovo/Desktop/data/t1/hard.xlsx",3,encoding="UTF-8")
#对y进行格式转换，但demodeclassLables是RSNNS里面的函数
y = decodeClassLabels(whole[,9])
x=as.matrix(whole[,1:8])
dnn <- sae.dnn.train(x, y, hidden = c(5, 5))




#-----------------------------大样本总体数据------------------------------------#
rm(list=ls())
cat("\014")             #清空控制台，快捷键control+L


whole<-read.xlsx("C:/Users/lenovo/Desktop/data/t1/hard.xlsx",3,encoding="UTF-8")
data1=t(whole[,1:8])


nl=3                          
unit.type = "logistic"       
Nx.patch=8                    ## 1.与特征相对应？
Ny.patch=31                   ## 2.？？？
N.input = Nx.patch*Ny.patch   
N.hidden = 8*8                ## 3.number of units in the hidden layer
lambda = 0.0002              
beta = 6                     
rho = 0.01                    
epsilon <- 0.001             
max.iterations = 2000        
autoencoder.object <- autoencode(X.train=data1,nl=nl,N.hidden=N.hidden,
                                 unit.type=unit.type,lambda=lambda,beta=beta,rho=rho,epsilon=epsilon,
                                 optim.method="BFGS",max.iterations=max.iterations,
                                 rescale.flag=TRUE,rescaling.offset=0.001)
summary(autoencoder.object)# 数据如何解读？
print(autoencoder.object)
## Report mean squared error for training and test sets:
cat("autoencode(): mean squared error for training set: ",
    round(autoencoder.object$mean.error.training.set,3),"\n")
## Visualize hidden units' learned features:
visualize.hidden.units(autoencoder.object,Nx.patch,Ny.patch)

X.output <- predict(autoencoder.object, X.input=data1, hidden.output=FALSE)$X.output 

op <- par(no.readonly = TRUE)  
par(mfrow=c(3,2),mar=c(2,2,2,2))
for (n in c(1,2,3)){
  ## input image:           4.?????????????????
  image(matrix(data1[n,],nrow=Ny.patch,ncol=Nx.patch),axes=FALSE,main="Input image",
        col=gray((0:20)/20))
  ## output image:
  image(matrix(X.output[n,],nrow=Ny.patch,ncol=Nx.patch),axes=FALSE,main="Output image",
        col=gray((0:20)/20))
}
par(op)  ## restore plotting par's
X.output
write.xlsx(t(X.output),"C:/Users/lenovo/Desktop/autoencoder1.xlsx","sheet 1")













































