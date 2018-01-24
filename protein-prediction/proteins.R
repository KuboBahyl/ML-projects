#######################################
#
# Author: Jakub Bahyl
#
# Subject: Intro to ML - MFF
#
# Theme: Protein modeling
#
#######################################

rm(list=ls())
#load(".RData")

library(data.table) # fread()
library(ggplot2) # nice boxplots

library(rpart) # tree models
library(rpart.plot) # nice tree graphs
library(randomForest) #random forest
library(e1071) # SVM

library(ROCR)		# ROC curves
library(crossval)   # model evaluation

library(rmarkdown) # report

#######################################
###### Preparing data
#######################################

data <- fread(file='plr.dataset.B.development.csv', header=T, sep="\t")
data <- as.data.frame(data)
features <- colnames(data[,-1])
str(data)
summary(data)

ids <- levels(factor(data$protein_id))

# dividing into train test 90:10
set.seed(666)
train <- subset( data, data$protein_id %in% sample(ids, size=0.9*length(ids))) 
test <- data [ !(row.names(data) %in% row.names(train)), ]
levels(factor(train$protein_id))
levels(factor(test$protein_id))

#######################################
###### Basic analysis
#######################################

# number of ids
length(ids)

# percentage of points with and without ligandibility class
classTable <- table(data$protein_id, data$class, dnn = c("protein id", "Able to ligand"))
rowSegments <-  rowSums(classTable)
classTable <- round(classTable*100 / rowSums(classTable), digits=2)
classTable <- cbind(rowSegments, classTable)
colnames(classTable) <- c('Segments', 'No', 'Yes')
classTable
rm(rowSegments)

#####################################
### Naive model (classifying according the majority output)
######################################

major <- "0"
naive.pred <- rep(major, nrow(test))
naive.conf <- confusionMatrix(test$class, naive.pred, "0")
naive.acc <- diagnosticErrors(naive.conf)[1]

#####################################
### Decision tree
####################################

set.seed(777)

# start with small complexity parameter
tree <- rpart(class ~ .,
              data=train[sample(nrow(train)), -1], 
              method="class",
              control = rpart.control(cp=0.0001))
rpart.plot(tree)
printcp(tree)
plotcp(tree)

# choose the best one (minimizes xerror - cross-validated error)
bestcp = tree$cptable[ which.min(tree$cptable[,"xerror"]), "CP" ]
tree.pruned <- prune(tree, cp=bestcp)
rpart.plot(tree.pruned, extra=3)

tree.conf <- confusionMatrix(test$class, predict(tree.pruned, test, type="class") , "0")
diagnosticErrors(tree.conf)[1]

cross = 10
folds <- cut( seq(1,length(sample(ids))), breaks=cross, labels=F )

crossTreeAcc <- c()

for (i in 1:cross) {
  cat("k-fold cross-validation: ", i ,". STEP... \n")
  
  indices <- which(folds==i)
  set.seed(i*666)
  cross.test <- subset( data[, -1], data$protein_id %in% ids[indices])
  cross.train <- data [ !(row.names(data) %in% row.names(cross.test)), -1]
  cross.train <- cross.train[sample(nrow(cross.train)), ]
  
  cat('Data divided into Train:', round(nrow(cross.train)/nrow(data)*100,2),
      '% and Test:', round(nrow(cross.test)/nrow(data)*100,2), '%. \n')
  
  cross.tree <- rpart(class ~ .,
                      data=cross.train, 
                      method="class",
                      control = rpart.control(cp=0.0001))
  
  
  cross.bestcp = cross.tree$cptable[ which.min(cross.tree$cptable[,"xerror"]), "CP" ]
  cross.tree.pruned <- prune(cross.tree, cp=cross.bestcp)
  
  cross.conf <- confusionMatrix(cross.test$class, predict(cross.tree.pruned, cross.test, type="class"), "0")
  cross.diag <- diagnosticErrors(cross.conf)
  
  crossTreeAcc[i] <- cross.diag[1]
  
  cat("Accuracy:",round(crossTreeAcc[i]*100, digits=2),"% \n")
}

### t-tests
tree.acc <- mean(crossTreeAcc)
t.test(crossTreeAcc, conf.level = 0.9)
t.test(crossTreeAcc, conf.level = 0.95)
t.test(crossTreeAcc, conf.level = 0.99)

# worse than naive classifier..

### importance of features

# treba sledovat vrchol stromu a najzelensie konce

#######################################
###### Ensemble methods
#######################################

set.seed(777)

# Random forest with default values
forest <- randomForest(as.factor(class) ~ ., 
                       data=train[sample(nrow(train)), -1] )
plot(forest, ylim=c(-1e-2,0.05))
print(forest) # view results 
forest.conf <- confusionMatrix(test$class, predict(forest, test, type="class"), "0")
diagnosticErrors(forest.conf)[1]

# Determining best tuning on small ntree (otherwise error=0)
bestmtry <- tuneRF(train[, -1], as.factor(train$class), stepFactor = 1.3, improve=1e-4 , ntree=10)
# the best one is mtry=15, ntree could be 300
forest.pruned <- randomForest(as.factor(class) ~ ., 
                              data=train[sample(nrow(train)), -1],
                              mtry=14,
                              nodesize=30,
                              ntree=500)
print(forest.pruned) # view results 
forest.conf <- confusionMatrix(test$class, predict(forest.pruned, test, type="class"), "0")
diagnosticErrors(forest.conf)[1]

crossForestAcc <-c()

for (i in 1:cross) {
  cat("k-fold cross-validation: ", i ,". STEP... \n")
  indices <- which(folds==i)
  
  set.seed(i*666)
  cross.test <- subset( data[, -1], data$protein_id %in% ids[indices])
  cross.train <- data [ !(row.names(data) %in% row.names(cross.test)), -1]
  cross.train <- cross.train[sample(nrow(cross.train)), ]
  
  cat('Data divided into Train:', round(nrow(cross.train)/nrow(data)*100,2),
      '% and Test:', round(nrow(cross.test)/nrow(data)*100,2), '%. \n')
  
  cross.forest <- randomForest(as.factor(class) ~ .,
                               cross.train,
                               mtry=14,
                               nodesize=30,
                               ntree=500)
  
  cross.conf <- confusionMatrix(cross.test$class, predict(cross.forest, cross.test, type="class"), "0")
  cross.diag <- diagnosticErrors(cross.conf)
  
  crossForestAcc[i] <- cross.diag[1]
  
  cat("Accuracy:",round(crossForestAcc[i]*100, digits=2),"% \n")
}

### t-tests
forest.acc <- mean(crossForestAcc)
t.test(crossForestAcc, conf.level = 0.9)
t.test(crossForestAcc, conf.level = 0.95)
t.test(crossForestAcc, conf.level = 0.99)


#######################################
###### Support Vector Machines
#######################################

## SVM linear
SVM.lin <- svm(as.factor(class) ~ ., 
               data=train[sample(nrow(train)), -1],
               kernel = "linear",
               cost = 1)

summary(SVM.lin)
SVMlin.conf <- confusionMatrix(test$class, predict(SVM.lin, test, type="class"), "0")
diagnosticErrors(SVMlin.conf)[1]
# SVM.lin is no different from naive model


## SVM polynomial, degree=2
SVM.deg2 <- svm(as.factor(class) ~ ., 
                data=train[sample(nrow(train)), -1],
                kernel = "polynomial",
                degree = 2,
                cost = 1)

SVMdeg2.conf <- confusionMatrix(test$class, predict(SVM.deg2, test, type="class"), "0")
diagnosticErrors(SVMdeg2.conf)[1]
# better than naive

## SVM radial basis
SVM.radial <- svm(as.factor(class) ~ ., 
                  data=train[sample(nrow(train)), -1],
                  kernel = "radial")

SVMradial.conf <- confusionMatrix(test$class, predict(SVM.radial, test, type="class"), "0")
diagnosticErrors(SVMradial.conf)[1]
# seems promising

## train the best model using cross validation
train.small <- subset( train, train$protein_id %in% sample(ids, size=5))
SVM.tuning <- tune.svm(as.factor(class) ~ ., 
                       data = train.small[sample(nrow(train.small)), -1],
                       kernel = "radial",
                       gamma=c(0.01, 0.02, 0.05, 0.1),
                       cost=c(0.5,1,2,5) )

print(SVM.tuning)
plot(SVM.tuning)

SVM.tuned <- svm(as.factor(class) ~ ., 
                 data = train[sample(nrow(train)), -1],
                 kernel = "radial",
                 gamma = 0.05,
                 cost = 2)

SVM.conf <- confusionMatrix(test$class, predict(SVM.tuned, test, type="class"), "0")
diagnosticErrors(SVM.conf)[1]

## 10-fold cross-validation with the best cost and gamma
crossSvmAcc <-c()

for (i in 1:cross) {
  cat("10-fold cross-validation: ", i ,". STEP... \n")
  indices <- which(folds==i)
  
  set.seed(i*666)
  cross.test <- subset( data[, -1], data$protein_id %in% ids[indices])
  cross.train <- data [ !(row.names(data) %in% row.names(cross.test)), -1]
  cross.train <- cross.train[sample(nrow(cross.train)), ]
  
  cross.SVM <- svm(as.factor(class) ~ ., 
                   data = cross.train[sample(nrow(cross.train)),],
                   kernel = "radial",
                   gamma = 0.05,
                   cost = 2)
  
  cross.conf <- confusionMatrix(cross.test$class, predict(cross.SVM, cross.test, type="class"), "0")
  cross.diag <- diagnosticErrors(cross.conf)
  
  crossSvmAcc[i] <- cross.diag[1]
  
  cat("Accuracy:",round(crossSvmAcc[i]*100, digits=2),"% \n")
}

### t-tests
SVM.acc <- mean(crossSvmAcc)
t.test(crossSvmAcc, conf.level = 0.9)
t.test(crossSvmAcc, conf.level = 0.95)
t.test(crossSvmAcc, conf.level = 0.99)

#######################################
###### Feature selection
#######################################

### Variable importance from RF
varImpPlot(forest.pruned, sort=T, main="Importance") 
importance <- forest.pruned$importance
featuresOrdered <- rownames(importance)[order(-importance)]

# forward selection (12 max) using SVM
forwardAcc <- c()

for (k in 13:16) {
  cat("Features selected: ", k ,"....\n")
  forward.features <- featuresOrdered[1:k]
  
  data.subset <- data[sample(nrow(data)), c("protein_id", forward.features, "class") ]
  
  crossSvmAcc <-c()
  
  for (i in 1:cross) {
    cat("10-fold cross-validation: ", i ,". STEP... \n")
    indices <- which(folds==i)
    
    set.seed(i*666)
    cross.test <- subset( data.subset[, -1], data.subset$protein_id %in% ids[indices])
    cross.train <- data.subset [ !(row.names(data.subset) %in% row.names(cross.test)), -1]
    cross.train <- cross.train[sample(nrow(cross.train)), ]
    
    cross.SVM <- svm(as.factor(class) ~ ., 
                     data = cross.train[sample(nrow(cross.train)),],
                     kernel = "radial",
                     gamma = 0.05,
                     cost = 2)
    
    cross.conf <- confusionMatrix(cross.test$class, predict(cross.SVM, cross.test, type="class"), "0")
    cross.diag <- diagnosticErrors(cross.conf)
    
    crossSvmAcc[i] <- cross.diag[1]
  }
  cat("Total accuracy:",round(mean(crossSvmAcc)*100, digits=2),"% \n")
  
  forwardAcc[k] <- mean(crossSvmAcc)
}

#######################################
###### ROC evaluation
#######################################

tree.pred <- predict(tree.pruned, test[,-37], type="prob")[,2]
tree.prediction <- prediction(tree.pred, test$class)
tree.roc <- performance(tree.prediction, measure = "tpr", x.measure = "fpr")
plot(tree.roc,
     main="ROC curves",
     colorize = T)
tree.auc <- performance(tree.prediction, measure = "auc")
round(tree.auc@y.values[[1]], 3)

forest.pred <- predict(forest.pruned, test[,-37], type="prob")[,2]
forest.prediction <- prediction(forest.pred, test$class)
forest.roc <- performance(forest.prediction, measure = "tpr", x.measure = "fpr")
plot(forest.roc,
     add=T,
     colorize = T)
forest.auc <- performance(forest.prediction, measure = "auc")
round(forest.auc@y.values[[1]], 3)

SVM.fitted <- attributes(predict(SVM.pruned, test, type ="response", decision.values=TRUE))$decision.values
SVM.prediction <- prediction(SVM.fitted, test$class, label.ordering = c("1","0") )
SVM.roc <- performance(SVM.prediction, measure = "tpr", x.measure = "fpr")
plot(SVM.roc,
     add=T,
     colorize = T)
SVM.auc <- performance(SVM.prediction, measure = "auc")
round(SVM.auc@y.values[[1]], 3)

abline(0,1)

#######################################
###### Final prediction
#######################################

blind <- fread(file='plr.dataset.B.test.blind.csv', header=T, sep="\t")
blind <- as.data.frame(blind)

# best model no1 - Random Forest tuned
forest.best <- randomForest(as.factor(class) ~ ., 
                            data=data[sample(nrow(data)), -1],
                            mtry=14,
                            nodesize=30,
                            ntree=500)

blind.forest.pred <- predict(forest.best, blind[,-1], type="class")

fileConn<-file("model1-forest.txt")
writeLines(as.character(blind.forest.pred), fileConn)
close(fileConn)

# best model no2 - SVM tuned
SVM.best <- svm(as.factor(class) ~ ., 
                data = data[sample(nrow(data)), -1],
                kernel = "radial",
                gamma = 0.05,
                cost = 2)

confusionMatrix(test$class, predict(SVM.best, test, type="class"), "0")
diagnosticErrors(confusionMatrix(test$class, predict(SVM.best, test, type="class"), "0"))[1]

blind.SVM.pred <- predict(SVM.best, blind[,-1], type="class")

fileConn<-file("model1-SVM.txt")
writeLines(as.character(blind.SVM.pred), fileConn)
close(fileConn)
