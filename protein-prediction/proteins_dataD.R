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
# load saved data
load("Dcaret.Rdata")

library(data.table) # quick data loading with fread()

library(rpart.plot) # nice tree graphs
library(caret) # ML methods
library(ada) # AdaBoost
library(gbm) # GBM

library(parallel) # distributed computing
library(doMC) # multicore “back-end” to the Caret

library(ROCR) # ROC curves

# turning on multicore computing
cores <- detectCores()
registerDoMC(cores = cores)

ttest_error <- function(data, conf) {
  t <- t.test(data, conf.level = conf)
  terror <- (t$conf.int[2] - t$conf.int[1]) / 2
  return(terror)
}

#######################################
###### Preparing data
#######################################

data <- fread(file='data/plr.dataset.D.development.csv', header=T, sep="\t")
data <- as.data.frame(data)

# renaming and changing values of ligand_distance to 0 if <= 2.6 and 1 otherwise
colnames(data)[37] <- "class"
data$class <- as.character(as.numeric(data$class<=2.6))
head(data$class)


features <- colnames(data[,c(-1,-37)])
sum(is.na(data)) # no NAs ^^
str(data)
summary(data)

ids <- levels(factor(data$protein_id))

# dividing into train:test = 90:10 with respect to protein ids
set.seed(111)
train <- subset(data[,-1], data$protein_id %in% sample(ids, size=0.9*length(ids))) 
test <- data[!(row.names(data) %in% row.names(train)), -1]

#######################################
###### Basic analysis
#######################################

# number of ids
length(ids)

# table with number of surface points and percentages of class distribution
classTable <- table(data$protein_id, data$class, dnn = c("protein id", "Able to ligand"))
rowSegments <-  rowSums(classTable)
classTable <- round(classTable*100 / rowSums(classTable), digits=2)
classTable <- cbind(rowSegments, classTable)
colnames(classTable) <- c('Segments', 'No', 'Yes')
classTable
rm(rowSegments)

#####################################
### Naive model (assigning class "0" to every point)
######################################

naive.acc <- 1 - sum(as.numeric(test$class)) / length(test$class)
naive.f1 <- 0

#######################################
###### Random forest
#######################################

set.seed(333)
train <- train[sample(nrow(train)), ]

# Searching for the best mtry parameter using 100 trees with cross-validation and OOB validation
forestcontrol <- trainControl(method = "oob")
forestmetric <- "Accuracy"
forestgrid <- expand.grid(mtry=c(5:20))
forest <- train(class ~ ., 
                data = train, 
                method = "rf",
                metric = forestmetric,
                trControl = forestcontrol,
                tuneGrid = forestgrid,
                ntree=100)
ggplot(forest)
print(forest)

# tuned model with best mtry and ntree=500
forest.tuned <- train(class ~ ., 
                      data = train, 
                      method = "rf",
                      metric = forestmetric,
                      trControl = forestcontrol,
                      tuneGrid = forest$bestTune,
                      ntree=500)

forest.conf <- confusionMatrix(predict(forest.tuned, test), test$class, positive = "1")
forest.conf$table
forest.acc <- forest.conf$overall["Accuracy"]
forest.f1 <- forest.conf$byClass["F1"]

### Variable importance from tuned RF
forestImp <- varImp(forest.tuned, scale=T)
ggplot(forestImp)
ggplot(forestImp, top=10)

impdata <- data.frame(forestImp$importance)
featuresOrdered <- rownames(impdata)[order(-impdata$Overall)]

### True 10-fold cross validation
k = 10
folds <- cut( seq(1,length(sample(ids))), breaks=k, labels=F )
crossForestAcc <- c()
crossForestF1 <- c()

for (i in 1:k) {
  cat("k-fold cross-validation: ", i ,". STEP... \n")
  
  indices <- which(folds==i)
  set.seed(i)
  cross.test <- subset( data[, -1], data$protein_id %in% ids[indices])
  cross.train <- data [ !(row.names(data) %in% row.names(cross.test)), -1]
  cross.train <- cross.train[sample(nrow(cross.train)), ]
  
  cat("Training RF with ntree=500... \n")
  forestcontrol <- trainControl(method = "oob")
  forestmetric <- "Accuracy"
  forestgrid <- expand.grid(mtry=16)
  forest <- train(class ~ ., 
                  data = cross.train, 
                  method = "rf",
                  metric = forestmetric,
                  trControl = forestcontrol,
                  tuneGrid = forestgrid,
                  ntree=500)
  
  forest.conf <- confusionMatrix(predict(forest, cross.test), cross.test$class, positive = "1")
  crossForestAcc[i] <- forest.conf$overall["Accuracy"]
  crossForestF1[i] <- forest.conf$byClass["F1"]
  
  cat("Accuracy:",round(crossForestAcc[i]*100, digits=2),"% \n")
  cat("F1 score:",round(crossForestF1[i]*100, digits=1),"% \n")
}

### t-tests
forest.acc <- mean(crossForestAcc)
ttest_error(crossForestAcc, 0.90)
ttest_error(crossForestAcc, 0.95)
ttest_error(crossForestAcc, 0.99)

forest.f1 <- mean(crossForestF1)
ttest_error(crossForestF1, 0.90)
ttest_error(crossForestF1, 0.95)
ttest_error(crossForestF1, 0.99)


#######################################
###### Support Vector Machines
#######################################

set.seed(444)
train <- train[sample(nrow(train)), ]

SVMcontrol <- trainControl(method = "cv", number=10)
SVMmetric <- "Accuracy"
SVM.lin <- train(class ~ ., 
                  data = train, 
                  method = "svmLinear",
                  metric = SVMmetric,
                  trControl = SVMcontrol)

SVMlin.conf <- confusionMatrix(predict(SVM.lin, test), test$class, positive = "1")
SVMlin.conf$table
SVMlin.conf$overall["Accuracy"]
SVMlin.conf$byClass["F1"]

SVM.poly <- train(class ~ ., 
                 data = train, 
                 method = "svmPoly",
                 metric = SVMmetric,
                 trControl = SVMcontrol,
                 degree = 2)

SVMpoly.conf <- confusionMatrix(predict(SVM.poly, test), test$class, positive = "1")
SVMpoly.conf$table
SVMpoly.conf$overall["Accuracy"]
SVMpoly.conf$byClass["F1"]

SVM.rad <- train(class ~ ., 
                 data = train, 
                 method = "svmRadial",
                 metric = SVMmetric,
                 trControl = SVMcontrol)

SVMrad.conf <- confusionMatrix(predict(SVM.rad, test), test$class, positive = "1")
SVMrad.conf$table
SVMrad.conf$overall["Accuracy"]
SVMrad.conf$byClass["F1"]

SVMgrid <- expand.grid(sigma=c(0.01, 0.02, 0.05, 0.1, 0.2, 0.5), C=c(1, 2, 5, 10, 12, 15, 18, 20, 22, 25))
SVM.exp <- train(class ~ ., 
                 data = train, 
                 method = "svmRadial",
                 metric = SVMmetric,
                 trControl = SVMcontrol,
                 tuneGrid = SVMgrid)
ggplot(SVM.exp)
print(SVM.exp)

SVMcontrol <- trainControl(method = "cv", number=10, classProbs=T)
SVM.tuned <- train(make.names(class) ~ ., 
                   data = train, 
                   method = "svmRadial",
                   metric = SVMmetric,
                   trControl = SVMcontrol,
                   tuneGrid = SVM.exp$bestTune)

SVM.conf <- confusionMatrix(predict(SVM.tuned, test), test$class, positive = "1")
SVM.conf$table
SVM.acc <- SVM.conf$overall["Accuracy"]
SVM.f1 <- SVM.conf$byClass["F1"]

### True 10-fold cross validation
k = 10
folds <- cut( seq(1,length(sample(ids))), breaks=k, labels=F )
crossSVMacc <- c()
crossSVMf1 <- c()

for (i in 1:k) {
  cat("k-fold cross-validation: ", i ,". STEP... \n")
  
  indices <- which(folds==i)
  set.seed(i)
  cross.test <- subset( data[, -1], data$protein_id %in% ids[indices])
  cross.train <- data [ !(row.names(data) %in% row.names(cross.test)), -1]
  cross.train <- cross.train[sample(nrow(cross.train)), ]
  
  cat("Training SVM with sigma=0.1 and cost=12... \n")
  SVM <- train(class ~ ., 
               data=cross.train[sample(nrow(cross.train)), -1], 
               method = "svmRadial",
               metric = SVMmetric,
               trControl = SVMcontrol,
               tuneGrid = SVM.exp$bestTune)
  
  SVM.conf <- confusionMatrix(predict(SVM, cross.test), cross.test$class, positive = "1")
  crossSVMacc[i] <- SVM.conf$overall["Accuracy"]
  crossSVMf1[i] <- SVM.conf$byClass["F1"]
  
  cat("Accuracy:",round(crossSVMacc[i]*100, digits=2),"% \n")
  cat("F1 score:",round(crossSVMf1[i]*100, digits=2),"% \n")
}

### t-tests
SVM.acc <- mean(crossSVMacc)
ttest_error(crossSVMacc, 0.90)
ttest_error(crossSVMacc, 0.95)
ttest_error(crossSVMacc, 0.99)

SVM.f1 <- mean(crossSVMf1)
ttest_error(crossSVMf1, 0.90)
ttest_error(crossSVMf1, 0.95)
ttest_error(crossSVMf1, 0.99)

#######################################
###### AdaBoost
#######################################
set.seed(555)
train <- train[sample(nrow(train)), ]

ABcontrol <- trainControl(method = "cv", number=10)
ABmetric <- "Accuracy"
ABgrid <- expand.grid(iter=c(100,200), maxdepth=c(12,15,20), nu=c(0.1,0.2,0.5))
AB <- train(class ~ ., 
             data = train, 
             method = "ada",
             metric = ABmetric,
             trControl = ABcontrol,
             tuneGrid = ABgrid)
ggplot(AB)
print(AB)

ABgrid <- expand.grid(iter=300, maxdepth=12, nu=0.5)
AB.tuned <- train(class ~ ., 
                  data = train, 
                  method = "ada",
                  metric = ABmetric,
                  trControl = ABcontrol,
                  tuneGrid = ABgrid)

AB.conf <- confusionMatrix(predict(AB.tuned, test), test$class, positive = "1")
AB.conf$table
AB.acc <- AB.conf$overall["Accuracy"]
AB.f1 <- AB.conf$byClass["F1"]

### True 10-fold cross validation
k = 10
folds <- cut( seq(1,length(sample(ids))), breaks=k, labels=F)
crossABacc <- c()
crossABf1 <- c()

for (i in 1:k) {
  cat("k-fold cross-validation: ", i ,". STEP... \n")
  
  indices <- which(folds==i)
  set.seed(i)
  cross.test <- subset( data[, -1], data$protein_id %in% ids[indices])
  cross.train <- data [ !(row.names(data) %in% row.names(cross.test)), -1]
  cross.train <- cross.train[sample(nrow(cross.train)), ]
  
  cat("Training AdaBoost with sigma=0.05 and cost=12... \n")
  ABcontrol <- trainControl(method = "cv", number=10)
  ABmetric <- "Accuracy"
  ABgrid <- expand.grid(iter=300, maxdepth=12, nu=0.5)
  AB <- train(class ~ ., 
               data=cross.train[sample(nrow(cross.train)), -1], 
               method = "ada",
               metric = ABmetric,
               trControl = ABcontrol,
               tuneGrid = ABgrid)
  
  AB.conf <- confusionMatrix(predict(AB, cross.test), cross.test$class, positive = "1")
  crossABacc[i] <- AB.conf$overall["Accuracy"]
  crossABf1[i] <- AB.conf$byClass["F1"]
  
  cat("Accuracy:",round(crossABacc[i]*100, digits=2),"% \n")
  cat("F1 score:",round(crossABf1[i]*100, digits=2),"% \n")
}

### t-tests
AB.acc <- mean(crossABacc)
ttest_error(crossABacc, 0.90)
ttest_error(crossABacc, 0.95)
ttest_error(crossABacc, 0.99)

AB.f1 <- mean(crossSVMf1)
ttest_error(crossABf1, 0.90)
ttest_error(crossABf1, 0.95)
ttest_error(crossABf1, 0.99)

#######################################
###### ROC evaluation
#######################################

forest.pred <- predict(forest.tuned, test[,-37], type="prob")[,2]
forest.prediction <- prediction(forest.pred, test$class)
forest.roc <- performance(forest.prediction, measure = "tpr", x.measure = "fpr")
plot(forest.roc,
     colorize = F)
forest.auc <- performance(forest.prediction, measure = "auc")
round(forest.auc@y.values[[1]], 3)

AB.pred <- predict(AB.tuned, test[,-37], type="prob")[,2]
AB.prediction <- prediction(AB.pred, test$class)
AB.roc <- performance(AB.prediction, measure = "tpr", x.measure = "fpr")
plot(AB.roc,
     add=T,
     colorize = F)
AB.auc <- performance(AB.prediction, measure = "auc")
round(AB.auc@y.values[[1]], 3)

SVM.pred <- predict(SVM.tuned, test, type="prob")[,2]
SVM.prediction <- prediction(SVM.pred, test$class)
SVM.roc <- performance(SVM.prediction, measure = "tpr", x.measure = "fpr")
plot(SVM.roc,
     add=T,
     colorize = F)
SVM.auc <- performance(SVM.prediction, measure = "auc")
round(SVM.auc@y.values[[1]], 3)

GBM.pred <- predict(GBM.tuned, test)
GBM.prob <- 1 - 1 / (1 + exp(-(GBM.pred-2.6)))
GBM.prediction <- prediction(GBM.prob, test$class)
GBM.roc <- performance(GBM.prediction, measure = "tpr", x.measure = "fpr")
plot(GBM.roc,
     add=T,
     colorize = T)
GBM.auc <- performance(GBM.prediction, measure = "auc")
round(GBM.auc@y.values[[1]], 3)

abline(0,1)

#######################################
###### Regression
#######################################

# reloading data with ligand_distance
data <- fread(file='data/plr.dataset.D.development.csv', header=T, sep="\t")
data <- as.data.frame(data)

# dividing into train:test = 90:10 with respect to protein ids
set.seed(111)
train <- subset(data[,-1], data$protein_id %in% sample(ids, size=0.9*length(ids))) 
test <- data[!(row.names(data) %in% row.names(train)), -1]

set.seed(123)
train <- train[sample(nrow(train)), ]

# GBM 
caretGrid <- expand.grid(interaction.depth=c(1, 3),
                         n.trees = (0:40)*50,
                         shrinkage=c(0.01, 0.001),
                         n.minobsinnode=10)

GBMgrid <- expand.grid(interaction.depth=c(10,12,15),
                       n.trees=c(0:10)*200, 
                       shrinkage=c(0.01, 0.02, 0.05),
                       n.minobsinnode=10)

GBMmetric <- "RMSE"
GBMcontrol <- trainControl(method="cv", number=10)

GBM <- train(ligand_distance ~ ., 
             data = as.matrix(train),
             method="gbm",
             distribution="gaussian",
             trControl = GBMcontrol,
             metric = GBMmetric, 
             tuneGrid = GBMgrid, 
             verbose = FALSE)

ggplot(GBM)
trellis.par.set(caretTheme())
plot(GBM)  
print(GBM)
GBM$bestTune

GBMgrid <- expand.grid(interaction.depth=15,
                       n.trees=5000, 
                       shrinkage=0.05,
                       n.minobsinnode=10)

GBM.tuned <- train(ligand_distance ~ ., 
                    data = as.matrix(train),
                    method="gbm",
                    distribution="gaussian",
                    trControl = GBMcontrol,
                    metric = GBMmetric, 
                    tuneGrid = GBMgrid, 
                    verbose = FALSE)    


GBM.pred <- predict(GBM.tuned, test)
GBM.prob <- 1 - 1 / (1 + exp(-(GBM.pred-2.6)))
GBM.class <- as.character(as.numeric(GBM.pred<=2.6))
test.class <- as.character(as.numeric(test$ligand_distance<=2.6))
head(GBM.class)

GBM.conf <- confusionMatrix(GBM.class, test.class, positive = "1")
GBM.conf$table
GBM.acc <- GBM.conf$overall["Accuracy"]
GBM.f1 <- GBM.conf$byClass["F1"]

#######################################
###### Final prediction
#######################################

blind <- fread(file='data/plr.dataset.D.test.blind.csv', header=T, sep="\t")
blind <- as.data.frame(blind)

# best model no1 - Random Forest tuned
ABgrid <- expand.grid(iter=300, maxdepth=12, nu=0.5)
AB.best <- train(class ~ ., 
                  data = data[,-1], 
                  method = "ada",
                  metric = ABmetric,
                  trControl = ABcontrol,
                  tuneGrid = ABgrid)

blind.AB.pred <- predict(AB.best, blind)

fileConn <- file("BlindD-AB.txt")
writeLines(as.character(blind.AB.pred), fileConn)
close(fileConn)

# best model no2 - SVM tuned
GBMgrid <- expand.grid(interaction.depth=15,
                       n.trees=5000, 
                       shrinkage=0.05,
                       n.minobsinnode=10)
GBM.best <- train(ligand_distance ~ ., 
                  data = as.matrix(data[,-1]),
                  method="gbm",
                  distribution="gaussian",
                  trControl = GBMcontrol,
                  metric = GBMmetric, 
                  tuneGrid = GBMgrid, 
                  verbose = FALSE)    

blind.GBM.pred <- predict(GBM.best, blind)
GBM.class <- as.character(as.numeric(blind.GBM.pred<=2.6))

fileConn <- file("GBM-dataD.txt")
writeLines(as.character(GBM.class), fileConn)
close(fileConn)
