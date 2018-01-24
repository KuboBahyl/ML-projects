
#################################################################################
#################################################################################

### ROC curves, AUC

### Caravan data set

### Barbora Hladka, Martin Holub

### http://ufal.mff.cuni.cz/course/npfl054

#################################################################################
#################################################################################


library(ISLR)		# data
library(rpart)		# decision trees
library(rpart.plot)	# drawing decision trees
library(e1071)		# SVM
library(ROCR)		# plotting ROC curve
library(crossval)   # evaluation


###############
## EXERCISE #1: Caravan data set processing
###############

data.X <-  scale(Caravan[,-86])
data.Y <-  Caravan$Purchase
data <- data.frame(data.X, data.Y)
colnames(data)[86] <- "Purchase"

# splitting the data into training and test sets 50:50

set.seed(99)
train <- data[sample(row.names(data), size = round(nrow(data)*0.5)), ]
train.X <- train[,-86]
train.Y <- train$Purchase

test <- data[!(row.names(data) %in% row.names(train)), ]
test.X <- test[,-86]
test.Y <- test$Purchase


###############
## EXERCISE #2: Build a Decision Tree Classifier
###############

dt <- rpart(formula = Purchase ~ .,
	data = train,
	control = rpart.control(minsplit = 20, minbucket = 1, cp = 0.008))

rpart.plot(dt, extra = 4)

###############
## EXERCISE #3: Build a Logistic Regression Classifier
###############

log <- glm(Purchase~. , data = train, family = binomial)

###############
## EXERCISE #4: Build SVM Classifiers
###############

# to plot ROC curve - work with fitted values instead of predicted classes
# fitted values => h(x) = theta_0 + theta_1*x_1 + theta_2*x_2 + ...
# i.e., decision.values = T

svm.1 <- svm(Purchase ~ . ,
	data = train,
	kernel = "radial",
	gamma = 0.1,
	decision.values = T)

svm.2 <- svm(Purchase ~ . ,
	data = train,
	kernel = "radial",
	gamma = 0.01,
	decision.values = T)

svm.3 <- svm(Purchase ~ . ,
	data = train,
	kernel = "radial",
	gamma = 0.001,
	decision.values = T)


###############
## EXERCISE #5: Plot ROC curves
###############

################
## EXERCISE #5.1: Decision trees
###############

# work with the predicted classes
dt.pred.class <- predict(dt, test[,-86], type = "class")
confusionMatrix(test.Y, dt.pred.class, "No")
diagnosticErrors(confusionMatrix(test.Y, dt.pred.class, "No"))

# work with the proportion of classes in the leaf nodes
dt.pred.prob  <- predict(dt, test[,-86], type="prob")[,2]

# check classes and probabilities with the trained decision tree
cbind(dt.pred.class, dt.pred.prob)

# plot ROC curve for the dt model
pred.dt <- prediction(dt.pred.prob,test.Y)
perf.dt <- performance(pred.dt,
		measure = "tpr",
		x.measure = "fpr")

plot(perf.dt,
	main="ROC curves: Data",
	colorize = T)

# check
str(perf.dt)
cutoffs <- data.frame(cut=perf.dt@alpha.values[[1]],
					  fpr=perf.dt@x.values[[1]], 
                      tpr=perf.dt@y.values[[1]])

################
## EXERCISE #5.1.1: Explain
###############

# > cutoffs[8,]
#          cut fpr tpr
# 8 0.03019943   1   1

################
## EXERCISE #5.2: Logistic regression
###############

log.probs <- predict(log, test, type ="response")
pred.log <- prediction(log.probs, test.Y)
perf.log <- performance(pred.log,
		measure = "tpr",
		x.measure = "fpr")
plot(perf.log,
	colorize = T,
	add=TRUE)

################
## EXERCISE #5.3: SVM
###############

# gamma = 0.1
fitted.1 <- attributes(predict(svm.1, test,type ="response", decision.values=TRUE))$decision.values
pred.1 <- prediction(fitted.1,
		test.Y,
		label.ordering = c("Yes","No") )

perf.1 <- performance(pred.1,
		measure = "tpr",
		x.measure = "fpr")

plot( perf.1,
	col = 4,
	add = TRUE )

# gamma = 0.01
fitted.2 <- attributes(predict(svm.2, test, type ="response", decision.values=TRUE))$decision.values

pred.2 <- prediction(fitted.2,
		test.Y,
		label.ordering = c("Yes","No") )

perf.2 <- performance(pred.2,
		measure = "tpr",
		x.measure = "fpr")

plot(perf.2,
	col = 5,
	add = TRUE )

# gamma = 0.001
fitted.3 <- attributes(predict(svm.3, test, type ="response", decision.values=TRUE))$decision.values

pred.3 <- prediction(fitted.3,
		test.Y,
		label.ordering = c("Yes","No") )

perf.3 <- performance(pred.3,
		measure = "tpr",
		x.measure = "fpr")

plot( perf.3,
	col = 6,
	add = TRUE  )

################
## EXERCISE #5.4: Random classifier
###############
abline(a = 0, b = 1)


################
## EXERCISE #6: AUC
###############

## decision trees
message("\nAUC-- DT")
perf.auc.dt <- performance(pred.dt, measure = "auc")
dt.auc <- round(perf.auc.dt@y.values[[1]], 3)

## logistic regression
message("\nAUC-- LogR")
perf.auc.log <- performance(pred.log, measure = "auc")
perf.auc.log@y.values
log.auc <- round(perf.auc.log@y.values[[1]], 3)

## SVM 
message("\nAUC-- SVM -- gamma = 0.1")
perf.auc.1 <- performance(pred.1, measure = "auc")
perf.auc.1@y.values
svm1.auc <- round(perf.auc.1@y.values[[1]], 3)

message("\nAUC-- SVM -- gamma = 0.01")
perf.auc.2 <- performance(pred.2, measure = "auc")
perf.auc.2@y.values
svm2.auc <- round(perf.auc.2@y.values[[1]], 3)

message("\nAUC-- SVM -- gamma = 0.001")
perf.auc.3 <- performance(pred.3, measure = "auc")
perf.auc.3@y.values
svm3.auc <- round(perf.auc.3@y.values[[1]], 3)

################
## EXERCISE #7: Finish plot -- add legend
###############
legend(0.6, 0.6, title="Classifiers", bty='n',
	c(	
	paste("dt"), 
	paste("log"),
	expression(paste("svm: ", gamma, " = ", 10^-1)),
	expression(paste("svm: ", gamma, " = ", 10^-2)),
	expression(paste("svm: ", gamma, " = ", 10^-3))),
	2:6)

legend(0.6,0.3, title="AUC", bty='n',
	c(dt.auc, log.auc, svm1.auc, svm2.auc, svm3.auc), 
	2:6)

################
## EXERCISE #8: Plot cut-off vs. ACC/TPR/FPR
###############

## decision trees
message("\nDT")
perf.acc.dt <- performance(pred.dt, measure = "acc", )
perf.tpr.dt <- performance(pred.dt, measure = "tpr")
perf.fpr.dt <- performance(pred.dt, measure = "fpr")

## logistic regression
message("\nLogR")
perf.acc.log <- performance(pred.log, measure = "acc", )
perf.tpr.log <- performance(pred.log, measure = "tpr")
perf.fpr.log <- performance(pred.log, measure = "fpr")

par(mfrow=c(2,3))
plot(perf.acc.dt, main="DT-ACC", col=2, lwd=2)
plot(perf.tpr.dt, main="DT-TPR", col=3, lwd=2)
plot(perf.fpr.dt, main="DT-FPR", col=4, lwd=2)

plot(perf.acc.log, main="LogR-ACC", col=2, lwd=2)
plot(perf.tpr.log, main="LogR-TPR", col=3, lwd=2)
plot(perf.fpr.log, main="LogR-FPR", col=4, lwd=2)


