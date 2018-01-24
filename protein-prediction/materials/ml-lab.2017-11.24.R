#################################################################################
#################################################################################

### Logistic regression, k-NN, Naive Bayes

### Caravan data set

### Data source:

## http://liacs.leidenuniv.nl/~puttenpwhvander/library/cc2000/data.html

### Barbora Hladka, Martin Holub

### http://ufal.mff.cuni.cz/course/npfl054

#################################################################################
#################################################################################

library(ISLR)     # Caravan data set
library(class)	  # kNN	
library(e1071)	  # Naive Bayes
library(crossval) # evaluation

# basic performance binary classification measures
perf <- function(true, pred, neg) {
# true -- true prediction
# pred -- output
# neg -- negative class
	cm <- confusionMatrix(true, pred, neg)
	acc <- diagnosticErrors(cm)[[1]]
	p <- diagnosticErrors(cm)[[4]]
	r <- diagnosticErrors(cm)[[2]]
	f1 <- 2*p*r/(p +r)
# accuracy -- acc = (TP+TN)/(FP+TN+TP+FN)
# sensitivity/recall -- sens = TP/(TP+FN) [true positive rate]
# specificity --  spec = TN/(FP+TN) [true negative rate]
# precision -- ppv = TP/(FP+TP) [positive predicted value]
# npv = TN/(TN+FN) [negative predicted value]
# lor = log(TP*TN/(FN*FP)) [diagnostic odds ratio (TP/FP)/(FN/TN)]
	
# output Accuracy, Precision, Recall, F1-measure
	return ( round(c(acc, p, r, f1),2) )
}

###############
##  EXERCISE #1: Scaling illustration
###############

# Differences in scaling may affect algorithms 
age <- c(20,25,29,31,36,38,45,52,56,65)
salary <- c(1000, 14000, 56000, 59000,  62000, 45000, 32000, 75000, 14000, 88000)
employee.data <- data.frame(age, salary)

# scale(x, center = TRUE, scale = TRUE)
employee.data.sc <- scale(employee.data)

# clustering
hca <- hclust(dist(employee.data))
hca.sc <- hclust(dist(employee.data.sc))

par(mfrow = c(2, 2))

yticks <- seq(1000, 100000, 5000)
xticks <- seq(20, 65, 2)
plot(employee.data,
	main = "No scaling",
	pch = 19,
	axes = FALSE)
axis(1, at = xticks, labels = xticks)
axis(2, at = yticks, labels = yticks, las = 2)
text(salary ~ age,
	labels = rownames(employee.data),
	data = employee.data,
	pos = 3,
	col='red',
	font = 2)
plot(hca)

plot(employee.data.sc,
	main = "After standardization",
	pch = 19)
text(salary ~ age,
	labels = rownames(employee.data),
	data = employee.data.sc,
	pos = 3,
	col='red',
	font = 2)
plot(hca.sc)

###############
## EXERCISE #2: Caravan data set processing
###############

dim(Caravan)

round(prop.table(table(Caravan$Purchase))*100,2)
# visualize
colors <- c("grey","yellow")
col <- colors
pie(table(Caravan$Purchase),
	main = "Customers of Caravan policy (Caravan$Purchase)",
	col = colors)
box()

# APLEZIER - number of boat policies
table(Caravan$APLEZIER)
table(Caravan$APLEZIER[Caravan$Purchase=="Yes"])
table(Caravan$Purchase)

# scaling 
data.X <-  scale(Caravan[,-86])
data.Y <-  Caravan$Purchase
data <- data.frame(data.X, data.Y)
colnames(data)[86] <- "Purchase"

# every column of data has a standard deviation of one and a mean of zero
# e.g.
var(data$MOSTYPE)
mean(data$MOSTYPE)
var(data$MAANTHUI)
mean(data$MOSTYPE)

# splitting the data into training and test sets 50:50
set.seed(1)
train <- data[sample(row.names(data), size = round(nrow(data)*0.5)), ]
test <- data[!(row.names(data) %in% row.names(train)), ]

train.X <- train[,-86]
train.Y <- train$Purchase

test.X <- test[,-86]
test.Y <- test$Purchase

###############
## EXERCISE #2.1: Are there any identical examples with different classification?
###############

###############
## EXERCISE #3: Build a Most Frequent Classifier (MFC)
###############

# most frequent class
which.max(table(train.Y))

# prediction
mfc.pred <- rep("No", length(test.Y))

# evaluation
perf.mfc <- perf(test.Y, mfc.pred, "No")

###############
## EXERCISE #4: Build a Logistic Regression Classifier
###############

# training
log <- glm(Purchase ~ . , data = train, family = binomial)

# prediction
log.probs <- predict(log, test, type ="response")

# probability cut-off 0.5
log.pred.05 <- rep("No", length(test.Y))
log.pred.05[log.probs > 0.5]="Yes"
table(log.pred.05,test.Y)

# probability cut-off 0.25
log.pred.025 <- rep("No", length(test.Y))
log.pred.025[log.probs > 0.25]="Yes"

# evaluation
# confusion matrices
cm.log.025 <- table(test.Y, log.pred.025)
cm.log.05 <- table(test.Y, log.pred.05)

perf.025 <- perf(test.Y, log.pred.025, "No")
perf.05 <- perf(test.Y, log.pred.05, "No")

###############
## EXERCISE #5: Build a k-NN Classifier
###############

## k = 1
# prediction
knn.1 <- knn(train.X , test.X, train.Y, k=1)
# evaluation
perf.1nn <- perf(test.Y, knn.1, "No")

## k = 5
# prediction
knn.5 <- knn(train.X , test.X, train.Y, k=5)
# evaluation
perf.5nn <- perf(test.Y, knn.5, "No")

###############
## EXERCISE #5.1: k=10 cross-validation
###############

cv.sample <- sample(1:nrow(train))
message("The number of all train data = ", length(cv.sample))

add.zeros <- 10 - nrow(train) %% 10
if(add.zeros < 10) cv.sample <- c(cv.sample, rep(0, add.zeros))
message("\nAdjusted cv.sample: ", length(cv.sample))
message("Tail of cv.sample:")
print(tail(cv.sample, 10))

## making 10 folds 
cv.index <- matrix(data = cv.sample, nrow = 10)
message("\nNumbers of data in folds:")
for(i in 1:10) print(length(cv.index[i,][cv.index[i,] > 0]))

message("\n*****   Doing 10-fold cross validation   *****\n")

## vectors with accuracies
m.knn.acc <- numeric(0)

k <- 10
 
for(i in 1:10){
  message(i, ". fold:")
  cv.train <- train[ - cv.index[i,][cv.index[i,] > 0], ]
  cv.test  <- train[ cv.index[i,][cv.index[i,] > 0], ]
  message("\t", "size(cv.train) = ", nrow(cv.train), "\tsize(cv.test) = ", nrow(cv.test))

  ### prediction -- m.knn
  m.knn <- knn(cv.train[,1:85], cv.test[,1:85], cv.train[,86], k = 10)
  acc <- sum(m.knn == cv.test[,86])/nrow(cv.test)
  m.knn.acc <- c(m.knn.acc, acc)
  message("\t", "m.knn accuracy = ", round(100*acc, 2), "%")
}

message("\n*****   Results of cross-validation process   *****\n")

message("m.knn -- cross-validation accuracies:")
print( round(m.knn.acc, 3) )
message()

###############
## EXERCISE #6: Build a Naive Bayes Classifier
###############

# training
nb <- naiveBayes(Purchase ~ PPERSAUT + MKOOPKLA + MRELGE + APLEZIER, data = train) 

# PPERSAUT Contribution car policies
# MKOOPKLA Purchasing power class
# MRELGE Married
# APLEZIER Number of boat policies

# prediction
nb.pred <- predict(nb, test, type="class")
# evaluation
perf.nb <- perf(test.Y, nb.pred, "No")

###############
## EXERCISE #6.1: Build a Naive Bayes Classifier using all features
###############

###############
## EXERCISE #7: Build a Decision Tree Classifier
###############

###############
## EXERCISE #8: Discuss performance of the classifiers
###############

print("Accuracy Precision Recall F-measure\n")
perf.mfc
perf.025
perf.05
perf.1nn
perf.5nn
perf.nb

###############
## EXERCISE #9: Feature selection
###############

cor <- cor(data.X, as.numeric(data.Y))
attr <- row.names(cor)

plot(unlist(cor),
	axes = FALSE,
	ylab="cor(X,Purchase)",
	xlab="")
axis(2)
abline(h = 0,
	col ="red",
	lwd = 2)
text(unlist(cor),
	labels = attr,
	pos = 3,
	cex = 0.7)

# Exclude the features with cor(X,Y) close to 0
# e.g.,

cor[which(attr == "MINK3045")]
cor[which(attr == "MINK123M")]
cor[which(attr == "PWABEDR")]

# MINK3045  Income 30-45.000
# MINK123M  Income >123.000
# PWABEDR   Contribution third party insurance (firms) ...






