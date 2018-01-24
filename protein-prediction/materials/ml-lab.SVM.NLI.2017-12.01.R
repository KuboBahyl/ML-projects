
#################################################################################
#################################################################################

### Support vector machines

### Native language identification

### Target class: class

### Barbora Hladka, Martin Holub

### MFF UK, 2016/17

### http://ufal.mff.cuni.cz/course/npfl054

#################################################################################
#################################################################################

library(e1071)

###############
## EXERCISE #1: Read and explore the NLi data
################

data <- read.table(file="http://ufal.mff.cuni.cz/~hladka/2015/docs/fv.c.1.gram.rel.traindev.csv",
	sep = "\t",
	header = T)

###############
## EXERCISE #1.1: Look at the top 5 rows of data
################

###############
## EXERCISE #1.2: Find out the target class distribution in data
##		target class: class
################

# split the data into training and test sets
train.indeces<- read.table(file="http://ufal.mff.cuni.cz/~hladka/2015/docs/train.txt")
train <- data[ data$file %in% t(as.vector(train.indeces)), ]

test.indeces <- read.table(file="http://ufal.mff.cuni.cz/~hladka/2015/docs/dev.txt")
test <-data[ data$file %in% t(as.vector(test.indeces)), ]

# remove file name
train$file <- NULL
test$file <- NULL

# check whether there is any constant column in training set
names(train[, sapply(train, function(v) var(v, na.rm=TRUE) == 0)])

# contingency table for distribution of target class in training set
ct <- table(train$class)
round(prop.table(ct),2)

###############
## EXERCISE #2: Train SVM with various kernels and do prediction on the test data
##		target value: class
################

## train linear kernel
model <- svm(class ~ ., train,
	kernel = "linear",
	cost = 1)

# do prediction
prediction <- predict(model, test, type = "class")
mean(prediction == test$class) # [1]  0.3963636 
table(prediction, test$class)

## train polynomial kernel, degree=2
model2 <- svm(class ~ ., train,
	kernel = "polynomial",
	degree = 2,
	cost = 1)

# do prediction
prediction2 <- predict(model2, test, type="class")
mean(prediction2==test$class) # [1] 0.3172727

prediction2.train <- predict(model2, train, type="class")
mean(prediction2.train==train$class)  # [1] 0.4630303

## train polynomial kernel, degree=3
model3 = svm(class ~ ., train,
	kernel = "polynomial",
	degree = 3,
	cost = 1)

# do prediction
prediction3 <- predict(model3, test, type="class")
mean(prediction3==test$class) # [1] 0.2490909

prediction3.train <- predict(model3, train, type="class")
mean(prediction3.train==train$class)  # [1] 0.4190909

## train radial basis kernel
model4 <- svm(class ~ ., train,
	kernel = "radial")

# do prediction
prediction4 <- predict(model4, test, type="class")
mean(prediction4==test$class) # [1] 0.4081818

###############
## EXERCISE #3: Illustrate tuning parameters and scaling
##		target value: class
################

# generate random sample of 2,000 examples from the training set
set.seed(99)
s <- sample(nrow(train)) 
indices.train <- s[1:2000]
train.small <- train[indices.train,]

# check constant columns
names(train.small[, sapply(train.small, function(v) var(v, na.rm=TRUE)==0)])

train.small$c.1.11 <- NULL
train.small$c.1.13 <- NULL
train.small$c.1.25 <- NULL

###############
## EXERCISE #3.1: Scaling
################ 

## train radial basis kernel with scaling
model.small.scale <- svm(class ~ ., train.small,
	kernel = "radial")

# do prediction
prediction.small.scale <- predict(model.small.scale, test, type = "class")
mean(prediction.small.scale==test$class)	# [1] 0.35 

## train radial basis kernel without scaling
model.small.noscale <- svm(class ~ ., train.small,
	scale = FALSE,
	kernel = "radial")

# do prediction
prediction.small.noscale <- predict(model.small.noscale, test, type = "class")
mean(prediction.small.noscale==test$class)	# [1] 0.09090909

###############
## EXERCISE #2.2: Tuning parameters
################ 

## train the best model using cross validation
model.tune <- tune.svm(class ~ ., 
		data=train.small,
		kernel = "radial",
		gamma = c(0.001, 0.005, 0.01, 0.015, 0.02),
		cost = c(0.5, 1, 5,  10))

# find the best cost and gamma
# 1, 0.01, resp.
print(model.tune)
plot(model.tune)

## run 10-fold cross-validation with the best cost and gamma
model.best <- svm(class ~ ., train.small, 
	kernel = "radial",
	gamma = 0.01,
	cost = 1,
	cross = 10)

model.best$accuracies
 [1] 33.0 27.5 31.0 33.5 28.0 29.0 29.0 33.5 33.0 34.5

model.best$tot.accuracy
# [1] 31.2

# prediction
prediction.best <- predict(model.best, test, type="class")
mean(prediction.best == test$class) # [1] 0.3472727

