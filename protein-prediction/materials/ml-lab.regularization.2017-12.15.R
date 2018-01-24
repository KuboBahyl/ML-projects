
#################################################################################
#################################################################################

### Regularization glmnet()
### http://cran.revolutionanalytics.com/web/packages/glmnet/glmnet.pdf

### College data set
### https://cran.r-project.org/web/packages/ISLR/ISLR.pdf

### Barbora Hladka, Martin Holub

### http://ufal.mff.cuni.cz/course/npfl054

#################################################################################
#################################################################################

library(e1071)
library(glmnet)
library(ISLR)

#######################
## function definitions
#######################

# add feature names into regularization plot

add.names <- function(fit) {
        L <- length(fit$lambda)
        x <- log(fit$lambda[L])
        y <- fit$beta[, L]
        labs <- names(y)
        text(x, y, 
		labels = labs,
		pos = 4,
		offset = 0.7)
}

################
## EXERCISE #1: Read data, get training and test sets
################

# training and test data
set.seed(123)

# do not work with Private, Accept, Enroll
college <- College[, -c(1,3,4)]

# split data into training and test data 50:50
train.ratio <- 0.5
train <- college[sample(row.names(college),
			size = round(nrow(college)*train.ratio)), ]
test <- college[!(row.names(college) %in% row.names(train)), ]

nrow(train)
nrow(test)

###############
## EXERCISE #2: Linear regression
##		        target value: Apps
###############

fit.lm <- lm(Apps ~ ., data = train)

summary(fit.lm)
# Coefficients	- features used for prediction
# Estimate		- values of hypothesis parameters theta_i
# Std. Error	- the average amount that the estimate differs from the actual value
# t value		- t-statistics used in testing whether a given parameter
#				is significantly different from zero
# Pr(>|t|)		- 2-tailed p-values used in testing the null hypothesis
#         		that the parameter is 0
# statistical codes
# annotation          p-value          significance level alpha
#    ***            [0, 0.001]                0.001
#     **         (0.001, 0.01]                0.01
#      *          (0.01, 0.05]                0.05
#      .           (0.05, 0.1]                0.1
#                     (0.1, 1]                1
# alpha = 0.001 (***)
# 	Parameter for F.Undergrad is significantly different from zero
# alpha = 0.01 (**)
#	Expend, Grad.Rate 
# alpha = 0.05 (*)
#	perc.alumni, Room.Board, 

# prediction
pred <- predict(fit.lm, test)

# mean squared error (MSE)
fit.lm.mse <- mean((test$Apps - pred)^2)

###############
## EXERCISE #2: Linear regression with regularization
##		        target value: Apps
###############

# right data format

x <- model.matrix(Apps ~ ., data = train)
y <- data.matrix(train$Apps)

x.test <- model.matrix(Apps ~ ., data = test)
y.test <- data.matrix(test$Apps)

# lambda values
grid <- 10^seq(4, -2, length = 100)
summary(grid)

###############
## EXERCISE #2.1: Lasso regularization
###############

# training
fit.lasso <- glmnet(x, y,
	alpha = 1,
	lambda = grid)

# explore fit
grid[50]
grid[60]

fit.lasso$lambda[50]
coef(fit.lasso)[,50]
sum(abs(coef(fit.lasso)[-1,50])) # l1-norm 	

fit.lasso$lambda[60]
coef(fit.lasso)[,60]
sum(abs(coef(fit.lasso)[-1,60]))

plot(fit.lasso, xvar = "lambda")
add.names(fit.lasso)
# each curve corresponds to a feature path
# number of non-zero parameters above at a given lambda 

print(fit.lasso)
# The Df column is the number of nonzero parameters
# %dev is the percent deviance explained

coef(fit.lasso, s = 0.1)	# s is lambda

# prediction with lambda = 50
pred.lasso <- predict(fit.lasso, newx = x.test, s = 50)
# mean squared error
fit.lasso.mse <- mean((test$Apps - pred.lasso)^2)

# prediction with lambda = 0, i.e., linear regression without regularization
pred.0 <- predict(fit.lasso, newx = x.test, s = 0)
# mean squared error
fit.lasso.0.mse <- mean((test$Apps - pred.0)^2)
# In general, if we want to fit a unergularized LR, then
# we should use the lm() function -- see Exercise #2


## run 10-cross-validation lasso
set.seed(123) # choice of the cross-validation folds is random
fit.lasso.cv <- cv.glmnet(x, y, alpha = 1, lambda = grid)

# explore fit
plot(fit.lasso.cv)
# cross-validation curve and 
# upper and lower standard deviation curves
# along the lambda sequence

fit.lasso.cv$name	# loss fucntion used in cross-validation
fit.lasso.cv$lambda.min # value of lambda that gives minimum mean cross-validated error 
fit.lasso.cv$lambda.1se # largest value of lambda such that error is within 1 standard error of the minimum

coef(fit.lasso.cv, s = "lambda.min")

plot(fit.lasso.cv$glmnet.fit, "lambda", label = TRUE)
add.names(fit.lasso.cv$glmnet.fit)

# prediction
lambda.best <- fit.lasso.cv$lambda.min
pred.lasso.cv <- predict(fit.lasso.cv, newx = x.test, s = lambda.best)
fit.lasso.cv.mse <- mean((test$Apps-pred.lasso.cv)^2)


###############
## EXERCISE #2.2: Ridge regression
###############

## run 10-cross-validation ridge regression
set.seed(123) # choice of the cross-validation folds is random
fit.ridge.cv <- cv.glmnet(x, y,
	alpha = 0,
	lambda = grid)

# prediction
lambda.best <- fit.ridge.cv$lambda.min
pred.ridge.cv <- predict(fit.ridge.cv, newx = x.test, s = lambda.best)
fit.ridge.cv.mse <- mean((test$Apps - pred.ridge.cv)^2)

## fit.lm, fit.ridge.cv, fit.lasso.cv

fit.lm.mse
fit.lasso.cv.mse
fit.ridge.cv.mse

# parameter values of fit.lm
lm <- as.matrix(coef(fit.lm)) #y Lasso 

# parameter values for lambda.min by Ridge regression
ridge <- coef(fit.ridge.cv, s = fit.ridge.cv$lambda.min)

# parameter values for lambda.min by Lasso regression
lasso <- coef(fit.lasso.cv, s = fit.lasso.cv$lambda.min)

## compare parameters
cbind2(ridge, lasso)

###############
## EXERCISE #3: Logistic regression
##		target value: newY
###############

# College data set
# Target value: newY
# create a binary attribute, newY,
# that contains a 1 if Apps contains a value above its median,
# and a 0 if Apps contains a value below its median

med <- median(train$Apps)
newY <- rep("one", nrow(train))
newY[train$Apps < med] <- "zero"
train.college <- data.frame(train, newY)
train.college <- train.college[, -1]

med <- median(test$Apps)
newY <- rep("one", nrow(test))
newY[test$Apps < med] <- "zero"
test.college <- data.frame(test, newY)
test.college <- test.college[, -1]

###############
## EXERCISE #3.1: without regularization
##		target value: newY
###############

# training
glm.fit <- glm(formula = newY ~ .,
	family = binomial,
	data = train.college)

# prediction
glm.probs <- predict(glm.fit,
	newdata = test.college,
	type = "response")

glm.pred <- rep("one", nrow(test.college))
glm.pred[glm.probs > 0.5] = "zero"

# evaluation
table(glm.pred, test.college$newY)
glm.fit.acc <- mean(glm.pred == test.college$newY)

###############
## EXERCISE #3.1: Lasso and Ridge regression
##		target value: newY
###############

## 10-cross-validation Ridge regression
x <- model.matrix(newY ~ ., data = train.college)
y <- data.matrix(train.college$newY)

x.test <- model.matrix(newY~., data=test.college)
y.test <- data.matrix(test.college$newY)

fit.log.ridge <- cv.glmnet(x, y,
	family = "binomial",
	alpha = 0,
	type.measure = "class", 
	lambda = grid)

# prediction
lambda.best <- fit.log.ridge$lambda.min

pred.log.ridge <- predict(fit.log.ridge,
	type = "class",
	newx = x.test,
	s = lambda.best)

# evaluation
fit.log.ridge.acc <- mean(pred.log.ridge == y.test)

## 10-cross-validation Lasso
fit.log.lasso <- cv.glmnet(x, y,
	family = "binomial",
	alpha = 1,
	lambda = grid)

# prediction
lambda.best <- fit.log.lasso$lambda.min

pred.log.lasso <- predict(fit.log.lasso,
	type = "class",
	newx = x.test,
	s = lambda.best)

# evaluation
fit.log.lasso.acc <- mean(pred.log.lasso == y.test)

# compare accuracy
fit.log.lasso.acc
fit.log.ridge.acc
glm.fit.acc
