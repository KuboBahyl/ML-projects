remove(list=objects())
movies <- read.csv("movies.csv", header=TRUE,
                   stringsAsFactors=FALSE)
# First look
hist(movies$imdb_score, nclass="FD", main="Histogram hodnoten? na IMDB", xlab="Hodnotenie",
     ylab="Frekvencia", xlim=c(2,9), col="gray")

# cutting
moviesDATA <- na.omit(movies)
moviesDATA <- moviesDATA[moviesDATA$imdb_score>=8 | moviesDATA$imdb_score<5,]

attach(moviesDATA)

##################################################################################
## Contingency tables ##

imdb_score[imdb_score>=8] = "good"
imdb_score[imdb_score!="good"] = "bad"

country[country!="USA"] = "Other"

title_year[title_year>=2000] = "new"
title_year[title_year!="new"] = "old"

odds <- function(O)
{
  O[1,1]*O[2,2]/(O[1,2]*O[2,1])
}

# country vs score
table1 <- table(country, imdb_score)
table1
mosaic(table1, dir=c("h", "v"), shade=TRUE)
chisq.test(table1, correct=FALSE)
or <- odds(table1)
D <- log(or) / sum(1/table1)
2*pnorm(D)

# country vs score vs year
table2 <- table(country, imdb_score, title_year)
table2
mosaic(table2, condvars = c(3))
or1 <- odds(table(imdb_score[title_year=="new"], country[title_year=="new"]))
or2 <- odds(table(imdb_score[title_year=="old"], country[title_year=="old"]))
woolf_test(table2)
breslowday.test(table2)

# country vs score for old years
imdb_score.old <- imdb_score[title_year=="old"]
country.old <- country[title_year=="old"]
table3 <- table(country.old, imdb_score.old)
table3
mosaic(table3, dir=c("h", "v"), shade=TRUE)
chisq.test(table3, correct=FALSE)
or <- odds(table3)
D <- log(or) / sum(1/table3)
2*pnorm(D)


## Breslow - Day test
breslowday.test <- function(x) {
  #Find the common OR based on Mantel-Haenszel
  or.hat.mh <- mantelhaen.test(x)$estimate
  #Number of strata
  K <- dim(x)[3]
  #Value of the Statistic
  X2.HBD <- 0
  #Value of aj, tildeaj and Var.aj
  a <- tildea <- Var.a <- numeric(K)
  
  for (j in 1:K) {
    #Find marginals of table j
    mj <- apply(x[,,j], MARGIN=1, sum)
    nj <- apply(x[,,j], MARGIN=2, sum)
    
    #Solve for tilde(a)_j
    coef <- c(-mj[1]*nj[1] * or.hat.mh, nj[2]-mj[1]+or.hat.mh*(nj[1]+mj[1]),
              1-or.hat.mh)
    sols <- Re(polyroot(coef))
    #Take the root, which fulfills 0 < tilde(a)_j <= min(n1_j, m1_j)
    tildeaj <- sols[(0 < sols) &  (sols <= min(nj[1],mj[1]))]
    #Observed value
    aj <- x[1,1,j]
    
    #Determine other expected cell entries
    tildebj <- mj[1] - tildeaj
    tildecj <- nj[1] - tildeaj
    tildedj <- mj[2] - tildecj
    
    #Compute \hat{\Var}(a_j | \widehat{\OR}_MH)
    Var.aj <- (1/tildeaj + 1/tildebj + 1/tildecj + 1/tildedj)^(-1)
    
    #Compute contribution
    X2.HBD <- X2.HBD + as.numeric((aj - tildeaj)^2 / Var.aj)
    
    #Assign found value for later computations
    a[j] <- aj ;  tildea[j] <- tildeaj ; Var.a[j] <- Var.aj
  }
  
  #Compute Tarone corrected test
  X2.HBDT <-as.numeric( X2.HBD -  (sum(a) - sum(tildea))^2/sum(Var.aj) )
  
  #Compute p-value based on the Tarone corrected test
  p <- 1-pchisq(X2.HBDT, df=K-1)
  
  res <- list(X2.HBD=X2.HBD,X2.HBDT=X2.HBDT,p=p)
  class(res) <- "bdtest"
  return(res)
}

############################################################################
############################################################################
############################################################################

remove(list=objects())
movies <- read.csv("movies.csv", header=TRUE,
                   stringsAsFactors=FALSE)

# cutting
moviesDATA <- na.omit(movies)
moviesTEST <- na.omit(movies)

moviesDATA <- moviesDATA[moviesDATA$imdb_score>=7.8 | moviesDATA$imdb_score<5,]
moviesTEST <- moviesTEST[(moviesTEST$imdb_score<7.8 & moviesTEST$imdb_score>=7.6) 
                         | (moviesTEST$imdb_score>=5 & moviesTEST$imdb_score<=5.2),]

# Only USA movies
moviesDATA <- moviesDATA[moviesDATA$country=="USA",]
moviesTEST <- moviesTEST[moviesTEST$country=="USA",]
attach(moviesDATA)


##################################################################################
## Logistic regression ##

imdb_score[imdb_score>=7.8] = "good"
imdb_score[imdb_score!="good"] = "bad"

# in millions of dollars
budget <- budget/1000000

# regarding to standard 2h
duration <- duration - 120

# in thausends of likes
actor_1_facebook_likes <- actor_1_facebook_likes/1000

# color or white film
color[color=="Color"] = "color"
color[color==" Black and White"] = "BW"

prob <- function(z)
{
  1/( 1 + exp(-z))
}

movieModel <- glm(as.factor(imdb_score)~1 + budget + duration + actor_1_facebook_likes + color,
                  family=binomial(link="logit"))

summary(movieModel)

betaHAT <- movieModel$coefficients
res.dev <- movieModel$deviance
null.dev <- movieModel$null.deviance
inv.I <-  summary(movieModel)$cov.unscaled

exp(betaHAT[1])

## Null model test
1-pchisq(null.dev - res.dev,df=5-1)

## Submodels

# H0: beta(budget) = 0
movieSubmodel1 <- glm(as.factor(imdb_score)~1 + duration + actor_1_facebook_likes + color,
                      family=binomial(link="logit"))
res.devSub1 <- movieSubmodel1$deviance
pchisq(res.devSub1-res.dev,df=5-4)

# H0: beta(color) = 0
movieSubmodel2 <- glm(as.factor(imdb_score)~1 + budget + duration + actor_1_facebook_likes,
                      family=binomial(link="logit"))
res.devSub2 <- movieSubmodel2$deviance
1 - pchisq(res.devSub2-res.dev,df=5-4)

dolnabeta4 <- betaHAT[5] - qnorm(0.975)*sqrt(inv.I[5,5])
hornabeta4 <- betaHAT[5] + qnorm(0.975)*sqrt(inv.I[5,5])
exp(dolnabeta4)
exp(betaHAT[5])
exp(hornabeta4)


## Probability

# no budget, 1.5h dur, 20k likes, color
a <- c(1,0,-30,20,1)
pHAT <- prob(t(a)%*%betaHAT)

p0 <- 0.5

V <- (t(a)%*%betaHAT - log(p0/(1-p0))) / sqrt(t(a) %*% inv.I %*% a)
pnorm(V)

pDolny <- prob(t(a)%*%betaHAT - qnorm(0.975)*sqrt(t(a) %*% inv.I %*% a))
pHorny <- prob(t(a)%*%betaHAT + qnorm(0.975)*sqrt(t(a) %*% inv.I %*% a))

## Validation


imdbTEST <- moviesTEST$imdb_score
n <- length(imdbTEST)

moviesTEST$color[moviesTEST$color=="Color"] = 1
moviesTEST$color[moviesTEST$color==" Black and White"] = 0
moviesTEST$color <- as.numeric(moviesTEST$color)

dataTEST <- cbind(
  rep(1, times=n),
  moviesTEST$budget/1000000,
  moviesTEST$duration - 120,
  moviesTEST$actor_1_facebook_likes/1000,
  moviesTEST$color
)

imdbTEST[imdbTEST>7] = "good"
imdbTEST[imdbTEST!="good"] = "bad"

pHAT <- prob(dataTEST%*%betaHAT)

tableTEST <- table(predpoved = pHAT>0.5  ,  skutocnost = imdbTEST=="good")

sensitivity <- tableTEST[2,2]/(tableTEST[2,2] + tableTEST[1,2])
specificity <- tableTEST[1,1]/(tableTEST[1,1] + tableTEST[2,1])
accuracy <- (tableTEST[1,1] + tableTEST[2,2]) / sum(tableTEST)

############################################################################
############################################################################
############################################################################

remove(list=objects())
movies <- read.csv("movies.csv", header=TRUE,
                   stringsAsFactors=FALSE)

# cutting
moviesDATA <- na.omit(movies)

# Only USA movies
moviesDATA <- moviesDATA[moviesDATA$country=="Canada",]

###################################################################################
## Bootstrap ##

imdb <- moviesDATA$imdb_score
n <- length(imdb)
hist(imdb, col="blue", nclass=10, main="Histogram of Canada movie reviews",
     xlab="IMDB reviews", ylab = "Frekvencia", xlim=c(0,10))

medianHAT <- median(imdb)

median.boot <- NULL
set.seed(10)
N <- 10000
for(i in 1:N){
  print(i)
  flush.console()
  places <- sample(1:n,size=n,replace=TRUE)
  median.boot[i] <- median(imdb[places])
}

hist(median.boot,col="red", main="Histogram medi8",
     xlab="medians", ylab="Frekvencia", xlim=c(5.5,7))

var.boot <- var(median.boot)

# vychylenie
biasHAT <- mean(median.boot) - medianHAT

medDolny <- medianHAT - biasHAT - qnorm(0.975)*sqrt(var.boot)
medHorny <- medianHAT - biasHAT + qnorm(0.975)*sqrt(var.boot)
