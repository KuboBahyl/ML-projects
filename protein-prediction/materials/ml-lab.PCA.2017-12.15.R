#################################################################################
#################################################################################

### Principal Component Analysis

### Auto data set

### Barbora Hladka, Martin Holub

### http://ufal.mff.cuni.cz/course/npfl054

#################################################################################
#################################################################################

library(ISLR)

################
## EXERCISE #1: Read and explore data
################

a <- Auto[c("mpg", "cylinders", "horsepower", "weight")]
apply(a, 2, mean)
apply(a, 2, var)

plot(a) # scatter plot matrix
cov(a) # covariation matrix
cor(a) # correlation matrix

##########
## EXERCISE #2: Principal Component Analysis
##########

## scaling 
pca <- prcomp(a, scale = TRUE)
names(pca)
# [1] "sdev"     "rotation" "center"   "scale"    "x" 

summary(pca)	# coefficients of the components
pca$rotation	# loadings
pca$center		# mean values
pca$scale
pca$x			# scores
boxplot(pca$x)


pca$sdev		# standard deviation of each PC
pca.var <- pca$sdev^2	# variance explained by each PC
pca.var
sum(pca.var)		# total variance explained by all PCs

# Proportional Variance Explained by each PC (PVE)
pve <- pca.var/sum(pca.var)

plot(pve,
	xlab = "Principal Component",
	main = "Scree plot: Auto data set",
	ylab = "Proportion of Variance Explained", 
	ylim = c(0,1),
	type = 'b')

plot(cumsum(pve),
	xlab = "Principal Component",
	ylab = "Cumulative Proportion of Variance Explained",
	main = "Scree plot: Auto data set",
	ylim = c(0,1),
	type = 'b')

install.packages('devtools')
library(devtools)
install_github("ggbiplot", "vqv")

library(ggbiplot)
g <- ggbiplot(pca, obs.scale = 1, var.scale = 1, 
              ellipse = TRUE, 
              circle = TRUE)
g <- g + scale_color_discrete(name = '')
g <- g + theme(legend.direction = 'horizontal', 
               legend.position = 'top')
print(g)

# biplot
biplot(pca,
	scale = T, # the arrows are scaled to represent the loadings
	main = "Biplot: scaled Auto data set",
	xlab = "First Principal Component",
	ylab = "Second Principal Component")

# biplot with the car names
biplot(pca,
	scale = T,
	xlabs = Auto$name,
	main = "Biplot: scaled Auto data set",
	xlab = "First Principal Component",
	ylab = "Second Principal Component")


## unscaling features
pca.un <- prcomp(a, scale = FALSE)
biplot(pca.un,
	scale = T,
	xlabs = rep("O", nrow(a)),
	main = "Biplot: unscaled Auto data set",
	xlab = "First Principal Component",
	ylab = "Second Principal Component")

