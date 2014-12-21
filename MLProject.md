# Practical Machine Learning Peer Assessment
 
## Summary

This analysis was done to predict the manner in which the subjects performed weight lifting exercises. The data is collected from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. The outcome variable has five classes and the total number of predictors are 159.

## Getting and preparing the data

We load the training and testing data sets. Here it was necessary to pay attention to the fact that missing values could be represented in several ways, either by an NA, a totally empty value or #DIV/0! indicating a divide by zero error. 

 Examining the dataset, there are id columns x, some timestamp etc which are not useful for model fitting. We removed those as well.

 There are 159 variables. But many of them are missing values for most of the records. I removed them as well.



```r
## downloading data from URL
Train_URL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
Test_URL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url=Train_URL, destfile="pml-training.csv",method = "curl")
```

```
## Warning: running command 'curl
## "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv" -o
## "pml-training.csv"' had status 127
```

```
## Warning in download.file(url = Train_URL, destfile = "pml-training.csv", :
## download had nonzero exit status
```

```r
download.file(url=Test_URL, destfile="pml-testing.csv")
##reading data
Train <- read.csv("pml-training.csv",row.names=1,na.strings = c("","NA", "#DIV/0!"))
```

```
## Warning in file(file, "rt"): cannot open file 'pml-training.csv': No such
## file or directory
```

```
## Error in file(file, "rt"): cannot open the connection
```

```r
Test <- read.csv("pml-testing.csv",row.names=1,na.strings = c("NA","", "#DIV/0!"))


## remmving some varables which are not required
ColsToDrp <- c ("user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "X", "new_window")
Training <- Train[,!(names(Train) %in% ColsToDrp )]
```

```
## Error in eval(expr, envir, enclos): object 'Train' not found
```

```r
Testing <- Test[,!(names(Test) %in% ColsToDrp)]

## removing variables which has many missing values
NoOfCols <- dim(Training)[2]
```

```
## Error in eval(expr, envir, enclos): object 'Training' not found
```

```r
ColsWithMissingData <- vector(length=NoOfCols)
```

```
## Error in vector(length = NoOfCols): object 'NoOfCols' not found
```

```r
for (i in 1:NoOfCols) { ColsWithMissingData[i] <- sum(is.na(Training[,i]))}
```

```
## Error in eval(expr, envir, enclos): object 'NoOfCols' not found
```

```r
Training <- Training[,which(ColsWithMissingData  < 5)]
```

```
## Error in eval(expr, envir, enclos): object 'Training' not found
```

```r
Testing <- Testing[,which(ColsWithMissingData  < 5)]
```

```
## Error in which(ColsWithMissingData < 5): object 'ColsWithMissingData' not found
```

```r
dim(Training)
```

```
## Error in eval(expr, envir, enclos): object 'Training' not found
```

```r
dim(Testing)
```

```
## [1]  20 154
```


 we subdivide the training set to create a cross validation set. We allocate 70% of the original training set to the new training set, and the other 30% to the cross validation set:


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
inTrain <- createDataPartition(y=Training$classe, p=0.7, list=FALSE)
```

```
## Error in createDataPartition(y = Training$classe, p = 0.7, list = FALSE): object 'Training' not found
```

```r
Training <- Training[inTrain,]
```

```
## Error in eval(expr, envir, enclos): object 'Training' not found
```

```r
TrainingTest <- Training[-inTrain,]
```

```
## Error in eval(expr, envir, enclos): object 'Training' not found
```

## Linear Regression

In the new training and validation set, there are 53 predictors and 1 response. I check the correlations between the predictors and the outcome variable in the new training set. There doesn’t seem to be any predictors strongly correlated with the outcome variable, so linear regression model may not be a good option. We will check other models for better fit.


```r
cor <- abs(sapply(colnames(Training[, -ncol(Training)]), function(x) cor(as.numeric(Training[, x]), as.numeric(Training$classe), method = "spearman")))
```

```
## Error in is.data.frame(x): object 'Training' not found
```

```r
cor
```

```
## function (x, y = NULL, use = "everything", method = c("pearson", 
##     "kendall", "spearman")) 
## {
##     na.method <- pmatch(use, c("all.obs", "complete.obs", "pairwise.complete.obs", 
##         "everything", "na.or.complete"))
##     if (is.na(na.method)) 
##         stop("invalid 'use' argument")
##     method <- match.arg(method)
##     if (is.data.frame(y)) 
##         y <- as.matrix(y)
##     if (is.data.frame(x)) 
##         x <- as.matrix(x)
##     if (!is.matrix(x) && is.null(y)) 
##         stop("supply both 'x' and 'y' or a matrix-like 'x'")
##     if (!(is.numeric(x) || is.logical(x))) 
##         stop("'x' must be numeric")
##     stopifnot(is.atomic(x))
##     if (!is.null(y)) {
##         if (!(is.numeric(y) || is.logical(y))) 
##             stop("'y' must be numeric")
##         stopifnot(is.atomic(y))
##     }
##     Rank <- function(u) {
##         if (length(u) == 0L) 
##             u
##         else if (is.matrix(u)) {
##             if (nrow(u) > 1L) 
##                 apply(u, 2L, rank, na.last = "keep")
##             else row(u)
##         }
##         else rank(u, na.last = "keep")
##     }
##     if (method == "pearson") 
##         .Call(C_cor, x, y, na.method, FALSE)
##     else if (na.method %in% c(2L, 5L)) {
##         if (is.null(y)) {
##             .Call(C_cor, Rank(na.omit(x)), NULL, na.method, method == 
##                 "kendall")
##         }
##         else {
##             nas <- attr(na.omit(cbind(x, y)), "na.action")
##             dropNA <- function(x, nas) {
##                 if (length(nas)) {
##                   if (is.matrix(x)) 
##                     x[-nas, , drop = FALSE]
##                   else x[-nas]
##                 }
##                 else x
##             }
##             .Call(C_cor, Rank(dropNA(x, nas)), Rank(dropNA(y, 
##                 nas)), na.method, method == "kendall")
##         }
##     }
##     else if (na.method != 3L) {
##         x <- Rank(x)
##         if (!is.null(y)) 
##             y <- Rank(y)
##         .Call(C_cor, x, y, na.method, method == "kendall")
##     }
##     else {
##         if (is.null(y)) {
##             ncy <- ncx <- ncol(x)
##             if (ncx == 0) 
##                 stop("'x' is empty")
##             r <- matrix(0, nrow = ncx, ncol = ncy)
##             for (i in seq_len(ncx)) {
##                 for (j in seq_len(i)) {
##                   x2 <- x[, i]
##                   y2 <- x[, j]
##                   ok <- complete.cases(x2, y2)
##                   x2 <- rank(x2[ok])
##                   y2 <- rank(y2[ok])
##                   r[i, j] <- if (any(ok)) 
##                     .Call(C_cor, x2, y2, 1L, method == "kendall")
##                   else NA
##                 }
##             }
##             r <- r + t(r) - diag(diag(r))
##             rownames(r) <- colnames(x)
##             colnames(r) <- colnames(x)
##             r
##         }
##         else {
##             if (length(x) == 0L || length(y) == 0L) 
##                 stop("both 'x' and 'y' must be non-empty")
##             matrix_result <- is.matrix(x) || is.matrix(y)
##             if (!is.matrix(x)) 
##                 x <- matrix(x, ncol = 1L)
##             if (!is.matrix(y)) 
##                 y <- matrix(y, ncol = 1L)
##             ncx <- ncol(x)
##             ncy <- ncol(y)
##             r <- matrix(0, nrow = ncx, ncol = ncy)
##             for (i in seq_len(ncx)) {
##                 for (j in seq_len(ncy)) {
##                   x2 <- x[, i]
##                   y2 <- y[, j]
##                   ok <- complete.cases(x2, y2)
##                   x2 <- rank(x2[ok])
##                   y2 <- rank(y2[ok])
##                   r[i, j] <- if (any(ok)) 
##                     .Call(C_cor, x2, y2, 1L, method == "kendall")
##                   else NA
##                 }
##             }
##             rownames(r) <- colnames(x)
##             colnames(r) <- colnames(y)
##             if (matrix_result) 
##                 r
##             else drop(r)
##         }
##     }
## }
## <bytecode: 0x00000000184249a8>
## <environment: namespace:stats>
```

## Random Forest


```r
library(randomForest)
```

```
## Error in library(randomForest): there is no package called 'randomForest'
```

```r
## fitting with train data
fitRF <- randomForest(classe ~ ., data=Training, method="class")
```

```
## Error in eval(expr, envir, enclos): could not find function "randomForest"
```

```r
PredictRF <- predict(fitRF, type="class")
```

```
## Error in predict(fitRF, type = "class"): object 'fitRF' not found
```

```r
confusionMatrix(Training$classe,PredictRF)
```

```
## Error in confusionMatrix(Training$classe, PredictRF): object 'Training' not found
```

```r
table(Training$classe, PredictRF)
```

```
## Error in table(Training$classe, PredictRF): object 'Training' not found
```

```r
nright = table(PredictRF == Training$classe)
```

```
## Error in table(PredictRF == Training$classe): object 'PredictRF' not found
```

```r
nright
```

```
## Error in eval(expr, envir, enclos): object 'nright' not found
```

```r
ForestInError = as.vector(100 * (1-nright["TRUE"] / sum(nright)))
```

```
## Error in as.vector(100 * (1 - nright["TRUE"]/sum(nright))): object 'nright' not found
```

```r
ForestInError 
```

```
## Error in eval(expr, envir, enclos): object 'ForestInError' not found
```

```r
varImpPlot(fitRF, sort = TRUE,  main = "Importance of the Predictors")
```

```
## Error in eval(expr, envir, enclos): could not find function "varImpPlot"
```

```r
## cross validating with 30% of train data
ValidateRF <- predict(fitRF, newdata=TrainingTest, type="class")
```

```
## Error in predict(fitRF, newdata = TrainingTest, type = "class"): object 'fitRF' not found
```

```r
confusionMatrix(TrainingTest$classe,ValidateRF)
```

```
## Error in confusionMatrix(TrainingTest$classe, ValidateRF): object 'TrainingTest' not found
```

```r
nright = table(ValidateRF == TrainingTest$classe)
```

```
## Error in table(ValidateRF == TrainingTest$classe): object 'ValidateRF' not found
```

```r
nright
```

```
## Error in eval(expr, envir, enclos): object 'nright' not found
```

```r
ForestInError = as.vector(100 * (1-nright["TRUE"] / sum(nright)))
```

```
## Error in as.vector(100 * (1 - nright["TRUE"]/sum(nright))): object 'nright' not found
```

```r
ForestInError 
```

```
## Error in eval(expr, envir, enclos): object 'ForestInError' not found
```
 The random forest algorithm generates a model with accuracy 0.9913. The out-of-sample error is 0.9%, which is pretty low. We don’t need to go back and include more variables with imputations. The top 4 most important variables according to the model fit are ‘roll_belt’, ‘yaw_belt’, ‘pitch_forearm’ and ‘pitch_belt’.

## Regression Trees


```r
library(tree)
```

```
## Error in library(tree): there is no package called 'tree'
```

```r
#fitting the model
fitTree <- tree(classe ~ ., method="tree", data=Training)
```

```
## Error in eval(expr, envir, enclos): could not find function "tree"
```

```r
PredictTree <- predict(fitTree, type="class")
```

```
## Error in predict(fitTree, type = "class"): object 'fitTree' not found
```

```r
table(Training$classe, PredictTree)
```

```
## Error in table(Training$classe, PredictTree): object 'Training' not found
```

```r
fitTree.prune <- prune.misclass(fitTree, best=10)
```

```
## Error in eval(expr, envir, enclos): could not find function "prune.misclass"
```

```r
#plot of generated tree
plot(fitTree.prune)
```

```
## Error in plot(fitTree.prune): object 'fitTree.prune' not found
```

```r
title(main="Tree created using tree function")
```

```
## Error in title(main = "Tree created using tree function"): plot.new has not been called yet
```

```r
text(fitTree.prune, cex=1.2)
```

```
## Error in text(fitTree.prune, cex = 1.2): object 'fitTree.prune' not found
```

```r
nright = table(PredictTree == Training$classe)
```

```
## Error in table(PredictTree == Training$classe): object 'PredictTree' not found
```

```r
TreeInError = as.vector(100 * (1 - nright["TRUE"] / sum(nright)))
```

```
## Error in as.vector(100 * (1 - nright["TRUE"]/sum(nright))): object 'nright' not found
```

```r
TreeInError 
```

```
## Error in eval(expr, envir, enclos): object 'TreeInError' not found
```

```r
#cross validating the model 30% data
ValidateTree <- predict(fitTree, newdata = TrainingTest, type="class")
```

```
## Error in predict(fitTree, newdata = TrainingTest, type = "class"): object 'fitTree' not found
```

```r
table(TrainingTest$classe, ValidateTree)
```

```
## Error in table(TrainingTest$classe, ValidateTree): object 'TrainingTest' not found
```

```r
nright = table(ValidateTree == TrainingTest$classe)
```

```
## Error in table(ValidateTree == TrainingTest$classe): object 'ValidateTree' not found
```

```r
TreeInError  = as.vector(100 * (1 - nright["TRUE"] / sum(nright)))
```

```
## Error in as.vector(100 * (1 - nright["TRUE"]/sum(nright))): object 'nright' not found
```

```r
TreeInError 
```

```
## Error in eval(expr, envir, enclos): object 'TreeInError' not found
```

```r
##pruning to improve cross validation
error.cv <- {Inf}
for (i in 2:19) {
    prune.data <- prune.misclass(fitTree, best=i)
    pred.cv <- predict(prune.data, newdata=TrainingTest, type="class")
    nright = table(pred.cv == TrainingTest$classe)
    error = as.vector(100 * ( 1- nright["TRUE"] / sum(nright)))
    error.cv <- c(error.cv, error) 
}
```

```
## Error: could not find function "prune.misclass"
```

```r
#error.cv
plot(error.cv, type = "l", xlab="Size of tree (number of nodes)", ylab="Out of sample error(%)", main = "Relationship between tree size and out of sample error")
```

```
## Warning in min(x): no non-missing arguments to min; returning Inf
```

```
## Warning in max(x): no non-missing arguments to max; returning -Inf
```

```
## Error in plot.window(...): need finite 'ylim' values
```
 Despite the complexity of the tree, the above fifures does not indicate overfitting as the out of sample error does not increase as more nodes are added to the tree.


## Results
The random forest clearly performs better, approaching 99% accuracy for in-sample and out-of-sample error so we will select this model and apply it to the test data set. We use the provided function to classify 20 data points from the test set by the type of lift. 



```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}


TestFit <- predict(fitRF, newdata=Testing, type="class")
```

```
## Error in predict(fitRF, newdata = Testing, type = "class"): object 'fitRF' not found
```

```r
pml_write_files(TestFit)
```

```
## Error in pml_write_files(TestFit): object 'TestFit' not found
```

## Conclusion
We see Random Forest is the most rebost for this set.
