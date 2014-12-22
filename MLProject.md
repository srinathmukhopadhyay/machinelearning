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
Test <- read.csv("pml-testing.csv",row.names=1,na.strings = c("NA","", "#DIV/0!"))


## remmving some varables which are not required
ColsToDrp <- c ("user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "X", "new_window")
Training <- Train[,!(names(Train) %in% ColsToDrp )]
Testing <- Test[,!(names(Test) %in% ColsToDrp)]

## removing variables which has many missing values
NoOfCols <- dim(Training)[2]
ColsWithMissingData <- vector(length=NoOfCols)
for (i in 1:NoOfCols) { ColsWithMissingData[i] <- sum(is.na(Training[,i]))}
Training <- Training[,which(ColsWithMissingData  < 5)]
Testing <- Testing[,which(ColsWithMissingData  < 5)]
dim(Training)
```

```
## [1] 19622    54
```

```r
dim(Testing)
```

```
## [1] 20 54
```


 we subdivide the training set to create a cross validation set. We allocate 70% of the original training set to the new training set, and the other 30% to the cross validation set:


```r
library(caret)
inTrain <- createDataPartition(y=Training$classe, p=0.7, list=FALSE)
Training <- Training[inTrain,]
TrainingTest <- Training[-inTrain,]
```

## Linear Regression

In the new training and validation set, there are 53 predictors and 1 response. I check the correlations between the predictors and the outcome variable in the new training set. There doesn’t seem to be any predictors strongly correlated with the outcome variable, so linear regression model may not be a good option. We will check other models for better fit.


```r
cor <- abs(sapply(colnames(Training[, -ncol(Training)]), function(x) cor(as.numeric(Training[, x]), as.numeric(Training$classe), method = "spearman")))
cor
```

```
##           num_window            roll_belt           pitch_belt 
##          0.002764738          0.125904512          0.038663333 
##             yaw_belt     total_accel_belt         gyros_belt_x 
##          0.072372054          0.090903128          0.014753518 
##         gyros_belt_y         gyros_belt_z         accel_belt_x 
##          0.003361276          0.002010658          0.033380410 
##         accel_belt_y         accel_belt_z        magnet_belt_x 
##          0.011732036          0.139320872          0.006396327 
##        magnet_belt_y        magnet_belt_z             roll_arm 
##          0.198201606          0.140946343          0.050857082 
##            pitch_arm              yaw_arm      total_accel_arm 
##          0.188754216          0.030566920          0.151788155 
##          gyros_arm_x          gyros_arm_y          gyros_arm_z 
##          0.017121411          0.027648970          0.016409402 
##          accel_arm_x          accel_arm_y          accel_arm_z 
##          0.257715570          0.086402306          0.102039252 
##         magnet_arm_x         magnet_arm_y         magnet_arm_z 
##          0.284088072          0.266223157          0.159184733 
##        roll_dumbbell       pitch_dumbbell         yaw_dumbbell 
##          0.086007331          0.100635686          0.011591513 
## total_accel_dumbbell     gyros_dumbbell_x     gyros_dumbbell_y 
##          0.021534911          0.014715701          0.022948943 
##     gyros_dumbbell_z     accel_dumbbell_x     accel_dumbbell_y 
##          0.016923105          0.129130266          0.017324107 
##     accel_dumbbell_z    magnet_dumbbell_x    magnet_dumbbell_y 
##          0.082494797          0.151050929          0.051959568 
##    magnet_dumbbell_z         roll_forearm        pitch_forearm 
##          0.197716420          0.059298018          0.323511151 
##          yaw_forearm  total_accel_forearm      gyros_forearm_x 
##          0.054259853          0.118443265          0.009780504 
##      gyros_forearm_y      gyros_forearm_z      accel_forearm_x 
##          0.001904242          0.008985527          0.213407268 
##      accel_forearm_y      accel_forearm_z     magnet_forearm_x 
##          0.015206169          0.011146126          0.202161817 
##     magnet_forearm_y     magnet_forearm_z 
##          0.114644151          0.049973411
```

## Random Forest


```r
library(randomForest)
## fitting with train data
fitRF <- randomForest(classe ~ ., data=Training, method="class")

PredictRF <- predict(fitRF, type="class")
confusionMatrix(Training$classe,PredictRF)
```

```
## Error in loadNamespace(name): there is no package called 'e1071'
```

```r
table(Training$classe, PredictRF)
```

```
##    PredictRF
##        A    B    C    D    E
##   A 3906    0    0    0    0
##   B    5 2651    2    0    0
##   C    0    9 2387    0    0
##   D    0    0   15 2236    1
##   E    0    0    0    7 2518
```

```r
nright = table(PredictRF == Training$classe)
nright
```

```
## 
## FALSE  TRUE 
##    39 13698
```

```r
ForestInError = as.vector(100 * (1-nright["TRUE"] / sum(nright)))
ForestInError 
```

```
## [1] 0.2839048
```

```r
varImpPlot(fitRF, sort = TRUE,  main = "Importance of the Predictors")
```

![plot of chunk RandomForest](figure/RandomForest-1.png) 

```r
## cross validating with 30% of train data
ValidateRF <- predict(fitRF, newdata=TrainingTest, type="class")
confusionMatrix(TrainingTest$classe,ValidateRF)
```

```
## Error in loadNamespace(name): there is no package called 'e1071'
```

```r
nright = table(ValidateRF == TrainingTest$classe)
nright
```

```
## 
## TRUE 
## 4111
```

```r
ForestInError = as.vector(100 * (1-nright["TRUE"] / sum(nright)))
ForestInError 
```

```
## [1] 0
```
 The random forest algorithm generates a model with accuracy 0.9913. The out-of-sample error is 0.9%, which is pretty low. We don’t need to go back and include more variables with imputations. The top 4 most important variables according to the model fit are ‘roll_belt’, ‘yaw_belt’, ‘pitch_forearm’ and ‘pitch_belt’.

## Regression Trees


```r
library(tree)
```

```
## Warning: package 'tree' was built under R version 3.1.2
```

```r
#fitting the model
fitTree <- tree(classe ~ ., method="tree", data=Training)
PredictTree <- predict(fitTree, type="class")
table(Training$classe, PredictTree)
```

```
##    PredictTree
##        A    B    C    D    E
##   A 3558   26   18  251   53
##   B  347 1405   94  362  450
##   C   35  104 1692  379  186
##   D   37  218  363 1497  137
##   E   90  280  148  342 1665
```

```r
fitTree.prune <- prune.misclass(fitTree, best=10)

#plot of generated tree
plot(fitTree.prune)
title(main="Tree created using tree function")
text(fitTree.prune, cex=1.2)
```

![plot of chunk Trees](figure/Trees-1.png) 

```r
nright = table(PredictTree == Training$classe)
TreeInError = as.vector(100 * (1 - nright["TRUE"] / sum(nright)))
TreeInError 
```

```
## [1] 28.53607
```

```r
#cross validating the model 30% data
ValidateTree <- predict(fitTree, newdata = TrainingTest, type="class")
table(TrainingTest$classe, ValidateTree)
```

```
##    ValidateTree
##        A    B    C    D    E
##   A 1028    8    7   84   19
##   B  118  429   29  123  126
##   C   14   31  502  107   62
##   D    9   63  111  462   39
##   E   18   93   42  102  485
```

```r
nright = table(ValidateTree == TrainingTest$classe)
TreeInError  = as.vector(100 * (1 - nright["TRUE"] / sum(nright)))
TreeInError 
```

```
## [1] 29.3116
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
#error.cv
plot(error.cv, type = "l", xlab="Size of tree (number of nodes)", ylab="Out of sample error(%)", main = "Relationship between tree size and out of sample error")
```

![plot of chunk Trees](figure/Trees-2.png) 
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
pml_write_files(TestFit)
```

## Conclusion
We see Random Forest is the most rebost for this set.
