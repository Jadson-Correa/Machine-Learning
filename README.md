# Practical Machine Learning - Coursera
#### Jadson
#### 03/08/2020

## Practical Machine Learning Project
## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: [http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).

## Data
The training data for this project are available here:

[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

The test data are available here:

[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

## Analysis
```{r}
setwd("C:/Users/Home/Desktop/coursera")
training <- read.csv("./pml-training.csv",na.strings=c("NA","#DIV/0!",""))
testing <- read.csv("./pml-testing.csv",na.strings=c("NA","#DIV/0!",""))
# Data dimensions
dim(training)
```
[1] 19622   160
```{r}
dim(testing)
```
[1]  20 160

```{r}
# First look at the data
head(training)
head(testing)
```

Cross-validation will be performed by spliting the training dataset into:

1. Training dataset:70% of the observations.

2. Testing dataset: 30% of the observations

```{r}
# load packages
#install.packages("caret")
#install.packages("randomForest")
library(caret)
library(randomForest)

set.seed(12345)
train <- createDataPartition(y=training$classe,p=0.7, list=FALSE)
# training dataset
training.set <- training[train,]
# testing dataset
testing.set <- training[-train,]
```
Training and testing data consist of 160 variables. The choice of specific predictors is based on removing near zero variance predictors, with the nearZeroVar function, and also variables containing many NAs.

```{r}
# Remove near zero variance predictors
ind.nzv = nearZeroVar(x = training, saveMetrics = T)
# Remove variables with more than 50% NA values
ind.NA = !as.logical(apply(training, 2, function(x){ mean(is.na(x)) >= 0.5}))
# Cleaning data
ind2 = ind.NA*1 + (!ind.nzv$nzv)*1
ind3 = ind2 == 2
sum(ind3)
```
[1] 59

```{r}
#View(data.frame(ind.NA, !ind.nzv$nzv, ind2, ind3))
training.set = training.set[,ind3]
testing.set = testing.set[, ind3]
training.set = training.set[, -1]
testing.set = testing.set[, -1]
testing = testing[,ind3]
testing = testing[,-1]
# Coerce the data into the same type in order to avoid "Matching Error" when calling random forest model, due to different levels in variables
for (i in 1:length(testing) ) {
  for(j in 1:length(training.set)) {
    if( length( grep(names(training.set[i]), names(testing)[j]) ) == 1)  {
      class(testing[j]) <- class(training.set[i])
    }      
  }      
}
# To get the same class between testing and training.set
testing = testing[,-ncol(testing)]
testing <- rbind(training.set[2, -58] , testing)
testing <- testing[-1,]
```

We will use two approaches to create a prediction model for the values of classe variable.

Firstly prediction with trees will be attempted, using the ‘rpart’ method and the caret package.

```{r}
# Prediction with Trees
# Build model
#install.packages("e1071")
library(e1071)
set.seed(12345)
tree.fit = train(y = training.set$classe,
                 x = training.set[,-ncol(training.set)],
                 method = "rpart")
# Plot classification tree
#install.packages("rattle")
library(rattle)
rattle::fancyRpartPlot(
  tree.fit$finalModel
)
```
