---
title: "Machine Learning Peer Project-Predictions using the Weight Lifting Exercises Dataset"
author: "Siddharth Chadha"

output: 
  html_document:
    keep_md: yes
  md_document:
  pdf_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, warning=FALSE, message=FALSE, fig.width=11, fig.height=7)
options(width=150)
library(lattice)
library(ggplot2)
library(plyr)
library(randomForest)
```

## Executive Summary

Based on a dataset provide by HAR [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har) we will try to train a predictive model to predict what exercise was performed using a dataset with 159 features



Work flow of the model:

1) Getting the data from the data source 
2) Cleaning and formatting data which is fit to pass the alorithm 
3) Doing EDA on the data to find key insights into it 
4) Spliting or data into train and test splits for training and validation
5) Generating a model and passing the data to train the model 
6) Once the model is made, we test the accuracy with the test data
7) Finally visualising the results of our predictions 

## Processing

We will change the automation to am = 0 
Also adding cylinders as an attribute 

```{r}
training.raw <- read.csv("pml-training.csv")
testing.raw <- read.csv("pml-testing.csv")
```

## EDA

Discribing out data points
```{r}
dim(training.raw)
```


Filling N/A values as they hinder the performance of the algortihm 

```{r}
max_perc_NA = 25
max_count_NA <- nrow(training.raw) / 100 * max_perc_NA
rmcol <- which(colSums(is.na(training.raw) | training.raw=="") > max_count_NA)
training.cleaned_1 <- training.raw[,-rmcol]
testing.cleaned_1 <- testing.raw[,-rmcol]
```

Removing timestamps

```{r}
rmcol <- grep("timestamp", names(training.cleaned_1))
training.cleaned_2 <- training.cleaned_1[,-c(1, rmcol )]
testing.cleaned_2 <- testing.cleaned_1[,-c(1, rmcol )]
```

Coverting all values into discrete values 
```{r}
level_classe <- levels(training.cleaned_2$classe)
training.cleaned_3 <- data.frame(data.matrix(training.cleaned_2))
training.cleaned_3$classe <- factor(training.cleaned_3$classe, labels=level_classe)
testing.cleaned_3 <- data.frame(data.matrix(testing.cleaned_2))
```

Exploring the data

```{r}
training.cleaned <- training.cleaned_3
testing.cleaned <- testing.cleaned_3
```


##EDA 

Since the test set provided is the the ultimate validation set, we will split the current training in a test and train set to work with.

```{r}
set.seed(169)
library(caret)
index_classe <- which(names(training.cleaned) == "classe")
partitions <- createDataPartition(y=training.cleaned$classe, p=0.7, list=FALSE)
training.subSetTrain <- training.cleaned[partitions, ]
training.subSetTest <- training.cleaned[-partitions, ]
```

What are some fields that have high correlations with the classe?

```{r}
correlations <- cor(training.subSetTrain[, -index_classe], as.numeric(training.subSetTrain$classe))
best_Correlations <- subset(as.data.frame(as.table(correlations)), abs(Freq)>0.3)
best_Correlations
```


```{r}
library(Rmisc)
library(ggplot2)
p1 <- ggplot(training.subSetTrain, aes(classe,pitch_forearm)) + 
  geom_boxplot(aes(fill=classe))
p2 <- ggplot(training.subSetTrain, aes(classe, magnet_arm_x)) + 
  geom_boxplot(aes(fill=classe))
multiplot(p1,p2,cols=2)
```


## Model selection 

To exclude attributes from pca or training, we will identify the data with high correlation

```{r}
library(corrplot)
corrMat <- cor(training.subSetTrain[, -index_classe])
high_corr <- findCorrelation(corrMat, cutoff=0.9, exact=TRUE)
ex_Columns <- c(high_corr, index_classe)
corrplot(corrMat, method="color", type="lower", order="hclust", tl.cex=0.70, tl.col="black", tl.srt = 45, diag = FALSE)
```



```{r}
pcaPreProcess.all <- preProcess(training.subSetTrain[, -index_classe], method = "pca", thresh = 0.99)
training.subSetTrain.pca.all <- predict(pcaPreProcess.all, training.subSetTrain[, -index_classe])
training.subSetTest.pca.all <- predict(pcaPreProcess.all, training.subSetTest[, -index_classe])
testing.pca.all <- predict(pcaPreProcess.all, testing.cleaned[, -index_classe])
pcaPreProcess.subset <- preProcess(training.subSetTrain[, -ex_Columns], method = "pca", thresh = 0.98)
training.subSetTrain.pca.subset <- predict(pcaPreProcess.subset, training.subSetTrain[, -ex_Columns])
training.subSetTest.pca.subset <- predict(pcaPreProcess.subset, training.subSetTest[, -ex_Columns])
testing.pca.subset <- predict(pcaPreProcess.subset, testing.cleaned[, -index_classe])
```

We wil use 180 random forests variations to get the best accuracy
If the tree is overfitted, we use true prunning methods to make the tree more generalised

```{r}
library(randomForest)
ntree <- 180  
start <- proc.time()
rfMod.cleaned <- randomForest(
  x=training.subSetTrain[, -index_classe], 
  y=training.subSetTrain$classe,
  xtest=training.subSetTest[, -index_classe], 
  ytest=training.subSetTest$classe, 
  ntree=ntree,
  keep.forest=TRUE,
  proximity=TRUE) 
proc.time() - start
start <- proc.time()
rfMod.exclude <- randomForest(
  x=training.subSetTrain[, -ex_Columns], 
  y=training.subSetTrain$classe,
  xtest=training.subSetTest[, -ex_Columns], 
  ytest=training.subSetTest$classe, 
  ntree=ntree,
  keep.forest=TRUE,
  proximity=TRUE) #do.trace=TRUE
proc.time() - start
start <- proc.time()
rfMod.pca.all <- randomForest(
  x=training.subSetTrain.pca.all, 
  y=training.subSetTrain$classe,
  xtest=training.subSetTest.pca.all, 
  ytest=training.subSetTest$classe, 
  ntree=ntree,
  keep.forest=TRUE,
  proximity=TRUE) 
proc.time() - start
start <- proc.time()
rfMod.pca.subset <- randomForest(
  x=training.subSetTrain.pca.subset, 
  y=training.subSetTrain$classe,
  xtest=training.subSetTest.pca.subset, 
  ytest=training.subSetTest$classe, 
  ntree=ntree,
  keep.forest=TRUE,
  proximity=TRUE) 
proc.time() - start
```

## Examining the model

We will check the accuracy of the four models we have created and pick the most efficient one.

```{r}
rfMod.cleaned
rfMod.cleaned.training.acc <- round(1-sum(rfMod.cleaned$confusion[, 'class.error']),3)
paste0("Accuracy on training: ",rfMod.cleaned.training.acc)
rfMod.cleaned.testing.acc <- round(1-sum(rfMod.cleaned$test$confusion[, 'class.error']),3)
paste0("Accuracy on testing: ",rfMod.cleaned.testing.acc)
rfMod.exclude
rfMod.exclude.training.acc <- round(1-sum(rfMod.exclude$confusion[, 'class.error']),3)
paste0("Accuracy on training: ",rfMod.exclude.training.acc)
rfMod.exclude.testing.acc <- round(1-sum(rfMod.exclude$test$confusion[, 'class.error']),3)
paste0("Accuracy on testing: ",rfMod.exclude.testing.acc)
rfMod.pca.all
rfMod.pca.all.training.acc <- round(1-sum(rfMod.pca.all$confusion[, 'class.error']),3)
paste0("Accuracy on training: ",rfMod.pca.all.training.acc)
rfMod.pca.all.testing.acc <- round(1-sum(rfMod.pca.all$test$confusion[, 'class.error']),3)
paste0("Accuracy on testing: ",rfMod.pca.all.testing.acc)
rfMod.pca.subset
rfMod.pca.subset.training.acc <- round(1-sum(rfMod.pca.subset$confusion[, 'class.error']),3)
paste0("Accuracy on training: ",rfMod.pca.subset.training.acc)
rfMod.pca.subset.testing.acc <- round(1-sum(rfMod.pca.subset$test$confusion[, 'class.error']),3)
paste0("Accuracy on testing: ",rfMod.pca.subset.testing.acc)
```



```{r}
par(mfrow=c(1,2)) 
varImpPlot(rfMod.exclude, cex=0.7, pch=16, main='Variable Importance Plot: rfMod.exclude')
plot(rfMod.exclude, , cex=0.7, main='Error vs No. of trees plot')
par(mfrow=c(1,1)) 
```

To really look in depth at the distances between predictions we can use MDSplot and cluster predictiosn and results

```{r}
start <- proc.time()
library(RColorBrewer)
palette <- brewer.pal(length(classeLevels), "Set1")
rfMod.mds <- MDSplot(rfMod.exclude, as.factor(classeLevels), k=2, pch=20, palette=palette)
library(cluster)
rfMod.pam <- pam(1 - rfMod.exclude$proximity, k=length(level_classe), diss=TRUE)
plot(
  rfMod.mds$points[, 1], 
  rfMod.mds$points[, 2], 
  pch=rfMod.pam$clustering+14, 
  col=alpha(palette[as.numeric(training.subSetTrain$classe)],0.49), 
  bg=alpha(palette[as.numeric(training.subSetTrain$classe)],0.19), 
  cex=0.5,
  xlab="x", ylab="y")
legend("bottomleft", legend=unique(rfMod.pam$clustering), pch=seq(15,14+length(classeLevels)), title = "PAM cluster")
  legend("topleft", legend=level_classe, pch = 18, col=palette, title = "Classification")
proc.time() - start
```

## Test Results
```{r}
predictions <- t(cbind(
    exclude=as.data.frame(predict(rfMod.exclude, testing.cleaned[, -excludeColumns]), optional=TRUE),
    cleaned=as.data.frame(predict(rfMod.cleaned, testing.cleaned), optional=TRUE),
    pcaAll=as.data.frame(predict(rfMod.pca.all, testing.pca.all), optional=TRUE),
    pcaExclude=as.data.frame(predict(rfMod.pca.subset, testing.pca.subset), optional=TRUE)
))
predictions
```


