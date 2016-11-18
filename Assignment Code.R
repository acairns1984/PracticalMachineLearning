#-----------------------------
#-----------------------------
#--- Pratical Machine Learning - Course Project

remove(list=ls())
library(caret)
library(e1071)
library(gbm)
library(randomForest)
library(ggplot2)

#------------------
#-- Import Data

#Load the data
setwd("U:/Coursera/Practical Machine Learning/Week 4 Project/")
train.data <- read.csv(file = "pml-training.csv", header = TRUE)

#------------------
#-- Clean Data

#Eliminate observations where "New_window" equals yes
summary(train.data[train.data$new_window == "no",])
train.data <- train.data[train.data$new_window == "no",]

#Eliminate variables where all observations are NAs
NAObs <- apply(train.data, 2, function(i) sum(!is.na(i)) == nrow(train.data))
train.data <- train.data[,names(NAObs[NAObs == TRUE])]

#Eliminate variables where all observations are NAs
CompleteObs <- apply(train.data, 2, function(i) length(i[i == ""]) == 0)
train.data <- train.data[, names(CompleteObs[CompleteObs == TRUE])]

#Drop Unneeded columns
ColToDelete <- c("X", "user_name", "new_window", "raw_timestamp", "num_window", "cvtd_timestamp")
train.data <- train.data[, -grep(paste(ColToDelete, collapse = "|"), colnames(train.data))]

#------------------
#-- Identify predictors 

### Model 1

set.seed(123)
fit.rf <- randomForest(classe ~ ., data = train.data, ntree = 50, mtry = 7, importance = T)

varimp <- importance(fit.rf, type = 2) #mean decrease in node purity
RankVarImp <- as.matrix(cbind(varimp[order(-varimp[,1]), ], Rank = 1:nrow(varimp)))
RankVarImp

varImpPlot(fit.rf, main = "Figure 1: Variable Importance")


### Loop Through various specifications

Iterations <- lapply(7:nrow(RankVarImp), function(i){
            print(i)
            varNames <- rownames(RankVarImp[1:i, ])
            set.seed(i)
            fit <- randomForest(train.data[,varNames], y = train.data$classe, ntree = 500, mtry = 7)
            k <- fit$confusion[,1:5]
            OOBestimate <- 1 - sum(diag(k))/sum(colSums(k))
            OOBestimate
      })

# Variable count with lowest OOB error
dat <- data.frame(NumofVar = 7:nrow(RankVarImp), OOBError = do.call(c, Iterations))
dat[dat$OOBError == min(dat$OOBError),]

# Plot
g <- ggplot(data = dat, aes(x = NumofVar, y = OOBError)) + geom_line(col = "blue")
g <- g + ggtitle("Figure 2: OOB Error by Variable Count") + xlab("Number of Predictors Included (According to Rank)")
g + ylab("OOB Error (1 - accuracy)")

### Final Model

set.seed(12345)
fit.rfFinal <- randomForest(classe ~ roll_belt + yaw_belt + pitch_forearm + magnet_dumbbell_z + pitch_belt + magnet_dumbbell_y
                            + roll_forearm + magnet_dumbbell_x + roll_dumbbell + accel_dumbbell_y + accel_belt_z
                            + magnet_belt_y + magnet_belt_z + accel_forearm_x + accel_dumbbell_z + roll_arm + gyros_belt_z
                            + magnet_forearm_z + total_accel_dumbbell + yaw_dumbbell + magnet_arm_x + yaw_arm + gyros_dumbbell_y
                            + magnet_belt_x + accel_forearm_z + accel_dumbbell_x + magnet_arm_y + magnet_forearm_y + magnet_forearm_x
                            + accel_arm_x + total_accel_belt + yaw_forearm + magnet_arm_z + pitch_arm + accel_arm_y
                            + accel_forearm_y + gyros_arm_y + gyros_arm_x + accel_arm_z + gyros_dumbbell_x + gyros_forearm_y
                            + gyros_belt_y + accel_belt_x + total_accel_forearm + total_accel_arm + gyros_belt_x
                            + gyros_dumbbell_z +  pitch_dumbbell + accel_belt_y
                            , data = train.data, ntree = 2500, mtry = 7, importance = T)

#Out of sample/OOB Error of Final Model
fit.rfFinal

### Make Test Set Predictions
test.data <- read.csv(file = "pml-testing.csv", header = TRUE)
pred.val <- predict(fit.rfFinal, newdata = test.data)
pred.val




