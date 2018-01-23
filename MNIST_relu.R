## MNIST Pre-processing - ReLu
## I'm going to try to remove columns with near zero variance.
## 15 x 5 x 3: Hidden Layers

library(caret)
library(dplyr)
library(sigmoid)

## Function to display digit... thanks to JunMa
displayDigit <- function(X){
        m <- matrix(unlist(X),nrow = 28,byrow = T)
        m <- t(apply(m, 2, rev))
        image(m,col=grey.colors(255))
}


setwd("C:/Users/Adam/Desktop/Kaggle_MNIST")

train <- read.csv("train.csv", header = TRUE) ## 42000 x 785

#displayDigit(train[15555,-1])

######################################################
## First, remove columns with near zero variance
######################################################


nzv <- nearZeroVar(train)

train <- train[, -nzv] ## 42000 x 253... now we have 253 columns instead of 784



############################################
## Split the data into train/dev sets
############################################

set.seed(999)
inTrain <- createDataPartition(y=train$label, p=0.7, list=F)

training_set <- train[inTrain, ]
dev_set <- train[-inTrain, ]


train <- training_set ## Rename back to train to avoid confusion

dim(train) ## 29403 x 253
dim(dev_set) ## 12597 x 253

############################################
## Separate labels from data
############################################

label <- train$label
label <- as.data.frame(label, ncol = 1) ## 29403 x 1
train.label <- label ## For assessing performance on training set

dev_label <- dev_set$label ## 12597 x 1
dev_label <- as.data.frame(dev_label, ncol = 1)
dev.label <- dev_label ## For assessing performance on dev set


train <- select(train, -c(label)) ## 29403 x 784
dev_set <- select(dev_set, -c(label)) ## 12597 x 784

## One-hot encode label
label <- data.frame(label) ## 29403 x 10
label$label <- as.factor(label$label)
label <- model.matrix(~ label + 0, data=label, 
                      contrasts.arg = lapply(label, contrasts, contrasts=FALSE))

dev_label <- data.frame(dev_label) ## 12597 x 10
dev_label$dev_label <- as.factor(dev_label$dev_label)
dev_label <- model.matrix(~ dev_label + 0, data=dev_label,
                          contrasts.arg = lapply(dev_label, contrasts, contrasts=FALSE))


#########################
## Transpose Matrices
#########################

train <- t(train) ## 252 x 29403
label <- t(label) ## 10 x 29403

dev_set <- t(dev_set) ## 252 x 12597
dev_label <- t(dev_label) ## 10 x 12597

train <- train/max(train) ## Scales by max rgb value = 255
dev_set <- dev_set/max(dev_set)

tic <- Sys.time()

########################################################
## Set up cross validation to select lambda
########################################################

train_accuracy <- list()
dev_accuracy <- list()
lambda.lst <- 10^seq(-5,5)
for (z in 1:length(lambda.lst)){
        
        lambda <- lambda.lst[z]
        

#############################################################
## Initialize Parameters - Xavier Initialization
#############################################################

w_1 <- matrix(rnorm(15*252, mean=0, sd=1), nrow = 15, ncol = 252,  byrow = TRUE) # 15x252
w_1 <- w_1 * sqrt(2/252) ## 15 x 252


b_1 <- matrix(rnorm(15, mean=0, sd=sqrt(2/252)), nrow = 15, ncol = 1,  byrow = TRUE) # 15x1

w_2 <- matrix(rnorm(5*15, mean=0, sd=1), nrow = 5, ncol = 15,  byrow = TRUE) ## 5x15
w_2 <- w_2 * sqrt(2/15) ## 5 x 15


b_2 <- matrix(rnorm(5, mean=0, sd=sqrt(2/15)), nrow = 5, ncol = 1,  byrow = TRUE) # 5x1


w_3 <- matrix(rnorm(3*5, mean=0, sd=1), nrow = 3, ncol = 5,  byrow = TRUE) # 3x5
w_3 <- w_3 * sqrt(2/5) ## 3 x 5


b_3 <- matrix(rnorm(3, mean=0, sd=sqrt(2/5)), nrow = 3, ncol = 1,  byrow = TRUE) # 3x1

w_4 <- matrix(rnorm(10*3, mean=0, sd=1), nrow=10, ncol=3, byrow=TRUE) ## 10x3
w_4 <- w_4 * sqrt(2/3) ## 10x3


b_4 <- matrix(rnorm(10, mean=0, sd=(2/3)), nrow=10, ncol=1, byrow=TRUE) ## 10 x 1



##############################################
## Begin 1st Loop - Iterate through epochs
##############################################

num_epochs <- 1000
alpha <- .05
cost <- NULL
m <- ncol(train)
#lambda <- .01 ## changed from .001


for (j in 1:num_epochs){


######################################
## Create mini-batches
######################################

shuffle <- sample(1:ncol(train))
shuffled_train <- train[,shuffle]
shuffled_label <- label[,shuffle]

num_complete_minibatches <- floor(m/32)

mini_batch_train <- list()
mini_batch_label <- list()
for (k in 1:(num_complete_minibatches+1)){
        
        if ((((k+1) * 32)+1) > m)
        {
                mini_batch_train[[k]] <- 
                shuffled_train[,((num_complete_minibatches * 32)+1) : m]
                
                mini_batch_label[[k]] <-
                shuffled_label[,((num_complete_minibatches * 32)+1) : m]
        }
        
        else if(k == 1) {
        
                mini_batch_train[[k]] <- 
                shuffled_train[, 1 : (k * 32)]
                
                mini_batch_label[[k]] <-
                shuffled_label[, 1 : (k * 32)]
            
        }
        
        else {
                mini_batch_train[[k]] <- 
                shuffled_train[,(((k-1)*32)+1) : (k * 32)]
                
                mini_batch_label[[k]] <-
                shuffled_label[,(((k-1)*32)+1) : (k * 32)]
        }
        
}






#######################################################
## Second iteration - Iterate through mini-batches
#######################################################


for (i in 1:length(mini_batch_train)){
        
        ############################################################
        ## Implement Forward Propagation
        ############################################################
        
        
        z_1 <- (w_1 %*% mini_batch_train[[i]])
        z_1 <- sweep(z_1, 1, b_1, "+") ## 15 x 32...usually
        a_1 <- matrix(pmax(0, z_1), nrow=15, 
                      ncol=ncol(mini_batch_train[[i]]), byrow=F) ## 15 x 32
        
        
        z_2 <- (w_2 %*% a_1) 
        z_2 <- sweep(z_2, 1, b_2, "+") ## 5 x 32
        a_2 <- matrix(pmax(0, z_2), nrow=5, 
                      ncol=ncol(mini_batch_train[[i]]), byrow=F) ## 5 x 32
        
        z_3 <- (w_3 %*% a_2) 
        z_3 <- sweep(z_3, 1, b_3, "+") ## 3 x 32
        a_3 <- matrix(pmax(0, z_3), nrow=3, 
                      ncol=ncol(mini_batch_train[[i]]), byrow=F) ## 3 x 32
        
        z_4 <- (w_4 %*% a_3) 
        z_4 <- sweep(z_4, 1, b_4, "+") ## 10 x 32
        t_4 <- exp(z_4) ## 10 x 32
        a_4 <- t(t(t_4) / colSums(t_4)) ## 10 x 32
        a_4[a_4 < 0] <- .000000000001 ## To avoid NaNs


        ########################################
        ## Compute Cost
        ########################################
        
        loss <- -colSums(mini_batch_label[[i]]*log(a_4))
        cost.tmp <- (1/ncol(mini_batch_train[[i]]))*sum(loss)
        
        ###############
        ## Regularize
        ###############
        
        ## Frobenius Norm
        frob.reg <- ((lambda/(2*ncol(mini_batch_train[[i]]))) * 
                        (norm(w_1, type=c("F")) + norm(w_2, type = c("F")) +
                         norm(w_3, type=c("F")) + norm(w_4, type = c("F"))))
        
        cost.tmp <- cost.tmp + frob.reg
        
        
        # check progress
        if (j%%100 == 0 | j == num_epochs){
                print(paste("epoch", j,': cost', cost.tmp))}
        
        if(j%%10 == 0 | j == num_epochs){
                cost <- c(cost, cost.tmp)}
        

        
        
        #####################################
        ## Implement Back Prop
        #####################################
        dz_4 <- a_4 - mini_batch_label[[i]]
        dw_4 <- (1/ncol(mini_batch_train[[i]])) * (dz_4 %*% t(a_3))
        ## Add regularization to backprop
        dw_4 <- (dw_4 + ((lambda/ncol(mini_batch_train[[i]])) * w_4)) 
        db_4 <- as.matrix((1/ncol(mini_batch_train[[i]])) * rowSums(dz_4), 
                          nrow=nrow(b_4), ncol=1)
        
        dz_3 <- (t(w_4) %*% dz_4) 
        dz_3[z_3 <= 0] <- 0
        dw_3 <- (1/ncol(mini_batch_train[[i]])) * (dz_3 %*% t(a_2))
        ## Add regularization to backprop
        dw_3 <- (dw_3 + ((lambda/ncol(mini_batch_train[[i]])) * w_3))
        db_3 <- as.matrix((1/ncol(mini_batch_train[[i]])) * rowSums(dz_3), 
                          nrow=nrow(b_3), ncol=1)
        
        
        dz_2 <- (t(w_3) %*% dz_3)
        dz_2[z_2 <= 0] <- 0
        dw_2 <- (1/ncol(mini_batch_train[[i]])) * (dz_2 %*% t(a_1))
        ## Add regularization to backprop
        dw_2 <- (dw_2 + ((lambda/ncol(mini_batch_train[[i]])) * w_2))
        db_2 <- as.matrix((1/ncol(mini_batch_train[[i]])) * rowSums(dz_2), 
                          nrow=nrow(b_2), ncol=1)
        
        dz_1 <- (t(w_2) %*% dz_2)
        dz_1[z_1 <= 0] <- 0
        dw_1 <- (1/ncol(mini_batch_train[[i]])) * 
                (dz_1 %*% t(mini_batch_train[[i]]))
        ## Add regularization to backprop
        dw_1 <- (dw_1 + ((lambda/ncol(mini_batch_train[[i]])) * w_1))
        db_1 <- as.matrix((1/ncol(mini_batch_train[[i]])) * rowSums(dz_1), 
                          nrow=nrow(b_1), ncol=1)


        ###################################
        ## Update Parameters
        ###################################
        
        w_4 <- (w_4 - (alpha*(dw_4)))
        b_4 <- (b_4 - (alpha*(db_4)))
        
        w_3 <- (w_3 - (alpha*(dw_3)))
        b_3 <- (b_3 - (alpha*(db_3)))
        
        w_2 <- (w_2 - (alpha*(dw_2)))
        b_2 <- (b_2 - (alpha*(db_2)))
        
        w_1 <- (w_1 - (alpha*(dw_1)))
        b_1 <- (b_1 - (alpha*(db_1)))
        
}


}




#####################################
## Map Cost Curve
#####################################


#x <- seq(1:length(cost))
#plot(x, cost, type = "l")


#####################################
## Assess performance on training set
#####################################

## One pass through forward prop


z_1 <- (w_1 %*% train)
z_1 <- sweep(z_1, 1, b_1, "+") ## 15 x 29403
a_1 <- matrix(pmax(0, z_1), nrow=15, 
              ncol=ncol(train), byrow=F) ## 15 x 29403


z_2 <- (w_2 %*% a_1) 
z_2 <- sweep(z_2, 1, b_2, "+") ## 5 x 29403
a_2 <- matrix(pmax(0, z_2), nrow=5, 
              ncol=ncol(train), byrow=F) ## 5 x 29403

z_3 <- (w_3 %*% a_2) 
z_3 <- sweep(z_3, 1, b_3, "+") ## 3 x 29403
a_3 <- matrix(pmax(0, z_3), nrow=3, 
              ncol=ncol(train), byrow=F) ## 3 x 29403

z_4 <- (w_4 %*% a_3) 
z_4 <- sweep(z_4, 1, b_4, "+") ## 10 x 29403
t_4 <- exp(z_4) ## 10 x 29403
a_4 <- t(t(t_4) / colSums(t_4)) ## 10 x 29403


y_hat <- rownames(a_4)[apply(a_4,2,which.max)]
y_hat <- gsub("[^0-9]", "", y_hat)
y_hat <- as.data.frame(as.numeric(y_hat), nrow=nrow(train.label), ncol=1)
colnames(y_hat) <- c("label")

#head(train.label)
#head(y_hat)

#table(y_hat)

acc <- merge(y_hat, train.label, by = label)
train_accuracy[[z]] <- nrow(acc) / nrow(train.label) 

## 95.82% after 1000 epochs @ lambda == .001
## 95.87% after 5000 epochs @ lambda == .001

## 92.01% after 1000 epochs @ lambda == .01


#################################
## Assess Performance on Dev Set
#################################

## One pass through forward prop


z_1 <- (w_1 %*% dev_set)
z_1 <- sweep(z_1, 1, b_1, "+") ## 15 x 12597
a_1 <- matrix(pmax(0, z_1), nrow=15, 
              ncol=ncol(dev_set), byrow=F) ## 15 x 12597


z_2 <- (w_2 %*% a_1) 
z_2 <- sweep(z_2, 1, b_2, "+") ## 5 x 12597
a_2 <- matrix(pmax(0, z_2), nrow=5, 
              ncol=ncol(dev_set), byrow=F) ## 5 x 12597

z_3 <- (w_3 %*% a_2) 
z_3 <- sweep(z_3, 1, b_3, "+") ## 3 x 12597
a_3 <- matrix(pmax(0, z_3), nrow=3, 
              ncol=ncol(dev_set), byrow=F) ## 3 x 12597

z_4 <- (w_4 %*% a_3) 
z_4 <- sweep(z_4, 1, b_4, "+") ## 10 x 12597
t_4 <- exp(z_4) ## 10 x 29403
a_4 <- t(t(t_4) / colSums(t_4)) ## 10 x 12597


y_hat <- rownames(a_4)[apply(a_4,2,which.max)]
y_hat <- gsub("[^0-9]", "", y_hat)
y_hat <- as.data.frame(as.numeric(y_hat), nrow=nrow(dev.label), ncol=1)
colnames(y_hat) <- c("label")

#head(dev.label)
#head(y_hat)

#table(y_hat)

acc <- merge(y_hat, dev.label, by = label)
dev_accuracy[[z]] <- nrow(acc) / nrow(dev.label) 

## 90.64% after 1000 epochs @ lambda == .001
## 90.12% after 5000 epochs @ lambda == .001 ... boo...

## 89.31% after 1000 epochs @ lambda == .01

}

## BEST ACCURACY IS AT LAMBDA = .1


##################################
## Assess processing performance
##################################

## No cross-validation
#toc <- Sys.time()
#run.time <- toc - tic 
## 18.45 minutes at 1000 epochs - Roughly 919k iterations
## 1.74 hours at 5000 epochs - roughly 4.6M iterations.

toc <- Sys.time()
run.time <- toc - tic 

## 1.06 DAYS


################################
## Build a "picker"
################################

dev.sample.index <- sample(1:ncol(dev_set),1)
dev.sample <- as.matrix(dev_set[,dev.sample.index]) 



## One pass through forward prop


z_1 <- (w_1 %*% dev.sample)
z_1 <- sweep(z_1, 1, b_1, "+") ## 15 x 1
a_1 <- matrix(pmax(0, z_1), nrow=15, 
              ncol=ncol(dev.sample), byrow=F) ## 15 x 1


z_2 <- (w_2 %*% a_1) 
z_2 <- sweep(z_2, 1, b_2, "+") ## 5 x 1
a_2 <- matrix(pmax(0, z_2), nrow=5, 
              ncol=ncol(dev.sample), byrow=F) ## 5 x 1

z_3 <- (w_3 %*% a_2) 
z_3 <- sweep(z_3, 1, b_3, "+") ## 3 x 1
a_3 <- matrix(pmax(0, z_3), nrow=3, 
              ncol=ncol(dev.sample), byrow=F) ## 3 x 1

z_4 <- (w_4 %*% a_3) 
z_4 <- sweep(z_4, 1, b_4, "+") ## 10 x 1
t_4 <- exp(z_4) ## 10 x 1
a_4 <- t(t(t_4) / colSums(t_4)) ## 10 x 1


y_hat <- rownames(a_4)[apply(a_4,2,which.max)]
y_hat <- gsub("[^0-9]", "", y_hat)
y_hat <- as.data.frame(as.numeric(y_hat), nrow=nrow(dev.sample), ncol=1)
colnames(y_hat) <- c("label")

print(paste("The predicted digit is:", y_hat))

## Assess truth

train2 <- read.csv("train.csv", header = TRUE) ## 42000 x 785

dev_set2 <- train2[-inTrain, ]

dev.sample2 <- dev_set2[dev.sample.index,]

displayDigit(dev.sample2[1,-1])


















