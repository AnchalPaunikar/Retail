library(dplyr)
library(car)
library(caret)
library(randomForest)
library(cvTools)
library(pROC)
setwd("C:/Users/ASUS/OneDrive/Desktop/Edvancer R/R")
source("createDummies.r")

setwd("C:/Users/ASUS/OneDrive/Desktop/Edvancer R/R/projectdb")

store.train = read.csv("store_train.csv")
store.test = read.csv("store_test.csv")
store.test$data = "test"
store.train$data = "train"

store.test$store = NA
store_all = rbind(store.train,store.test)
store_all$store = as.factor(store_all$store)
View(store_all)
glimpse(store_all)
sum(is.na(store_all))
attach(store_all)
names(store.train)
# table(countytownname, countyname)
# tapply(countyname,countytownname , unique )
# table(state_alpha


#data cleaning-----------------------
store_all = store_all %>% select(-countyname, -countytownname,-Areaname, 
                            -state_alpha, -storecode)


store_all$population[is.na(store_all$population)]=round(median(store_all$population,na.rm=T),0)
store_all$country[is.na(store_all$country)]=round(median(store_all$country,na.rm=T),0)


#store_all = createDummies(store_all,"storecode", 50)
store_all = createDummies(store_all,"store_Type", 0)

store.test = store_all %>% filter(data == "test") %>% select(-data,-store)
store.train = store_all %>% filter(data =="train") %>% select(-data)

#data preparation-----------------------

sum(is.na(store.train))
colSums(is.na(store.train))
store.train$population = ifelse(is.na(store.train$population),
                                round(mean(store.train$population,na.rm =T)),store.train$population)

store.test$population = ifelse(is.na(store.test$population),
                                round(mean(store.test$population,na.rm =T)),store.test$population)

store.test$country = ifelse(is.na(store.test$country),
                               round(mean(store.test$country,na.rm =T)),store.test$country)
sum(is.na(store.test))
#model---------------------------------

set.seed(21)
s=sample(1:nrow(store.train),0.75*nrow(store.train))
train_75=store.train[s,] 
test_25=store.train[-s,] 

for_vif = lm(store~. -Id, data = train_75)

for_vif=lm(store~.-Id-sales0-sales2-sales3-sales1,data=train_75)
sort(vif(for_vif),decreasing = T)[1:3]
summary(for_vif)

fit=glm(store~.-Id-sales0-sales2-sales3-sales1,data=train_75) #32 predictor var
fit=step(fit)
summary(fit)
fit=glm(store ~ sales4 + CouSub + population + storecode_METRO12620N23019 + 
          storecode_METRO14460MM1120,data=train_75) #32 predictor var

library(pROC)
scoreLG=predict(fit,newdata =test_25,type = "response")
roccurve=roc(test_25$store,scoreLG) 
auc(roccurve)

rf.model3= randomForest(as.factor(store)~.-Id,data=train_75)
test.score3=predict(rf.model3,newdata=test_25,type="prob")[,2]
auc(roc(test_25$store,test.score3))



train_75$store = as.numeric(train_75$store == 1)
train_75$store = as.factor(train_75$store)

subset_paras=function(full_list_para,n=10){  #n=10 is default, you can give higher value
  
  all_comb=expand.grid(full_list_para)
  
  s=sample(1:nrow(all_comb),n)
  
  subset_para=all_comb[s,]
  
  return(subset_para)
}


param=list(mtry=c(2,3.741,3),
           ntree=c(100,200,500,150,175,275,250,300), 
           maxnodes=c(200,250,275,375,325,350,100,300),
           nodesize=c(5,10,20,12)       
)

mycost_auc=function(store,yhat){  #Real #Predicted
  roccurve=pROC::roc(store,yhat)
  score=pROC::auc(roccurve)
  return(score)
}  

store.train$store = as.factor(store.train$store)
num_trial=1152
my_params=subset_paras(param,num_trial)

myauc = 0
for (i in 1:num_trial) {
  print(paste0("starting iteration: ", i))
  param = my_params[i,]
  
  
  k = cvTuning(randomForest, store~. , 
               data = store.train,
               tuning = param,
               folds = cvFolds(nrow(store.train), K = 10, type ="random"),
               cost = mycost_auc,
               seed = 2,
               predictArgs = list(type="prob"))
  
  score.this = k$cv[,2]
  
  if(score.this > myauc){
    myauc = score.this
    print(myauc)
    best_param = param
    print(best_param)
  }
  print(paste0("iteration", i, "completed"))
}; best_param; myauc


best_param = data.frame(mtry =3,
                        ntree = 275,
                        maxnodes =250 ,
                        nodesize = 20 )
# 
# mtry ntree maxnodes nodesize
#     3   150      300       20
# [1] 0.8196232
# mtry ntree maxnodes nodesize
#     3   200      300       12
# [1] 0.8206724

# 
# mtry ntree maxnodes nodesize
#     3   500      300       20
# [1] 0.8181867
# mtry ntree maxnodes nodesize
#    3   500      500        5
# [1] 0.8151922

# mtry ntree maxnodes nodesize
#  3.741    50      300       20
# [1] 0.8151621

# mtry ntree maxnodes nodesize
#  3.741   150      500       10
# [1] 0.8154377

# mtry ntree maxnodes nodesize
#     3   275      250       20
# [1] 0.8204526


best_param = data.frame(mtry =20 ,
                        ntree =200,
                        maxnodes =100 ,
                        nodesize = 10 )
# mtry ntree maxnodes nodesize
# 2029   35   100      100       12
# [1] 0.8012028



store.rf.final = randomForest(as.factor(store)~.-Id, data = train_75,
                              mtry = best_param$mtry,
                              ntree = best_param$ntree,
                              maxnodes = best_param$maxnodes,
                              nodesize = best_param$nodesize);store.rf.final


val.score = predict(store.rf.final, newdata = test_25, type = "prob")
val.pred = ifelse(val.score > 0.5, 1, 0)

train.score = predict(store.rf.final, newdata = train_75, type = "prob")
train.pred = ifelse(train.score > 0.5, 1, 0)

test.score = predict(store.rf.final, newdata = store.test, type = "prob")
test.pred = ifelse(test.score > 0.5, 1, 0)
price1 = test.pred[,2]

auc_score=auc(roc(test_25$store,val.score[,2])); auc_score
#write.csv(price1, "abac.csv", row.names = F)
write.table(price1, "82.17store.csv", col.names = "Store", row.names = F)
?write.csv

score=predict(store.rf.final,newdata= store.test, type="prob")[,1]; score
write.csv(test.pred[,1], "pred.csv", row.names = T)
write.csv(auc_score, "store.csv")
 
# fit_s = glm(store~.-State-store_Type_SupermarketType1-sales3-store_Type_SupermarketType3-
#               store_Type_GroceryStore-Id-sales4-country-sales0, 
#             family = "binomial", data = train1_s1)
# round(sort((summary(fit_s)$coefficients)[,4]),4)
# 
# 
# val.score = predict(fit_s, newdata = train2_s2, type = "response")
# val.pred = ifelse(val.score > 0.5, 1, 0)
# val.pred[1:9]
# 
# 
# 
# train.score = predict(fit_s,newdata = train1_s1, type = "response")
# train.pred = ifelse(train.score > 0.5, 1, 0)
# train.pred[1:9]


# for training data
# train_pred <- ifelse(train.score > 0.5, 1, 0)
# 
# # for  validation data
# 
# val_pred <- ifelse(val.score > 0.5, 1, 0)


# creating confusion matrix
train_cm = confusionMatrix(factor(train.pred),
                           factor(train1_s1$store))

val_cm =  confusionMatrix(factor(val.pred),
                          factor(train2_s2$store))

real = train1_s1$store

TP = sum(real == 1 & train.pred == 1); TP
TN = sum(real == 0 & train.pred == 0);TN
FN = sum(real == 0 & train.pred == 1);FN
FP = sum(real == 1 & train.pred == 0);FP

P = TP + FN;P
N = TN + FP;N
 
accuracy = (TP + TN)/(P+N); accuracy
Sn = TP/P
Sp = TN/N
precision = TP/(TP + FP)
recall = Sn

KS = (TP/P) - (FP/N);KS

mtry ntree maxnodes nodesize
  4   500      300       15
[1] 0.8178858
















