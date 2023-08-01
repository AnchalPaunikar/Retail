library(dplyr)
library(car)
library(caret)
library(randomForest)
library(cvTools)
library(pROC)

#create dummies function
source("createDummies.r")


#IMPORTING DATA INTO R
store.train = read.csv("store_train.csv")
store.test = read.csv("store_test.csv")

#Adding target variable "store" in test data
store.test$store = NA

#Adding new column to differentiate between train and test data after combining
store.test$data = "test"
store.train$data = "train"

#Combining train and test data
store_all = rbind(store.train,store.test)

#converting target variable into factor type
store_all$store = as.factor(store_all$store)
View(store_all)
glimpse(store_all)

#Finding NA values
sum(is.na(store_all))

names(store.train)
#finding correlation
table(countytownname, countyname)
 tapply(countyname,countytownname , unique )



#data cleaning-----------------------
#removing least important column for prediction
store_all = store_all %>% select(-countyname, -countytownname,-Areaname, 
                            -state_alpha, -storecode)


#creating dummies
store_all = createDummies(store_all,"store_Type", 0)

#Splitting train and test data from combined data
store.test = store_all %>% filter(data == "test") %>% select(-data,-store)
store.train = store_all %>% filter(data =="train") %>% select(-data)

#data preparation-----------------------

sum(is.na(store.train))
colSums(is.na(store.train))
#filling na values with median of particular column
store.train$population = ifelse(is.na(store.train$population),
                                round(mean(store.train$population,na.rm =T)),store.train$population)

store.test$population = ifelse(is.na(store.test$population),
                                round(mean(store.test$population,na.rm =T)),store.test$population)

store.test$country = ifelse(is.na(store.test$country),
                               round(mean(store.test$country,na.rm =T)),store.test$country)
sum(is.na(store.test))
#linear model---------------------------------

#Splitting the data set into Train75 anad train25
set.seed(21)
s=sample(1:nrow(store.train),0.75*nrow(store.train))
train_75=store.train[s,] 
test_25=store.train[-s,] 

for_vif = lm(store~. -Id, data = train_75)

for_vif=lm(store~.-Id-sales0-sales2-sales3-sales1,data=train_75)
sort(vif(for_vif),decreasing = T)[1:3]
summary(for_vif)

#Generalized linear model----------------------------------------------
fit=glm(store~.-Id-sales0-sales2-sales3-sales1,data=train_75) #32 predictor var
fit=step(fit)
summary(fit)
fit=glm(store ~ sales4 + CouSub + population + storecode_METRO12620N23019 + 
          storecode_METRO14460MM1120,data=train_75) #32 predictor var


scoreLG=predict(fit,newdata =test_25,type = "response")
roccurve=roc(test_25$store,scoreLG) 
auc(roccurve)

#predicting random forest model
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


score=predict(store.rf.final,newdata= store.test, type="prob")[,1]; score

 #creating confusion matrix
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

