'''
CS 579: Online Social Network Analysis
Project II
Group Members: Vishwa Babariya, Kanchi Patel
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
train_copy=train.copy()
test_copy=test.copy()

#Preprocessing the data
train=train[["title1_en","title2_en","label"]]
test=test[["title1_en","title2_en"]]

train["title1_en"]=train["title1_en"].str.lower()
train["title2_en"]=train["title2_en"].str.lower()
test["title1_en"]=test["title1_en"].str.lower()
test["title2_en"]=test["title2_en"].str.lower()
train_size=int(len(train))

tfid_vectorizer=TfidfVectorizer(stop_words="english")
train_data=train["title1_en"]+""+train["title2_en"]
tfid_vectorizer.fit(train_data)
train1=tfid_vectorizer.transform(train["title1_en"])
train2=tfid_vectorizer.transform(train["title2_en"])
xtrain=StandardScaler(with_mean=False).fit_transform(train1-train2)
ytrain=train_copy["label"]

test1=tfid_vectorizer.transform(test["title1_en"])
test2=tfid_vectorizer.transform(test["title2_en"])
xtest=StandardScaler(with_mean=False).fit_transform(test1-test2)

#Training the Logistic Regression Model
svc=SVC(kernel="linear",C=1,max_iter=50000)
svc.fit(xtrain,ytrain)
ytest=svc.predict(xtest)

xtrainx,xtrainy,ytrainx,ytrainy=train_test_split(xtrain,ytrain,test_size=0.2,random_state=579)
svc1=SVC(kernel="linear",C=1,max_iter=50000)
svc1.fit(xtrainx,ytrainx)
pred=svc1.predict(xtrainy)
accuracy=accuracy_score(ytrainy,pred)
print("Accuracy: ",accuracy)

result=test_copy[["id"]].copy()
for x in range(len(ytest)):
    result.loc[x,"label"]=ytest[x]
result.to_csv("SVM_ytest.csv",index=False)
