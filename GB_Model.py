'''
CS 579: Online Social Network Analysis
Project II
Group Members: Vishwa Babariya, Kanchi Patel
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
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

cnt_vectorizer=CountVectorizer(stop_words="english")
train_data=train["title1_en"]+""+train["title2_en"]
xtrain=cnt_vectorizer.fit_transform(train_data)
ytrain=train["label"]
test_data=test["title1_en"]+""+test["title2_en"]
xtest=cnt_vectorizer.transform(test_data)

#Training the Gradient Boosting
gboost=GradientBoostingClassifier(random_state=579)
gboost.fit(xtrain,ytrain)
ytest=gboost.predict(xtest)

xtrainx,xtrainy,ytrainx,ytrainy=train_test_split(xtrain,ytrain,test_size=0.2,random_state=579)
gboost1=GradientBoostingClassifier(random_state=579)
gboost1.fit(xtrainx,ytrainx)
pred=gboost1.predict(xtrainy)
accuracy=accuracy_score(ytrainy,pred)
print("Accuracy: ",accuracy)

result=test_copy[["id"]].copy()
for x in range(len(ytest)):
    result.loc[x,"label"]=ytest[x]
result.to_csv("GB_ytest.csv",index=False)
