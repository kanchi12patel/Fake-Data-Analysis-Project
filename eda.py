'''
CS 579: Online Social Network Analysis
Project II
Group Members: Vishwa Babariya, Kanchi Patel
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

#Taking a look as to what the data looks like
print(train.head(5))
print()
print("Column names: ",list(train))

#Number of rows and columns
n_row,n_col=train.shape
print("Number of rows in train dataset: ",n_row)
print("Number of columns in train dataset: ",n_col)
n_row,n_col=test.shape
print()
print("Number of rows in test dataset: ",n_row)
print("Number of columns in test dataset: ",n_col)

#Checking for any missing values
print("\nMissing Values in Train Dataset: ")
print(train.isna().sum())
print("\nMissing Values in Test Dataset: ")
print(test.isna().sum())

#Count of Label column
label_cnt=train["label"].value_counts()
bg=label_cnt.plot(kind="bar",color="pink")
bg.set_title("Label Distribution")
bg.set_xlabel("Label",rotation=90)
bg.set_ylabel("Frequency")
for x,y in enumerate(label_cnt):
    bg.text(x,y+1,str(y),horizontalalignment="center")
plt.show()

