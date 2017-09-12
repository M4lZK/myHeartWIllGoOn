import pandas as pd
import numpy as np
import random

#load train data
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

#load test data
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

#problem 1
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
#print(train["Age"].median())

#problem 2
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode())
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test["Embarked"] = test["Embarked"].fillna(test["Embarked"].mode())
#print(train["Embarked"].mode())

#problem 3
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Sex"] = train["Sex"].fillna(train["Sex"].mode())
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Sex"] = test["Sex"].fillna(test["Sex"].mode())

#logistic regression classifier
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def train(model,data_train,y):
    predicted = sigmoid(np.dot(data_train,model))
    error = np.array(y - predicted) #error (891,1)
    model += r*np.dot(data_train.T,error)

def predicted(model,data_test):
    predicted = sigmoid(np.dot(data_test,model))
    return predicted


data_train = np.array(train[["Pclass","Sex","Age","Embarked"]].values)
data_train[np.isnan(data_train)] = 0
data_test = np.array(test[["Pclass","Sex","Age","Embarked"]].values)
data_test[np.isnan(data_test)] = 0
y_train = np.array(train[["Survived"]].values)
#y_test = np.array(test[["Survived"]].values)

random.seed(0)
model = np.random.random((4,1))
r = 0.1

for x in range(0,100000):
    train(model,data_train,y_train)
    
answer = predicted(model,data_test).round()

to_csv = pd.DataFrame(columns= ['PassengerId','Survived'])
to_csv['PassengerId'] = test['PassengerId']
to_csv['Survived'] = answer
to_csv.to_csv('titanic.csv',index=False)