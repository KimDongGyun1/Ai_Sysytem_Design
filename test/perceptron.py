from sklearn import datasets
from sklearn.linear_model import Perceptron 
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

test=pd.read_csv("archive5/Test set.csv")
train=pd.read_csv('archive5/Training set.csv')
train.head()

test['Sex'].replace('Female',0, inplace=True)
test['Sex'].replace('Male',1, inplace=True)
train['Sex'].replace('Female',0, inplace=True)
train['Sex'].replace('Male',1, inplace=True)

yt=np.array(train['Sex'])
xt=train.drop(['Sex'], axis=1)
xt=np.array(xt)

scaler=MinMaxScaler()
xt=scaler.fit_transform(xt)

# 60퍼센트는 훈련할때 쓰고 40프로는 test할 때 쓴다. dataset.data는 x(feature)이고 dataset.target y(labal)이다
x_train,x_test,y_train,y_test = train_test_split(xt, yt, test_size=0.3)

p = Perceptron(max_iter=100, eta0=1)
p.fit(x_train, y_train) # epoch, learing rate

# y값의 예측값이 들어간다
res = p.predict(x_test)

conf = np.zeros((2,2))
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1
print(conf)

correct = 0
for i in range(2):
    correct += conf[i][i]
accuracy = correct/len(res)
print("Accuracy is", accuracy*100, "%.")
