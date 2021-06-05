from sklearn.linear_model import Perceptron 
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

# 70퍼센트는 훈련할때 쓰고 30프로는 test할 때 쓴다. xt는 x(feature)이고 yt는 y(labal)이다
x_train,x_test,y_train,y_test = train_test_split(xt, yt, test_size=0.3)

p = Perceptron(max_iter=100, eta0=0.1) # 데이터를 백번을 훑어봄. 학습률은 0.1로 정함 
p.fit(x_train, y_train) # x(feature) y(label)를 학습 

# y값의 예측값이 들어간다
res = p.predict(x_test)

# 혼동함수를 만든다. lable과 예측값이 맞아떨어질때 대각선에 덧셈이 된다.
conf = np.zeros((2,2))
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1
print(conf)

# 맞은것을 합산하여 correct에 넣는다
correct = 0
for i in range(2):
    correct += conf[i][i] 
accuracy = correct/len(res) # 정확률 전체 갯수로 나눔  
print("테스트 집합에 대한 정확률", accuracy*100, "%.")
# Perceptron은 선형 분류밖에 하지 못함. 그렇기에 선형 분리가 불가능한 데이터에서는 높은 오류율을 보임.
