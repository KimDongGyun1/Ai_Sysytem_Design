from sklearn import datasets
from sklearn import svm # svm이라는 학습모델 사용
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


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

# 60퍼센트는 훈련할때 쓰고 40프로는 test할 때 쓴다. xt는 x(feature)이고 yt는 y(labal)이다
#random_state=1 이라고 하면 바로 이 random 함수의 seed 값을 고정시키기 때문에 여러번 수행하더라도
#같은 레코드를 추출합니다. random 함수의 seed값을 random_state라고 생각하시면 됩니다.
x_train,x_test,y_train,y_test = train_test_split(xt, yt, test_size=0.3)

s = svm.SVC(gamma=1,C=100) # SVC는 분류를 해주는 머신러닝 기법 s에 빈모델을 만들어줌
s.fit(x_train, y_train) # 학습 (모델링) 

res = s.predict(x_test)

# 훈련집합/테스트집합 나누기는 랜덤하게 샘플링 하므로 우연히 높거나 낮은 정확률을 얻게 분할될 수 있음.
#k겹 교차검증 k개의 성능을 평균하여 신뢰도를 높임.
#accuracy = cross_val_score(s, xt, yt, cv=5)
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


