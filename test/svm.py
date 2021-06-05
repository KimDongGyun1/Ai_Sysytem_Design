from sklearn import svm # svm이라는 학습모델 사용 svm은 여백을 최대로 하는 최적 모델을 찾아줌
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score # knn


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

# C를 크게 설정하면 여백이 줄어들어 훈련 집합에 대한 에러가 줄어듦 -> 일반화 능력이 떨어짐
# C를 작게 설정하면 여백이 늘어나고 훈련 집에에 대한 에러를 허용해줌 -> 일반화 능력이 좋아짐
s = svm.SVC(gamma=1,C=100) # SVC는 분류를 해주는 머신러닝 기법 s에 빈모델을 만들어줌

# 훈련집합/테스트집합 나누기는 랜덤하게 샘플링 하므로 우연히 높거나 낮은 정확률을 얻게 분할될 수 있음.
#k겹 교차검증 k개의 성능을 평균하여 신뢰도를 높임.
accuracy = cross_val_score(s, xt, yt, cv=5)
print(accuracy)
print("테스트 집합에 대한 정확률", accuracy.mean()*100, "%.")
