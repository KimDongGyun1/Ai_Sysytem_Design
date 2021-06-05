# 다층 퍼셉트론은 퍼셉트론 2개로 특징 공간을 세 개로 나눌 수 있다. 선형적으로 분류할 수 없는 문제를 풀 수 있다.
from sklearn.neural_network import MLPClassifier
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

# 은닉층의 노드가 많아지면 복잡한 데이터 모델링에는 유리하지만 과잉 적합 가능성이 있다.
mlp = MLPClassifier(hidden_layer_sizes=(150),#은닉층의 노드수가 150개다.
                    learning_rate_init=0.1, # 첫 번째 learning_rate에 0.1로 준다.
                    batch_size=32, # 미니 배치에 각각 크기를 32개 샘플로 구성한다.
                    solver='sgd', # 미니배치방법
                    verbose=True)
mlp.fit(x_train, y_train) # x(feature) y(label)를 학습 

# y값의 예측값이 들어간다
res = mlp.predict(x_test)

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


