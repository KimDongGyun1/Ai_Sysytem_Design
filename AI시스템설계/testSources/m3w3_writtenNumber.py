from sklearn import datasets
from sklearn import svm # svm이라는 학습모델 사용

digit = datasets.load_digits()

s = svm.SVC(gamma=0.1, C=10) # SVC는 분류를 해주는 머신러닝 기법 s에 빈모델을 만들어줌
# digit.data, digit.target를 던져주면 fit이라는 함수가 s라는 모델을 학습을 시켜줌
s.fit(digit.data, digit.target) # 학습 (모델링) 

new_d = [digit.data[0], digit.data[1], digit.data[2]]
results = s.predict(new_d) # new_d 가지고 예측을 한게 results에 들어간다.

print("예측값: ", results)
# 예측값과 참값을 비교
print("참값: ", digit.target[0], digit.target[1], digit.target[2])

results_2 = s.predict(digit.data) # 모든 fit를 가지고 preditct해서 results_2에 넣는다.
correct = [i for i in range(len(results_2)) if results_2[i] == digit.target[i]]
accuracy = len(correct)/len(results_2)
print("정확도: ", accuracy*100, "%") # 퍼센트로 표현해주는 것

# 훈련을 한 데이터로 예측을 했더니 정확히 나왔다라는 것을 알 수 있음.
