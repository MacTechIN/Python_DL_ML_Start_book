# -*- coding: utf-8 -*-
"""
pip install joblib
pip install mglearn
"""

from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics 

from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_breast_cancer

import mglearn
import matplotlib.pyplot as plt


#sklean datasets : load_breast_cancer 
#sklean datasets : train_test_split() 학습 테스트 샘플 나누기 
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
   cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []

# 1에서 10까지 n_neighbors를 적용
neighbors_settings = range(1, 11) 




for n_neighbors in neighbors_settings: #이웃의 갯수를 1~10으로 나눠 트레이닝 시킴 
    # 모델 생성
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    
    # 훈련 세트 정확도 저장
    training_accuracy.append(clf.score(X_train, y_train))
    
    # 일반화 정확도 저장
    test_accuracy.append(clf.score(X_test, y_test))

#훈련후 그래프 그림 

plt.plot(neighbors_settings, training_accuracy, label="training_accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test_accuracy")
plt.ylabel("정확도")
plt.xlabel("n_neighbors")
plt.legend()
