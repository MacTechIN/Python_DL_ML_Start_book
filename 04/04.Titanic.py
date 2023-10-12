#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:03:33 2023

@author: sl
"""

    
# 데이터 로딩 

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns


#%%

# 데이터 로딩 

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
submission = pd.read_csv("./submission.csv")


#%%
print(train.shape, test.shape, submission.shape)

#출력 : (891, 12) (418, 11) (418, 2)


#%%
train.head(3)  # 11개 Column title 
print(train.keys())
"""
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
"""
#%%
test.head(3)
print(test.keys())

"""
Index(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
       'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
"""
# 문제 : test 승객ID에 대한 survived 를 예측하고, 그 예측값을 Survived에 입력해야 한다. 
 
#%%
print(submission.head(3))
submission.keys()
"""
   PassengerId  Survived
0          892         0
1          893         1
2          894         0
Out[21]: Index(['PassengerId', 'Survived'], dtype='object')
"""

#%%
# EDA (데이터 탐색 )

# 1. 데이터 구조  ; info() , isnull() , duplicated () 

train.info()

#%%
train.describe(include= 'all')


#%%
# 결측잡 확인 : missingno 라이브러리 사용 

# !pip install missingno

# 막대 그래프로 train 에 그래프로 나타낸다. 

#%%
import missingno as msno 

msno.bar(train, figsize=(10,5), color=(0.7,0.2,0.2))
msno.matrix(test, figsize=(10,5),color=(0.7,0.2,0.2))
plt.show()

#%%
# 2-4 상관 관계 분석 

corr_train = train[['Pclass','Age','SibSp','Parch','Fare','Survived']]

plt.figure(figsize=(8, 8))
sns.set(font_scale=0.8)
sns.heatmap(corr_train.corr(), annot=True, cbar=True);
plt.show()


#%%
#3-1 데이터의 결합 

train['TrainSplit'] = 'Train'
test['TrainSplit'] = 'Test'

data = pd.concat([train, test],axis=0)
print(data.shape)

#%%
## 3-2 데이터 전처리 
# 결측값 처리 

# 숫자형 데이터 'Pclass','Age','SibSp','Parch','Fare','Survived'만 모아 data_num 을 만든다. 
data_num = data.loc[:,['Pclass','Age','SibSp','Parch','Fare','Survived']]

print(data_num.head())

#%%
# 결측값 처리 , age = 경균값 mean, fare는 최빈값 

checkAgeResult = data_num['Age'].isnull()
print(data_num[checkAgeResult])

#%%
data_num['Age'] = data_num['Age'].fillna(data_num['Age'].mean())

print(data_num['Age'])

#%%

#Fair 는 152번 1개의 결측데이터 확인 가능하다. 
f_result = data_num['Fare'].isnull()
print(data_num[f_result])

#%%

#mode()에 리턴 값이 시리즈 이므로 mode()[0] 으로 최빈수를 구할수 있다. 
print(type (data_num['Fare'].mode()))

print(data_num['Fare'].mode()[0])

#%%
data_num['Fare'] = data_num['Fare'].fillna(data_num['Fare'].mode()[0])


#%%
data_num.isnull().sum()

#%%
# X_train 에 들어갈 Column 을 모아 selectec_feature_columns = [] 라고 만들자 

selected_features_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

# y_train 은 예측을 해야할 컬럼 이므로 'Survived' 가 될 것이다. 


