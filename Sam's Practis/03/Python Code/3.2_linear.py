#!/usr/bin/env python
# coding: utf-8

# ## 일차함수 관계식 찾기

# x 변수, y 변수 데이터 만들기 (리스트 객체)

# In[1]:


x = [-3,  31,  -11,  4,  0,  22, -2, -5, -25, -14]
y = [ -2,   32,   -10,   5,  1,   23,  -1,  -4, -24,  -13]
print(x)
print(y)


# 그래프 그리기 (matplotlib)

# In[2]:


import matplotlib.pyplot as plt
plt.plot(x, y)
plt.show()


# - 판다스 데이터프레임 만들기

# In[3]:


import pandas as pd
df = pd.DataFrame({'X':x, 'Y':y})
df.shape


# In[4]:


df.head()


# In[5]:


df.tail()


# 머신러닝 - 사이킷런 *LinearRegression* 모형

# In[6]:

#%%
train_features = ['X']
target_cols = ['Y']

X_train = df.loc[:, train_features]
y_train = df.loc[:, target_cols]
#%%
X_train1 = df.loc[:, 'X']
y_train1 = df.loc[:, 'Y']
#%%

X_train2 = df.loc[:, ['X']]
y_train2 = df.loc[:, ['Y']]

print(X_train.shape, y_train.shape)

#%%

from sklearn.linear_model import LinearRegression


lr = LinearRegression()
#lr.fit(X_train, y_train)

# 훈련데이터는 2차원 형태의 pandas DataFrame 이어야 한다. 

lr.fit(X_train1, y_train1)

#%%

import inspect 

print(inspect.getfile(LinearRegression))


# In[8]:


lr.coef_, lr.intercept_


# In[9]:


print ("기울기: ", lr.coef_[0][0])
print ("y절편: ", lr.intercept_[0])


# In[10]:  
#예측 시키기 위한 테스트 
    
import numpy as np


X_new = np.array(11).reshape(1, 1)

print(X_new)

#예측 
X_pred = lr.predict(X_new)



# In[11]:

# 11부터 16-1까지 배열 생성 
X_test = np.arange(11, 16, 1).reshape(-1, 1) #-1인 이유 : 크기에 맞게 행을 생성 , 자동으로 5행 생성 
 
X_test

print(X_test)

# In[12]:


y_pred = lr.predict(X_test)
y_pred


