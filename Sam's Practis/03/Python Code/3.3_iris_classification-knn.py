#!/usr/bin/env python
# coding: utf-8

# # 데이터셋 불러오기

# In[1]:


# 라이브러리 환경
import pandas as pd
import numpy as np


# In[2]:


# skleran 데이터셋에서 iris 데이터셋 로딩
from sklearn import datasets 
iris = datasets.load_iris()

# iris 데이터셋은 딕셔너리 형태이므로, key 값을 확인
iris.keys()

print(iris)
# In[3]:


# DESCR 키를 이용하여 데이터셋 설명(Description) 출력
print(iris['DESCR'])


# In[4]:


# target 속성의 데이터셋 크기
print("데이터셋 크기:", iris['target'].shape)

# target 속성의 데이터셋 내용
print("데이터셋 내용: \n", iris['target'])


# In[5]:


# data 속성의 데이터셋 크기

print("데이터셋 타입:", type(iris['data']))
print("데이터셋 크기:", iris['data'].shape)
print("\t 행의 갯수 :", iris['data'].shape[0])
print("\t 열의 갯수 :", iris['data'].shape[1])


# data 속성의 데이터셋 내용 (첫 7개 행을 추출)
print("데이터셋 내용: \n", iris['data'][:7, :])


# In[6]:


# data 속성을 판다스 데이터프레임으로 변환 data array 데이터셋 타입: <class 'numpy.ndarray'>

print(iris['data'])
print(iris['feature_names'])

#%%

df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
print("데이터프레임의 형태:", df.shape)
df.head()


# In[7]:


# 열(column) 이름을 간결하게 변경 길이  cm 뺌  
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df.head(5) # head 5개


# In[8]:


# Target 열 추가 및 데이터 삽입 
df['Target'] = iris['target']
print('데이터셋의 크기: ', df.shape)
df.head()


# # 데이터 탐색(EDA)

# In[9]:


# 데이터프레임의 기본정보
df.info()


# In[10]:


# 통계정보 요약
df.describe()


# In[11]:


# 결측값 확인 , 결측치가 없음
# 컬럼 별로 집계  
df.isnull().sum()


# In[12]:


#중복 데이터 확인 
#행별 중복 유무 : 중복 (True)

    
print(df.duplicated()) # 

df.loc[df.duplicated(), : ]



#%%

#행별     
temp = df[df.duplicated(keep=False)]

print(temp)


#%%



#%%

# 중복 데이터 확인
df.duplicated().sum()


# In[13]:


# 중복 데이터 출력
df.loc[df.duplicated(), :]

df.iloc[142]

# In[14]:


# 중복 데이터 모두 출력
df.loc[(df.sepal_length==5.8)&(df.petal_width==1.9), :]

df.loc[(df.sepal_length==5.8)&(df.petal_length==5.1), :]

#%%

# dfs = df[['sepal_length','sepal_width','petal_length','petal_width','Target']].sort_values()

#dfs = df['sepal_length_1'] = df['sepal_length'].sort_values()

dfs = df.sort_values(['sepal_length', 'sepal_width','petal_length', 'petal_width',  'Target'])

print(dfs)

#%%


#항목별 소팅 참조 

df_sepal_length = df['sepal_lenght'].sort_values()
df_sepal_width = df['sepal_width'].sort_values()
df_petal_length = df['petal_lenght'].sort_values()
df_petal_width = df['petal_width'].sort_values()
df_Target = df['Target'].sort_values() 



# In[15]:


# 중복 데이터 제거
df = df.drop_duplicates()
df.loc[(df.sepal_length==5.8)&(df.petal_width==1.9), :]


# In[16]:


# 변수 간의 상관관계 분석
df.corr()


# In[17]:


# 시각화 라이브러리 설정

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)


# In[18]:


# 상관계수 히트맵
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()


# In[19]:


# Target 값의 분포 - value_counts 함수
df['Target'].value_counts()


# In[20]:


# sepal_length 값의 분포 - hist 함수
plt.hist(x='sepal_length', data=df)
plt.show()


# In[21]:


# sepal_widgth 값의 분포 - displot 함수 (histogram)
sns.displot(x='sepal_width', kind='hist', data=df)
plt.show()


# In[22]:


# petal_length 값의 분포 - displot 함수 (kde 밀도 함수 그래프)
sns.displot(x='petal_width', kind='kde', data=df)
plt.show()


# In[23]:


# 품종별 sepal_length 값의 분포 비교
sns.displot( x='sepal_length', hue='Target', kind='kde', data=df)
plt.show()


# In[24]:


# 나머지 3개 피처 데이터를 한번에 그래프로 출력
for col in ['sepal_width', 'petal_length', 'petal_width']:
    sns.displot(x=col, hue='Target', kind='kde', data=df)
plt.show()


# In[25]:


# 두 변수 간의 관계
sns.pairplot(df, hue = 'Target', height = 2.5, diag_kind = 'kde')
plt.show()


# # Baseline 모델 학습

# #### 학습용-테스트 데이터셋 분리하기

# In[26]:


from sklearn.model_selection import train_test_split

X_data = df.loc[:, 'sepal_length':'petal_width']
y_data = df.loc[:, 'Target']

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    test_size=0.2, 
                                                    shuffle=True, 
                                                    random_state=20)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)






# ### KNN

# In[27]:


# 모델 학습
from sklearn.neighbors import KNeighborsClassifier

#이웃의 갯수를 7로 정함. 
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)


# In[28]:


# 예측
y_knn_pred = knn.predict(X_test)
print("예측값: ", y_knn_pred[:5])


# In[29]:


#SVM (Support Vector Machine)

# 성능 평가
from sklearn.metrics import accuracy_score

knn_acc = accuracy_score(y_test, y_knn_pred)
print("Accuracy: %.4f" % knn_acc)


# ### SVM

# In[30]:


#SVM (Support Vector Machine)
#RBF : Radial Basic Function = 가우시안 함수에 기반한 비선형 대응, 경계가 불분명한 경우 사용하면 좋다.

# 모델 학습
from sklearn.svm import SVC

svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)


# In[31]:


# 예측
y_svc_pred = svc.predict(X_test)
print("예측값: ", y_svc_pred[:5])
# 성능 평가
svc_acc = accuracy_score(y_test, y_svc_pred)
print("Accuracy: %.4f" % svc_acc)



#%%






# ### 로지스틱 회귀

# 시그모이드 함수의 출력값 : 0~1 사이의 값 

# 1에 가까우면 해당 클래스로 분류 
# 0에 가까우면 분류에서 제외 



# In[32]:


# 모델 학습
from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression()
lrc.fit(X_train, y_train)


# In[33]:


# 예측
y_lrc_pred = lrc.predict(X_test)
print("예측값: ", y_lrc_pred[:5])
# 성능 평가
lrc_acc = accuracy_score(y_test, y_lrc_pred)
print("Accuracy: %.4f" % lrc_acc)


# In[34]:

#모든 정답에 대한 확률값 
# 확률값 예측
y_lrc_prob = lrc.predict_proba(X_test)
y_lrc_prob


# ### 의사결정나무

# In[35]:


# 모델 학습 및 예측
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=3, random_state=20)
dtc.fit(X_train, y_train)


# In[36]:


# 예측
y_dtc_pred = dtc.predict(X_test)
print("예측값: ", y_dtc_pred[:5])
# 성능 평가
dtc_acc = accuracy_score(y_test, y_dtc_pred)
print("Accuracy: %.4f" % dtc_acc)


# #앙상블 모델

# ### 보팅

# In[37]:


# Hard Voting 모델 학습 및 예측
from sklearn.ensemble import VotingClassifier
hvc = VotingClassifier(estimators=[('KNN', knn), ('SVM', svc), ('DT', dtc)], 
                       voting='hard')
hvc.fit(X_train, y_train)
# 예측
y_hvc_pred = hvc.predict(X_test)
print("예측값: ", y_hvc_pred[:5])
# 성능 평가
hvc_acc = accuracy_score(y_test, y_hvc_pred)
print("Accuracy: %.4f" % hvc_acc)


# ### 배깅 (랜덤포레스트)

# In[38]:


# 모델 학습 및 검증
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=20)
rfc.fit(X_train, y_train)
# 예측
y_rfc_pred = rfc.predict(X_test)
print("예측값: ", y_rfc_pred[:5])
# 모델 성능 평가
rfc_acc = accuracy_score(y_test, y_rfc_pred)
print("Accuracy: %.4f" % rfc_acc)


# ### 부스팅 (XGBoost)

# In[39]:


# 모델 학습 및 예측
from xgboost import XGBClassifier
xgbc = XGBClassifier(n_estimators=50, max_depth=3, random_state=20)
xgbc.fit(X_train, y_train)
# 예측
y_xgbc_pred = xgbc.predict(X_test)
print("예측값: ", y_xgbc_pred[:5])
# 모델 성능 평가
xgbc_acc = accuracy_score(y_test, y_xgbc_pred)
print("Accuracy: %.4f" % xgbc_acc)


# # 교차 검증 (Cross-Validation)

# ### Hold out 교차 검증

# In[40]:


# 검증용 데이터셋 분리
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, 
                                            test_size=0.3, 
                                            shuffle=True, 
                                            random_state=20)
print(X_tr.shape, y_tr.shape)
print(X_val.shape, y_val.shape)


# In[41]:


# 학습
rfc = RandomForestClassifier(max_depth=3, random_state=20)
rfc.fit(X_tr, y_tr)
# 예측
y_tr_pred = rfc.predict(X_tr)
y_val_pred = rfc.predict(X_val)
# 검증
tr_acc = accuracy_score(y_tr, y_tr_pred)
val_acc = accuracy_score(y_val, y_val_pred)
print("Train Accuracy: %.4f" % tr_acc)
print("Validation Accuracy: %.4f" % val_acc)


# In[42]:


# 테스트 데이터 예측 및 평가
y_test_pred = rfc.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print("Test Accuracy: %.4f" % test_acc)


# ### K-Fold 교차 검증

# In[43]:


# 데이터셋을 5개의 Fold로 분할하는 KFold 클래스 객체 생성
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=20)
# 훈련용 데이터와 검증용 데이터의 행 인덱스를 각 Fold별로 구분하여 생성
num_fold = 1
for tr_idx, val_idx in kfold.split(X_train): 
    print("%s Fold----------------------------------" % num_fold)
    print("훈련: ", len(tr_idx), tr_idx[:10])
    print("검증: ", len(val_idx), val_idx[:10])
    num_fold = num_fold + 1


# In[44]:


# 훈련용 데이터와 검증용 데이터의 행 인덱스를 각 Fold별로 구분하여 생성
val_scores = []
num_fold = 1
for tr_idx, val_idx in kfold.split(X_train, y_train):
    # 훈련용 데이터와 검증용 데이터를 행 인덱스 기준으로 추출
    X_tr, X_val = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
    # 학습
    rfc = RandomForestClassifier(max_depth=5, random_state=20)
    rfc.fit(X_tr, y_tr)
    # 검증
    y_val_pred = rfc.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)  
    print("%d Fold Accuracy: %.4f" % (num_fold, val_acc))
    val_scores.append(val_acc)   
    num_fold += 1  


# In[45]:


# 평균 Accuracy 계산
import numpy as np
mean_score = np.mean(val_scores)
print("평균 검증 Accuraccy: ", np.round(mean_score, 4))

