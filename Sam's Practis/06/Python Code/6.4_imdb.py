#!/usr/bin/env python
# coding: utf-8

# # 1. 환경 설정

# In[1]:


# 라이브러리 설정
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import random

# 랜덤 시드 고정
SEED=12
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)  


# # 2. 데이터셋 준비

# In[2]:


# IMDb 영화 리뷰 데이터셋
from tensorflow.keras import datasets
imdb = datasets.imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000, index_from=3)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape) 


# In[3]:


# 첫번째 리뷰 - 벡터
print(X_train[0])


# In[4]:


# 첫번째 리뷰 - 벡터 길의 (원소 개수)
len(X_train[0])


# In[5]:


word_index = imdb.get_word_index()
word_index


# In[6]:


# 숫자 벡터를 텍스트로 변환
def decode_review_vector(review_vector):
    index_to_word = {value:key for key, value in word_index.items()}
    decoded_review = ' '.join([index_to_word.get(idx - 3, '?') for idx in review_vector])
    return decoded_review

# 첫번째 리뷰 디코딩
decode_review_vector(X_train[0])


# In[7]:


# 첫번째 리뷰의 정답 레이블 
y_train[0]


# # 3. 데이터 전처리

# In[8]:


# 각 리뷰의 단어 개수 분포
review_length = [len(review) for review in X_train]
sns.displot(review_length);


# In[9]:


# Padding
from tensorflow.keras.preprocessing import sequence
X_train_pad = sequence.pad_sequences(X_train, maxlen=250)
X_test_pad = sequence.pad_sequences(X_test, maxlen=250)

print(X_train_pad[0])


# # 4. 모델 학습

# In[10]:


# 모델 정의
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU

def build_model(model_type='RNN'):
    model = Sequential()
    # Embedding
    model.add(Embedding(input_dim=10000, output_dim=128))
    
    # RNN
    if model_type=='RNN':
        model.add(SimpleRNN(units=64, return_sequences=True)) 
        model.add(SimpleRNN(units=64)) 
    # LSTM
    elif model_type=='LSTM':
        model.add(LSTM(units=64, return_sequences=True)) 
        model.add(LSTM(units=64)) 
    # GRU
    elif model_type=='GRU':
        model.add(GRU(units=64, return_sequences=True)) 
        model.add(GRU(units=64))    

    # Dense Classifier
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile
    model.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])

    return model


# In[11]:


rnn_model = build_model('RNN')
rnn_model.summary()


# In[12]:


rnn_history = rnn_model.fit(X_train_pad, y_train, batch_size=32, epochs=10,
                        validation_split=0.1, verbose=2) 


# In[13]:


# 20 epoch 까지 손실함수와 정확도를 그래프로 나타내는 함수

def plot_metrics(history, start=1, end=20):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Loss: 손실 함수
    axes[0].plot(range(start, end+1), history.history['loss'][start-1:end], 
                label='Train')
    axes[0].plot(range(start, end+1), history.history['val_loss'][start-1:end], 
                label='Validation')
    axes[0].set_title('Loss')
    axes[0].legend()
    # Accuraccy: 예측 정확도
    axes[1].plot(range(start, end+1), history.history['accuracy'][start-1:end], 
                label='Train')
    axes[1].plot(range(start, end+1), history.history['val_accuracy'][start-1:end], 
                label='Validation')
    axes[1].set_title('Accuracy')
    axes[1].legend()
plt.show()

# 그래프 그리기
plot_metrics(history=rnn_history, start=1, end=10)    


# In[14]:


# LSTM 모델 적용
lstm_model = build_model('LSTM')

lstm_history = lstm_model.fit(X_train_pad, y_train, batch_size=32, epochs=10,
                        validation_split=0.1, verbose=0) 

plot_metrics(history=lstm_history, start=1, end=10)   


# In[15]:


# GRU 모델 적용
gru_model = build_model('GRU')
gru_model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

gru_history = gru_model.fit(X_train_pad, y_train, batch_size=32, epochs=10,
                        validation_split=0.1, verbose=0) 

plot_metrics(history=gru_history, start=1, end=10)   


# In[15]:




