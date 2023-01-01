# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: anormaly_detection
#     language: python
#     name: anormaly_detection
# ---

# #### [Path] 

import os
Root_Path = 'C:/Users/User/Anormaly_Detection'

# #### [Settings]

# +
# read wav 
from scipy.io.wavfile import read as read_wav
import os

# Preprcessing
import random
import pandas as pd
import numpy as np
import matplotlib.pylab as plt 
import os
from glob import glob

# librosa
import librosa
import librosa.display
import IPython.display as ipd

# Train - Test Split
from sklearn.model_selection import train_test_split

# Modeling
from sklearn.ensemble import IsolationForest

# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore') 
# -

# 설정 값 
CFG = {
    'SR':16000,   # Sampling Rate 
    'N_MFCC':128, # MFCC 벡터를 추출할 개수 (<=128)
    'SEED':41     # Random Seed 
}


# +
# Seed 
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED']) # Seed 고정
# -

# #### [Read] Info Dataset (Contain Paths) #### 

train_df = pd.read_csv('./train.csv') # 모두 정상 Sample
test_df = pd.read_csv('./test.csv')

# #### [Check] Data Row #### 

# +
# 전체 Row 개수 확인
print('Train Wav File - Row Cnt: ',len(train_df))
print('Test Wav File  - Row Cnt : ',len(test_df))

# # SAMPLE ID 개수 확인 
# print('Train Wav File - SAMPLE_ID : ',len(train_df['SAMPLE_ID'].unique()))
# print('Train Wav File - SAMPLE_ID : ',len(test_df['SAMPLE_ID'].unique()))
# print()

# # Fan Type 별 데이터 확인 - Train
# print('Train Fan Type 0 : ',len(train_df[train_df['FAN_TYPE']==0]))
# print('Train Fan Type 2 : ',len(train_df[train_df['FAN_TYPE']==2]))
# print()

# # Fan Type 별 데이터 확인 - Test
# print('Test Fan Type 0 : ',len(train_df[train_df['FAN_TYPE']==0]))
# print('Test Fan Type 2 : ',len(train_df[train_df['FAN_TYPE']==2]))
# print()
# -

# #### [Check] Sampling Rate #### 

# +
# def get_sampling_rate(df):
#     sample_rate_lst = []
#     for path in tqdm(df['SAMPLE_PATH']):
#         sampling_rate, data=read_wav(path)
#         sample_rate_lst.append(sampling_rate)
#     return sample_rate_lst

# +
# train_sample_rate = get_sampling_rate(train_df)
# test_sample_rate = get_sampling_rate(test_df)

# +
# print('Train Sample Rate :', set(train_sample_rate)) # 16,000
# print('Test Sample Rate  :', set(test_sample_rate))  # 16,000
# -

# ### Preprocessing ### 

# ### MFCC ###

def get_mfcc_feature(df):
    features = []
    sr_lst = []
    shape_y = []
    for path in tqdm(df['SAMPLE_PATH']):
        # librosa패키지를 사용하여 wav 파일 load
        y, sr = librosa.load(path, sr=CFG['SR']) # 16,000
        
        # librosa패키지를 사용하여 mfcc 추출
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG['N_MFCC']) # 128

        sr_lst.append(sr)
        shape_y.append(y.shape) # 160000
        
        y_feature = []
        # 추출된 MFCC들의 평균을 Feature로 사용
        for e in mfcc:
            y_feature.append(np.mean(e))
        features.append(y_feature)
    return features,sr_lst,shape_y


train_features,train_st,train_y_shape = get_mfcc_feature(train_df)
test_features,test_st,test_y_shape = get_mfcc_feature(test_df)

# ### Normalization ### 

# - 데이터의 음량이 제 각각인 경우, Normalization 하여 amplitude(진폭) 를 -1 에서 1로 조정 
# - 현재 데이터의 음량이 제 각각인지 명확하게 확인되지는 않으나, 우선 정규화 시행해봄 
# - (Isolation Forest 기준 정확도 3% 향상)

# 직관적으로는 Wav File마다 소리 크기의 차이가 느껴짐 (비교)
# Case 1) 
ipd.Audio(train_df['SAMPLE_PATH'][0])
# print('FAN_TYPE : ',train_df['FAN_TYPE'][0]) # FAN_TYPE : 2

# Case 2)
ipd.Audio(train_df['SAMPLE_PATH'][1])
# print('FAN_TYPE : ',train_df['FAN_TYPE'][1]) # FAN_TYPE : 0

def Normalization(audio):
    audio_np = np.array(audio)
    normed_wav = audio_np / max(np.abs(audio_np))
    return normed_wav


# 음량 Normalization
norm_train_features = [Normalization(train_feature) for train_feature in train_features]
norm_test_features = [Normalization(test_feature) for test_feature in test_features]


# ### Mu-Law Encoding ### 

# - 작은 소리 변화에 민감하게 반응할 수 있도록 Mu-Law 인코딩 
# - wave 값을 표현할 때 작은 값에는 높은 분별력을 큰 값에는 낮은 분별력을 갖도록 함 
# - 아날로그 데이터를 디지털(이진코드)로 변환할 때 사용하는 북미 표준 방식
# - (Isolation Forest 0.79로 향상)

def mu_law(x,mu=255) : 
    return np.sign(x)*np.log(1+mu*np.abs(x))/np.log(1+mu)


# +
# # Mu-law 인코딩 예시 
# x = np.linspace(-1,1,1000)
# x_mu = mu_law(x)

# plt.figure(figsize=(6,4))
# plt.plot(x)
# plt.plot(x_mu)
# plt.show()
# -

mu_law_train = [mu_law(train_feature) for train_feature in train_features]
mu_law_test = [mu_law(test_feature) for test_feature in test_features]

model_train = mu_law_train
model_test  = mu_law_test

# ### Modeling ### 

# #### Isolation Forest #### 

model = IsolationForest(n_estimators=200, max_samples=256, contamination='auto', random_state=CFG['SEED'], verbose=0)
model.fit(model_train)


def get_pred_label(model_pred):
    # IsolationForest 모델 출력 (1:정상, -1:불량) 이므로 (0:정상, 1:불량)로 Label 변환
    model_pred = np.where(model_pred == 1, 0, model_pred)
    model_pred = np.where(model_pred == -1, 1, model_pred)
    return model_pred


test_pred = model.predict(model_test) # model prediction
test_pred = get_pred_label(test_pred)

submit = pd.read_csv('./sample_submission.csv')

submit['LABEL'] = test_pred
submit.head()

submit.to_csv('./result/submit_normalization_mu.csv', index=False)
