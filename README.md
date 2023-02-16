# 개요 ( Playground Series - Season 3 Episode 6 )
<br>
<div align="center">
  platforms & Languages
</div>
<div align="center"> 
  <img src="https://img.shields.io/badge/python-blue?style=flat&logo=Languages&logoColor=white"/>
  <img src="https://img.shields.io/badge/colab-orange?style=flat&logo=Languages&logoColor=white"/>	
</div>
<br>
<br>


1.   데이터 준비 및 분석
  - 데이터 준비
  - 시각화 및 분석

2.   베이스 모델
  - xgb 베이스 모델 제작
  - feature_importance_ 확인

3.   피처 튜닝
  - selector_best 5가지 선정하여 학습하여 결과 확인 후 피처 선정

4.   모델 선정
  - 대표 모델 gbm, xgb, lgbm 학습후 모델 선정

5.   하이퍼 파라미터 튜닝
  - Optuna를 사용하여 하이퍼 파라미터 튜닝

6.   풀데이터 활용
  - 선정된 피처, 모델, 파라미터를 사용해 풀데이터로 학습

# 1. 데이터 준비 및 분석
##  1. 사용한 패키지
- 기본 라이브러리
import pandas as pd 
import numpy as np 

- 시각화
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

- 데이터 정제
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

- 학습 모델
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb

- 피처 튜닝
from sklearn.feature_selection import f_regression, SelectKBest

## 데이터 정보
<img src='./img/data.png'>

## 컬럼 분포도
<img src='./img/data_bar.png'>

## bar차트
<img src='./img/price_bar.png'>

## regplot 차트
<img src='./img/regplot.png'>

## heatmap 상관관계
<img src='./img/heatmap.png'>
- 고유값 40개 미만 컬럼에서는 made가 변별력있는 데이터로 보였으나 pointplot 결과 관련성이 보이지 않았다.
- 고유값이 많은 컬럼에서는 squareMeters이 가장 관련있는 데이터로 보여졌으며 다른컬럼에서는 산포도에서 정답에 필요한 데이터로 보이지 않았음

## 이상치제거
- train = train.query("made != 10_000") 
- train = train.query("squareMeters < 1e6")
- train = train.query("floors < 1000")
- train = train.query("garage < 1500")

- 미제거 train = train.query("basement < 20_000")
- 미제거 train = train.query("attic < 20_000")

## 피처제거
id
cityCode
cityPartRange

# 2. 베이스 모델
## 베이스 모델 학습
XGBRegressor 사용
제출 : 십만 몇점

### feature_importance
<img src='./img/xgb_params_impor.png'>

# 3. 피처 튜닝
- SelectKBest

## 피처 튜닝
SelectKBest 사용
제출 : 십이만몇점
모든 컬럼사용 결정

# 4. 모델 선정
## 모델 선정
GBM 15만 8천
XGB 15만
LGBM 16만 9천

# 5. 하이퍼 파라미터 튜닝
- Optuna XGB
## 스코어 및 파라미터
- 점수 : 146496
- 파라미터 {
'learning_rate': 0.45,
'n_estimators': 900,
'max_depth': 7}

# 6. 풀데이터 활용
- 점수 : 측정
