# Playground Series - Season 3 Episode 6
<br>
<div align="center">
  :smirk: platforms & Languages
</div>
<div align="center"> 
  <img src="https://img.shields.io/badge/python-blue?style=flat&logo=Languages&logoColor=white"/>
  <img src="https://img.shields.io/badge/colab-orange?style=flat&logo=Languages&logoColor=white"/>	
</div>
<br>


1.   데이터 준비 및 분석
  - 데이터 준비
  - 시각화 및 분석
  
<div><br></div>

2.   베이스 모델
  - xgb 베이스 모델 제작
  - feature_importance_ 확인

<div><br></div>

3.   스케일링

<div><br></div>

4.   피처 튜닝
  - selector_best 5가지 선정하여 학습하여 결과 확인 후 피처 선정

<div><br></div>

5.   모델 선정
  - 대표 모델 gbm, xgb, lgbm 학습후 모델 선정

<div><br></div>

6.   하이퍼 파라미터 튜닝
  - Optuna를 사용하여 하이퍼 파라미터 튜닝

<div><br></div>

7.   풀데이터 활용
  - 선정된 피처, 모델, 파라미터를 사용해 풀데이터로 학습

<div><br></div>
- 검증 : RMSE

<div><br></div>
<div><br></div>

# 1. 데이터 준비 및 분석

<div><br></div>

##  사용한 패키지
- 기본 라이브러리
- import pandas as pd 
- import numpy as np 

<div><br></div>

- 시각화
- import matplotlib.pyplot as plt
- %matplotlib inline
- import seaborn as sns

<div><br></div>

- 데이터 정제
- from sklearn.model_selection import train_test_split
- from sklearn.metrics import mean_squared_error

<div><br></div>

- 학습 모델
- from xgboost import XGBRegressor
- from lightgbm import LGBMRegressor
- import xgboost as xgb

<div><br></div>

- 피처 튜닝
- from sklearn.feature_selection import f_regression, SelectKBest

<div><br></div>

## 데이터 정보
- 결측치가 없음
- 모두 수치형
- 컬럼 개수가 많은것과 적은것으로 분류해 막대차트와 산포도로 표현 하면 좋아보임
- 컬럼별 의미
<img src='./img/data.png'>

<div><br></div>

## 컬럼 분포도
- 이상치를 가지고 있는 컬럼이 보임
- 예상 이상치 컬럼 : squareMeters, floors, cityCode, basement, attic, garage
<img src='./img/data_bar.png'>

<div><br></div>

## bar차트
- made를 제외하고는 변동폭이 없음
<img src='./img/price_bar.png'>

<div><br></div>

## regplot 차트
- 이상치 제거후 산포도
- squareMeters는 매우 관련있어보이고 나머지는 거의 관련 없음으로 예상
<img src='./img/regplot.png'>

<div><br></div>

## heatmap 상관관계
- squareMeters만 상관관계가 있음으로 보임 ( 상관계수 0.59 )
<img src='./img/heatmap.png'>

## EDA 결과 ( 미완 )
- 고유값 40개 미만 컬럼에서는 made가 변별력있는 데이터로 보였으나 pointplot 결과 관련성이 보이지 않았다.
- 고유값이 많은 컬럼에서는 squareMeters이 가장 관련있는 데이터로 보여졌으며 다른컬럼에서는 산포도에서 정답에 필요한 데이터로 보이지 않았음

<div><br></div>

## 이상치제거
- 시각화한 자료를 바탕으로 이상치 기준 선정 후 제거
- train = train.query("made != 10_000") 
- train = train.query("squareMeters < 1e6")
- train = train.query("floors < 1000")
- train = train.query("garage < 1500")

- 미제거 train = train.query("basement < 20_000")
- 미제거 train = train.query("attic < 20_000")

<div><br></div>

## 피처제거
- 관련성이 적은 피처중 베이스모델에서 점수가 낮은 피처 제거
- id : index 값이므로 제거
- cityCode : 제거시 점수 상승
- cityPartRange : 제거시 점수 상승

<div><br></div>

# 2. 베이스 모델
- XGBRegressor 
<div><br></div>

## 베이스 모델 학습
제출 : 101510 점

<div><br></div>

### feature_importance
<img src='./img/xgb_params_impor.png'>

<div><br></div>

# 3. 스케일링 ( 미완 ) 실제 값들의 크기 나열후 정리
- MinMax
- MaxAbs
- Standard

<div><br></div>

# 4. 피처 튜닝  ( 미완 ) 점수
- SelectKBest

<div><br></div>

## 피처 튜닝
- 베이스모델 학습후 파라미터별 중요도의 대부분이 squareMeters 이였기에 중요도로 피처를 줄여서 학습
- SelectKBest 사용하여 5개의 중요 피처를 선별하여 학습
- 제출 : 십이만몇점
- 점수 상승이 없었기에 모든 컬럼사용 결정

<div><br></div>

# 5. 모델 선정

<div><br></div>

## 모델 선정
- GBM, XGB, LGBM 3가지 모델로 선별 결과 가장점수가 높은 XGB모델 사용 결정
- GBM 158829
- XGB 151210
- LGBM 169071

<div><br></div>

# 6. 하이퍼 파라미터 튜닝 ( 미완 ) 옵튜나 파라미터와 학습량 기제 
- Optuna XGB

<div><br></div>

## 스코어 및 파라미터
- 점수 : 146496
- 파라미터 {
'learning_rate': 0.45,
'n_estimators': 900,
'max_depth': 7}

<div><br></div>

# 7. 풀데이터 활용 ( 점수 측정  하기 )
- 점수 : 측정
