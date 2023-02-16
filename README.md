# 개요 ( Playground Series - Season 3 Episode 6 )
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

<img src='./img/heatmap.png'>

## EDA 결과
- 고유값 40개 미만 컬럼에서는 made가 변별력있는 데이터로 보였으나 pointplot 결과 관련성이 보이지 않았다.
- 고유값이 많은 컬럼에서는 squareMeters이 가장 관련있는 데이터로 보여졌으며 다른컬럼에서는 산포도에서 정답에 필요한 데이터로 보이지 않았음

<div><br></div>

## 이상치제거
- train = train.query("made != 10_000") 
- train = train.query("squareMeters < 1e6")
- train = train.query("floors < 1000")
- train = train.query("garage < 1500")

- 미제거 train = train.query("basement < 20_000")
- 미제거 train = train.query("attic < 20_000")

<div><br></div>

## 피처제거
- id
- cityCode
- cityPartRange

<div><br></div>

# 2. 베이스 모델

<div><br></div>

## 베이스 모델 학습
XGBRegressor 사용
제출 : 십만 몇점

<div><br></div>

### feature_importance
<img src='./img/xgb_params_impor.png'>

<div><br></div>

# 3. 스케일링
- MinMax
- MaxAbs
- Standard

<div><br></div>

# 4. 피처 튜닝
- SelectKBest

<div><br></div>

## 피처 튜닝
SelectKBest 사용
제출 : 십이만몇점
모든 컬럼사용 결정

<div><br></div>

# 5. 모델 선정

<div><br></div>

## 모델 선정
GBM 15만 8천
XGB 15만
LGBM 16만 9천

<div><br></div>

# 6. 하이퍼 파라미터 튜닝
- Optuna XGB

<div><br></div>

## 스코어 및 파라미터
- 점수 : 146496
- 파라미터 {
'learning_rate': 0.45,
'n_estimators': 900,
'max_depth': 7}

<div><br></div>

# 7. 풀데이터 활용
- 점수 : 측정
