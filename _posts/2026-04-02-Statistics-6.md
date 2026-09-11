---
layout: single
title: "OneHotEncoder와 pd.get_dummies의 완벽한 비교 및 차이점 정리"
date: 2026-04-11 20:00:00 +0900
categories: [데이터 분석 & 통계 (Statistics)]
tags: [Pandas, Scikit-learn, OneHotEncoder, get_dummies, MachineLearning, 전처리]
---

머신러닝이나 데이터 분석을 진행할 때 범주형 변수를 수치형으로 변환하기 위해 가장 흔하게 사용하는 두 가지 도구가 있습니다. 바로 Pandas의 `pd.get_dummies()`와 Scikit-learn의 `OneHotEncoder()`입니다.

처음에는 "둘 다 원핫 인코딩을 해주는 함수 아닌가?" 하고 가볍게 넘어가기 쉽지만, **실무 머신러닝 파이프라인 구축** 관점에서는 두 도구의 동작 방식과 철학이 매우 다릅니다. 이번 포스팅에서는 두 방식의 결정적인 차이점과 새로운 값이 들어왔을 때의 처리 방식, 그리고 반환 타입까지 상세히 알아보겠습니다.

---

## 1. 개요: `get_dummies` vs `OneHotEncoder`

*   **`pd.get_dummies()`**: 데이터를 볼 때마다 그 데이터에 존재하는 고유값(Category)들을 기준으로 **무조건 컬럼을 새로 만들어냅니다.** 따라서 테스트 데이터에 학습 때 없던 새로운 카테고리가 등장하면, 그 새로운 값을 위한 컬럼이 뒤쪽에 툭 하고 추가됩니다.
*   **`OneHotEncoder()`**: `.fit()`을 하는 순간 학습 데이터에 있던 카테고리 종류를 확정(고정)해 버립니다. 이후 `.transform()`을 할 때는 학습 때 알던 카테고리들만 기준으로 변환을 수행합니다.

---

## 2. 새로운 변수(카테고리)가 들어올 때의 차이

머신러닝 모델을 배포한 이후, 실제 서비스 환경에서는 학습 때 보지 못했던 새로운 데이터(예: 새로운 카테고리)가 들어올 수 있습니다. 이때 두 방식은 완전히 다르게 대처합니다.

### 📌 `OneHotEncoder`의 경우 (새로운 값이 들어올 때)

기본 설정 상태(`handle_unknown='error'`)에서는 학습 때 보지 못한 새로운 값이 들어오면 에러를 발생시킵니다. 하지만 실무에서는 새로운 값이 들어올 수 있으므로 보통 아래처럼 옵션을 줍니다.

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
```

이렇게 `handle_unknown='ignore'`로 설정하면, 새로운 값이 들어왔을 때 에러를 내지 않고 **해당 원소에 해당하는 모든 원핫 벡터를 전부 0으로 처리**해 버립니다.
*   뒤에 새로운 컬럼이 툭 튀어나오는 것이 아니라, 기존에 고정되어 있던 컬럼 구조를 유지한 채 새로운 값에 대해서는 모든 자리를 0으로 채우는 방식입니다.
*   연산 시 해당 행 데이터의 값이 전부 0이 되므로, 모델 입장에서 이 행은 "어떤 카테고리에도 속하지 않는 상태"로 인식되어 계산됩니다.

### 📌 `get_dummies()`의 경우

새로운 값이 들어오면 에러가 나는 대신, **새로운 컬럼이 우르르 추가**됩니다.
*   예를 들어 학습 데이터에 `['사과', '바나나']`만 있었다가 테스트 데이터에 `['포도']`가 새로 들어오면, 포도 컬럼이 뒤에 새로 생기면서 기존 컬럼들과 함께 총 3개의 컬럼(`[1, 0, 0]`, `[0, 1, 0]`, `[0, 0, 1]`)으로 늘어납니다.
*   **주의점:** 머신러닝 모델은 학습할 때 사용한 입력 컬럼의 개수(차원)와 테스트할 때 들어오는 입력 컬럼의 개수가 정확히 일치해야 합니다. 따라서 `get_dummies()`를 학습용과 테스트용에 각각 따로 쓰면 컬럼 개수가 안 맞아 에러가 나거나 엉뚱한 예측을 하게 됩니다.

---

## 3. `handle_unknown='ignore'` 사용 시 발생하는 치명적인 한계와 해결책

만약 `drop='first'` 등을 적용한 상태에서 `OneHotEncoder`를 사용하고 새로운 모르는 값이 들어와 모든 컬럼이 `0`으로 채워졌다고 가정해 봅시다.

```text
결과 예시: 1 / 0 / 0 / 0 / 0 / 0 
(앞의 1은 성별, 뒤의 5개 0은 모든 색상 컬럼이 꺼진 상태)
```

여기서 뒤의 5개 컬럼이 모두 `0`인 이유는 두 가지 중 하나입니다.
1. 원래 드롭되었던 **기준 색상(기본 색상)**이 들어온 경우
2. 학습 때 보지 못한 완전히 **새로운 색상(예: 보라색)**이 들어와서 `ignore` 처리된 경우

컴퓨터 입장에서는 결과 행에 전부 0이 찍혀 있으므로 이 둘을 구별할 방법이 없습니다. 

### 💡 실무 해결 팁
1. **원본 데이터프레임 병행 관리:** 인코딩된 0과 1 데이터만 모델에 넣되, 원본 문자열 데이터(예: 색상: 보라색)는 원본 컬럼이나 별도 로그로 보관하여 추적 가능하게 합니다.
2. **'기타(Other)' 카테고리 미리 만들기:** 데이터 전처리 단계에서 희귀한 값들은 미리 '기타'라는 하나의 카테고리로 묶어버립니다. 이러면 `color_기타` 컬럼이 생기므로 기본 색상과 새로운 값을 완벽하게 구분할 수 있습니다.

---

## 4. 반환 타입의 차이: DataFrame vs NumPy Array

두 도구는 기본적으로 반환해 주는 데이터 타입도 다릅니다.

*   **`pd.get_dummies()`**: 이름에 `pd`가 들어간 것 처럼, 실행하면 바로 **Pandas 데이터프레임(DataFrame)** 형태로 결과를 반환하며 컬럼 이름도 예쁘게 자동 생성됩니다.
*   **`OneHotEncoder()`**: Scikit-learn의 도구이므로, 기본적으로는 **NumPy 2차원 배열(Array)** 형태로 결과를 반환하며 컬럼 이름이 따로 붙지 않습니다.

### 🔄 상호 변환 방법

#### 1) OneHotEncoder 결과를 DataFrame으로 변환하기
실무에서는 컬럼 이름 확인과 데이터 병합을 위해 데이터프레임 형태로 다루는 것이 훨씬 편리합니다.

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# 인코더 생성 및 학습
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_array = encoder.fit_transform(df[['color']])

# 인코딩된 컬럼 이름 가져오기 후 DataFrame 변환
column_names = encoder.get_feature_names_out(['color'])
df_encoded = pd.DataFrame(encoded_array, columns=column_names)
```

#### 2) get_dummies 결과를 2차원 배열(NumPy Array)로 변환하기
`pd.get_dummies()`로 만든 데이터프레임 뒤에 `.to_numpy()`를 붙여주면 컬럼 이름과 인덱스가 떨어져 나가고 순수한 2차원 배열만 남게 됩니다.

```python
import pandas as pd

df_dummies = pd.get_dummies(df, drop_first=True)

# 최신 판다스 권장 방식인 to_numpy() 사용 (머신러닝 입력용으로 .astype(int) 권장)
array_data = df_dummies.to_numpy().astype(int)
print(type(array_data))  # <class 'numpy.ndarray'>
```

---

## 요약

*   **`pd.get_dummies()`**: 탐색적 데이터 분석(EDA)이나 시각화, 간단한 전처리에 매우 편리하지만, 학습/테스트 데이터 파이프라인 관리 시 컬럼 개수 불일치 문제를 조심해야 합니다.
*   **`OneHotEncoder()`**: 배포(Production) 환경에서 학습 때의 컬럼 구조를 완벽하게 고정하고 안전하게 유지해야 하는 머신러닝 모델 파이프라인에 최적화되어 있습니다.
