---
title: "rPPG-Toolbox PhysNet 연구 코드 최적화 보고서"
author: "고급파이썬프로그래밍 최종 과제"
date: "2026-06-17"
---

# rPPG-Toolbox PhysNet 연구 코드 최적화 보고서

## 1. 연구 코드 소개

본 보고서는 실제 연구 파이프라인에서 사용한 `rPPG-Toolbox`의 PhysNet 코드를 대상으로 한다. 기준 설정 파일은 `UBFC-PHYS_PHYSNET.yaml`이다. PhysNet은 얼굴 영상에서 원격 광용적맥파(remote photoplethysmography, rPPG)를 추정하는 딥러닝 모델이며, 입력 영상 clip으로부터 BVP/rPPG 시계열을 예측한 뒤 FFT 또는 peak detection 기반 심박 지표로 평가한다.

본 과제에서는 원본 연구 코드와 실제 UBFC-PHYS 데이터셋을 직접 수정하거나 접근하지 않았다. 대신 원본 코드의 핵심 병목을 분석하고, 동일한 계산 구조를 재현한 독립 구현을 `advanced_python_physnet_safe/`에 작성했다. 성능 비교는 실제 데이터 대신 synthetic data로 수행했다. 이는 연구 데이터 경로와 개인정보성 subject 정보를 공개하지 않으면서도 최적화 전후를 비교할 수 있게 하기 위한 선택이다.

### 1.1 파이프라인 내 위치

| 구분 | 역할 | 원본 코드 위치 |
|---|---|---|
| 실행 진입점 | config 로드, DataLoader 생성, trainer 선택 | `main.py` |
| 모델 | 3D CNN 기반 PhysNet forward | `neural_methods/model/PhysNet.py` |
| 학습/검증/테스트 | train/valid/test loop, checkpoint, metric 호출 | `neural_methods/trainer/PhysnetTrainer.py` |
| 손실 함수 | rPPG와 BVP label 간 negative Pearson loss | `neural_methods/loss/PhysNetNegPearsonLoss.py` |
| 데이터 로딩/전처리 | `.npy` clip loading, face crop, normalization, chunking | `dataset/data_loader/BaseLoader.py`, `dataset/data_loader/UBFCPHYSLoader.py` |
| 평가 | HR, SNR, MACC 등 평가 지표 계산 | `evaluation/metrics.py`, `evaluation/post_process.py` |

### 1.2 입력과 출력

`UBFC-PHYS_PHYSNET.yaml` 기준으로 PhysNet은 다음 구조의 데이터를 사용한다.

| 항목 | 형태 |
|---|---|
| video clip | `[B, 3, 128, 128, 128]` |
| label / predicted rPPG | `[B, 128]` |
| frame size | `128 x 128` |
| clip length | `128` |
| batch size | config 기준 `4` |

출력은 예측 rPPG 시계열과 평가 지표다. 평가 지표에는 MAE, RMSE, MAPE, Pearson, SNR, MACC가 포함된다.

## 2. 기존 코드 문제점 분석

### 2.1 학습 루프의 반복 손실 계산

PhysNet의 negative Pearson loss는 학습 batch마다 호출된다. 원본 구현은 batch dimension을 Python loop로 순회한다. 이 구조에서는 batch size가 커질수록 Python interpreter overhead가 선형으로 증가한다.

**원본 코드: `neural_methods/loss/PhysNetNegPearsonLoss.py`**

```python
def forward(self, preds, labels):       
    loss = 0
    for i in range(preds.shape[0]):
        sum_x = torch.sum(preds[i])               
        sum_y = torch.sum(labels[i])             
        sum_xy = torch.sum(preds[i]*labels[i])       
        sum_x2 = torch.sum(torch.pow(preds[i],2))  
        sum_y2 = torch.sum(torch.pow(labels[i],2)) 
        N = preds.shape[1]
        pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
        loss += 1 - pearson
        
        
    loss = loss/preds.shape[0]
    return loss
```

문제의 핵심은 수식 자체가 아니라 실행 방식이다. `preds`와 `labels`는 이미 tensor이므로 batch 전체에 대해 한 번에 sum을 계산할 수 있다.

### 2.2 평가 단계 MACC의 O(T^2) 계산

원본 MACC 계산은 모든 lag에 대해 `np.roll`과 `np.corrcoef`를 반복한다. 길이 `T`인 신호에서 lag가 `T-1`개이고, 각 lag마다 길이 `T` 상관계수를 계산하므로 지배 연산은 대략 `O(T^2)`이다.

**원본 코드: `evaluation/post_process.py`**

```python
def _compute_macc(pred_signal, gt_signal):
    pred = deepcopy(pred_signal)
    gt = deepcopy(gt_signal)
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    min_len = np.min((len(pred), len(gt)))
    pred = pred[:min_len]
    gt = gt[:min_len]
    lags = np.arange(0, len(pred)-1, 1)
    tlcc_list = []
    for lag in lags:
        cross_corr = np.abs(np.corrcoef(
            pred, np.roll(gt, lag))[0][1])
        tlcc_list.append(cross_corr)
    macc = max(tlcc_list)
    return macc
```

이 함수는 test 단계에서 subject/window별로 반복 호출된다. 따라서 전체 학습보다 짧은 평가 단계에서도 긴 signal이나 많은 window에서는 병목이 될 수 있다.

### 2.3 detrend의 반복 matrix inverse

원본 `_detrend`는 호출될 때마다 identity matrix, second-difference matrix, dense inverse를 새로 만든다. 같은 signal length와 같은 `lambda_value`를 반복 사용하는 평가에서는 중복 계산이다.

**원본 코드: `evaluation/post_process.py`**

```python
def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal
```

이 구현은 수학적으로는 명확하지만, 같은 길이의 신호를 반복 처리할 때 projection matrix를 매번 재구성하므로 시간과 메모리를 낭비한다.

### 2.4 구조적 문제

`PhysnetTrainer`는 학습, 검증, 테스트, checkpoint 저장, metric 호출을 모두 담당한다. `BaseLoader` 역시 raw reading, crop, normalization, chunking, multiprocessing, file-list 생성, runtime loading을 한 클래스에서 수행한다. 이번 작업에서는 원본을 수정하지 않았지만, 단일책임원칙 관점에서는 분리 후보다.

## 3. 적용한 수업 개념

### 3.1 E. 딥러닝 연구 코드 최적화

negative Pearson loss는 딥러닝 학습 루프에서 반복 호출되는 tensor 연산이다. 모델 크기나 데이터 크기를 줄이지 않고, 같은 입력과 같은 수식을 batch-wise tensor 연산으로 바꾸는 것이므로 딥러닝 연구 코드 최적화에 해당한다.

### 3.2 A. 자료구조 및 복잡도 기반 개선

MACC는 모든 lag를 순회하는 반복 correlation 구조였다. 이를 FFT 기반 circular correlation으로 바꾸면 모든 lag의 상관 구조를 한 번에 계산할 수 있다. 기존 방식은 `O(T^2)` 성격이고, FFT 기반 방식은 일반적으로 `O(T log T)` 성격이다. 단순히 함수를 바꾼 것이 아니라 지배 연산 자체를 바꾼 개선이다.

### 3.3 D. Decorator 및 caching

detrend projection matrix는 같은 `(signal_length, lambda_value)`에 대해 재사용할 수 있다. 개선 코드에서는 `functools.lru_cache` decorator를 사용했다. 이는 decorator를 형식적으로 붙인 것이 아니라, 평가 단계의 반복 matrix inverse를 제거하기 위한 caching 정책이다.

### 3.4 미적용 항목과 이유

Iterator/generator 기반 전처리 streaming과 class/SRP refactoring도 의미 있는 후보였다. 예를 들어 `BaseLoader.chunk`는 모든 chunk를 list로 materialize한 뒤 저장하고, `PhysnetTrainer`는 여러 책임을 동시에 수행한다. 그러나 이번 작업에서는 원본 데이터와 원본 코드에 영향을 주지 않는 것이 우선이었으므로, 실제 전처리/트레이너 통합 변경이 필요한 B/C 항목은 구현하지 않고 분석 후보로만 남겼다.

## 4. 개선 과정

개선 코드는 원본 파일을 수정하지 않고 아래 파일에 독립 구현했다.

- `advanced_python_physnet_safe/physnet_safe_optimizations.py`
- `advanced_python_physnet_safe/benchmark_coursework.py`
- `advanced_python_physnet_safe/results/benchmark_results.csv`

### 4.1 Negative Pearson loss 개선

**개선 코드**

```python
class VectorizedNegPearson(nn.Module):
    """Batch-vectorized equivalent of PhysNet's negative Pearson loss."""

    def forward(self, preds, labels):
        preds = preds.view(preds.shape[0], -1)
        labels = labels.view(labels.shape[0], -1)

        sum_x = torch.sum(preds, dim=1)
        sum_y = torch.sum(labels, dim=1)
        sum_xy = torch.sum(preds * labels, dim=1)
        sum_x2 = torch.sum(preds.pow(2), dim=1)
        sum_y2 = torch.sum(labels.pow(2), dim=1)

        n = preds.shape[1]
        numerator = n * sum_xy - sum_x * sum_y
        denominator = torch.sqrt(
            (n * sum_x2 - sum_x.pow(2)) * (n * sum_y2 - sum_y.pow(2))
        )
        eps = torch.finfo(preds.dtype).eps
        pearson = numerator / denominator.clamp_min(eps)
        return torch.mean(1 - pearson)
```

기존 코드와 동일한 Pearson 수식을 사용하지만, sample별 반복을 제거했다. 결과 동등성은 `torch.allclose`로 확인했다.

### 4.2 MACC 개선

**개선 코드**

```python
def fft_circular_macc(pred_signal, gt_signal):
    pred = np.asarray(pred_signal, dtype=np.float64).reshape(-1)
    gt = np.asarray(gt_signal, dtype=np.float64).reshape(-1)
    min_len = min(pred.size, gt.size)
    pred = pred[:min_len]
    gt = gt[:min_len]
    if min_len < 2:
        return float("nan")

    pred_centered = pred - np.mean(pred)
    gt_centered = gt - np.mean(gt)
    pred_norm = np.linalg.norm(pred_centered)
    gt_norm = np.linalg.norm(gt_centered)
    if pred_norm == 0.0 or gt_norm == 0.0:
        return float("nan")

    corr = scipy.fft.ifft(scipy.fft.fft(pred_centered) * np.conj(scipy.fft.fft(gt_centered))).real
    corr = np.abs(corr / (pred_norm * gt_norm))
    return float(np.max(corr[: min_len - 1]))
```

원본이 lag `0`부터 `len(pred)-2`까지만 보므로 개선 코드도 같은 범위의 최대값을 사용했다. 따라서 benchmark에서 원본 참조 구현과 같은 값을 내는지 확인할 수 있다.

### 4.3 detrend 개선

**개선 코드**

```python
@lru_cache(maxsize=32)
def _detrend_projection(signal_length, lambda_value):
    signal_length = int(signal_length)
    lambda_value = float(lambda_value)
    h = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    d_mat = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    return h - np.linalg.inv(h + (lambda_value**2) * np.dot(d_mat.T, d_mat))


def cached_detrend(input_signal, lambda_value):
    input_signal = np.asarray(input_signal)
    projection = _detrend_projection(input_signal.shape[0], float(lambda_value))
    return np.dot(projection, input_signal)
```

원본과 같은 projection matrix를 사용하지만, matrix 생성 결과를 cache한다. 같은 길이의 signal을 반복 처리하는 평가 상황에서 효과가 크다.

### 4.4 고려했지만 적용하지 않은 대안

| 대안 | 미적용 이유 |
|---|---|
| 원본 `PhysNetNegPearsonLoss.py` 직접 수정 | 원본 연구 코드 안전 보존 요구 |
| `PhysnetTrainer` class 분리 | 원본 학습 파이프라인 전반에 영향 |
| `BaseLoader.chunk` generator streaming | raw preprocessing과 파일 저장 정책에 영향 |
| 실제 UBFC-PHYS 데이터 benchmark | 민감 경로와 데이터 접근 금지 조건 |
| AMP 또는 `torch.compile` | GPU 환경과 모델 전체 실행 검증 필요 |

## 5. 최적화 결과

### 5.1 측정 조건

측정은 `advanced_python_physnet_safe/benchmark_coursework.py`로 수행했다.

| 항목 | 값 |
|---|---|
| 반복 횟수 | warm-up 1회 후 본 측정 10회 |
| random seed | 100 |
| 시간 측정 | `time.perf_counter` |
| peak memory 측정 | `tracemalloc` |
| OS | Linux 5.15.0-139-generic |
| Python | 3.8.20 |
| NumPy | 1.22.0 |
| SciPy | 1.5.2 |
| PyTorch | 2.1.2+cu121 |
| device | CPU |
| 실제 데이터 사용 여부 | 사용하지 않음 |

### 5.2 Negative Pearson loss 결과

| 입력 | Before 평균±표준편차 ms | After 평균±표준편차 ms | 개선 배율 | Correct |
|---|---:|---:|---:|---|
| B=4,T=128 | 0.1521 ± 0.0344 | 0.0447 ± 0.0036 | 3.40x | True |
| B=16,T=128 | 0.5448 ± 0.0150 | 0.0438 ± 0.0028 | 12.43x | True |
| B=64,T=128 | 2.1442 ± 0.0119 | 0.0469 ± 0.0030 | 45.75x | True |
| B=64,T=512 | 2.4575 ± 0.5431 | 0.1104 ± 0.0133 | 22.26x | True |

batch size가 커질수록 개선 폭이 커졌다. 이는 원본의 Python loop 횟수가 batch size에 비례하기 때문이다.

### 5.3 MACC 결과

| 입력 | Before 평균±표준편차 ms | After 평균±표준편차 ms | 개선 배율 | Correct |
|---|---:|---:|---:|---|
| T=128 | 8.3623 ± 0.0768 | 0.0570 ± 0.0062 | 146.63x | True |
| T=256 | 16.9391 ± 0.0944 | 0.0614 ± 0.0064 | 275.88x | True |
| T=512 | 34.9527 ± 0.1442 | 0.0665 ± 0.0066 | 525.86x | True |
| T=1024 | 70.3462 ± 0.4283 | 0.0758 ± 0.0067 | 927.87x | True |

입력 길이가 증가할수록 개선 배율이 커졌다. 이는 `O(T^2)` 반복 correlation을 FFT 기반 계산으로 바꾸었기 때문이다.

### 5.4 detrend 결과

| 입력 | Before 평균±표준편차 ms | After 평균±표준편차 ms | 개선 배율 | Correct |
|---|---:|---:|---:|---|
| T=32 | 0.2115 ± 0.0175 | 0.0055 ± 0.0008 | 38.70x | True |
| T=64 | 130.5220 ± 64.6791 | 0.0061 ± 0.0014 | 21300.83x | True |
| T=128 | 596.5495 ± 55.7484 | 2.4590 ± 5.8683 | 242.60x | True |
| T=256 | 1775.3165 ± 111.4675 | 0.0189 ± 0.0025 | 93906.19x | True |

### 5.5 peak memory 비교

| 항목 | 입력 | Before peak KiB | After peak KiB |
|---|---|---:|---:|
| loss | B=64,T=128 | 1.182 | 0.938 |
| MACC | T=1024 | 82.648 | 64.730 |
| detrend | T=256 | 2055.391 | 2.094 |

detrend의 memory 감소가 특히 크다. 원본은 dense matrix와 inverse를 매번 생성하지만, after는 cache된 projection matrix를 재사용하기 때문이다.

### 5.6 코드 구조 변화

| 항목 | Before | After |
|---|---|---|
| negative Pearson | batch loop가 손실 함수 내부에 직접 있음 | `VectorizedNegPearson`으로 독립 구현 |
| MACC | lag 반복과 corrcoef 호출이 한 함수에 결합 | FFT 기반 계산 함수로 분리 |
| detrend | projection matrix 생성과 적용이 매 호출 결합 | projection 생성 cache와 적용 함수 분리 |
| benchmark | 수동 확인 중심 | CSV와 환경 JSON 자동 저장 |

## 6. 결과 해석 및 한계

### 6.1 왜 빨라졌는가

Negative Pearson loss는 Python loop를 제거하고 tensor 연산으로 바꾸었기 때문에 빨라졌다. 특히 batch size가 큰 경우 loop overhead가 크게 줄어든다.

MACC는 계산 복잡도 자체가 줄었다. 원본은 lag마다 상관계수를 다시 계산했지만, FFT 기반 방식은 모든 circular lag의 correlation을 한 번에 구한다. 따라서 signal length가 커질수록 after의 이점이 커진다.

Detrend는 반복되는 matrix inverse를 cache했다. 동일한 length와 lambda 값이 반복될수록 cache hit가 늘어나며, 이 경우 matrix inverse 비용을 한 번만 지불하면 된다.

### 6.2 어떤 조건에서 효과가 큰가

| 개선 | 효과가 큰 조건 |
|---|---|
| VectorizedNegPearson | batch size가 클 때 |
| FFT MACC | 평가 signal/window 길이가 길 때 |
| cached detrend | 같은 window length가 반복될 때 |

### 6.3 trade-off

FFT MACC는 기존 lag 범위와 circular correlation 정의를 맞추도록 구현했지만, correlation 정의를 조금이라도 바꾸면 결과가 달라질 수 있다. 따라서 실제 코드에 반영할 때는 원본 metric과의 동등성 test가 반드시 필요하다.

`lru_cache`는 memory를 사용해 시간을 줄인다. 본 구현에서는 `maxsize=32`로 제한하여 cache가 무한히 커지지 않게 했다. 하지만 매우 다양한 window length를 처리하는 데이터셋에서는 cache hit가 낮아질 수 있다.

독립 구현은 원본 코드의 안전성을 보장하지만, 실제 end-to-end pipeline에 바로 반영된 것은 아니다. GitHub 제출 단계에서는 `src/before`, `src/after` 또는 commit history로 전후 비교가 명확히 보이도록 구성해야 한다.

### 6.4 한계

첫째, 실제 UBFC-PHYS 데이터는 사용하지 않았다. 따라서 본 결과는 원본 수식과 synthetic 입력에 대한 microbenchmark 결과이며, 전체 학습 시간 개선률은 직접 주장하지 않는다.

둘째, CPU에서 측정했다. GPU 학습에서는 loss vectorization의 절대 시간과 상대 개선률이 달라질 수 있다. 다만 Python loop 제거 자체는 GPU에서도 의미가 있다.

셋째, class/SRP나 generator streaming은 분석 후보로만 다루었다. 원본 pipeline 안전성을 우선했기 때문에 이번 구현에서는 A, D, E 세 항목에 집중했다.

## 7. GitHub 제출 구성 제안

GitHub 업로드 시에는 다음 구조를 권장한다.

```text
physnet-advanced-python-final/
├── README.md
├── requirements.txt
├── src/
│   ├── before/
│   │   ├── PhysNetNegPearsonLoss_before.py
│   │   └── post_process_before.py
│   └── after/
│       └── physnet_safe_optimizations.py
├── benchmark/
│   └── run_benchmark.py
├── results/
│   ├── benchmark_results.csv
│   └── environment.json
└── report/
    ├── report.md
    └── report.pdf
```

현재 서버 작업물에서는 원본을 보존하기 위해 `advanced_python_physnet_safe/` 안에 독립 구현을 두었다. GitHub 제출용 저장소에서는 원본 코드의 핵심 부분을 `src/before/`에 민감 경로 없이 복사하고, 개선 구현을 `src/after/`에 두면 과제 지시사항의 “최적화 전후 비교 가능성”을 더 명확히 충족할 수 있다.

## 8. 결론

본 과제는 PhysNet 연구 코드에서 반복적으로 호출되는 손실 함수와 평가 함수를 분석하고, 수업에서 배운 자료구조/복잡도, decorator caching, 딥러닝 tensor vectorization 개념을 적용했다. 세 개선 모두 동일 synthetic 입력에서 원본 참조 구현과 결과 동등성을 확인했고, 10회 반복 benchmark로 평균, 표준편차, peak memory를 기록했다.

가장 큰 개선은 evaluation 단계의 detrend와 MACC에서 나타났다. 이는 단순한 코드 정리가 아니라 지배 연산과 중복 계산을 직접 줄였기 때문이다. 학습 단계에서는 negative Pearson loss vectorization이 batch size 증가에 따라 큰 효과를 보였다. 실제 데이터와 원본 코드를 건드리지 않는 안전한 방식으로 진행했기 때문에, 다음 단계에서는 이 구조를 GitHub 제출용 `before/after` 폴더와 PDF 보고서로 정리하면 된다.
