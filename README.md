# Python, C 교차검증 15비트로 진행
Python 87%
C      83.90% (839/1000)

# 시간 측정 (비트: 15, 정확도: 87.30% (873/1000))
데이터 로드: 0.218670 seconds
스파이크 인코딩: 0.785961 seconds
포아송 인코딩: 5.350856 seconds 
막전위: 147.214734 seconds
판별: 0.001788 seconds

# 정확도 측정
frequency float형태를 중요도가 높은 정수형태로 변경하고 이미지 100개에 대해서 비트수 조절을 진행했습니다.
float형태 92% (92/100)
15비트    92% (92/100) 
6비트     92% (92/100) 주파수의 최대 범위가 1~50이기 떄문에 6비트까지는 차이가 없습니다.
5비트     92% (92/100)
4비트     86% (86/100)
2비트     86% (86/100)
1비트     22% (22/100)

# 포아송 인코딩
ChatGPT
일단 다 가능하나 각자의 강점과 단점이 있다.
1. 픽셀마다 랜덤 배열 생성: 
- 학습 및 추론의 정확도를 최우선으로 고려할 때.
- 독립적인 픽셀 정보를 보존하고 싶을 때.

2. 하나의 이미지에 대해 동일한 랜덤 배열 사용:
- 계산 효율성을 높이고 싶을 때.
- 픽셀 간 비교가 중요하지 않을 때.

3. 모든 이미지에 대해 동일한 랜덤 배열 사용:
- 테스트 반복성(reproducibility)이 필요하거나, 빠른 프로토타이핑을 위해 성능보다 단순화가 중요한 경우.

Claude
여러 방법이 가능하지만, 일반적으로 포아송 인코딩에서는 각 픽셀마다 독립적인 랜덤 배열을 생성하고,
이 패턴을 모든 이미지에 동일하게 적용합니다.
포아송 인코딩: 5.350856 seconds -> 포아송 인코딩: 1.216050 seconds

스파이크 인코딩: 0.785961 seconds

1. 24*24 크롭
2. 크롭후 시간 측정
3. 포아송 인코딩 다시 구현

Epoch: 0 Testing Accuracy: 0.21 Time passed: 574.11 seconds
Epoch: 1 Testing Accuracy: 0.88 Time passed: 1090.28 seconds
Epoch: 2 Testing Accuracy: 0.88 Time passed: 1596.56 seconds
Epoch: 3 Testing Accuracy: 0.86 Time passed: 2106.85 seconds
Epoch: 4 Testing Accuracy: 0.76 Time passed: 2604.51 seconds
Epoch: 5 Testing Accuracy: 0.69 Time passed: 3095.48 seconds
Epoch: 6 Testing Accuracy: 0.71 Time passed: 3589.89 seconds
Epoch: 7 Testing Accuracy: 0.79 Time passed: 4089.98 seconds
Epoch: 8 Testing Accuracy: 0.76 Time passed: 4587.85 seconds
Epoch: 9 Testing Accuracy: 0.79 Time passed: 5091.8 seconds
Epoch: 10 Testing Accuracy: 0.79 Time passed: 5594.75 seconds
Epoch: 11 Testing Accuracy: 0.76 Time passed: 6091.78 seconds
Epoch: 12 Testing Accuracy: 0.79 Time passed: 6592.98 seconds
Epoch: 13 Testing Accuracy: 0.76 Time passed: 7085.56 seconds
Epoch: 14 Testing Accuracy: 0.76 Time passed: 7584.04 seconds
Epoch: 15 Testing Accuracy: 0.74 Time passed: 8107.96 seconds
Epoch: 16 Testing Accuracy: 0.76 Time passed: 8611.24 seconds
Epoch: 17 Testing Accuracy: 0.76 Time passed: 9109.21 seconds
Epoch: 18 Testing Accuracy: 0.74 Time passed: 9605.46 seconds
Epoch: 19 Testing Accuracy: 0.79 Time passed: 10101.03 seconds
Epoch: 20 Testing Accuracy: 0.81 Time passed: 10593.54 seconds
Epoch: 21 Testing Accuracy: 0.79 Time passed: 11090.08 seconds
Epoch: 22 Testing Accuracy: 0.79 Time passed: 11581.12 seconds
Epoch: 23 Testing Accuracy: 0.79 Time passed: 12079.19 seconds
Epoch: 24 Testing Accuracy: 0.79 Time passed: 12569.84 seconds
Finished Training. Saved Weights and Labels.