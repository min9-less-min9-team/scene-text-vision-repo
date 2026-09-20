title: [Task] PC-A: Qwen3.5-9B QLoRA 1 epoch → 27B와 불일치율·오라클 상한 측정
labels: task,day1

### 목표
27B와 다른 크기의 같은 계열 모델을 학습해 앙상블 후보로 쓸 가치가 있는지 판정한다.

### 실행 자리
PC-A (5070 Ti)

### 입력·의존성
9B 스윕 종료 후. 검증 분할 이슈. `--mode all --model-id Qwen/Qwen3.5-9B --precision 4bit --epochs 1 --batch-size 1 --grad-accum 8`.

### 완료 기준
검증 정확도, 27B 파인튜닝과의 불일치율, 오라클 상한(둘 중 하나라도 맞는 비율)을 댓글로. 판정 기준: 불일치 ≥ 8% 이고 정확도가 27B −1%p 이내면 멤버 채택, 5% 미만이면 폐기.

### 산출물 위치
Drive/scene-text-vision-runs/ft9b_v1/ (adapter/, test_probs.npz, valid_predictions.csv)

### 마감
9/21 22:00

1회차 34위 기준 4B QLoRA 1 epoch이 16GB에서 약 3~4시간. 9B는 그 두 배로 잡는다.

### 체크리스트
- [ ] 같은 검증 분할(시드·id 목록)을 사용했다
- [ ] 확률 npz를 Drive에 저장했다
- [ ] 결과 수치와 판정(채택/조건부/폐기)을 댓글로 남겼다
- [ ] 효과가 없었다면 Insight 이슈에 "닫은 길"로 기록했다
