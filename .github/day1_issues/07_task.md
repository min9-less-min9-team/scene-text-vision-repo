title: [Task] Colab: Qwen3.5-27B LoRA 1 epoch + TTA4 → 두 번째 제출
labels: task,day1

### 목표
직전 대회 1위 레시피로 27B를 1 epoch 학습하고 검증 정확도·불확실도를 남긴다.

### 실행 자리
Colab (A100)

### 입력·의존성
제로샷 이슈 완료, 9B 스윕의 해상도·프롬프트 결론. `--mode all --epochs 1 --lr 1e-4 --lora-r 16 --lora-alpha 32 --valid-size 0.1 --tta-perms 4`.

### 완료 기준
`valid_predictions.csv`(gold 포함)와 qtype별 정확도, 불확실도 상위 25% 정확도를 댓글로. `submission_test.csv` 제출 후 Public 점수와 제로샷 제출 대비 다른 문항 수 D 기록. adapter와 `test_probs.npz`를 Drive에 저장.

### 산출물 위치
Drive/scene-text-vision-runs/ft27b_v1/ (adapter/, test_probs.npz, valid_predictions.csv, submission_test.csv)

### 마감
9/21 18:00

OOM이면 `--precision auto`가 8bit로 내려간다. 그래도 OOM이면 `--max-visual-tokens 768`. 학습 산출물은 반드시 Drive.

### 체크리스트
- [ ] 같은 검증 분할(시드·id 목록)을 사용했다
- [ ] 확률 npz를 Drive에 저장했다
- [ ] 결과 수치와 판정(채택/조건부/폐기)을 댓글로 남겼다
- [ ] 효과가 없었다면 Insight 이슈에 "닫은 길"로 기록했다
