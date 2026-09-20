title: [Task] Colab: Qwen3.5-27B smoke → 제로샷 검증 400 + test TTA2 → 첫 제출(형식 확정)
labels: task,day1

### 목표
파이프라인이 실제 데이터에서 끝까지 도는지 확인하고, 첫 제출로 ID 순서·대소문자·파일 형식을 확정한다. 27B 제로샷 정확도로 이후 우선순위를 정한다(`docs/DAY1_PLAN.md` 4절).

### 실행 자리
Colab (A100)

### 입력·의존성
규정 확인 이슈와 데이터 업로드. `notebooks/colab_runbook.ipynb` 0~3절. 컬럼명이 다르면 `--id-col/--path-col/--question-col/--choice-cols/--answer-col`.

### 완료 기준
`--mode smoke` 통과 → `--mode zeroshot`으로 검증 400장 정확도와 qtype별 정확도 댓글, `submission_zeroshot.csv` 제출 후 Public 점수 댓글. 해상도 1024 vs 1536 토큰 비교 결과 1줄.

### 산출물 위치
Drive/scene-text-vision-runs/zs27b/ (zeroshot_probs.npz, submission_zeroshot.csv)

### 마감
9/21 11:30

CU 절약: 제로샷 test는 TTA2까지만. A100 배정이 안 되면 `Qwen/Qwen3.5-9B`로 대체하고 댓글에 명시.

### 체크리스트
- [ ] 같은 검증 분할(시드·id 목록)을 사용했다
- [ ] 확률 npz를 Drive에 저장했다
- [ ] 결과 수치와 판정(채택/조건부/폐기)을 댓글로 남겼다
- [ ] 효과가 없었다면 Insight 이슈에 "닫은 길"로 기록했다
