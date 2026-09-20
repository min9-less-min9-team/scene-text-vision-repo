title: [Task] PC-A: Qwen3.5-9B 제로샷 해상도·프롬프트 스윕 → 27B 학습 설정 확정
labels: task,day1

### 목표
27B를 여러 번 학습할 CU가 없으므로, 해상도(visual token)와 프롬프트 스타일은 9B로 먼저 고른다.

### 실행 자리
PC-A (5070 Ti)

### 입력·의존성
검증 분할 이슈. `scripts/vqa_textmc.py --mode zeroshot --model-id Qwen/Qwen3.5-9B --precision 4bit --max-test-samples 400`.

### 완료 기준
검증 400장에서 `--max-visual-tokens` {원본 근처, 1.5배, 2배} × `--prompt-style` {text, generic} 6조합의 정확도 표(전체·유형별)를 댓글로. 층·상호·숫자 유형에서 해상도 이득이 있는지 명시. 27B 학습에 쓸 설정 1개를 결론으로.

### 산출물 위치
Drive/scene-text-vision-runs/sweep9b/ (조합별 zeroshot_valid_probs.npz)

### 마감
9/21 12:00

1회차 2위: 원본 해상도 이상은 무효였으나 이는 개수 세기 결론. 작은 글자에서는 다를 수 있어 재검증한다(`docs/DAY1_PLAN.md` 5절).

### 체크리스트
- [ ] 같은 검증 분할(시드·id 목록)을 사용했다
- [ ] 확률 npz를 Drive에 저장했다
- [ ] 결과 수치와 판정(채택/조건부/폐기)을 댓글로 남겼다
- [ ] 효과가 없었다면 Insight 이슈에 "닫은 길"로 기록했다
