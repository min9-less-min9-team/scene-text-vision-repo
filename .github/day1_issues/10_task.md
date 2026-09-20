title: [Task] PC-C: crop 스윕 (상단/하단/5분할 + grounding bbox 2-pass) 층·위치 유형 집계
labels: task,day1

### 목표
인접 간판 혼동과 층·위치 질문에 crop이 도움이 되는지, 어떤 crop이 좋은지 9B로 판정한다.

### 실행 자리
PC-C (5070 Ti)

### 입력·의존성
검증 분할 이슈, 9B 스윕의 해상도 결론. `--crop-mode topbottom|five`. 추가로 모델에게 대상 간판 bbox를 묻고 crop해 재채점하는 2-pass(TextCoT식, `docs/PAPERS.md` 1절)는 간단한 스크립트로.

### 완료 기준
검증 400장 + 층·위치 유형 100장에서 원본 vs 각 crop 방식의 정확도 표를 댓글로. 유형별로 +2%p 이상인 방식만 채택 후보. 불확실 문항(margin < 0.15)에만 적용했을 때의 이득도 별도 계산.

### 산출물 위치
Drive/scene-text-vision-runs/crop9b/ (방식별 probs.npz)

### 마감
9/21 22:00

### 체크리스트
- [ ] 같은 검증 분할(시드·id 목록)을 사용했다
- [ ] 확률 npz를 Drive에 저장했다
- [ ] 결과 수치와 판정(채택/조건부/폐기)을 댓글로 남겼다
- [ ] 효과가 없었다면 Insight 이슈에 "닫은 길"로 기록했다
