title: [Task] 09:00: 규정·데이터 설명 읽고 요약 댓글 (팀전 여부, 제출 횟수, 외부 데이터, API)
labels: task,day1

### 목표
공개되는 Overview·Data·Rules를 전원이 읽고, 전략에 영향을 주는 항목만 한 곳에 요약한다.

### 실행 자리
GPU 불필요

### 입력·의존성
https://www.kaggle.com/competitions/ssafy-16-2-ai-9-21-9-28 의 Overview, Data, Rules 탭. 1회차 규정은 `docs/PLAN.md` 2-5절.

### 완료 기준
댓글에 다음 항목 확정: 개인전/팀전과 인원, 1일 제출 횟수, Public/Private 비율, 마감 시각, 외부 데이터·사전학습 가중치·API 허용 여부, 디스커션·발표 점수 유무, 데이터 컬럼 구성(`id, path, question, a~d, answer`인지), dev.csv 유무와 응답 형식, 베이스라인 노트북 유무. 1회차와 달라진 점은 굵게.

### 산출물 위치
이 이슈 댓글. 필요하면 `docs/PLAN.md` 2-5절 갱신.

### 마감
9/21 09:30

### 체크리스트
- [ ] 같은 검증 분할(시드·id 목록)을 사용했다
- [ ] 확률 npz를 Drive에 저장했다
- [ ] 결과 수치와 판정(채택/조건부/폐기)을 댓글로 남겼다
- [ ] 효과가 없었다면 Insight 이슈에 "닫은 길"로 기록했다
