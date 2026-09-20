title: [Task] PC-B: 이계열 모델 게이트 (A.X 4.0 VL Light, InternVL3.5-8B) 불일치율 ≥ 8% 판정
labels: task,day1

### 목표
Qwen 계열과 다르게 틀리는 모델을 찾는다. 한국어 특화 VLM(SKT A.X 4.0 VL Light, Apache 2.0)을 1순위로 본다(`docs/PAPERS.md` 6절).

### 실행 자리
PC-B (5070 Ti)

### 입력·의존성
검증 분할 이슈. 후보: `skt/A.X-4.0-VL-Light`(8B), `OpenGVLab/InternVL3_5-8B`. `vqa_textmc.py`가 해당 모델을 못 읽으면 프로세서 인자만 바꿔 제로샷만 먼저.

### 완료 기준
각 후보의 검증 400장 제로샷 정확도, 27B 제로샷과의 불일치율, 오라클 상한, a/b/c/d 단일 토큰 여부, 장당 추론 시간을 댓글로. 통과(불일치 ≥ 8%, 정확도 −3%p 이내) 모델은 QLoRA 1 epoch까지 진행.

### 산출물 위치
Drive/scene-text-vision-runs/gate_<model>/

### 마감
9/21 22:00

VARCO-VISION-2.0-14B는 CC BY-NC-ND라 파인튜닝 불가, 제로샷 멤버로만 검토. 규정 이슈에서 허용 여부 확인 후.

### 체크리스트
- [ ] 같은 검증 분할(시드·id 목록)을 사용했다
- [ ] 확률 npz를 Drive에 저장했다
- [ ] 결과 수치와 판정(채택/조건부/폐기)을 댓글로 남겼다
- [ ] 효과가 없었다면 Insight 이슈에 "닫은 길"로 기록했다
