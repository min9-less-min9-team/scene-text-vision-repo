title: [Task] 09:00 전: PC 4대 환경 통일 + Qwen3.5-4B/9B 사전 다운로드
labels: task,day1

### 목표
대회 시작 전에 5070 Ti PC 4대의 학습·추론 환경을 같은 버전으로 맞추고 모델 가중치를 미리 받아 둔다.

### 실행 자리
PC-A, PC-B, PC-C, PC-D (각자 자기 PC)

### 입력·의존성
`requirements-pc.txt`, `nvidia-smi`로 확인한 CUDA 버전(5070 Ti는 Blackwell이라 cu128 이상 PyTorch 필수). Windows는 `PYTHONUTF8=1`.

### 완료 기준
각 PC에서 `python -c "import torch, transformers, peft, bitsandbytes"` 성공, `python scripts/vqa_textmc.py --help` 출력 확인, `Qwen/Qwen3.5-4B`와 `Qwen/Qwen3.5-9B` 다운로드 완료. PC별 torch/transformers/peft/bitsandbytes 버전을 이 이슈 댓글에 기록.

### 산출물 위치
각 PC 로컬. 버전 표는 이 이슈 댓글.

### 마감
9/21 09:00

주의: flash-attn 대신 sdpa. Colab 담당은 런타임에서 A100(High-RAM) 배정 여부와 남은 CU를 댓글로.

### 체크리스트
- [ ] 같은 검증 분할(시드·id 목록)을 사용했다
- [ ] 확률 npz를 Drive에 저장했다
- [ ] 결과 수치와 판정(채택/조건부/폐기)을 댓글로 남겼다
- [ ] 효과가 없었다면 Insight 이슈에 "닫은 길"로 기록했다
