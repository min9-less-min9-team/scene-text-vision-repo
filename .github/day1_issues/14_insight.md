title: [Insight] 논문 조사 결론: 아직 안 써본 신규 접근 4가지와 폐기 목록
labels: insight,day1

### 한 줄 요약
2024~2026 논문은 기존 골격을 정당화한다. 신규로 검증할 가치가 있는 것은 한국어 특화 VLM 멤버, PriDe 편향 보정, grounding 기반 crop 2-pass, 무이미지 대조 채점 네 가지다.

### 종류
논문·자료 조사

### 출처
`docs/PAPERS.md` (ViCrop ICLR25, TextCoT, HiDe, PriDe ICLR24, 2026 순서 민감도 감사, 2026 VLM TTS, ImageCLEF 2026 MCQ 1위, VCD, Visual-RFT, OCR-augmented bilingual VQA 2025-10, KRETA EMNLP25, VARCO-VISION-2.0, A.X 4.0 VL Light)

### 핵심 내용
유효: 로짓 채점(2026 greedy 논문), 순열 TTA(2026 감사: 최신 MLLM도 순서 뒤집힘 24~50%), 파인튜닝.
신규: (1) A.X 4.0 VL Light 8B Apache 2.0, OutdoorKorean 97.3. (2) PriDe: 5% 샘플 ×4로 사전확률 추정, 비용 1.15배. (3) 모델에게 bbox 묻고 crop 재채점(TextCoT), 고해상도 모델에선 이득 축소(LLaVA-NeXT +3.5). (4) 이미지 없는 로짓을 빼는 VCD식 보정(텍스트만 0.6인 데이터라 시도 가치).
폐기: OCR 텍스트 첨부(한국어 장면 VQA 66→65), RL 파인튜닝(SFT 대비 이득 근거 약함), 전체 문항 CoT·self-consistency(지각 과제에서 기준선 이하), PRM 검증기(8.7배 비용에 −0.39pp), zoom RL 모델, 합성 데이터, 어댑터 병합.

### 우리 과제에의 판정
조건부 (검증 후 결정)

### 다음 액션
PC-B(한국어 VLM), PC-C(crop 2-pass), PC-D(PriDe·대조 채점) Task에서 검증.
