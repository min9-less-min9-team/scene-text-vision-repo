# SSAFY 16기 2회차 AI 챌린지 (텍스트 이미지 4지선다 VQA) 사전 계획

작성 2026-09-20, 갱신 2026-09-21 (Kaggle 디스커션·영문권 대회 조사 반영). 대회 페이지: https://www.kaggle.com/competitions/ssafy-16-2-ai-9-21-9-28 (9/21 09:00 과제·데이터 공개, 9/28 마감, 참가 등록 994명).
실행 도구는 `src/textmc/`(`vqa_textmc.py`, `inspect_data.py`, `blend_probs.py`, `ocr_extract.py`)와 `notebooks/colab_runbook.ipynb`에 있습니다. 첫날 실행 순서는 `docs/DAY1_PLAN.md`, 논문 조사는 `docs/PAPERS.md`를 보세요.

---

## 0. 한 줄 결론

**"Qwen3.5-27B + 보기 셔플 LoRA(정답 글자 1토큰만 학습) + a/b/c/d next-token 확률 + 보기 순서 TTA"** 가 직전 두 대회 1·2위의 공통 레시피이고, 텍스트 VQA에서는 여기에 **원본 해상도에 맞춘 visual token 설정, 불확실 문항만 crop/고해상도 재추론, 층·상호 등 질문 유형별 프롬프트 힌트, 계열이 다른 모델의 로그확률 앙상블**을 얹는 것이 핵심입니다. 첫 2시간 안에 zero-shot 제출로 파이프라인을 검증하고, 그 다음 LoRA 학습으로 점수를 올립니다.

---

## 1. 참고 범위 원칙

- 실제 대회 자료는 **최종(Private) 10위 이내** 참가자의 글·코드만 근거로 삼았습니다. 그 밖의 글은 순위와 함께 "제외"로만 표기합니다.
- 영문권 대회는 공개된 1~3위 리포트가 있는 것만 정리했습니다.

## 2. SSAFY 대회 상위권 코드·디스커션 분석

### 2-1. 16기 1회차 (재활용 VQA, 개인전 954명) 최종 리더보드 상위 10위

| 순위 | Private | 참가자 | 공개 자료 |
|---:|---:|---|---|
| 1 | 0.95703 | 대전_5반_김기현 | GitHub [kimkihyun1/ssafy-ai-challenge-vqa](https://github.com/kimkihyun1/ssafy-ai-challenge-vqa) (디스커션 글 없음) |
| 2 | 0.94915 | 대전_6반_이선재 (Kaggle 닉 Vongole) | 디스커션 "[VLM 파인튜닝 파이프라인] 공유합니다" + GitHub [Sunjae-L22/vlm-vqa-pipeline](https://github.com/Sunjae-L22/vlm-vqa-pipeline) |
| 3 | 0.94875 | 서울_18반_정회성 | 없음 |
| 4 | 0.94639 | 서울_18반_배소연 | 없음 |
| 5 | 0.94599 | 광주_5반_한기헌 | 없음 |
| 6 | 0.94521 | 서울_19반_김원겸 | 없음 |
| 7 | 0.94521 | 서울_7반_한범구 | 없음 |
| 8 | 0.94442 | 광주_4반_안준영 | 없음 |
| 9 | 0.94284 | 대전_2반_길용진 | 없음 |
| 10 | 0.94245 | 서울_18반_김성현 | 없음 |

디스커션 19개 글 중 10위 이내 작성자의 글은 2위(Vongole) 1건뿐입니다. 나머지는 14위(dlwnsgur0708, count 분석·Tip 2건), 16위(이창엽), 18위(김진영 4건), 29위(유지원, OOF reliability routing), 57위(류효정), 59위(김윤성), 84위, 314위, 563위 글이라 참고 근거에서 제외했습니다.

### 2-2. 1위 (kimkihyun1) 요약

Qwen3.5-27B LoRA(r16/α32, lr 1e-4, 1 epoch, batch 1×accum 8, visual token ≤1024, Colab A100 80GB bf16), 학습 시 보기 무작위 셔플 + 정답 글자 1토큰에만 LM loss, TTA4(라틴 방진 순열) 확률 평균 → Public 0.9507. 불확실 1,000문항에 TTA24·hires 1536 추가, 불확실 500문항에 LoRA OFF(원본 모델) 확률을 조건부(margin<0.15, 원본 확신≥0.60, TTA 일치≥0.75, gap≤0.12)로 50:50 결합 → Public 0.9527, Private 0.9570.

### 2-3. 2위 (이선재 / Vongole) 디스커션·레포 요약 — 이번 대회에 가장 직접적으로 쓸 내용

단계별 Public 상승: 텍스트만 0.600 → Qwen3-VL-8B 제로샷 + 해상도 1024 + 선지 순환 TTA 0.893 → QLoRA 파인튜닝(32분) 0.929 → 이계열 모델(Qwen2.5-VL-7B) 앙상블 0.939 → 질문 유형별 가중치 0.941 → Qwen3-VL-32B(4bit) 투입 0.945~0.948.

효과가 있었던 것:
1. **해상도부터 확인.** 원본 720×960인데 프로세서 기본값이 더 작게 줄임. `max_pixels`를 원본 수준으로 올린 한 줄이 +4.75%p. **원본 이상으로 키우는 것은 무효과**(1024→1280 동일). 이번 텍스트 VQA도 `inspect_data.py`의 해상도 통계부터 보고 토큰 예산을 정할 것.
2. **QLoRA 파인튜닝이 단일 최대 레버**(+3.55%p). 파인튜닝 허용이면 프롬프트·TTA보다 먼저.
3. **선지 순환 TTA** rot4가 rot1 대비 +1.5%p. 학습에도 같은 회전 증강.
4. **앙상블 멤버는 성능이 아니라 "다르게 틀리는가"로 선택.** 챔피언과 불일치 ≥8% + 정확도 −1%p 이내면 멤버 가치. 같은 계열 재학습·시드 변경(불일치 4%)은 기여 0. Qwen2.5-VL-7B(더 약함)가 +26문항, Qwen3-VL-32B가 +17문항. 결합은 로그확률 가중평균이면 충분(투표·온도 스케일링 전부 열세).
5. **질문 유형별 가중치 분리**(개수형/비개수형)로 +5문항. 유형 집합이 겹치지 않아 독립 최적화 가능.
6. **A100 40GB에서 Qwen3-VL-32B dense는 4bit(19.5GB)로 QLoRA까지 가능.** MoE(30B-A3B)는 bitsandbytes가 `nn.Linear`만 양자화해 불가. GPU 메모리 판정은 총량이 아니라 로드 전후 델타로.
7. **노이즈 문턱 2σ ≈ √D**: 두 제출이 D문항 다르면 √D문항 이내 점수 차는 측정 불가. Public 1문항 = 0.000394(2,537문항). 문턱 미만 차이의 실험에 제출을 쓰지 말 것(`blend_probs.py`가 이 값을 출력).
8. **최종 2개 선택 = Public 최고 + 헤지**(30문항 이상 다르면서 다른 후보들과 평균적으로 가장 가까운 것).
9. **홀드아웃 용도 구분**: 단일 모델 우열은 홀드아웃(500장 이상)으로, 앙상블 구조·가중치는 LB로 판단(87~674장 홀드아웃 결론을 LB가 4번 뒤집음).

닫은 길(다시 밟지 말 것): CoT 생성 추론 −2.33%p(세는 문장은 완벽한데 지각이 틀림), 원본 이상 업스케일, 좌우/상하 뒤집기·회전 TTA, 파인튜닝 후 프롬프트 변형 TTA(민감도 소멸, −1.67%p), 에폭 2(과적합), 비전타워 LoRA, 하드 라우팅·투표·온도 스케일링, 정답 위치 사전확률.

환경 함정: `pip uninstall -y torchao`(peft 0.16+ 충돌), pandas/numpy/pillow 업그레이드 금지(`google.colab` 파손), 4bit 모델은 `merge_and_unload()` 금지, 산출물은 반드시 Drive에(VM 회수로 어댑터 2개 유실), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

2위 레포 `docs/LEARNINGS.md`는 "2차는 5인 9일"을 전제로 팀 운영법(계열별 분담, 검증 분할·npy 규격·닫은 길 목록 공유, 제출 예산 관리자 1명)을 적어 두었습니다. 이번 2회차가 팀전이면 그대로 적용합니다.

### 2-4. 15기 2회차 1위 (MinjuJangg/ai_chall, 팀전 193팀, Private 0.9764)

Qwen3.5 choice-CE LoRA(4개 보기 토큰 CE), 보기 셔플·TTA, confidence/margin 기반 보수적 브랜치 병합, 취약 qtype(count)만 Grounding DINO/SAM crop + InternVL 재추론, qtype별 가중 앙상블. (같은 대회의 다른 레포 `dong99u/ssafy_ai_challenge`는 Public 0.918로 10위권이 아니라 제외.)

### 2-5. 16기 1회차 규정 (2회차에도 이어질 가능성이 높음, 9/21 09:00 공개분으로 반드시 재확인)

개인전 1인 1팀 / 외부 데이터 사용 가능(공개 데이터여야 함) / 데이터 증강 허용 / HF 사전학습 모델 사용 가능, LoRA·양자화 허용 / **API 호출 추론 금지** / **하루 최대 제출 20회** / 테스트 데이터 사전 접근·유출 금지. 평가 항목에 리더보드 외 디스커션 점수·발표 점수가 있었음(인사이트 글 작성이 가점).

---

## 3. 영문권 유사 대회 상위권 접근 (공개 리포트 기준)

| 대회 | 형식 | 상위권 접근 | 우리 과제에의 시사점 |
|---|---|---|---|
| **LAVA Challenge 2024 (ACCV 워크숍)** 다이어그램·간트차트·도면 4지선다 VQA, MMMU 지표 | 1위 WAS 0.85, 2위 MMLAB-UIT 0.84, 3위 V1olet 0.82 | 1위: MMMU에서 후보 3개 중 최고인 **Qwen2-VL** 선택 + 프롬프트 엔지니어링 + **투표 기반 앙상블**. 3위: 사전학습 VLM 여러 개의 앙상블 | 파인튜닝 없이도 강한 VLM + 앙상블로 승부. 우리는 파인튜닝 허용이라 그 위에 얹는 형태 |
| **Kaggle Data Science Osaka Winter 2024** arXiv PDF 기반 4지선다 문서 QA, Accuracy | 1위 Private 0.895 (Qwen2.5-32B-AWQ + 번역 + RAG) | 선택지 순서 **4가지 순열로 logprob를 뽑아 평균**하는 TTA를 문서 QA에도 그대로 사용. Mistral-Small-24B와 앙상블 시 0.90 | 보기 순서 TTA는 언어·도메인을 가리지 않는 표준 기법. 계열이 다른 두 번째 모델이 +0.5%p |
| **Kaggle LLM Science Exam 2023** 5지선다 과학 문제, MAP@3 | 1위 H2O (Private 0.933) | RAG + LoRA로 학습한 7B/13B 6개 앙상블. **보기를 한 개씩 넣는 이진 채점**으로 위치 편향 제거. 다지선다 + 120개 순열 TTA도 시도했으나 이진 방식보다 불안정. 다른 보기들의 평균 로짓을 헤드 입력에 추가하는 트릭 | 위치 편향은 순열 TTA로도 완전히 안 없어짐. 보기 텍스트가 매우 비슷할 때는 "각 보기가 정답인가" 이진 채점 브랜치도 앙상블 후보 |
| **TextVQA Challenge 2021** 장면 텍스트 자유답변 | 1위 Team Mia 56.16% | T5-3B 기반, OCR 토큰 + 객체 특징을 하나의 시퀀스로, 900만 장면 텍스트 이미지로 사전학습(MLM + 상대위치 예측), 적대적 학습, 앙상블, **fuzzy 문자열 매칭 후처리** | 지금은 VLM이 OCR 파이프라인을 대체. 남는 교훈은 "OCR 결과와 보기의 근사 문자열 매칭" 후처리 개념 |
| **TextVQA Challenge 2020** | 1위 SMA 45.5% | 객체·텍스트 관계 그래프 어텐션 + M4C 방식 포인터 디코딩, 고성능 OCR(SBD-Trans) + ST-VQA 사전학습 | OCR 품질이 곧 성능이던 시절. VLM 시대에는 해상도가 그 역할 |
| **ICDAR 2019 ST-VQA** | 1위 VTA | BERT 질문 인코딩 + BUTD VQA | 참고 가치 낮음 |
| **VizWiz VQA 2024 (CVPR)** 시각장애인 촬영 사진 VQA(텍스트 읽기 비중 높음) | 1위 SLCV(Shopline), 2위 V1olet, 3위 KNU-HomerunBall | 공개 리포트 없음(수상만 확인) | V1olet이 LAVA 3위와 동일 팀: 사전학습 VLM 앙상블로 두 대회 입상 |
| **Kaggle Benetech 2023** 차트 이미지 → 데이터 추출 | 상위권 DePlot/MatCha(pix2struct) | 합성 데이터 대량 생성, 2단계(도메인 적응→특화) 학습, 산점도는 별도 검출기 | 텍스트/숫자 읽기 과제에서 **합성 데이터 + 단계적 파인튜닝**이 유효. 규정상 외부 공개 데이터가 허용되면 간판 합성 데이터도 선택지 |

공통 결론: (1) 4지선다는 생성 대신 보기 토큰 확률 채점, (2) 보기 순서 순열 TTA, (3) 계열이 다른 모델의 확률 평균, (4) 위치 편향 대응. SSAFY 1·2위와 영문권 상위권이 같은 결론에 도달해 있으므로 이 골격을 흔들 이유가 없습니다.

---

## 4. 텍스트 VQA에서 추가로 신경 쓸 점

| 이슈 | 근거 | 대응 (스크립트 옵션) |
|---|---|---|
| 해상도 | 2위: 원본 해상도까지만 효과. 1위: 불확실 문항에 1536 토큰 재추론. Qwen3.5는 32px/token | `inspect_data.py`로 원본 크기 확인 → `--max-visual-tokens`를 원본 픽셀 수 근처로. 그 이상은 불확실 문항 재추론에만 |
| 대상 간판 vs 주변 간판 혼동 | AICOSS 상권 VQA 캠프 주요 오답 | 시스템 프롬프트에 "조건에 맞지 않는 다른 간판 무시"(구현됨), 불확실 문항 `--crop-mode five`/`topbottom` |
| 층(floor)·위치 질문 | VLM이 층 세기에 약함 | qtype `floor` 힌트(구현됨). 취약하면 상단/하단 crop 뷰 병행 |
| 오답 보기가 한 글자만 다른 근사 문자열 | 텍스트 VQA 출제 관행 | 철자 대조 지시(구현됨), choice-CE 학습. LLM Science Exam식 이진 채점 브랜치는 앙상블 후보 |
| OCR 힌트 | ViSignVQA: OCR 첨부가 소형 모델에 큰 효과, 강한 VLM에는 작고 오류 전파 위험 | `ocr_extract.py` → `--ocr-csv`. 검증 A/B 후 보조 브랜치로만 |
| 숫자(전화·가격·시간) | 한 자리 오독 = 오답 | qtype 힌트, 고해상도 재추론 우선 |
| 위치 편향 | 1·2위, Osaka, LLM Science Exam 모두 언급 | 셔플 학습 + TTA(구현됨) |
| CoT | 2위 −2.33%p | 쓰지 않음 |

---

## 5. 모델 선택과 GPU별 실행 플랜

로컬 PC는 Intel Arc(CUDA 없음)이므로 **모든 학습·추론은 Colab(또는 Kaggle)**. Colab Pro 100 CU/월, A100 약 15 CU/h, T4 약 1.8 CU/h. A100 80GB는 "High-RAM" 옵션.

| 배정 GPU | 주력 모델 | 정밀도 | 권장 옵션 |
|---|---|---|---|
| A100 80GB | `Qwen/Qwen3.5-27B` | bf16 (OOM 시 자동 8bit) | `--max-visual-tokens 1024~1280 --batch-size 1 --grad-accum 8` |
| A100 40GB | `Qwen/Qwen3.5-27B` 4bit **또는** `Qwen3-VL-32B-Instruct` 4bit(2위 검증) 또는 `Qwen3.5-9B` bf16 | 4bit / bf16 | 32B dense 4bit ≈ 19.5GB, BS1/ACC16 |
| L4 24GB | `Qwen3.5-9B` 4bit 또는 `Qwen3.5-4B` bf16 | 4bit | `--max-visual-tokens 1024` |
| T4 16GB | `Qwen3.5-4B` | 4bit (fp16) | `--max-visual-tokens 768~1024` |

앙상블 멤버 후보(계열이 달라야 함): Qwen3-VL-32B/8B, Qwen2.5-VL-7B(2위에서 +26문항), InternVL3.5-8B, Gemma 계열. 같은 모델 시드 변경은 금지.

Qwen3.5-27B(OCRBench 89.4, 한국어 포함 32개 언어 OCR)가 1순위, Qwen3.6-27B는 동급, Qwen3.8-27B(2026-08, OmniDocBench 91.1)는 시간이 남을 때 zero-shot 비교만.

주의: Qwen3.5 계열은 `pip install git+https://github.com/huggingface/transformers` + `peft>=0.20` 필요. 설치 후 런타임 재시작.

---

## 6. D-Day 실행 순서 (9/21 09:00 공개 → 9/28 마감)

### 0~30분: 규정·데이터 확인 (GPU 켜기 전에)
- [ ] 규정: 개인전/팀전(팀이면 인원), 외부 데이터·API·사전학습 가중치, **1일 제출 횟수**, Public/Private 비율, 마감 시각, 디스커션·발표 점수 여부.
- [ ] `inspect_data.py` → 컬럼, 정답 분포, qtype 분포, **원본 해상도**, train/test 이미지 중복, 이미지당 질문 수.
- [ ] 이미지 20장 눈으로 확인: 글자 크기, 간판 수, 층 표기, 보기 유사도.
- [ ] 텍스트만(이미지 없이) 하한선을 잡아 두면 "모델이 이미지를 쓰는지" 판별 가능(1회차 하한 0.57~0.60).

### 30분~2시간: 파이프라인 검증 + 첫 제출
```
!python vqa_textmc.py --mode dryrun --root $ROOT
!python vqa_textmc.py --mode smoke  --root $ROOT --model-id Qwen/Qwen3.5-27B --out-dir $RUNS/smoke
!python vqa_textmc.py --mode zeroshot --root $ROOT --model-id Qwen/Qwen3.5-27B --tta-perms 2 --out-dir $RUNS/zs27b
```
- zero-shot 검증 400개로 해상도 1024 vs 1536 비교, test 제출로 ID 순서·형식 확정.

### 2~6시간: LoRA 학습 1회차 (검증 분할 유지)
```
!python vqa_textmc.py --mode all --root $ROOT --model-id Qwen/Qwen3.5-27B \
    --epochs 1 --lr 1e-4 --lora-r 16 --lora-alpha 32 --max-visual-tokens 1280 \
    --valid-size 0.1 --tta-perms 4 --out-dir $RUNS/ft27b_v1
```
- 산출물: `valid_predictions.csv`, `test_predictions_detailed.csv`, `submission_test.csv`, `test_probs.npz`(앙상블용, 반드시 보관).

### 6시간~: 오답 분석 → 선택적 재추론 → 결합
1. qtype별 정확도, `unc_score` 상위 25% 정확도 확인.
2. 불확실 id 목록 → 고해상도 / 멀티크롭 / LoRA OFF / OCR 힌트 브랜치 → `blend_probs.py`로 조건부 결합(불일치율·노이즈 문턱 출력 확인).
3. 검증에서 오른 조합만 제출. 하루 20회 중 **5회는 마지막 날 미세조정용으로 남김**.

### 팀전일 경우 첫날 추가 할 일 (2위 플레이북)
- 사람마다 **다른 계열** 모델을 맡는다(예: A Qwen3.5-27B 주력, B Qwen3-VL-32B 4bit, C Qwen2.5-VL-7B·InternVL3.5, D OCR 힌트·crop 브랜치, E 검증·앙상블·제출 관리).
- 공유 3종: 검증 분할(시드·id 목록), 확률 `.npz` 규격(행 = test.csv 순서), 닫은 길 목록 파일.
- 제출 예산 관리자 1명. "챔피언과 몇 문항 다른가 ≥ √D"를 통과한 후보만 제출.

### 이후 날
- Day 2~3: 해상도·epoch·lr·loss 비교(검증 1%p 이상만 채택), 두 번째 계열 모델 학습, 유형별 가중치 사다리(0.15/0.30/0.50, 가운데부터).
- 마지막 날: 전체 train 재학습(`--valid-size 0`) → 검증된 결합 규칙 적용 → 최종 2개 = Public 최고 + 헤지.

---

## 7. 제출 전 체크리스트
- [ ] 행 수 == sample_submission, id 집합 일치, 정답 값 형식(a~d / A~D)
- [ ] 정답 분포가 한 글자로 쏠리지 않는지
- [ ] 챔피언 제출과 다른 문항 수 D, 그리고 √D 문턱 확인
- [ ] adapter·probs npz·detailed csv를 Drive에 보관

---

## 8. 리스크와 대응
| 리스크 | 대응 |
|---|---|
| transformers 버전으로 Qwen3.5 로드 실패 | git main 설치 + 재시작. `AutoModelForMultimodalLM` → `AutoModelForImageTextToText` 자동 시도 |
| A100 미배정 | 5절 표대로 축소. 9B bf16 / 4B 4bit 경로 준비됨 |
| OOM | `--precision auto` bf16→8bit 폴백, `--max-visual-tokens 768` |
| Colab 끊김 | adapter 저장 후 `--mode infer --adapter ...` 재개, TTA perm 캐시 재사용 |
| a/b/c/d 비단일 토큰 | 시작 시 assert, 공백 변형 재시도 |
| 형식이 다름(보기 5개, 자유답변) | `CHOICES`/`PERM_BANK` 수정. 자유답변이면 generate + 정규화 매칭으로 전환(별도 작업) |
| 작은 검증셋 착시 | 단일 모델 비교는 500장 이상, 앙상블 구조는 LB로 |

---

## 9. 준비된 파일
| 파일 | 역할 |
|---|---|
| `vqa_textmc.py` | dryrun / smoke / zeroshot / train / infer / all. 컬럼 자동 감지, 그룹 검증 분할, Qwen3.5 LoRA(타깃 자동 탐지, 비전 타워 제외), 보기 셔플, choice-CE·answer-LM loss, next-token 확률, TTA(최대 24), 멀티크롭, LoRA OFF, 불확실도 |
| `inspect_data.py` | 첫 30분 데이터 점검(원본 해상도 포함, 로컬 실행 가능) |
| `blend_probs.py` | 확률 결합(weighted/gated/replace) + 불일치율·노이즈 문턱 출력 + 제출 생성 |
| `ocr_extract.py` | EasyOCR·PaddleOCR id→OCR 텍스트 CSV |
| `colab_runbook.ipynb` | 위 순서를 셀로 배치한 Colab 노트북 |

---

## 10. 링크
- 16기 2회차 Kaggle: https://www.kaggle.com/competitions/ssafy-16-2-ai-9-21-9-28
- 16기 1회차 Kaggle(리더보드·디스커션): https://www.kaggle.com/competitions/ssafy-16-1-ai
- 1위 코드: https://github.com/kimkihyun1/ssafy-ai-challenge-vqa
- 2위 코드·플레이북: https://github.com/Sunjae-L22/vlm-vqa-pipeline (`docs/LEARNINGS.md`)
- 15기 2회차 1위: https://github.com/MinjuJangg/ai_chall
- LAVA 2024: https://lava-workshop.github.io/archive/2024
- Osaka Winter 2024 1위: https://www.kaggle.com/competitions/data-science-osaka-winter-2024/writeups/l-1st-place-solution
- LLM Science Exam 1위: https://www.kaggle.com/competitions/kaggle-llm-science-exam/writeups/team-h2o-llm-studio-1st-place-solution
- TextVQA 2021 1위 논문: https://arxiv.org/abs/2106.15332
- Qwen3.5-27B: https://huggingface.co/Qwen/Qwen3.5-27B
