# 상남자특) 앙상블 안함: 단일모델 public 0.97229 솔루션

앙상블 없이 **단일 모델**로 public 0.97229 달성한 방법 공유합니다.

## 1. 모델
- [unsloth/gemma-4-31B-it-unsloth-bnb-4bit](https://huggingface.co/unsloth/gemma-4-31B-it-unsloth-bnb-4bit)
- Google Gemma 4 31B Instruct를 Unsloth가 4bit(bitsandbytes)로 양자화한 버전
- LoRA: r=16, alpha=32, attention + MLP layer
- 정답 글자(a/b/c/d) 1토큰에만 loss를 거는 방식, 각 stage 1 epoch

## 2. 프롬프트: 2회 반복
- 논문: *[Prompt Repetition Improves Non-Reasoning LLMs](https://arxiv.org/pdf/2512.14982)*
- 같은 프롬프트를 두 번 이어 붙여 입력하는 것 만으로도 reasoning 없이 바로 답하는 모델에서, 뒤쪽 토큰이 앞쪽 내용을 한 번 더 참조할 수 있어 성능이 오른다는 내용

![Prompt Repetition](../assets/prompt_repetition.png)

```python
# 프롬프트 (2회 반복)
SYSTEM_PROMPT = (
    "당신은 이미지를 보고 질문에 답하는 시각 질의응답 도우미입니다. "
    "a, b, c, d 중 정확히 한 글자로만 답하고, 설명은 하지 마세요."
)
USER_INSTRUCTION = "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
PROMPT_REPEAT = 2
PROMPT_REPEAT_SEP = "\n\n"
```

## 3. 2-Stage 학습 (외부 → 내부)
학습을 한 번만 하란 법이 있을까요?
→ [관련 Discussion](https://www.kaggle.com/competitions/ssafy-16-2-ai-9-21-9-28/discussion/742691)

- Stage 1에서 한국어 이미지 속 글자 읽기 능력을 먼저 키우고
- Stage 2에서 대회 데이터 형식에 맞춰 마무리

### Stage 1: 외부 데이터 (증강 없음)
- [ByteDance/MTVQA](https://huggingface.co/datasets/ByteDance/MTVQA)
  - 다국어 텍스트 중심 VQA 벤치마크 (간판, 문서 등 이미지 속 글자 관련 질문)
  - 한국어(KO) 부분만 사용
  - 원래 주관식이라 **4지선다로 변환**: 오답 선지는 같은 이미지의 다른 답, 부족하면 유형(숫자 포함 여부)과 길이가 비슷한 다른 답으로 채움
- [tabtoyou/KRETA](https://huggingface.co/datasets/tabtoyou/KRETA)
  - 텍스트가 많은 이미지에서 한국어 읽기/추론을 평가하는 VQA 벤치마크
  - 객관식 형식이라 그대로 사용

![external_internal_exp](../assets/external_internal_exp.png)
(실험 결과, QWEN-3-VL-9B-unsloth-bnb-4bit)

### Stage 2: 내부 데이터
1. **Dev는 3:0, 3:1, 3:1:1 투표 패턴만 선택**
   - 다수결이 명확한 샘플만 써서 라벨 노이즈 최소화
2. **Qwen 3.8 27B Inference-only로 1차 필터링**
   - 학습 없이 추론만 돌려 선지별 확률 계산
   - 정답이 확률 상위 2개(Top 2) 안에도 못 들면 → 라벨이 틀렸거나 매우 어려운 문제일 가능성이 높음
   - 이 샘플들만 골라 손 라벨링 (대략 5~600개)
3. **손 라벨링 결과 반영**
   - 쓰레기 문제(정답 불명확, 이미지로 판단 불가 등) → **삭제**
   - 어려운 문제 → 선지 순서를 뒤집어 **증강** (abcd → dcba), 위치 편향 없이 내용으로 풀게 유도
4. **이미지 변환 증강** (원본 + 3종 = 4배)
   - compression: JPEG quality 70으로 재압축 → 저화질 사진 대응
   - blur: Gaussian blur (radius 1.0) → 초점 흐린 사진 대응
   - affine: 이동 ±3%, 기울임 ±5° → 비스듬히 찍은 사진 대응

![image_trans_exp](../assets/image_trans_exp.png)
(실험 결과, QWEN-3-VL-4B-unsloth-bnb-4bit)

## 4. 추론
- 추론 2회: 선지 순서를 순환 이동(shift 0, 2)해서 두 번 추론 후 확률 평균
