title: [Insight] 16기 1회차 1위·2위 레시피 요약과 우리 골격
labels: insight,day1

### 한 줄 요약
직전 대회 1·2위는 같은 골격(로짓 채점 + 보기 셔플 LoRA 1 epoch + 순열 TTA + 이계열 앙상블)을 썼고, 우리는 이를 그대로 출발점으로 삼는다.

### 종류
직전 대회 상위권 분석

### 출처
1위 https://github.com/kimkihyun1/ssafy-ai-challenge-vqa , 2위 https://github.com/Sunjae-L22/vlm-vqa-pipeline (docs/LEARNINGS.md), 정리: `docs/PLAN.md` 2절

### 핵심 내용
1위(Private 0.957): Qwen3.5-27B LoRA r16 lr1e-4 1 epoch, 정답 글자 1토큰 loss, TTA4, 불확실 500문항에 LoRA OFF 확률 조건부 결합.
2위(0.949): 해상도 설정 한 줄 +4.75%p, QLoRA 32분 +3.55%p, 회전 TTA +1.5%p, 이계열 모델(Qwen2.5-VL-7B) +26문항, 32B dense 4bit는 A100 40GB에서 학습 가능(MoE 불가), 노이즈 문턱 √D, 최종 2개 = Public 최고 + 헤지.
닫은 길: CoT −2.3%p, 뒤집기 TTA, 파인튜닝 후 프롬프트 변형 TTA, 에폭 2, 비전타워 LoRA, 투표·온도 스케일링.

### 우리 과제에의 판정
채택

### 다음 액션
Colab 제로샷·LoRA Task와 PC-A 9B Task가 이 레시피를 그대로 실행한다.
