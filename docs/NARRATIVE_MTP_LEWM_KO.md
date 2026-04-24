# Narrative-MTP / LeWM 적용 메모

이 패치는 장면 단위 생성 구조를 유지하면서 모델이 단순히 다음 문장만 쓰는 것이 아니라 이야기 상태, 미래 회수 조건, 모순 위험을 함께 학습하도록 만드는 기능입니다.

## 반영한 기능

- `PlanOutput`에 미래 상태 예측, 역방향 선행 조건, 회수 목표, 모순 위험 필드를 추가했습니다.
- 플래너 프롬프트가 Story Bible, 최근 장면, State Tracker, Narrative KG를 보고 다음 장면의 미래 상태까지 계획하도록 확장됐습니다.
- 장면 생성 전 관련 Narrative KG edge를 간단한 어휘 검색으로 골라 메모리 컨텍스트와 학습 데이터에 넣습니다.
- LLM critic 외에 rule-based guardrail을 추가해 필수 포함 누락, 금지 요소 노출, placeholder, 지나치게 짧은 장면을 잡습니다.
- `max_revision_passes`가 실제 반복 revision loop로 동작합니다.
- 학습 데이터셋에 `multi_target_sft.jsonl`을 추가했습니다. 이 파일은 장면 본문뿐 아니라 요약, 상태 변화, KG edge, 미래 예측을 JSON target으로 학습시킵니다.
- QLoRA 학습 스크립트는 system/user prompt 토큰을 loss에서 제외하고 assistant 답변 토큰만 학습합니다.

## 생성되는 학습 데이터

스토리 데이터셋 export 폴더에는 다음 파일이 생성됩니다.

- `accepted_sft.jsonl`: 일반 장면 본문 SFT 데이터
- `multi_target_sft.jsonl`: Narrative-MTP 구조 target SFT 데이터
- `prompt_only_teacher.jsonl`: 교사 모델 증류용 prompt-only 샘플
- `pairwise_dpo.jsonl`: 선호학습용 chosen/rejected 쌍
- `hard_negative.jsonl`: 실패 후보와 consistency issue 기록

`run_one_click_training()`은 기본적으로 `accepted_sft.jsonl`과 `multi_target_sft.jsonl`을 학습 파일로 사용합니다. 증류가 켜져 있고 `distilled_sft.jsonl`이 생성되면 그 파일을 앞에 붙여 함께 학습합니다.

## 기대 효과

이 방식은 Transformer 구조 자체를 바꾸는 완전한 MTP head 학습이 아닙니다. 대신 로컬-first QLoRA 흐름에 맞춰 SFT target을 확장해 모델이 장면 본문과 함께 미래 상태, 회수 목표, 모순 위험을 예측하도록 유도합니다.

작은 GPU에서도 적용 가능하고, 장편 소설에서 중요한 개연성, 복선 회수, 상태 일관성을 강화하는 현실적인 절충안입니다.

## 런타임 흐름

1. Story Bible은 정적 설정을 보관합니다.
2. State Tracker는 장면이 진행되면서 변하는 동적 상태를 보관합니다.
3. Narrative KG는 사건, 인과, 의도, 복선 edge를 보관합니다.
4. 장면 생성 전 현재 요청과 관련 높은 KG edge만 골라 프롬프트에 넣습니다.
5. 생성 결과는 LLM 평가와 규칙 기반 guardrail을 모두 통과해야 accepted pool에 들어갑니다.
6. 실패 후보는 hard-negative pool에 저장되어 이후 학습 데이터로 쓸 수 있습니다.
