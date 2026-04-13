# Research Applied

이 저장소에서 실제로 반영한 연구 아이디어 정리.

## 1) 월드 모델 → abstract-state scorer
raw reconstruction 모델 대신, 서사에서는 **다음 문장**보다 **다음 장면이 세계 상태를 어떻게 바꾸는가**가 더 중요하다고 보고 `world_model.py`를 만들었어.

점수 축:
- plausibility
- novelty
- surprise
- state_change_size
- transition gap
- location/time anchor
- knowledge drift

즉, 생성 모델은 prose를 쓰고 world model은 **후보 장면의 상태 전이 품질을 판정**해.

## 2) long-form planning
고정된 거대 개요보다:
- global outline
- adaptive scene plan
- next-scene memory bundle
- unresolved-thread pressure

이 네 가지를 묶어 장편의 중반 붕괴를 조금 더 막는 쪽으로 구현했어.

## 3) critic + revision
consistency critic / creativity critic / world model score를 합쳐 후보를 고르고, medium/high severity 이슈가 있으면 revision을 거쳐 다시 평가해.

## 4) self-improving data loop
별도 데이터 풀:
- prompt_only
- accepted
- pairwise
- hard_negative
- world_model_transitions

그래서 후속 SFT, DPO, distillation, critic tuning으로 이어지게 만들었어.

## 5) 4060 최적화 방향
- runtime unit을 scene로 제한
- recent scene memory만 사용
- role-wise 모델 alias 지원
- optional response cache
- single-worker job queue
- GGUF / OpenAI-compatible local server 전제
