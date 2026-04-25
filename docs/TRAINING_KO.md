# CoNarrative 학습 가이드

이 문서는 CoNarrative Studio에서 만든 장면 데이터를 꺼내고, 교사 모델로 좋은 예시와 평가를 추가한 뒤, 학생 모델을 LoRA/QLoRA로 학습하는 흐름을 설명합니다.

## 현재 정책

- 생성은 가능한 한 최신 `final_adapter` 학습 모델을 자동으로 사용합니다.
- 실제 생성 슬롯은 story별 `active_adapter`입니다. run 폴더는 기록과 롤백용으로만 남깁니다.
- Gemma E2B/E4B는 일반 생성 모델이 아니라 학습 때 쓰는 교사 모델로만 사용합니다.
- 학생 모델 기본값은 8GB Windows 환경에서 더 안전한 `Qwen/Qwen2.5-1.5B-Instruct`입니다.
- 사용자가 더 큰 모델을 입력해도 메모리 오류가 나면 1.5B, 0.5B 순서로 자동 재시도합니다.
- 새 학습은 후보 어댑터를 만든 뒤 품질 게이트를 통과해야 `active_adapter`로 승격됩니다.
- 기존 `active_adapter`와 같은 학생 모델로 학습하면 기존 어댑터에서 이어학습합니다.
- 교사 모델 기본값은 `google/gemma-4-E2B-it`이고, UI에서 `google/gemma-4-E4B-it`로 바꿀 수 있습니다.

## 한 번에 쓰는 법

브라우저 UI에서 스토리를 선택한 뒤 `Training` 패널을 사용합니다.

1. `학습 환경 자동 준비`를 누릅니다.
2. 학생 모델은 기본값 그대로 두거나 원하는 작은 모델로 바꿉니다.
3. 교사 모델을 E2B 또는 E4B 중 선택합니다.
4. 교사 모델 Base URL은 비워둬도 됩니다. 실행 중인 Gemma E2B/E4B 서버가 있으면 자동 탐색합니다.
5. `원클릭 학습하고 바로 연결`을 누릅니다.
6. 학습이 끝나면 최신 학습 모델 서버가 자동으로 켜지고 다음 생성부터 자동 우선 사용됩니다.

교사 모델 Base URL을 직접 입력해야 할 때의 예시:

```text
http://127.0.0.1:8080/v1
```

## 학습 데이터 구성

원클릭 학습은 다음 데이터를 분리해서 만듭니다.

- `accepted_sft.jsonl`: 사용자가 accepted한 실제 장면 본문
- `multi_target_sft.jsonl`: 장면 본문, 요약, 상태 변화, KG edge, 미래 예측을 함께 학습하는 MTP 데이터
- `prompt_only_teacher.jsonl`: 교사 모델에게 보낼 프롬프트 전용 데이터
- `distilled_sft.jsonl`: 교사 모델이 다시 쓴 좋은 장면 예시
- `teacher_coached_sft.jsonl`: 교사 모델이 만든 좋은 예시와 평가 JSON
- `pairwise_dpo.jsonl`: chosen/rejected 선호학습용 데이터
- `hard_negative.jsonl`: 실패 후보와 consistency issue

## active_adapter 구조

학습할 때마다 `runs/STORY_ID/날짜-모델/final_adapter`는 계속 생깁니다. 이것은 실험 기록입니다. 생성에 직접 쓰는 모델은 `active_adapters/STORY_ID/final_adapter` 하나입니다.

흐름은 다음과 같습니다.

1. 새 학습 결과를 run 폴더에 후보로 저장합니다.
2. 후보에 adapter 설정과 학습 성공 기록이 있는지 품질 게이트로 검사합니다.
3. 통과하면 후보를 `active_adapter`로 복사해 승격합니다.
4. 생성 서버를 재시작해서 새 active 모델만 사용합니다.
5. 다음 학습 때 학생 모델이 같으면 기존 active adapter에서 이어학습합니다.

## 용량 관리와 학습 상태

원클릭 학습은 성공 후 자동으로 오래된 학습 산출물을 정리합니다.

- 실제 생성 모델인 `active_adapter`는 삭제하지 않습니다.
- 성공 run은 기본 2개만 남깁니다.
- 실패 run과 중간에 실패한 후보 폴더는 기본적으로 정리합니다.
- 데이터셋과 active 모델 용량은 UI의 학습 상태 카드에 표시됩니다.

UI에서는 다음 정보를 초보자용으로 보여줍니다.

- 현재 학습 모델이 있는지
- 성공 학습 횟수와 실패 횟수
- 품질 게이트 점수
- active, run 기록, 데이터셋이 차지하는 용량
- 다음 학습이 이어학습인지 새 학습인지

필요하면 `학습 용량 정리` 버튼을 눌러 같은 정리 정책을 수동으로 다시 실행할 수 있습니다.

## 과적합 방지용 교사 코칭

`teacher_coached_sft.jsonl`은 단순히 같은 장면을 외우게 하지 않기 위한 데이터입니다. 교사 모델은 같은 요청을 보고 다른 좋은 예시를 만들고, 왜 좋은지 평가하며, 학생 모델이 일반화해야 할 원칙을 JSON으로 남깁니다.

이 데이터는 작은 accepted 데이터셋만 반복 학습할 때 생기는 문체 고착, 전개 반복, 특정 표현 암기를 줄이는 데 목적이 있습니다.

## 수동 학습 예시

```powershell
python scripts/train_qlora.py `
  --train-file `
    configs/workspace/training/datasets/YOUR_STORY_ID/distilled_sft.jsonl `
    configs/workspace/training/datasets/YOUR_STORY_ID/accepted_sft.jsonl `
    configs/workspace/training/datasets/YOUR_STORY_ID/multi_target_sft.jsonl `
    configs/workspace/training/datasets/YOUR_STORY_ID/teacher_coached_sft.jsonl `
  --model-name Qwen/Qwen2.5-1.5B-Instruct `
  --output-dir configs/workspace/training/runs/YOUR_STORY_ID/manual-run `
  --epochs 1 `
  --per-device-batch-size 1 `
  --gradient-accumulation-steps 16 `
  --profile auto
```

## 주의

- 교사 모델은 별도 OpenAI-compatible 서버로 떠 있어야 증류와 코칭이 동작합니다.
- 교사 Base URL을 비워두면 교사 증류/코칭은 건너뛰고 accepted/MTP 데이터만으로 학습합니다.
- 학습 모델 서버는 첫 연결 때 모델을 GPU에 올리므로 시간이 걸릴 수 있습니다.
- 8GB급 GPU에서는 자동으로 낮은 VRAM 프로파일이 적용됩니다.
- Windows에서 `os error 1455` 또는 페이징 파일 오류가 나면 원클릭 학습은 더 작은 학생 모델로 자동 재시도합니다.
