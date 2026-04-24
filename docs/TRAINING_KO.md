# CoNarrative 학습 가이드

이 문서는 CoNarrative Studio에서 만든 장면 데이터를 꺼내고, 교사 모델로 좋은 예시와 평가를 추가한 뒤, 학생 모델을 LoRA/QLoRA로 학습하는 흐름을 설명합니다.

## 현재 정책

- 생성은 가능한 한 최신 `final_adapter` 학습 모델을 자동으로 사용합니다.
- Gemma E2B/E4B는 일반 생성 모델이 아니라 학습 때 쓰는 교사 모델로만 사용합니다.
- 학생 모델 기본값은 `Qwen/Qwen2.5-3B-Instruct`입니다.
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
  --model-name Qwen/Qwen2.5-3B-Instruct `
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
