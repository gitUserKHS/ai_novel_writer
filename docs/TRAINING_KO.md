# CoNarrative 학습 가이드

이 문서는 CoNarrative Studio에서 만든 장면 데이터를 꺼내서, 더 나은 소설 모델로 증류하고, `google/gemma-4-E2B-it` 기반 LoRA/QLoRA 학습까지 이어가는 최소 경로를 설명합니다.

## 목표

- 현재 앱에서 만든 `accepted`, `prompt_only`, `pairwise`, `hard_negative` 풀을 분리 유지한 채 export합니다.
- `prompt_only` 풀을 상위 교사 모델로 다시 작성해서 증류용 SFT 데이터셋을 만듭니다.
- `accepted_sft.jsonl` 또는 `distilled_sft.jsonl`로 `Gemma 4 E2B` LoRA/QLoRA 학습을 돌립니다.

## 왜 이렇게 하나요

- 작은 모델이 바로 소설을 쓰게 하면 한국어 품질, 지시 이행, 개연성이 흔들리기 쉽습니다.
- 먼저 상위 모델이 한국어 장면을 잘 쓰게 만들고, 그 결과를 학생 모델이 따라가게 하는 편이 더 안정적입니다.
- `accepted`와 `pairwise`를 섞어두지 않고 따로 유지해야 나중에 SFT, DPO, 거부 샘플 분석을 분리해서 할 수 있습니다.

## 권장 기본 모델

- 학생 모델: `google/gemma-4-E2B-it`
- 교사 모델: 현재 더 강한 OpenAI-compatible 모델
  예시: 더 큰 Qwen, 더 큰 Gemma, Claude/OpenAI 호환 게이트웨이 뒤 모델

## 권장 실행 환경

- 가장 권장: Linux 또는 WSL2 + NVIDIA CUDA
- Windows 네이티브에서도 일부는 되지만, `bitsandbytes` 기반 4bit QLoRA는 환경에 따라 잘 안 될 수 있습니다.
- 지금 파이썬 환경에서 `torch.cuda.is_available()`가 `False`라면, GPU가 있어도 CUDA용 PyTorch가 아직 안 깔린 상태일 가능성이 큽니다.

## 1. 학습용 데이터 export

앱에서 스토리와 장면을 어느 정도 만든 뒤 아래 명령을 실행합니다.

```powershell
python -m conarrative.cli --config configs/demo.yaml export-dataset --story-id YOUR_STORY_ID
```

기본 출력 폴더:

```text
workspace/training/YOUR_STORY_ID/
```

생성 파일:

- `accepted_sft.jsonl`
- `prompt_only_teacher.jsonl`
- `pairwise_dpo.jsonl`
- `hard_negative.jsonl`
- `manifest.json`

## 2. 상위 교사 모델로 증류

`prompt_only_teacher.jsonl`은 아직 답이 없는 프롬프트 묶음입니다. 이 파일을 더 강한 모델에게 보내서 한국어 장면을 다시 받습니다.

```powershell
python scripts/distill_openai_compatible.py `
  --input-file workspace/training/YOUR_STORY_ID/prompt_only_teacher.jsonl `
  --output-file workspace/training/YOUR_STORY_ID/distilled_sft.jsonl `
  --base-url http://127.0.0.1:1234/v1 `
  --model YOUR_TEACHER_MODEL `
  --api-key not-needed `
  --resume
```

설명:

- `--base-url`: OpenAI-compatible 서버 주소
- `--model`: 교사 모델 이름
- `--resume`: 중간에 끊겨도 이어서 실행

## 3. QLoRA 학습

먼저 학습용 의존성을 설치합니다.

```powershell
pip install -r requirements-train.txt
```

그 다음 아래처럼 실행합니다.

```powershell
python scripts/train_qlora.py `
  --train-file workspace/training/YOUR_STORY_ID/distilled_sft.jsonl workspace/training/YOUR_STORY_ID/accepted_sft.jsonl `
  --model-name google/gemma-4-E2B-it `
  --output-dir workspace/training_runs/gemma4e2b-novel-qlora `
  --epochs 1 `
  --per-device-batch-size 1 `
  --gradient-accumulation-steps 16 `
  --profile auto
```

증류 없이 바로 내부 승인 샘플만으로 시작하고 싶으면:

```powershell
python scripts/train_qlora.py `
  --train-file workspace/training/YOUR_STORY_ID/accepted_sft.jsonl `
  --model-name google/gemma-4-E2B-it `
  --output-dir workspace/training_runs/gemma4e2b-accepted-only
```

## 어떤 파일을 먼저 쓰면 되나요

처음 시작:

- `accepted_sft.jsonl`
- `distilled_sft.jsonl`

그 다음 확장:

- `pairwise_dpo.jsonl`을 나중에 preference tuning용으로 사용
- `hard_negative.jsonl`로 실패 패턴 분석

## 추천 순서

1. 앱에서 30~100개 장면 정도 만든다.
2. `export-dataset`으로 데이터셋을 뽑는다.
3. `prompt_only_teacher.jsonl`을 강한 교사 모델로 증류한다.
4. `distilled_sft.jsonl + accepted_sft.jsonl` 중심으로 첫 LoRA를 학습한다.
5. 결과를 로컬 모델 서버에 올려서 CoNarrative 기본 모델로 연결한다.
6. 다시 생성한 결과를 모아 2차 학습을 반복한다.

## 품질 팁

- 증류 교사에게는 반드시 "한국어로만", "장면 본문만", "개연성 유지", "기억 문맥 엄수"를 강하게 걸어야 합니다.
- 처음부터 장편 전체를 한 번에 학습시키지 말고, 장면 단위 데이터로 시작하는 편이 안정적입니다.
- 작은 모델에 너무 긴 시퀀스를 넣으면 느려지고 일관성이 흔들릴 수 있으니, 첫 실험은 `4096` 전후가 무난합니다.
- 8GB급 GPU에서는 앱이 자동으로 `max_seq_length=2048`, `LoRA r=8`, `alpha=16` 프로파일을 사용합니다. 품질 실험은 안정화 후 `quality` 프로파일이나 더 긴 시퀀스로 올리세요.
- 학습 시작 전에 LM Studio, Ollama, 브라우저 GPU 작업처럼 VRAM을 잡아먹는 프로그램을 닫으면 모델 로딩 실패가 줄어듭니다.
- 한 번에 너무 많은 스타일을 섞지 말고, 한국어 문체 목표를 먼저 좁히는 편이 좋습니다.

## 주의

- 이 리포는 학습 데이터 export와 학습 실행 스크립트까지는 넣었지만, 실제 학습 성공 여부는 GPU 환경과 설치된 PyTorch/CUDA 조합에 영향을 받습니다.
- 특히 Windows 네이티브에서는 4bit QLoRA 환경이 까다로울 수 있으니, 문제가 나면 WSL2 또는 Linux를 우선 고려하는 것이 안전합니다.
