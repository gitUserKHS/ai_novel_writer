---
title: CoNarrative Studio
emoji: "book"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# CoNarrative Studio

CoNarrative Studio는 장면 단위로 장편 소설을 만드는 로컬 우선 스토리 생성 도구입니다.

초보자 기준 기본 사용 흐름은 단순합니다.

1. 실행한다.
2. 브라우저에서 `http://127.0.0.1:8000/`을 연다.
3. 상단 입력창에 만들고 싶은 소설 프롬프트를 한 번 입력한다.
4. 첫 장면, 다음 장면, 학습을 UI 버튼으로 진행한다.

기본 빠른 시작은 내장 스토리 엔진으로 바로 동작합니다. 직접 학습한 어댑터가 있으면 생성 시 최신 학습 모델을 자동 우선 사용합니다.

## 가장 쉬운 실행

Windows에서는 아래 파일을 더블클릭합니다.

```powershell
scripts\run_demo.bat
```

스크립트가 자동으로 처리하는 작업:

- `.venv` 가상환경 생성
- 필요한 패키지 설치
- 데모 작업공간 초기화
- 서버 실행

브라우저에서 `http://127.0.0.1:8000/`을 열고 프롬프트를 입력하면 됩니다.

Linux 또는 macOS에서는 아래 명령을 실행합니다.

```bash
bash scripts/run_demo.sh
```

## 수동 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m conarrative.cli --config configs/demo.yaml init
python -m conarrative.cli --config configs/demo.yaml serve --host 127.0.0.1 --port 8000
```

Windows PowerShell에서는 활성화 명령만 아래처럼 바꿉니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

## 브라우저에서 쓰는 법

- 첫 화면 상단 입력창에 프롬프트를 입력합니다.
- `이 프롬프트로 시작`을 누르면 스토리와 첫 장면이 생성됩니다.
- 이후에는 `다음 장면 자동 생성`을 누르면 장면 단위로 이어집니다.
- 로컬 모델을 자동 탐색하려면 `로컬 모델 자동 연결`을 누릅니다.
- 직접 학습한 모델이 있으면 다음 생성부터 최신 학습 모델이 자동 우선 사용됩니다.
- 고급 설정은 화면 아래쪽의 `고급 설정` 패널에 있습니다.

## 학습 모델 사용

UI에서 스토리를 선택하고 `원클릭 학습` 패널을 사용합니다.

1. `학습 환경 자동 준비`를 누릅니다.
2. 필요하면 Hugging Face 토큰을 입력합니다.
3. 필요하면 교사 모델을 선택합니다. Base URL은 비워두면 자동 탐색합니다.
4. `원클릭 학습하고 바로 연결`을 누릅니다.
5. 학습이 끝나면 생성된 `final_adapter`가 자동으로 생성 모델에 연결됩니다.

자동 처리 내용:

- 학습용 Python 3.12 가상환경 생성
- CUDA PyTorch, transformers, peft, bitsandbytes 설치
- 현재 스토리의 accepted, prompt-only, pairwise, hard-negative 데이터셋 export
- MTP/LeWM용 `multi_target_sft.jsonl` export
- 실행 중인 Gemma E2B/E4B 교사 모델 자동 탐색
- Gemma E2B/E4B 교사 모델로 prompt-only 데이터 증류
- 교사 모델이 만든 좋은 예시와 평가를 `teacher_coached_sft.jsonl`로 추가
- LoRA/QLoRA 학습 실행
- 학습 완료 어댑터를 OpenAI-compatible 생성 서버로 자동 연결

자세한 학습 설명은 [docs/TRAINING_KO.md](docs/TRAINING_KO.md)를 참고하세요.

## MTP/LeWM 반영 사항

현재 버전은 `ai_novel_writer_mtp_lewm_complete.zip`의 핵심 구조를 반영했습니다.

- 장면 계획에 미래 상태 예측, 역방향 전제 조건, 회수 목표, 모순 위험을 저장합니다.
- 메모리 프롬프트에 관련 Narrative KG 엣지를 함께 넣습니다.
- 생성 결과를 LLM 평가와 규칙 기반 검증으로 같이 점검합니다.
- 실패한 장면은 여러 번 수정 패스를 돌 수 있습니다.
- 학습 데이터에 메모리 스냅샷과 멀티 타깃 SFT 데이터를 포함합니다.
- QLoRA 학습은 프롬프트 토큰 손실을 마스킹하고 assistant 답변만 학습하도록 처리합니다.

자세한 설계 설명은 [docs/NARRATIVE_MTP_LEWM_KO.md](docs/NARRATIVE_MTP_LEWM_KO.md)를 참고하세요.

## 로컬 모델 연결

직접 로컬 모델 서버를 붙이고 싶을 때는 `openai_compatible` provider를 사용합니다.

예시는 `configs/local_backend_example.yaml`에 있습니다.

```yaml
backend:
  provider: openai_compatible
  base_url: http://127.0.0.1:8080/v1
  model: your-local-model-name
  api_key: not-needed
```

백엔드는 `/v1/models`와 `/v1/chat/completions`를 제공해야 합니다.

## Hugging Face Spaces

이 저장소는 Docker Space로 배포할 수 있게 정리되어 있습니다.

필요하면 아래 환경 변수를 Space에 넣습니다.

- `CONARRATIVE_PROVIDER=openai_compatible`
- `CONARRATIVE_BASE_URL`
- `CONARRATIVE_MODEL`
- `CONARRATIVE_API_KEY`

Spaces에서는 `app.py`가 실행되고 기본적으로 `0.0.0.0:$PORT`에서 열립니다. 데이터는 `/data/conarrative` 아래에 저장됩니다.

## 테스트

```bash
pytest -q
```
