---
title: CoNarrative Studio
emoji: "📚"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# CoNarrative Studio

CoNarrative Studio는 장면 단위로 이야기를 만드는 로컬 우선 스토리 스튜디오입니다.

지금 버전의 기본 사용 흐름은 복잡한 설정이 아니라 아래 한 줄입니다.

1. 실행한다
2. 브라우저를 연다
3. 프롬프트 한 줄을 넣는다
4. 스토리, 아웃라인, 첫 장면이 바로 만들어진다

기본 빠른 시작은 내장 스토리 엔진을 사용하므로 모델 연결이 없어도 바로 동작합니다.

## 가장 쉬운 실행 방법

Windows에서는 `scripts/run_demo.bat` 를 더블클릭하면 됩니다.

이 스크립트가 자동으로:

1. `.venv` 가상환경 생성
2. 필요한 패키지 설치
3. 데모 워크스페이스 초기화
4. 서버 실행

브라우저에서 `http://127.0.0.1:8000/` 를 열면 상단 입력창에 프롬프트만 넣어서 바로 시작할 수 있습니다.

Linux 또는 macOS에서는 아래를 실행하면 됩니다.

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

Windows PowerShell에서는 `source .venv/bin/activate` 대신 아래를 쓰면 됩니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

## 브라우저에서 하는 일

브라우저를 열면 가장 먼저 보이는 것은 빠른 시작 입력창입니다.

- 프롬프트를 입력합니다.
- `이 프롬프트로 시작` 버튼을 누릅니다.
- 스토리와 첫 장면이 바로 생성됩니다.
- 이후에는 `다음 장면 자동 생성` 버튼만 눌러 이어갈 수 있습니다.

고급 설정은 아래쪽 `고급 설정` 패널 안에 숨겨져 있습니다.

## 로컬 모델 연결은 선택 사항

기본 빠른 시작은 모델 연결 없이 동작합니다.

직접 로컬 모델 서버를 붙이고 싶을 때만 `openai_compatible` provider를 쓰면 됩니다.

예시는 `configs/local_backend_example.yaml` 에 있습니다.

```yaml
backend:
  provider: openai_compatible
  base_url: http://127.0.0.1:8080/v1
  model: your-local-model-name
  api_key: not-needed
```

백엔드는 `/v1/models` 와 `/v1/chat/completions` 를 제공해야 합니다.

## Hugging Face Spaces

이 저장소는 Docker Space로 바로 올릴 수 있게 정리되어 있습니다.

필요하면 아래 환경 변수를 Space에 넣으면 됩니다.

- `CONARRATIVE_PROVIDER=openai_compatible`
- `CONARRATIVE_BASE_URL`
- `CONARRATIVE_MODEL`
- `CONARRATIVE_API_KEY`

Spaces는 `app.py` 를 실행하고 기본적으로 `0.0.0.0:$PORT` 에서 뜹니다. 데이터는 `/data/conarrative` 아래에 저장합니다.

## 테스트

```bash
pytest -q
```
