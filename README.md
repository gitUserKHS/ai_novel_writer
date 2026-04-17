---
title: CoNarrative Studio
emoji: "📚"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# CoNarrative Studio

CoNarrative Studio는 장편 이야기를 씬 단위로 생성하는 로컬 우선 도구입니다. 웹 UI, CLI, 스토리 바이블, 상태 추적, 서사 그래프, 내보내기, 평가 기능을 제공합니다.

## 초보자용 한 줄 실행

Windows에서는 `scripts/run_demo.bat`를 더블클릭하면 됩니다.

이 파일이 자동으로 하는 일은 다음과 같습니다.

1. `.venv` 가상환경 생성
2. 필요한 패키지 설치
3. 데모 워크스페이스 초기화
4. 샘플 스토리 생성
5. 아웃라인 생성
6. 샘플 씬 1개 실행
7. 결과를 브라우저에서 열 수 있게 서버 실행

브라우저는 `http://127.0.0.1:8000/` 를 엽니다.

Linux 또는 macOS에서는 `bash scripts/run_demo.sh` 를 실행하면 됩니다.

## 수동 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m conarrative.cli --config configs/demo.yaml init
python -m conarrative.cli --config configs/demo.yaml serve --host 127.0.0.1 --port 8000
```

## 가장 자주 쓰는 기능

```bash
python -m conarrative.cli --config configs/demo.yaml create-story --input-file examples/story.yaml
python -m conarrative.cli --config configs/demo.yaml outline --story-id moon-theater --scene-count 4
python -m conarrative.cli --config configs/demo.yaml run-scene --story-id moon-theater --input-file examples/scene1.yaml --print-text
python -m conarrative.cli --config configs/demo.yaml export --story-id moon-theater
python -m conarrative.cli --config configs/demo.yaml evaluate --story-id moon-theater
```

## 로컬 백엔드 연결

로컬 모델 서버를 쓰려면 `configs/local_backend_example.yaml` 을 복사해서 아래처럼 맞추면 됩니다.

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

필요하면 Space 변수나 Secret을 아래처럼 넣으세요.

- `CONARRATIVE_PROVIDER=openai_compatible`
- `CONARRATIVE_BASE_URL`
- `CONARRATIVE_MODEL`
- `CONARRATIVE_API_KEY`

Spaces는 `app.py` 를 실행하고, 기본적으로 `0.0.0.0:$PORT` 에서 뜹니다. 데이터는 `/data/conarrative` 아래에 저장합니다.

## 테스트

```bash
pytest -q
```

## GitHub에 올리는 방법

현재 이 폴더는 git 저장소가 아닙니다. `push` 하려면 먼저 저장소를 초기화하고 원격을 연결해야 합니다.

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <YOUR_GITHUB_REPO_URL>
git branch -M main
git push -u origin main
```

이미 GitHub 저장소를 만들어 둔 상태라면 `git remote add origin ...` 부분만 맞게 넣으면 됩니다.
