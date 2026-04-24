const uiState = {
  stories: [],
  selectedStoryId: null,
  selectedSceneId: null,
  storyDetail: null,
  scenes: [],
  outline: [],
  artifacts: [],
  state: {},
  datasets: {},
  kg: [],
  modelCatalog: { options: [], current: null, detail: "" },
  trainingEnv: null,
  trainingJobId: null,
  trainingJob: null,
  trainingPollTimer: null,
  trainedAdapters: [],
  generationJob: null,
  generationEventSource: null,
};

const els = {
  healthPill: document.getElementById("health-pill"),
  refreshBtn: document.getElementById("refresh-btn"),
  quickstartPrompt: document.getElementById("quickstart-prompt"),
  quickstartSceneCount: document.getElementById("quickstart-scene-count"),
  quickstartWordCount: document.getElementById("quickstart-word-count"),
  modelPicker: document.getElementById("model-picker"),
  quickstartBtn: document.getElementById("quickstart-btn"),
  autoConnectBtn: document.getElementById("auto-connect-btn"),
  refreshModelsBtn: document.getElementById("refresh-models-btn"),
  quickstartNote: document.getElementById("quickstart-note"),
  storyList: document.getElementById("story-list"),
  storyTitle: document.getElementById("story-title"),
  storySummary: document.getElementById("story-summary"),
  continueBtn: document.getElementById("continue-btn"),
  exportBtn: document.getElementById("export-btn"),
  evaluateBtn: document.getElementById("evaluate-btn"),
  deleteStoryBtn: document.getElementById("delete-story-btn"),
  trainingReadyChip: document.getElementById("training-ready-chip"),
  trainingSummary: document.getElementById("training-summary"),
  trainingBaseModel: document.getElementById("training-base-model"),
  trainingEpochs: document.getElementById("training-epochs"),
  trainingHfToken: document.getElementById("training-hf-token"),
  trainingTeacherBaseUrl: document.getElementById("training-teacher-base-url"),
  trainingTeacherModel: document.getElementById("training-teacher-model"),
  trainingTeacherVariants: document.getElementById("training-teacher-variants"),
  trainingSetupBtn: document.getElementById("training-setup-btn"),
  trainingStartBtn: document.getElementById("training-start-btn"),
  trainedModelPicker: document.getElementById("trained-model-picker"),
  trainedRefreshBtn: document.getElementById("trained-refresh-btn"),
  trainedUseBtn: document.getElementById("trained-use-btn"),
  trainingTeacherNote: document.getElementById("training-teacher-note"),
  trainingLog: document.getElementById("training-log"),
  generationStream: document.getElementById("generation-stream"),
  outlineList: document.getElementById("outline-list"),
  sceneCount: document.getElementById("scene-count"),
  sceneList: document.getElementById("scene-list"),
  sceneDetail: document.getElementById("scene-detail"),
  storyStats: document.getElementById("story-stats"),
  artifactList: document.getElementById("artifact-list"),
  settingsForm: document.getElementById("settings-form"),
  testSettingsBtn: document.getElementById("test-settings-btn"),
  settingsResult: document.getElementById("settings-result"),
  stateViewer: document.getElementById("state-viewer"),
  datasetViewer: document.getElementById("dataset-viewer"),
  kgViewer: document.getElementById("kg-viewer"),
};

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function compactError(error) {
  const raw = error?.message || String(error || "알 수 없는 오류");
  return raw.length > 420 ? `${raw.slice(0, 420)}...` : raw;
}

function formatGib(value) {
  if (typeof value !== "number" || Number.isNaN(value) || value <= 0) {
    return "";
  }
  return `${value.toFixed(1)} GiB`;
}

function formatTrainingSummary(env) {
  const parts = [];
  if (env.ready) {
    parts.push("학습 환경 준비 완료.");
  } else {
    parts.push("학습 환경 준비가 필요합니다.");
  }
  const total = formatGib(env.cuda_total_gib);
  const free = formatGib(env.cuda_free_gib);
  if (total) {
    parts.push(`GPU ${env.gpu_name || "CUDA"}: ${free ? `${free} 여유 / ` : ""}${total} 전체.`);
  } else if (env.gpu_available) {
    parts.push(`GPU ${env.gpu_name || "NVIDIA"} 감지됨.`);
  }
  if (env.training_profile) {
    parts.push(`자동 프로파일: ${env.training_profile}.`);
  }
  if (env.detail) {
    parts.push(env.detail);
  }
  return parts.join(" ");
}

function updateStoryButtons(enabled) {
  els.continueBtn.disabled = !enabled;
  els.exportBtn.disabled = !enabled;
  els.evaluateBtn.disabled = !enabled;
  els.deleteStoryBtn.disabled = !enabled;
  els.trainingSetupBtn.disabled = !enabled;
  els.trainingStartBtn.disabled = !enabled;
  els.trainedRefreshBtn.disabled = !enabled;
  els.trainedUseBtn.disabled = !enabled || !uiState.trainedAdapters.length;
}

function renderEmptyStoryState() {
  uiState.selectedStoryId = null;
  uiState.selectedSceneId = null;
  uiState.storyDetail = null;
  uiState.scenes = [];
  uiState.outline = [];
  uiState.artifacts = [];
  uiState.state = {};
  uiState.datasets = {};
  uiState.kg = [];
  uiState.trainedAdapters = [];
  els.storyTitle.textContent = "스토리를 선택하거나 새로 시작하세요";
  els.storySummary.textContent = "왼쪽 목록에서 기존 스토리를 열거나 위에서 프롬프트 한 줄로 새 스토리를 시작할 수 있습니다.";
  els.outlineList.innerHTML = `<div class="empty-state">빠른 시작을 실행하면 여기에 아웃라인이 생깁니다.</div>`;
  els.sceneList.innerHTML = `<div class="empty-state">아직 생성된 장면이 없습니다.</div>`;
  els.sceneDetail.textContent = "장면을 선택하면 본문이 여기 표시됩니다.";
  els.storyStats.innerHTML = `<div class="empty-state">선택된 스토리가 없습니다.</div>`;
  els.artifactList.innerHTML = `<div class="empty-state">아직 생성된 파일이 없습니다.</div>`;
  els.stateViewer.textContent = "{}";
  els.datasetViewer.textContent = "{}";
  els.kgViewer.textContent = "[]";
  els.sceneCount.textContent = "0개";
  renderTrainedAdapters();
  updateStoryButtons(false);
  renderTrainingPanel();
}

function renderHealth(health) {
  if (health.backend_ok && health.provider === "openai_compatible") {
    els.healthPill.textContent = `로컬 모델 연결됨: ${health.model}`;
    return;
  }
  if (health.provider === "mock") {
    els.healthPill.textContent = "내장 스토리 엔진 사용 중";
    return;
  }
  els.healthPill.textContent = "실시간 모델 연결 안 됨";
}

function renderStories() {
  if (!uiState.stories.length) {
    els.storyList.innerHTML = `<div class="empty-state">아직 저장된 스토리가 없습니다.</div>`;
    return;
  }
  els.storyList.innerHTML = uiState.stories
    .map(
      (story) => `
        <button class="story-card ${story.id === uiState.selectedStoryId ? "active" : ""}" data-story-id="${escapeHtml(story.id)}">
          <strong>${escapeHtml(story.title)}</strong>
          <span>${escapeHtml(story.genre)}</span>
          <span>${escapeHtml(story.tone)}</span>
        </button>
      `,
    )
    .join("");
  els.storyList.querySelectorAll("[data-story-id]").forEach((button) => {
    button.addEventListener("click", () => selectStory(button.dataset.storyId));
  });
}

function renderSummary() {
  if (!uiState.storyDetail) {
    renderEmptyStoryState();
    return;
  }
  const story = uiState.storyDetail.story;
  const themes = (story.themes || []).join(", ") || "자동 추출 없음";
  const characters = (story.characters || []).join(", ") || "자동 기본값";
  els.storyTitle.textContent = story.title;
  els.storySummary.innerHTML = `
    <p>${escapeHtml(story.premise)}</p>
    <div class="summary-meta">
      <span>장르: ${escapeHtml(story.genre)}</span>
      <span>톤: ${escapeHtml(story.tone)}</span>
      <span>테마: ${escapeHtml(themes)}</span>
      <span>인물: ${escapeHtml(characters)}</span>
    </div>
  `;
}

function renderOutline() {
  if (!uiState.outline.length) {
    els.outlineList.innerHTML = `<div class="empty-state">아직 아웃라인이 없습니다.</div>`;
    return;
  }
  els.outlineList.innerHTML = uiState.outline
    .map(
      (card, index) => `
        <article class="outline-card ${card.status === "used" ? "used" : ""}">
          <div class="outline-order">${index + 1}</div>
          <div>
            <h3>${escapeHtml(card.title)}</h3>
            <p>${escapeHtml(card.summary_request)}</p>
            <div class="outline-meta">
              <span>${escapeHtml(card.time_label)}</span>
              <span>${escapeHtml(card.location)}</span>
              <span>${escapeHtml(card.status)}</span>
            </div>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderSceneDetail(scene) {
  if (!scene) {
    els.sceneDetail.textContent = "장면을 선택하면 본문이 여기 표시됩니다.";
    return;
  }
  const candidateMarkup = (scene.candidates || [])
    .map(
      (candidate) => `
        <details class="candidate-block">
          <summary>Candidate ${candidate.candidate_index} / score ${candidate.score}</summary>
          <pre class="code-box inline-box">${escapeHtml(candidate.text)}</pre>
        </details>
      `,
    )
    .join("");
  els.sceneDetail.innerHTML = `
    <h3>${escapeHtml(scene.title)}</h3>
    <p class="scene-copy">${escapeHtml(scene.summary)}</p>
    <div class="summary-meta">
      <span>POV: ${escapeHtml(scene.pov)}</span>
      <span>장소: ${escapeHtml(scene.location)}</span>
      <span>시간: ${escapeHtml(scene.time_label)}</span>
    </div>
    <pre class="code-box">${escapeHtml(scene.accepted_text)}</pre>
    <div class="detail-block">
      <h4>Critic Snapshot</h4>
      <pre class="code-box inline-box">${escapeHtml(JSON.stringify(scene.consistency, null, 2))}</pre>
    </div>
    <div class="detail-block">
      <h4>Candidate Drafts</h4>
      ${candidateMarkup || `<div class="empty-state">저장된 후보가 없습니다.</div>`}
    </div>
  `;
}

function renderScenes() {
  els.sceneCount.textContent = `${uiState.scenes.length}개`;
  if (!uiState.scenes.length) {
    els.sceneList.innerHTML = `<div class="empty-state">아직 생성된 장면이 없습니다.</div>`;
    els.sceneDetail.textContent = "장면을 선택하면 본문이 여기 표시됩니다.";
    return;
  }
  if (!uiState.selectedSceneId || !uiState.scenes.some((scene) => scene.id === uiState.selectedSceneId)) {
    uiState.selectedSceneId = uiState.scenes[uiState.scenes.length - 1].id;
  }
  els.sceneList.innerHTML = uiState.scenes
    .map(
      (scene) => `
        <button class="scene-card ${scene.id === uiState.selectedSceneId ? "active" : ""}" data-scene-id="${escapeHtml(scene.id)}">
          <strong>Scene ${scene.scene_index}</strong>
          <span>${escapeHtml(scene.title)}</span>
          <span>${escapeHtml(scene.time_label)}</span>
        </button>
      `,
    )
    .join("");
  els.sceneList.querySelectorAll("[data-scene-id]").forEach((button) => {
    button.addEventListener("click", () => {
      uiState.selectedSceneId = button.dataset.sceneId;
      renderScenes();
    });
  });
  renderSceneDetail(uiState.scenes.find((scene) => scene.id === uiState.selectedSceneId));
}

function renderStoryStats() {
  if (!uiState.storyDetail) {
    els.storyStats.innerHTML = `<div class="empty-state">선택된 스토리가 없습니다.</div>`;
    return;
  }
  const stats = [
    ["현재 장면 수", uiState.scenes.length],
    ["마지막 장면 번호", uiState.state.last_scene_index || 0],
    ["열린 떡밥", (uiState.state.active_threads || []).length],
    ["회수된 떡밥", (uiState.state.resolved_threads || []).length],
    ["Accepted 데이터", uiState.datasets.accepted || 0],
    ["Prompt-only 데이터", uiState.datasets.prompt_only || 0],
  ];
  els.storyStats.innerHTML = stats
    .map(
      ([label, value]) => `
        <div class="stat-card">
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(value)}</strong>
        </div>
      `,
    )
    .join("");
  els.stateViewer.textContent = JSON.stringify(uiState.state, null, 2);
  els.datasetViewer.textContent = JSON.stringify(uiState.datasets, null, 2);
  els.kgViewer.textContent = JSON.stringify(uiState.kg, null, 2);
}

function renderArtifacts() {
  if (!uiState.selectedStoryId || !uiState.artifacts.length) {
    els.artifactList.innerHTML = `<div class="empty-state">아직 생성된 파일이 없습니다.</div>`;
    return;
  }
  els.artifactList.innerHTML = uiState.artifacts
    .map(
      (artifact) => `
        <article class="artifact-card">
          <strong>${escapeHtml(artifact.artifact_type)}</strong>
          <span>${escapeHtml(artifact.created_at)}</span>
          <a href="/api/stories/${encodeURIComponent(uiState.selectedStoryId)}/artifacts/${artifact.id}/download">다운로드</a>
        </article>
      `,
    )
    .join("");
}

function renderTrainingPanel() {
  const env = uiState.trainingEnv;
  const hasStory = Boolean(uiState.selectedStoryId);
  const currentTeacher = `${els.trainingTeacherModel?.value || "google/gemma-4-E2B-it"} @ ${
    els.trainingTeacherBaseUrl?.value?.trim() || "자동 탐색"
  }`;
  if (!env) {
    els.trainingReadyChip.textContent = "학습 환경 확인 중";
    els.trainingSummary.textContent = "학습 환경 상태를 불러오는 중입니다.";
    els.trainingTeacherNote.textContent = `학습 교사 모델: ${currentTeacher}. 비워두면 실행 중인 Gemma E2B/E4B를 자동 탐색합니다.`;
    els.trainingLog.textContent = "학습 로그가 여기 표시됩니다.";
    els.trainingSetupBtn.disabled = true;
    els.trainingStartBtn.disabled = true;
    return;
  }
  els.trainingReadyChip.textContent = env.ready ? "학습 환경 준비 완료" : "학습 환경 준비 필요";
  els.trainingSummary.textContent = formatTrainingSummary(env);
  els.trainingTeacherNote.textContent = `학습 교사 모델: ${currentTeacher}. 학습 후 최신 어댑터가 자동으로 생성 모델에 연결됩니다.`;
  if (uiState.trainingJob && uiState.trainingJob.story_id === uiState.selectedStoryId) {
    const logs = (uiState.trainingJob.logs || []).map((item) => `[${item.time}] ${item.message}`);
    const header = [
      `job: ${uiState.trainingJob.kind}`,
      `status: ${uiState.trainingJob.status}`,
      `progress: ${Math.round((uiState.trainingJob.progress || 0) * 100)}%`,
      `message: ${uiState.trainingJob.message}`,
      uiState.trainingJob.error_text ? `error: ${uiState.trainingJob.error_text}` : "",
      "",
    ].filter(Boolean);
    els.trainingLog.textContent = header.concat(logs).join("\n");
  } else {
    els.trainingLog.textContent = env.ready
      ? "학습 환경이 준비되었습니다. 스토리를 선택한 뒤 '딸깍 학습 시작'을 누르세요."
      : "먼저 '학습 환경 자동 준비'를 눌러 학습용 Python 3.12/CUDA 환경을 만드세요.";
  }
  els.trainingSetupBtn.disabled = !hasStory;
  els.trainingStartBtn.disabled = !hasStory;
  els.trainedRefreshBtn.disabled = !hasStory;
  els.trainedUseBtn.disabled = !hasStory || !uiState.trainedAdapters.length;
  renderTrainedAdapters();
}

function renderTrainedAdapters() {
  const adapters = uiState.trainedAdapters || [];
  if (!adapters.length) {
    els.trainedModelPicker.innerHTML = `<option value="">아직 학습된 모델이 없습니다</option>`;
    els.trainedUseBtn.disabled = true;
    return;
  }
  els.trainedModelPicker.innerHTML = adapters
    .map((adapter, index) => {
      const label = `${adapter.run_id || `run ${index + 1}`} / ${adapter.base_model || "base model"} / ${adapter.training_profile || "profile"}`;
      return `<option value="${escapeHtml(adapter.adapter_dir)}">${escapeHtml(label)}</option>`;
    })
    .join("");
  els.trainedUseBtn.disabled = !uiState.selectedStoryId;
}

function stopTrainingJobPolling() {
  if (uiState.trainingPollTimer) {
    clearTimeout(uiState.trainingPollTimer);
    uiState.trainingPollTimer = null;
  }
}

function stopGenerationStream() {
  if (uiState.generationEventSource) {
    uiState.generationEventSource.close();
    uiState.generationEventSource = null;
  }
}

function renderGenerationJob(job) {
  uiState.generationJob = job;
  const logs = job.logs || [];
  const streamText = logs
    .filter((item) => String(item.message || "").startsWith("STREAM_"))
    .map((item) => String(item.message).replace(/^STREAM_[A-Z_]+:/, ""))
    .join("");
  const statusLines = logs
    .filter((item) => !String(item.message || "").startsWith("STREAM_"))
    .slice(-10)
    .map((item) => `[${item.time}] ${item.message}`);
  const header = [
    `job: ${job.kind}`,
    `status: ${job.status}`,
    `progress: ${Math.round((job.progress || 0) * 100)}%`,
    job.error_text ? `error: ${job.error_text}` : "",
    "",
  ].filter(Boolean);
  const body = streamText
    ? ["실시간 초안", "", streamText.slice(-12000), "", "진행 로그", ...statusLines]
    : ["진행 로그", ...statusLines];
  els.generationStream.classList.add("active");
  els.generationStream.textContent = header.concat(body).join("\n");
}

function waitForJobStream(jobId, renderer = renderGenerationJob) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (job) => {
      if (settled) {
        return;
      }
      settled = true;
      stopGenerationStream();
      if (job.status === "failed") {
        reject(new Error(job.error_text || job.message || "job failed"));
        return;
      }
      resolve(job);
    };
    if (window.EventSource) {
      stopGenerationStream();
      const source = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);
      uiState.generationEventSource = source;
      source.onmessage = (event) => {
        const job = JSON.parse(event.data);
        if (job.error) {
          reject(new Error(job.error));
          source.close();
          return;
        }
        renderer(job);
        if (job.status === "succeeded" || job.status === "failed") {
          finish(job);
        }
      };
      source.onerror = () => {
        source.close();
        pollJobUntilDone(jobId, renderer).then(resolve).catch(reject);
      };
      return;
    }
    pollJobUntilDone(jobId, renderer).then(resolve).catch(reject);
  });
}

async function pollJobUntilDone(jobId, renderer) {
  while (true) {
    const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
    renderer(job);
    if (job.status === "succeeded") {
      return job;
    }
    if (job.status === "failed") {
      throw new Error(job.error_text || job.message || "job failed");
    }
    await new Promise((resolve) => setTimeout(resolve, 750));
  }
}

function renderModelCatalog() {
  const catalog = uiState.modelCatalog || { options: [], current: null, detail: "" };
  const currentKey = catalog.current ? `${catalog.current.base_url}||${catalog.current.model}` : "mock";
  const builtinOption = `<option value="mock" ${currentKey === "mock" ? "selected" : ""}>내장 스토리 엔진</option>`;
  const remoteOptions = (catalog.options || [])
    .map((option) => {
      const value = `${option.base_url}||${option.model}`;
      const label = `${option.source} / ${option.model}`;
      return `<option value="${escapeHtml(value)}" ${value === currentKey ? "selected" : ""}>${escapeHtml(label)}</option>`;
    })
    .join("");
  els.modelPicker.innerHTML = builtinOption + remoteOptions;
}

async function loadHealth() {
  renderHealth(await api("/api/health"));
}

async function loadTrainingEnvironment() {
  uiState.trainingEnv = await api("/api/training/environment");
  renderTrainingPanel();
}

async function loadTrainedAdapters() {
  if (!uiState.selectedStoryId) {
    uiState.trainedAdapters = [];
    renderTrainedAdapters();
    return;
  }
  const data = await api(`/api/training/adapters?story_id=${encodeURIComponent(uiState.selectedStoryId)}`);
  uiState.trainedAdapters = data.items || [];
  renderTrainedAdapters();
}

async function loadSettings() {
  const settings = await api("/api/runtime-settings");
  for (const [key, value] of Object.entries(settings)) {
    if (els.settingsForm.elements[key]) {
      els.settingsForm.elements[key].value = value;
    }
  }
}

async function loadModelCatalog() {
  uiState.modelCatalog = await api("/api/runtime-settings/models");
  renderModelCatalog();
  renderTrainingPanel();
}

async function loadStories() {
  const data = await api("/api/stories");
  uiState.stories = data.items || [];
  if (uiState.selectedStoryId && !uiState.stories.some((story) => story.id === uiState.selectedStoryId)) {
    uiState.selectedStoryId = null;
  }
  renderStories();
  if (!uiState.stories.length) {
    renderEmptyStoryState();
    return;
  }
  if (!uiState.selectedStoryId && uiState.stories.length) {
    await selectStory(uiState.stories[0].id);
  }
}

async function selectStory(storyId) {
  uiState.selectedStoryId = storyId;
  renderStories();
  updateStoryButtons(true);
  const storyPath = encodeURIComponent(storyId);
  const [detail, scenes, state, datasets, kg, artifacts] = await Promise.all([
    api(`/api/stories/${storyPath}`),
    api(`/api/stories/${storyPath}/scenes`),
    api(`/api/stories/${storyPath}/state`),
    api(`/api/stories/${storyPath}/datasets`),
    api(`/api/stories/${storyPath}/kg`),
    api(`/api/stories/${storyPath}/artifacts`),
  ]);
  uiState.storyDetail = detail;
  uiState.outline = detail.outline || [];
  uiState.scenes = scenes.items || [];
  uiState.state = state;
  uiState.datasets = datasets;
  uiState.kg = kg.items || [];
  uiState.artifacts = artifacts.items || [];
  renderSummary();
  renderOutline();
  renderScenes();
  renderStoryStats();
  renderArtifacts();
  await loadTrainedAdapters();
  renderTrainingPanel();
}

async function handleQuickstart() {
  const prompt = els.quickstartPrompt.value.trim();
  if (!prompt) {
    alert("프롬프트를 입력하세요.");
    return;
  }
  els.generationStream.classList.add("active");
  els.generationStream.textContent = "생성 job을 시작하는 중입니다...";
  const started = await api("/api/quickstart/job", {
    method: "POST",
    body: JSON.stringify({
      prompt,
      scene_count: Number(els.quickstartSceneCount.value || 4),
      desired_length_words: Number(els.quickstartWordCount.value || 900),
    }),
  });
  els.quickstartNote.textContent = "생성 중입니다. 아래 스트림에서 진행 상황을 확인하세요.";
  const job = await waitForJobStream(started.job.id);
  const result = job.result || {};
  els.quickstartNote.textContent = result.detail || "스토리를 만들었습니다.";
  uiState.selectedStoryId = job.story_id;
  await Promise.all([loadStories(), loadHealth(), loadModelCatalog()]);
  await selectStory(job.story_id);
}

async function handleAutoConnect() {
  const result = await api("/api/runtime-settings/auto-connect", { method: "POST" });
  els.quickstartNote.textContent = result.detail || "모델 연결 상태를 확인했습니다.";
  await Promise.all([loadSettings(), loadHealth(), loadModelCatalog()]);
}

async function handleModelSelection() {
  const selected = els.modelPicker.value;
  if (selected === "mock") {
    await api("/api/runtime-settings/select-model", {
      method: "PUT",
      body: JSON.stringify({ provider: "mock", base_url: "", model: "" }),
    });
    els.quickstartNote.textContent = "내장 스토리 엔진으로 전환했습니다.";
  } else {
    const [baseUrl, model] = selected.split("||");
    await api("/api/runtime-settings/select-model", {
      method: "PUT",
      body: JSON.stringify({
        provider: "openai_compatible",
        base_url: baseUrl,
        model,
      }),
    });
    els.quickstartNote.textContent = `모델을 ${model}로 변경했습니다.`;
  }
  await Promise.all([loadSettings(), loadHealth(), loadModelCatalog()]);
}

async function handleContinue() {
  if (!uiState.selectedStoryId) {
    return;
  }
  els.generationStream.classList.add("active");
  els.generationStream.textContent = "다음 장면 생성 job을 시작하는 중입니다...";
  const started = await api(`/api/stories/${encodeURIComponent(uiState.selectedStoryId)}/continue/job`, {
    method: "POST",
    body: JSON.stringify({
      desired_length_words: Number(els.quickstartWordCount.value || 900),
    }),
  });
  const job = await waitForJobStream(started.id);
  const result = job.result || {};
  els.quickstartNote.textContent = result.detail || "다음 장면을 만들었습니다.";
  await Promise.all([loadStories(), loadHealth(), loadModelCatalog()]);
  await selectStory(job.story_id);
}

async function handleExport() {
  if (!uiState.selectedStoryId) {
    return;
  }
  const result = await api(`/api/stories/${encodeURIComponent(uiState.selectedStoryId)}/export`, {
    method: "POST",
  });
  els.quickstartNote.textContent = `내보내기 완료: ${result.artifact.path}`;
  await selectStory(uiState.selectedStoryId);
}

async function handleEvaluate() {
  if (!uiState.selectedStoryId) {
    return;
  }
  const result = await api(`/api/stories/${encodeURIComponent(uiState.selectedStoryId)}/evaluate`, {
    method: "POST",
  });
  els.quickstartNote.textContent = `평가 완료: scene ${result.report.scene_count}, consistency ${result.report.average_consistency_score}`;
  await selectStory(uiState.selectedStoryId);
}

async function handleDeleteStory() {
  if (!uiState.selectedStoryId || !uiState.storyDetail) {
    return;
  }
  stopTrainingJobPolling();
  const storyId = uiState.selectedStoryId;
  const storyTitle = uiState.storyDetail.story?.title || storyId;
  const confirmed = window.confirm(`'${storyTitle}' 스토리를 삭제할까요?\n장면, 아웃라인, 상태, 내보낸 파일도 함께 삭제됩니다.`);
  if (!confirmed) {
    return;
  }
  const result = await api(`/api/stories/${encodeURIComponent(storyId)}`, {
    method: "DELETE",
  });
  uiState.selectedStoryId = null;
  uiState.selectedSceneId = null;
  els.quickstartNote.textContent = `${result.title || storyTitle} 스토리를 삭제했습니다.`;
  await loadStories();
}

async function pollTrainingJob(jobId) {
  uiState.trainingJobId = jobId;
  const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
  uiState.trainingJob = job;
  renderTrainingPanel();
  if (job.status === "queued" || job.status === "running") {
    stopTrainingJobPolling();
    uiState.trainingPollTimer = setTimeout(() => {
      pollTrainingJob(jobId).catch((error) => {
        console.error(error);
        els.trainingLog.textContent = `학습 job 조회 실패: ${error.message}`;
      });
    }, 2500);
    return;
  }
  stopTrainingJobPolling();
  await loadTrainingEnvironment();
  if (uiState.selectedStoryId) {
    await selectStory(uiState.selectedStoryId);
  } else {
    renderTrainingPanel();
  }
}

async function handleTrainingSetup() {
  if (!uiState.selectedStoryId) {
    return;
  }
  uiState.trainingJob = null;
  renderTrainingPanel();
  const job = await api(`/api/stories/${encodeURIComponent(uiState.selectedStoryId)}/training/setup`, {
    method: "POST",
    body: JSON.stringify({ force_reinstall: false }),
  });
  els.quickstartNote.textContent = "학습 환경 준비를 시작했습니다.";
  await pollTrainingJob(job.id);
}

async function handleTrainingStart() {
  if (!uiState.selectedStoryId) {
    return;
  }
  uiState.trainingJob = null;
  renderTrainingPanel();
  const job = await api(`/api/stories/${encodeURIComponent(uiState.selectedStoryId)}/training/auto`, {
    method: "POST",
    body: JSON.stringify({
      base_model: els.trainingBaseModel.value.trim() || "Qwen/Qwen2.5-3B-Instruct",
      hf_token: els.trainingHfToken.value.trim(),
      epochs: Number(els.trainingEpochs.value || 1.0),
      use_distillation: true,
      teacher_base_url: els.trainingTeacherBaseUrl.value.trim(),
      teacher_model: els.trainingTeacherModel.value || "google/gemma-4-E2B-it",
      teacher_coaching: true,
      teacher_variants_per_prompt: Number(els.trainingTeacherVariants.value || 1),
    }),
  });
  els.quickstartNote.textContent = "원클릭 학습을 시작했습니다. 완료되면 최신 학습 모델을 자동으로 연결합니다.";
  await pollTrainingJob(job.id);
  await loadTrainedAdapters();
  await Promise.all([loadSettings(), loadHealth(), loadModelCatalog()]);
}

async function handleUseTrainedModel() {
  if (!uiState.selectedStoryId) {
    return;
  }
  const adapterDir = els.trainedModelPicker.value;
  if (!adapterDir) {
    els.quickstartNote.textContent = "먼저 학습을 완료한 모델을 선택하세요.";
    return;
  }
  els.quickstartNote.textContent = "학습 모델 서버를 시작하고 생성 모델로 연결하는 중입니다. 첫 실행은 모델 로딩 때문에 시간이 걸릴 수 있습니다.";
  const result = await api("/api/training/adapters/use", {
    method: "POST",
    body: JSON.stringify({
      story_id: uiState.selectedStoryId,
      adapter_dir: adapterDir,
      port: 5001,
    }),
  });
  els.quickstartNote.textContent = result.detail || "학습 모델을 생성 모델로 연결했습니다.";
  await Promise.all([loadSettings(), loadHealth(), loadModelCatalog()]);
}

async function handleSaveSettings(event) {
  event.preventDefault();
  const form = new FormData(els.settingsForm);
  await api("/api/runtime-settings", {
    method: "PUT",
    body: JSON.stringify({
      provider: form.get("provider"),
      base_url: form.get("base_url"),
      model: form.get("model"),
      api_key: form.get("api_key"),
      timeout_seconds: Number(form.get("timeout_seconds") || 180),
      candidate_count: Number(form.get("candidate_count") || 3),
      temperature_planner: Number(form.get("temperature_planner") || 0.2),
      temperature_writer: Number(form.get("temperature_writer") || 0.85),
      temperature_critic: Number(form.get("temperature_critic") || 0.2),
      temperature_revision: Number(form.get("temperature_revision") || 0.4),
    }),
  });
  els.settingsResult.textContent = "설정을 저장했습니다.";
  await Promise.all([loadHealth(), loadModelCatalog()]);
}

async function handleTestSettings() {
  const form = new FormData(els.settingsForm);
  const result = await api("/api/runtime-settings/test", {
    method: "POST",
    body: JSON.stringify({
      provider: form.get("provider"),
      base_url: form.get("base_url"),
      model: form.get("model"),
      api_key: form.get("api_key"),
      timeout_seconds: Number(form.get("timeout_seconds") || 180),
      candidate_count: Number(form.get("candidate_count") || 3),
      temperature_planner: Number(form.get("temperature_planner") || 0.2),
      temperature_writer: Number(form.get("temperature_writer") || 0.85),
      temperature_critic: Number(form.get("temperature_critic") || 0.2),
      temperature_revision: Number(form.get("temperature_revision") || 0.4),
    }),
  });
  els.settingsResult.textContent = result.ok ? `연결 성공: ${result.detail}` : `연결 실패: ${result.detail}`;
  await loadHealth();
}

function setBusy(button, busyText, callback) {
  return async (...args) => {
    const previousText = button.textContent;
    button.disabled = true;
    button.textContent = busyText;
    try {
      await callback(...args);
    } catch (error) {
      console.error(error);
      els.quickstartNote.textContent = `오류: ${compactError(error)}`;
    } finally {
      button.disabled = false;
      button.textContent = previousText;
    }
  };
}

function bindEvents() {
  els.refreshBtn.addEventListener("click", async () => {
    await Promise.all([loadHealth(), loadSettings(), loadStories(), loadModelCatalog(), loadTrainingEnvironment()]);
    if (uiState.selectedStoryId) {
      await selectStory(uiState.selectedStoryId);
    }
  });
  els.quickstartBtn.addEventListener("click", setBusy(els.quickstartBtn, "생성 중...", handleQuickstart));
  els.autoConnectBtn.addEventListener("click", setBusy(els.autoConnectBtn, "연결 중...", handleAutoConnect));
  els.refreshModelsBtn.addEventListener("click", setBusy(els.refreshModelsBtn, "새로고침 중...", loadModelCatalog));
  els.continueBtn.addEventListener("click", setBusy(els.continueBtn, "작성 중...", handleContinue));
  els.exportBtn.addEventListener("click", setBusy(els.exportBtn, "내보내는 중...", handleExport));
  els.evaluateBtn.addEventListener("click", setBusy(els.evaluateBtn, "평가 중...", handleEvaluate));
  els.deleteStoryBtn.addEventListener("click", setBusy(els.deleteStoryBtn, "삭제 중...", handleDeleteStory));
  els.trainingSetupBtn.addEventListener("click", setBusy(els.trainingSetupBtn, "학습 환경 준비 중...", handleTrainingSetup));
  els.trainingStartBtn.addEventListener("click", setBusy(els.trainingStartBtn, "학습 시작 중...", handleTrainingStart));
  els.trainedRefreshBtn.addEventListener("click", setBusy(els.trainedRefreshBtn, "새로고침 중...", loadTrainedAdapters));
  els.trainedUseBtn.addEventListener("click", setBusy(els.trainedUseBtn, "연결 중...", handleUseTrainedModel));
  els.trainingTeacherBaseUrl.addEventListener("input", renderTrainingPanel);
  els.trainingTeacherModel.addEventListener("change", renderTrainingPanel);
  els.trainingTeacherVariants.addEventListener("input", renderTrainingPanel);
  els.modelPicker.addEventListener("change", async () => {
    els.modelPicker.disabled = true;
    try {
      await handleModelSelection();
    } finally {
      els.modelPicker.disabled = false;
    }
  });
  els.settingsForm.addEventListener("submit", handleSaveSettings);
  els.testSettingsBtn.addEventListener("click", setBusy(els.testSettingsBtn, "테스트 중...", handleTestSettings));
}

async function boot() {
  bindEvents();
  renderEmptyStoryState();
  await Promise.all([loadHealth(), loadSettings(), loadStories(), loadModelCatalog(), loadTrainingEnvironment()]);
}

boot().catch((error) => {
  console.error(error);
  alert(`초기화 실패: ${error.message}`);
});
