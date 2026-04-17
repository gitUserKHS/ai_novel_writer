const uiState = {
  stories: [],
  selectedStoryId: null,
  selectedSceneId: null,
  scenes: [],
  outline: [],
  artifacts: [],
  currentJobId: null,
  jobTimer: null,
};

const els = {
  healthPanel: document.getElementById("health-panel"),
  storyList: document.getElementById("story-list"),
  storyForm: document.getElementById("story-form"),
  clearStoryFormBtn: document.getElementById("clear-story-form-btn"),
  selectedStoryBadge: document.getElementById("selected-story-badge"),
  saveBibleBtn: document.getElementById("save-bible-btn"),
  bibleStaticFacts: document.getElementById("bible-static-facts"),
  bibleRules: document.getElementById("bible-rules"),
  bibleMotifs: document.getElementById("bible-motifs"),
  bibleVoiceNotes: document.getElementById("bible-voice-notes"),
  generateOutlineBtn: document.getElementById("generate-outline-btn"),
  outlineCount: document.getElementById("outline-count"),
  outlineList: document.getElementById("outline-list"),
  sceneForm: document.getElementById("scene-form"),
  generateSceneBtn: document.getElementById("generate-scene-btn"),
  jobLog: document.getElementById("job-log"),
  jobStatusPill: document.getElementById("job-status-pill"),
  sceneList: document.getElementById("scene-list"),
  sceneDetail: document.getElementById("scene-detail"),
  sceneCountPill: document.getElementById("scene-count-pill"),
  stateViewer: document.getElementById("state-viewer"),
  datasetViewer: document.getElementById("dataset-viewer"),
  kgViewer: document.getElementById("kg-viewer"),
  artifactList: document.getElementById("artifact-list"),
  refreshArtifactsBtn: document.getElementById("refresh-artifacts-btn"),
  refreshMemoryBtn: document.getElementById("refresh-memory-btn"),
  exportBtn: document.getElementById("export-btn"),
  evaluateBtn: document.getElementById("evaluate-btn"),
  settingsForm: document.getElementById("settings-form"),
  settingsResult: document.getElementById("settings-result"),
  refreshHealthBtn: document.getElementById("refresh-health-btn"),
  refreshStoriesBtn: document.getElementById("refresh-stories-btn"),
  testSettingsBtn: document.getElementById("test-settings-btn"),
  openNewStoryBtn: document.getElementById("open-new-story-btn"),
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

function splitCsv(value) {
  return String(value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function splitLines(value) {
  return String(value || "")
    .split(/\n+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function setStoryMode(mode) {
  els.storyForm.dataset.mode = mode;
}

function getStoryMode() {
  return els.storyForm.dataset.mode || "create";
}

function formToStoryPayload() {
  const form = new FormData(els.storyForm);
  return {
    title: form.get("title") || "Untitled Story",
    genre: form.get("genre") || "literary fiction",
    tone: form.get("tone") || "lyrical and emotionally grounded",
    themes: splitCsv(form.get("themes")),
    characters: splitCsv(form.get("characters")),
    forbidden_facts: splitCsv(form.get("forbidden_facts")),
    target_length_scenes: Number(form.get("target_length_scenes") || 12),
    premise: form.get("premise") || "",
    notes: form.get("notes") || "",
  };
}

function fillStoryForm(story) {
  els.storyForm.elements.title.value = story.title || "";
  els.storyForm.elements.genre.value = story.genre || "";
  els.storyForm.elements.tone.value = story.tone || "";
  els.storyForm.elements.themes.value = (story.themes || []).join(", ");
  els.storyForm.elements.characters.value = (story.characters || []).join(", ");
  els.storyForm.elements.forbidden_facts.value = (story.forbidden_facts || []).join(", ");
  els.storyForm.elements.target_length_scenes.value = story.target_length_scenes || 12;
  els.storyForm.elements.premise.value = story.premise || "";
  els.storyForm.elements.notes.value = story.notes || "";
}

function clearStoryForm() {
  els.storyForm.reset();
  els.storyForm.elements.target_length_scenes.value = 12;
  els.selectedStoryBadge.textContent = "새 스토리";
  setStoryMode("create");
}

function fillBible(bible) {
  els.bibleStaticFacts.value = (bible.static_facts || []).join("\n");
  els.bibleRules.value = (bible.rules || []).join("\n");
  els.bibleMotifs.value = (bible.motifs || []).join("\n");
  els.bibleVoiceNotes.value = (bible.voice_notes || []).join("\n");
}

function biblePayload() {
  return {
    static_facts: splitLines(els.bibleStaticFacts.value),
    rules: splitLines(els.bibleRules.value),
    motifs: splitLines(els.bibleMotifs.value),
    voice_notes: splitLines(els.bibleVoiceNotes.value),
    reference_snippets: [],
  };
}

function scenePayload() {
  const form = new FormData(els.sceneForm);
  return {
    title: form.get("title") || "",
    pov: form.get("pov") || "",
    goal: form.get("goal") || "",
    location: form.get("location") || "",
    time_label: form.get("time_label") || "",
    summary_request: form.get("summary_request") || "",
    beats: splitLines(form.get("beats")),
    must_include: splitCsv(form.get("must_include")),
    must_avoid: splitCsv(form.get("must_avoid")),
    emotion_targets: splitCsv(form.get("emotion_targets")),
    desired_length_words: Number(form.get("desired_length_words") || 900),
    outline_card_id: form.get("outline_card_id") || null,
  };
}

function fillSceneFormFromOutline(card) {
  els.sceneForm.elements.title.value = card.title || "";
  els.sceneForm.elements.pov.value = card.pov || "";
  els.sceneForm.elements.goal.value = card.goal || "";
  els.sceneForm.elements.location.value = card.location || "";
  els.sceneForm.elements.time_label.value = card.time_label || "";
  els.sceneForm.elements.summary_request.value = card.summary_request || "";
  els.sceneForm.elements.beats.value = (card.beats || []).join("\n");
  els.sceneForm.elements.must_include.value = (card.must_include || []).join(", ");
  els.sceneForm.elements.must_avoid.value = (card.must_avoid || []).join(", ");
  els.sceneForm.elements.outline_card_id.value = card.id || "";
}

function renderHealth(data) {
  const badge = data.backend_ok ? "🟢" : "🟡";
  els.healthPanel.innerHTML = `
    <div>${badge} <strong>${escapeHtml(data.status)}</strong></div>
    <div>provider: <strong>${escapeHtml(data.provider)}</strong></div>
    <div>model: <strong>${escapeHtml(data.model)}</strong></div>
    <div>detail: ${escapeHtml(data.detail || "")}</div>
  `;
}

function renderStories() {
  if (!uiState.stories.length) {
    els.storyList.innerHTML = `<div class="muted small-text">아직 스토리가 없어. 왼쪽 아래가 아니라 위쪽 메타 카드에서 새로 만들면 돼.</div>`;
    return;
  }
  els.storyList.innerHTML = uiState.stories
    .map(
      (story) => `
        <button class="story-item ${story.id === uiState.selectedStoryId ? "active" : ""}" data-story-id="${escapeHtml(story.id)}">
          <strong>${escapeHtml(story.title)}</strong>
          <div class="scene-meta">${escapeHtml(story.genre)} · ${escapeHtml(story.tone)}</div>
          <div class="scene-meta">scene 목표 ${story.target_length_scenes}개</div>
        </button>
      `,
    )
    .join("");
  els.storyList.querySelectorAll("[data-story-id]").forEach((button) => {
    button.addEventListener("click", () => selectStory(button.dataset.storyId));
  });
}

function renderOutline(cards) {
  uiState.outline = cards || [];
  if (!uiState.outline.length) {
    els.outlineList.innerHTML = `<div class="muted small-text">아직 outline이 없어. scene 수를 정하고 생성 버튼을 눌러줘.</div>`;
    return;
  }
  els.outlineList.innerHTML = uiState.outline
    .map(
      (card) => `
      <div class="outline-card">
        <div>
          <strong>${escapeHtml(card.title)}</strong>
          <div class="outline-meta">${escapeHtml(card.time_label)} · ${escapeHtml(card.location)} · ${escapeHtml(card.status)}</div>
        </div>
        <div>${escapeHtml(card.summary_request)}</div>
        <div class="outline-meta">POV ${escapeHtml(card.pov)} / Goal ${escapeHtml(card.goal)}</div>
        <button class="secondary-btn" data-outline-id="${escapeHtml(card.id || "")}">scene 폼에 불러오기</button>
      </div>
    `,
    )
    .join("");
  els.outlineList.querySelectorAll("[data-outline-id]").forEach((button) => {
    button.addEventListener("click", () => {
      const card = uiState.outline.find((item) => item.id === button.dataset.outlineId);
      if (card) fillSceneFormFromOutline(card);
    });
  });
}

function renderScenes() {
  els.sceneCountPill.textContent = `${uiState.scenes.length} scenes`;
  if (!uiState.scenes.length) {
    els.sceneList.innerHTML = `<div class="muted small-text">아직 생성된 scene이 없어.</div>`;
    els.sceneDetail.textContent = "scene 생성 후 상세 정보를 여기서 확인할 수 있어.";
    return;
  }
  els.sceneList.innerHTML = uiState.scenes
    .map(
      (scene) => `
      <button class="scene-card ${scene.id === uiState.selectedSceneId ? "active" : ""}" data-scene-id="${escapeHtml(scene.id)}">
        <strong>Scene ${scene.scene_index}: ${escapeHtml(scene.title)}</strong>
        <div class="scene-meta">${escapeHtml(scene.time_label)} · ${escapeHtml(scene.location)}</div>
        <div class="scene-meta">${escapeHtml(scene.summary)}</div>
      </button>
    `,
    )
    .join("");
  els.sceneList.querySelectorAll("[data-scene-id]").forEach((button) => {
    button.addEventListener("click", () => {
      uiState.selectedSceneId = button.dataset.sceneId;
      renderScenes();
      const scene = uiState.scenes.find((item) => item.id === uiState.selectedSceneId);
      renderSceneDetail(scene);
    });
  });
  if (!uiState.selectedSceneId) {
    uiState.selectedSceneId = uiState.scenes[uiState.scenes.length - 1].id;
  }
  const selected = uiState.scenes.find((item) => item.id === uiState.selectedSceneId) || uiState.scenes[uiState.scenes.length - 1];
  renderSceneDetail(selected);
}

function renderSceneDetail(scene) {
  if (!scene) {
    els.sceneDetail.textContent = "상세 정보가 없어.";
    return;
  }
  const candidateSections = (scene.candidates || [])
    .map(
      (cand) => `
      <details>
        <summary>Candidate ${cand.candidate_index} · score ${cand.score} ${cand.accepted ? "(accepted)" : ""}</summary>
        <pre class="code-box">${escapeHtml(cand.text)}</pre>
      </details>
    `,
    )
    .join("");
  els.sceneDetail.innerHTML = `
    <h3>Scene ${scene.scene_index}: ${escapeHtml(scene.title)}</h3>
    <p class="scene-meta">POV ${escapeHtml(scene.pov)} · ${escapeHtml(scene.time_label)} · ${escapeHtml(scene.location)}</p>
    <p class="scene-meta">Goal: ${escapeHtml(scene.goal)}</p>
    <h4>Accepted text</h4>
    <pre class="code-box">${escapeHtml(scene.accepted_text)}</pre>
    <h4>Plan</h4>
    <pre class="code-box">${escapeHtml(JSON.stringify(scene.plan, null, 2))}</pre>
    <h4>Consistency</h4>
    <pre class="code-box">${escapeHtml(JSON.stringify(scene.consistency, null, 2))}</pre>
    <h4>Creativity</h4>
    <pre class="code-box">${escapeHtml(JSON.stringify(scene.creativity, null, 2))}</pre>
    <h4>Revision</h4>
    <pre class="code-box">${escapeHtml(JSON.stringify(scene.revision, null, 2))}</pre>
    <h4>Candidates</h4>
    ${candidateSections || `<div class="muted">없음</div>`}
  `;
}

function renderArtifacts(items) {
  uiState.artifacts = items || [];
  if (!uiState.artifacts.length) {
    els.artifactList.innerHTML = `<div class="muted small-text">아직 생성된 artifact가 없어.</div>`;
    return;
  }
  els.artifactList.innerHTML = uiState.artifacts
    .map(
      (artifact) => `
        <div class="artifact-item">
          <strong>${escapeHtml(artifact.artifact_type)}</strong>
          <div class="artifact-meta">${escapeHtml(artifact.created_at)} · ${escapeHtml(artifact.path)}</div>
          <a href="/api/stories/${encodeURIComponent(uiState.selectedStoryId)}/artifacts/${artifact.id}/download">다운로드</a>
        </div>
      `,
    )
    .join("");
}

function renderJob(job) {
  els.jobStatusPill.textContent = `${job.status} · ${(job.progress * 100).toFixed(0)}%`;
  const lines = (job.logs || []).map((entry) => `[${entry.time || ""}] ${entry.message}`).join("\n");
  els.jobLog.textContent = lines || job.message || "대기 중";
}

function updateStoryDependentButtons(enabled) {
  [
    els.saveBibleBtn,
    els.generateOutlineBtn,
    els.generateSceneBtn,
    els.refreshMemoryBtn,
    els.refreshArtifactsBtn,
    els.exportBtn,
    els.evaluateBtn,
  ].forEach((button) => {
    button.disabled = !enabled;
  });
}

async function loadHealth() {
  const data = await api("/api/health");
  renderHealth(data);
}

async function loadSettings() {
  const settings = await api("/api/runtime-settings");
  for (const [key, value] of Object.entries(settings)) {
    if (els.settingsForm.elements[key]) {
      els.settingsForm.elements[key].value = value;
    }
  }
}

async function loadStories() {
  const data = await api("/api/stories");
  uiState.stories = data.items || [];
  if (uiState.selectedStoryId && !uiState.stories.find((s) => s.id === uiState.selectedStoryId)) {
    uiState.selectedStoryId = null;
  }
  renderStories();
}

async function selectStory(storyId) {
  uiState.selectedStoryId = storyId;
  uiState.selectedSceneId = null;
  renderStories();
  updateStoryDependentButtons(true);
  const detail = await api(`/api/stories/${encodeURIComponent(storyId)}`);
  fillStoryForm(detail.story);
  fillBible(detail.bible);
  setStoryMode("update");
  els.selectedStoryBadge.textContent = storyId;
  renderOutline(detail.outline || []);
  await reloadStoryResources();
}

async function reloadStoryResources() {
  if (!uiState.selectedStoryId) return;
  const storyId = encodeURIComponent(uiState.selectedStoryId);
  const [scenesRes, stateRes, datasetsRes, kgRes, artifactsRes] = await Promise.all([
    api(`/api/stories/${storyId}/scenes`),
    api(`/api/stories/${storyId}/state`),
    api(`/api/stories/${storyId}/datasets`),
    api(`/api/stories/${storyId}/kg`),
    api(`/api/stories/${storyId}/artifacts`),
  ]);
  uiState.scenes = scenesRes.items || [];
  renderScenes();
  els.stateViewer.textContent = JSON.stringify(stateRes, null, 2);
  els.datasetViewer.textContent = JSON.stringify(datasetsRes, null, 2);
  els.kgViewer.textContent = JSON.stringify(kgRes.items || [], null, 2);
  renderArtifacts(artifactsRes.items || []);
}

async function handleStorySubmit(event) {
  event.preventDefault();
  const payload = formToStoryPayload();
  if (!payload.title || !payload.premise) {
    alert("제목과 프레미스는 넣어줘.");
    return;
  }
  let story;
  if (getStoryMode() === "update" && uiState.selectedStoryId) {
    story = await api(`/api/stories/${encodeURIComponent(uiState.selectedStoryId)}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
  } else {
    story = await api("/api/stories", { method: "POST", body: JSON.stringify(payload) });
  }
  await loadStories();
  await selectStory(story.id);
}

async function handleSaveBible() {
  if (!uiState.selectedStoryId) return;
  await api(`/api/stories/${encodeURIComponent(uiState.selectedStoryId)}/bible`, {
    method: "PUT",
    body: JSON.stringify(biblePayload()),
  });
  await selectStory(uiState.selectedStoryId);
}

async function handleGenerateOutline() {
  if (!uiState.selectedStoryId) return;
  const count = Number(els.outlineCount.value || 6);
  const res = await api(`/api/stories/${encodeURIComponent(uiState.selectedStoryId)}/outline/generate`, {
    method: "POST",
    body: JSON.stringify({ scene_count: count }),
  });
  renderOutline(res.items || []);
}

async function handleGenerateScene() {
  if (!uiState.selectedStoryId) return;
  const payload = scenePayload();
  if (!payload.pov || !payload.goal || !payload.location || !payload.time_label || !payload.summary_request) {
    alert("POV, Goal, Location, Time label, 장면 요청은 꼭 넣어줘.");
    return;
  }
  const job = await api(`/api/stories/${encodeURIComponent(uiState.selectedStoryId)}/scenes/generate`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  uiState.currentJobId = job.id;
  renderJob(job);
  startJobPolling();
}

function stopJobPolling() {
  if (uiState.jobTimer) {
    clearInterval(uiState.jobTimer);
    uiState.jobTimer = null;
  }
}

function startJobPolling() {
  stopJobPolling();
  uiState.jobTimer = setInterval(async () => {
    if (!uiState.currentJobId) return;
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(uiState.currentJobId)}`);
      renderJob(job);
      if (job.status === "succeeded") {
        stopJobPolling();
        await reloadStoryResources();
      } else if (job.status === "failed") {
        stopJobPolling();
        alert("scene 생성이 실패했어. 로그를 확인해줘.");
      }
    } catch (error) {
      stopJobPolling();
      console.error(error);
    }
  }, 1200);
}

async function handleExport() {
  if (!uiState.selectedStoryId) return;
  await api(`/api/stories/${encodeURIComponent(uiState.selectedStoryId)}/export`, { method: "POST" });
  await reloadStoryResources();
}

async function handleEvaluate() {
  if (!uiState.selectedStoryId) return;
  const res = await api(`/api/stories/${encodeURIComponent(uiState.selectedStoryId)}/evaluate`, { method: "POST" });
  await reloadStoryResources();
  els.datasetViewer.textContent += `\n\nLatest evaluation:\n${JSON.stringify(res.report, null, 2)}`;
}

async function handleSaveSettings(event) {
  event.preventDefault();
  const form = new FormData(els.settingsForm);
  const payload = {
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
  };
  await api("/api/runtime-settings", { method: "PUT", body: JSON.stringify(payload) });
  els.settingsResult.textContent = "저장됨";
  await loadHealth();
}

async function handleTestSettings() {
  const form = new FormData(els.settingsForm);
  const payload = {
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
  };
  const res = await api("/api/runtime-settings/test", { method: "POST", body: JSON.stringify(payload) });
  els.settingsResult.textContent = res.ok ? `연결 성공: ${res.detail}` : `연결 실패: ${res.detail}`;
  await loadHealth();
}

function bindEvents() {
  els.storyForm.addEventListener("submit", handleStorySubmit);
  els.clearStoryFormBtn.addEventListener("click", () => {
    uiState.selectedStoryId = null;
    uiState.selectedSceneId = null;
    clearStoryForm();
    renderStories();
    updateStoryDependentButtons(false);
  });
  els.openNewStoryBtn.addEventListener("click", () => {
    clearStoryForm();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
  els.saveBibleBtn.addEventListener("click", handleSaveBible);
  els.generateOutlineBtn.addEventListener("click", handleGenerateOutline);
  els.generateSceneBtn.addEventListener("click", handleGenerateScene);
  els.exportBtn.addEventListener("click", handleExport);
  els.evaluateBtn.addEventListener("click", handleEvaluate);
  els.refreshArtifactsBtn.addEventListener("click", reloadStoryResources);
  els.refreshMemoryBtn.addEventListener("click", reloadStoryResources);
  els.settingsForm.addEventListener("submit", handleSaveSettings);
  els.refreshHealthBtn.addEventListener("click", loadHealth);
  els.refreshStoriesBtn.addEventListener("click", loadStories);
  els.testSettingsBtn.addEventListener("click", handleTestSettings);
}

async function boot() {
  bindEvents();
  clearStoryForm();
  updateStoryDependentButtons(false);
  await Promise.all([loadHealth(), loadSettings(), loadStories()]);
}

boot().catch((error) => {
  console.error(error);
  alert(`초기화 실패: ${error.message}`);
});
