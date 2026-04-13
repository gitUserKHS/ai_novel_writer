const state = {
  selectedStoryId: null,
  stories: [],
  jobs: [],
  currentSceneId: null,
  uiPresets: {},
};

const storyTemplates = {
  "moon-theater": {
    title: "달빛 극장",
    genre: "mystery fantasy",
    tone: "lyrical and tense",
    premise: "폐쇄된 극장에 남겨진 은색 실 조각을 따라가며, 서윤은 사라진 동생이 남긴 마지막 동선을 장면 단위로 복원해야 한다.",
    themes: ["기억", "선택", "상실"],
    characters: ["서윤", "민호", "유진"],
    constraints: ["시간 이동 없음", "죽은 인물 부활 없음", "각 장면은 실마리를 하나 이상 전진시킬 것"],
    target_scene_count: 6,
    target_word_count: 8000,
  },
  "glass-harbor": {
    title: "유리 항만",
    genre: "science fiction thriller",
    tone: "precise and cold",
    premise: "수중 항만의 검열 기록을 훔친 하린은 감시망보다 먼저 진실에 도달해야 하며, 각 장면은 추적과 은폐의 균형을 유지해야 한다.",
    themes: ["감시", "정체성", "기록"],
    characters: ["하린", "도윤", "세라"],
    constraints: ["설정 위반 금지", "기술 정보는 장면 목적과 연결", "씬마다 긴장 상승 유지"],
    target_scene_count: 6,
    target_word_count: 7800,
  },
  "red-archive": {
    title: "붉은 기록 보관소",
    genre: "historical mystery",
    tone: "warm and melancholic",
    premise: "봉인된 기록 보관소에서 삭제된 이름을 되살리려는 archivist가 권력과 가족사를 동시에 마주한다.",
    themes: ["기록", "충성", "복원"],
    characters: ["윤서", "재겸", "명하"],
    constraints: ["시대 배경 유지", "초현실 전개 금지", "감정선은 누적식으로 전개"],
    target_scene_count: 5,
    target_word_count: 7600,
  },
};

const runtimePresets = {
  mock: {
    provider: "mock",
    base_url: "http://127.0.0.1:8080/v1",
    api_key: "not-needed",
    model: "mock-story-engine",
    temperature: 0.9,
    critic_temperature: 0.2,
    max_tokens: 2048,
    cache_responses: false,
    role_models: {},
  },
  "qwen-native": {
    provider: "ollama",
    base_url: "http://127.0.0.1:11434",
    api_key: "ollama",
    model: "qwen3:4b",
    temperature: 0.7,
    critic_temperature: 0.0,
    max_tokens: 1200,
    cache_responses: true,
    role_models: {},
  },
  "qwen-local": {
    provider: "ollama",
    base_url: "http://127.0.0.1:11434",
    api_key: "ollama",
    model: "qwen3:4b",
    temperature: 0.7,
    critic_temperature: 0.0,
    max_tokens: 1200,
    cache_responses: true,
    role_models: {
      planner: "qwen3:4b",
      writer: "qwen3:4b",
      consistency_critic: "outputs/training_qwen3_4b_critic_consistency",
      creativity_critic: "qwen3:4b",
      world_model: "outputs/training_qwen3_4b_world_model",
      revision: "qwen3:4b",
      extractor: "qwen3:4b",
    },
  },
  "qwen-strict": {
    provider: "ollama",
    base_url: "http://127.0.0.1:11434",
    api_key: "ollama",
    model: "qwen3:4b",
    temperature: 0.65,
    critic_temperature: 0.0,
    max_tokens: 1200,
    cache_responses: true,
    role_models: {
      planner: "qwen3:4b",
      writer: "qwen3:4b",
      consistency_critic: "outputs/training_qwen3_4b_critic_consistency",
      creativity_critic: "qwen3:4b",
      world_model: "outputs/training_qwen3_4b_world_model",
      revision: "qwen3:4b",
      extractor: "qwen3:4b",
    },
  },
};

const presetKinds = {
  runtime: {
    selectId: "runtime-preset-select",
    label: "runtime",
    collect: () => collectRuntimePayload(document.getElementById("runtime-form")),
    apply: (payload) => fillRuntimeForm(payload),
  },
  one_click: {
    selectId: "one-click-preset-select",
    label: "one-click",
    collect: () => collectOneClickPayload(document.getElementById("one-click-form")),
    apply: (payload) => fillOneClickForm(payload),
  },
  generalist: {
    selectId: "generalist-preset-select",
    label: "generalist",
    collect: () => collectGeneralistPayload(document.getElementById("generalist-form")),
    apply: (payload) => fillGeneralistForm(payload),
  },
  training: {
    selectId: "training-preset-select",
    label: "training",
    collect: () => collectTrainingPayload(document.getElementById("training-form")),
    apply: (payload) => fillTrainingForm(payload),
  },
  hf_publish: {
    selectId: "hf-publish-preset-select",
    label: "hf-publish",
    collect: () => collectHFPublishPayload(document.getElementById("hf-publish-form")),
    apply: (payload) => fillHFPublishForm(payload),
  },
  hf_pull: {
    selectId: "hf-pull-preset-select",
    label: "hf-pull",
    collect: () => collectHFPullPayload(document.getElementById("hf-pull-form")),
    apply: (payload) => fillHFPullForm(payload),
  },
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function shortText(text, limit = 180) {
  const normalized = String(text ?? "").replace(/\s+/g, " ").trim();
  if (!normalized) return "";
  return normalized.length > limit ? `${normalized.slice(0, limit - 1)}…` : normalized;
}

function formatScore(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return value.toFixed(3);
}

function toPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "0%";
  return `${Math.round(value * 100)}%`;
}

function numberOrNull(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return null;
  }
  return numeric;
}

function formatDateTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

function basename(value) {
  const parts = String(value ?? "").split(/[\\/]/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : String(value ?? "");
}

function logLine(message, tone = "info") {
  const panel = document.getElementById("log-panel");
  const row = document.createElement("div");
  row.className = "log-line";
  row.dataset.tone = tone;
  row.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  panel.prepend(row);
}

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  const res = await fetch(path, { ...options, headers });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return await res.json();
  }
  return await res.text();
}

function lines(text) {
  return String(text ?? "")
    .split(/\n+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function setStatusPill(element, text, tone = "neutral") {
  element.textContent = text;
  element.className = `status-pill ${tone}`;
}

function setSelectedTab(tabName) {
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${tabName}`);
  });
}

function ensureStorySelected() {
  if (!state.selectedStoryId) {
    throw new Error("먼저 스토리를 선택하거나 생성하세요.");
  }
}

function renderEmpty(container, message) {
  container.innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
}

function fillCreateStoryForm(payload) {
  const form = document.getElementById("create-story-form");
  form.title.value = payload.title || "";
  form.genre.value = payload.genre || "";
  form.tone.value = payload.tone || "";
  form.premise.value = payload.premise || "";
  form.themes.value = (payload.themes || []).join("\n");
  form.characters.value = (payload.characters || []).join("\n");
  form.constraints.value = (payload.constraints || []).join("\n");
  form.target_scene_count.value = String(payload.target_scene_count || 6);
  form.target_word_count.value = String(payload.target_word_count || 8000);
}

function applyStoryTemplate(templateKey) {
  const template = storyTemplates[templateKey];
  if (!template) return;
  fillCreateStoryForm(template);
  logLine(`샘플 템플릿 적용: ${template.title}`);
}

function fillRuntimeForm(payload) {
  const form = document.getElementById("runtime-form");
  form.provider.value = payload.provider || "mock";
  form.base_url.value = payload.base_url || "";
  form.api_key.value = payload.api_key || "";
  form.model.value = payload.model || "";
  form.temperature.value = String(payload.temperature ?? 0.9);
  form.critic_temperature.value = String(payload.critic_temperature ?? 0.2);
  form.max_tokens.value = String(payload.max_tokens ?? 2048);
  form.cache_responses.checked = !!payload.cache_responses;
  form.role_models_json.value = JSON.stringify(payload.role_models || {}, null, 2);
}

function applyRuntimePreset(presetKey) {
  const preset = runtimePresets[presetKey];
  if (!preset) return;
  fillRuntimeForm(preset);
  document.getElementById("runtime-result").textContent = JSON.stringify(preset, null, 2);
  logLine(`runtime preset 적용: ${presetKey}`);
}

function fillOneClickForm(payload) {
  const form = document.getElementById("one-click-form");
  form.preset.value = payload.preset || "qwen-native";
  form.mode.value = payload.mode || "smoke";
  form.story_file.value = payload.story_file || "examples/story.yaml";
  form.scene_file.value = payload.scene_file || "examples/scene_smoke.yaml";
  form.train_action.value = payload.train_action || "skip";
  form.train_preset.value = payload.train_preset || "none";
  form.scene_limit.value = payload.scene_limit || 0;
  form.run_tests.checked = !!payload.run_tests;
}

function fillGeneralistForm(payload) {
  const form = document.getElementById("generalist-form");
  form.preset.value = payload.preset || "qwen-loop";
  form.mode.value = payload.mode || "smoke";
  form.story_dir.value = payload.story_dir || "examples/story_pack";
  form.story_offset.value = payload.story_offset || 0;
  form.story_limit.value = payload.story_limit || 0;
  form.scene_limit.value = payload.scene_limit || 0;
  form.train_action.value = payload.train_action || "skip";
  form.train_preset.value = payload.train_preset || "none";
  form.resume.checked = payload.resume !== false;
}

function fillTrainingForm(payload) {
  const form = document.getElementById("training-form");
  form.config.value = payload.config || "configs/training_qwen3_4b_sft_smoke.yaml";
  form.train_file.value = payload.train_file || "";
  form.eval_file.value = payload.eval_file || "";
  form.output_dir.value = payload.output_dir || "";
  form.model_name_or_path.value = payload.model_name_or_path || "";
  form.dry_run.checked = !!payload.dry_run;
  form.print_config.checked = !!payload.print_config;
}

function fillHFPublishForm(payload) {
  const form = document.getElementById("hf-publish-form");
  form.source_dir.value = payload.source_dir || "outputs/training_qwen3_4b_sft";
  form.repo_id.value = payload.repo_id || "";
  form.namespace.value = payload.namespace || "";
  form.project.value = payload.project || "conarrative";
  form.role.value = payload.role || "";
  form.base_model.value = payload.base_model || "";
  form.stage.value = payload.stage || "";
  form.repo_type.value = payload.repo_type || "model";
  form.path_in_repo.value = payload.path_in_repo || "";
  form.revision.value = payload.revision || "";
  form.commit_message.value = payload.commit_message || "";
  form.release_tag.value = payload.release_tag || "";
  form.release_prefix.value = payload.release_prefix || "v";
  form.bump.value = payload.bump || "patch";
  form.tag_message.value = payload.tag_message || "";
  form.private.checked = !!payload.private;
  form.exclude_checkpoints.checked = payload.exclude_checkpoints !== false;
  form.auto_tag.checked = payload.auto_tag !== false;
  form.ignore_patterns.value = (payload.ignore_patterns || []).join("\n");
}

function fillHFPullForm(payload) {
  const form = document.getElementById("hf-pull-form");
  form.repo_id.value = payload.repo_id || "";
  form.repo_type.value = payload.repo_type || "model";
  form.local_dir.value = payload.local_dir || "outputs/hf_download";
  form.revision.value = payload.revision || "";
  form.allow_patterns.value = (payload.allow_patterns || []).join("\n");
  form.ignore_patterns.value = (payload.ignore_patterns || []).join("\n");
}

function collectRuntimePayload(form) {
  let roleModels = {};
  try {
    roleModels = JSON.parse(form.role_models_json.value || "{}");
  } catch (error) {
    throw new Error("Role models JSON 형식이 올바르지 않습니다.");
  }
  return {
    provider: form.provider.value,
    base_url: form.base_url.value,
    api_key: form.api_key.value,
    model: form.model.value,
    temperature: Number(form.temperature.value || 0),
    critic_temperature: Number(form.critic_temperature.value || 0),
    max_tokens: Number(form.max_tokens.value || 0),
    cache_responses: !!form.cache_responses.checked,
    role_models: roleModels,
    timeout_seconds: 120,
    extra_headers: {},
    cache_dir: "workspace/cache",
  };
}

function collectOneClickPayload(form) {
  return {
    preset: form.preset.value,
    mode: form.mode.value,
    train_action: form.train_action.value,
    train_preset: form.train_preset.value,
    story_file: form.story_file.value,
    scene_file: form.scene_file.value,
    scene_limit: numberOrNull(form.scene_limit.value),
    run_tests: !!form.run_tests.checked,
  };
}

function collectGeneralistPayload(form) {
  return {
    preset: form.preset.value,
    mode: form.mode.value,
    train_action: form.train_action.value,
    train_preset: form.train_preset.value,
    story_dir: form.story_dir.value,
    story_offset: Number(form.story_offset.value || 0),
    story_limit: numberOrNull(form.story_limit.value),
    scene_limit: numberOrNull(form.scene_limit.value),
    resume: !!form.resume.checked,
  };
}

function collectTrainingPayload(form) {
  return {
    config: form.config.value,
    train_file: form.train_file.value || null,
    eval_file: form.eval_file.value || null,
    output_dir: form.output_dir.value || null,
    model_name_or_path: form.model_name_or_path.value || null,
    dry_run: !!form.dry_run.checked,
    print_config: !!form.print_config.checked,
  };
}

function collectHFPublishPayload(form) {
  return {
    source_dir: form.source_dir.value,
    repo_id: form.repo_id.value,
    namespace: form.namespace.value || "",
    project: form.project.value || "conarrative",
    role: form.role.value || "",
    base_model: form.base_model.value || "",
    stage: form.stage.value || "",
    repo_type: form.repo_type.value,
    path_in_repo: form.path_in_repo.value || "",
    revision: form.revision.value || null,
    commit_message: form.commit_message.value || null,
    release_tag: form.release_tag.value || "",
    release_prefix: form.release_prefix.value || "v",
    bump: form.bump.value || "patch",
    tag_message: form.tag_message.value || "",
    private: !!form.private.checked,
    exclude_checkpoints: !!form.exclude_checkpoints.checked,
    auto_tag: !!form.auto_tag.checked,
    ignore_patterns: lines(form.ignore_patterns.value),
  };
}

function collectHFPullPayload(form) {
  return {
    repo_id: form.repo_id.value,
    repo_type: form.repo_type.value,
    local_dir: form.local_dir.value,
    revision: form.revision.value || null,
    allow_patterns: lines(form.allow_patterns.value),
    ignore_patterns: lines(form.ignore_patterns.value),
  };
}

function readRuntimeRoleModels() {
  const textarea = document.getElementById("runtime-form").role_models_json;
  try {
    return JSON.parse(textarea.value || "{}");
  } catch (error) {
    throw new Error("Runtime role_models JSON 형식이 올바르지 않습니다.");
  }
}

function writeRuntimeRoleModels(payload) {
  const textarea = document.getElementById("runtime-form").role_models_json;
  textarea.value = JSON.stringify(payload || {}, null, 2);
}

function bindRuntimeRoleModel(role, repoId) {
  const roleModels = readRuntimeRoleModels();
  roleModels[role] = repoId;
  writeRuntimeRoleModels(roleModels);
  setSelectedTab("runtime");
  logLine(`runtime role 연결: ${role} -> ${repoId}`);
}

function presetOptions(kind) {
  return state.uiPresets[kind] || [];
}

function presetSelect(kind) {
  return document.getElementById(presetKinds[kind].selectId);
}

function selectedPresetName(kind) {
  return presetSelect(kind).value || "";
}

function getPresetRecord(kind, name) {
  return presetOptions(kind).find((item) => item.name === name) || null;
}

function refreshPresetSelect(kind) {
  const select = presetSelect(kind);
  const currentValue = select.value;
  const options = presetOptions(kind);
  select.innerHTML = `<option value="">저장된 preset 선택</option>`;
  options.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.name;
    option.textContent = `${item.name} · ${formatDateTime(item.saved_at)}`;
    select.appendChild(option);
  });
  if (options.some((item) => item.name === currentValue)) {
    select.value = currentValue;
  }
}

async function refreshUiPresets() {
  const data = await api("/api/ui-presets");
  state.uiPresets = data.items || {};
  Object.keys(presetKinds).forEach((kind) => refreshPresetSelect(kind));
}

async function loadSavedPreset(kind) {
  const name = selectedPresetName(kind);
  if (!name) {
    throw new Error("먼저 저장된 preset을 선택하세요.");
  }
  const record = getPresetRecord(kind, name);
  if (!record) {
    throw new Error("선택한 preset을 찾지 못했습니다.");
  }
  presetKinds[kind].apply(record.payload || {});
  logLine(`${presetKinds[kind].label} preset 불러오기: ${name}`);
}

async function saveCurrentPreset(kind) {
  const name = window.prompt(`${presetKinds[kind].label} preset 이름`, selectedPresetName(kind) || "");
  if (!name) {
    return;
  }
  const payload = presetKinds[kind].collect();
  await api("/api/ui-presets", {
    method: "POST",
    body: JSON.stringify({ kind, name, payload }),
  });
  await refreshUiPresets();
  presetSelect(kind).value = name;
  logLine(`${presetKinds[kind].label} preset 저장: ${name}`);
}

async function deleteSavedPreset(kind) {
  const name = selectedPresetName(kind);
  if (!name) {
    throw new Error("삭제할 preset을 먼저 선택하세요.");
  }
  if (!window.confirm(`preset '${name}' 을(를) 삭제할까요?`)) {
    return;
  }
  await api(`/api/ui-presets/${encodeURIComponent(kind)}/${encodeURIComponent(name)}`, { method: "DELETE" });
  await refreshUiPresets();
  logLine(`${presetKinds[kind].label} preset 삭제: ${name}`);
}

function renderStories() {
  const container = document.getElementById("story-list");
  container.innerHTML = "";

  if (!state.stories.length) {
    renderEmpty(container, "아직 스토리가 없습니다. 왼쪽 폼에서 하나 만들어 보세요.");
    return;
  }

  state.stories.forEach((story) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `story-card ${state.selectedStoryId === story.id ? "active" : ""}`;
    button.innerHTML = `
      <strong>${escapeHtml(story.title)}</strong>
      <div class="meta-row">
        <span class="badge">${escapeHtml(story.genre)}</span>
        <span class="badge">${escapeHtml(story.status)}</span>
        <span class="badge">target ${escapeHtml(story.target_scene_count)}</span>
      </div>
      <p class="muted">${escapeHtml(shortText(story.premise, 120))}</p>
    `;
    button.addEventListener("click", async () => {
      state.selectedStoryId = story.id;
      state.currentSceneId = null;
      renderStories();
      await refreshStory();
    });
    container.appendChild(button);
  });
}

async function loadStories() {
  const data = await api("/api/stories");
  state.stories = data.items || [];
  if (!state.selectedStoryId && state.stories.length) {
    state.selectedStoryId = state.stories[0].id;
  }
  if (state.selectedStoryId && !state.stories.some((item) => item.id === state.selectedStoryId)) {
    state.selectedStoryId = state.stories.length ? state.stories[0].id : null;
  }
  document.getElementById("story-count-pill").textContent = `stories · ${state.stories.length}`;
  renderStories();
  if (state.selectedStoryId) {
    await refreshStory();
  }
}

async function refreshHealth() {
  const data = await api("/api/health");
  const tone = data.status === "ok" ? "ok" : "warn";
  setStatusPill(document.getElementById("health-pill"), `health · ${data.status}`, tone);
  setStatusPill(document.getElementById("runtime-pill"), `${data.provider} · ${data.model}`, "neutral");
}

async function refreshRuntime() {
  const data = await api("/api/runtime-settings");
  fillRuntimeForm(data);
  document.getElementById("runtime-result").textContent = JSON.stringify(data, null, 2);
}

function renderStoryStats(detail, scenes, artifacts) {
  const sceneCount = scenes.length;
  const avgConsistency = sceneCount
    ? scenes.reduce((sum, scene) => sum + Number(scene.consistency?.score || 0), 0) / sceneCount
    : null;
  const avgWorld = sceneCount
    ? scenes.reduce((sum, scene) => sum + Number(scene.consistency?.world_plausibility_score || 0), 0) / sceneCount
    : null;

  const cards = [
    { label: "Scenes", value: `${sceneCount}/${detail.story.target_scene_count}` },
    { label: "Consistency", value: formatScore(avgConsistency) },
    { label: "World Plausibility", value: formatScore(avgWorld) },
    { label: "Artifacts", value: String(artifacts.length) },
  ];

  const container = document.getElementById("story-stat-grid");
  container.innerHTML = cards
    .map(
      (card) => `
        <article class="stat-card">
          <span class="stat-label">${escapeHtml(card.label)}</span>
          <strong class="stat-value">${escapeHtml(card.value)}</strong>
        </article>
      `,
    )
    .join("");
}

function renderOverview(detail, outlineItems, scenes, artifacts, datasetCounts) {
  document.getElementById("story-title").textContent = detail.story.title;
  document.getElementById("story-meta").textContent =
    `${detail.story.genre} · ${detail.story.tone} · ${detail.story.status} · active threads ${detail.state.active_threads.length}`;

  renderStoryStats(detail, scenes, artifacts);

  const premisePanel = document.getElementById("overview-premise");
  premisePanel.innerHTML = `
    <p>${escapeHtml(detail.story.premise)}</p>
    <div class="meta-row">
      <span class="badge">themes · ${escapeHtml((detail.story.themes || []).join(", "))}</span>
      <span class="badge">characters · ${escapeHtml((detail.story.characters || []).join(", "))}</span>
      <span class="badge">location · ${escapeHtml(detail.state.current_location || "-")}</span>
      <span class="badge">time · ${escapeHtml(detail.state.current_time_label || "-")}</span>
    </div>
  `;

  const outlineDone = outlineItems.filter((item) => item.status === "done").length;
  const progress = outlineItems.length ? outlineDone / outlineItems.length : 0;
  const outlinePanel = document.getElementById("outline-progress-panel");
  if (!outlineItems.length) {
    renderEmpty(outlinePanel, "아직 outline이 없습니다. 상단의 Outline 생성 버튼을 누르세요.");
  } else {
    outlinePanel.innerHTML = `
      <div class="outline-progress-card">
        <strong>${outlineDone}/${outlineItems.length} scenes completed</strong>
        <div class="progress-track"><span class="progress-fill" style="width:${toPercent(progress)}"></span></div>
      </div>
      ${outlineItems
        .map(
          (card) => `
            <div class="outline-card">
              <strong>${escapeHtml(card.scene_index)}. ${escapeHtml(card.title)}</strong>
              <div class="meta-row">
                <span class="badge">${escapeHtml(card.pov)}</span>
                <span class="badge">${escapeHtml(card.location)}</span>
                <span class="badge">${escapeHtml(card.status)}</span>
              </div>
              <p class="muted">${escapeHtml(shortText(card.goal, 120))}</p>
            </div>
          `,
        )
        .join("")}
    `;
  }

  const recentPanel = document.getElementById("recent-scenes-panel");
  if (!scenes.length) {
    renderEmpty(recentPanel, "아직 생성된 장면이 없습니다.");
  } else {
    recentPanel.innerHTML = scenes
      .slice(-4)
      .reverse()
      .map(
        (scene) => `
          <article class="recent-scene-card">
            <strong>${escapeHtml(scene.scene_index)}. ${escapeHtml(scene.title)}</strong>
            <div class="meta-row">
              <span class="badge">consistency ${formatScore(Number(scene.consistency?.score || 0))}</span>
              <span class="badge">world ${formatScore(Number(scene.consistency?.world_plausibility_score || 0))}</span>
            </div>
            <p>${escapeHtml(scene.summary || shortText(scene.accepted_text, 160))}</p>
          </article>
        `,
      )
      .join("");
  }

  document.getElementById("dataset-panel").textContent = JSON.stringify(datasetCounts || {}, null, 2);
}

function renderBible(bible) {
  const form = document.getElementById("bible-form");
  form.static_facts.value = (bible.static_facts || []).join("\n");
  form.rules.value = (bible.rules || []).join("\n");
  form.forbidden.value = (bible.forbidden || []).join("\n");
  form.motifs.value = (bible.motifs || []).join("\n");
}

function renderScenes(items) {
  const list = document.getElementById("scene-list");
  list.innerHTML = "";

  if (!items.length) {
    renderEmpty(list, "아직 scene이 없습니다. 다음 Scene 생성 또는 Auto Novel을 실행하세요.");
    return;
  }

  items.forEach((scene) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "scene-card";
    button.innerHTML = `
      <strong>${escapeHtml(scene.scene_index)}. ${escapeHtml(scene.title)}</strong>
      <div class="meta-row">
        <span class="badge">${escapeHtml(scene.location || "-")}</span>
        <span class="badge">${escapeHtml(scene.time_label || "-")}</span>
        <span class="badge">score ${formatScore(Number(scene.consistency?.score || 0))}</span>
      </div>
      <p>${escapeHtml(scene.summary || shortText(scene.accepted_text, 140))}</p>
    `;
    button.addEventListener("click", async () => {
      state.currentSceneId = scene.id;
      await openSceneDetail(scene.id);
    });
    list.appendChild(button);
  });
}

function renderSceneDetail(detail) {
  document.getElementById("scene-detail-summary").innerHTML = `
    <strong>${escapeHtml(detail.scene_index)}. ${escapeHtml(detail.title)}</strong>
    <div class="meta-row">
      <span class="badge">${escapeHtml(detail.pov || "-")}</span>
      <span class="badge">${escapeHtml(detail.location || "-")}</span>
      <span class="badge">${escapeHtml(detail.time_label || "-")}</span>
      <span class="badge">consistency ${formatScore(Number(detail.consistency?.score || 0))}</span>
      <span class="badge">world ${formatScore(Number(detail.consistency?.world_plausibility_score || 0))}</span>
    </div>
    <p>${escapeHtml(detail.accepted_text || "")}</p>
  `;

  const candidatePanel = document.getElementById("scene-candidates");
  const candidates = detail.candidates || [];
  if (!candidates.length) {
    renderEmpty(candidatePanel, "이 장면에는 저장된 후보 정보가 없습니다.");
  } else {
    candidatePanel.innerHTML = candidates
      .map(
        (candidate) => `
          <article class="candidate-card ${candidate.accepted ? "accepted" : "rejected"}">
            <div class="meta-row">
              <span class="badge">${candidate.accepted ? "accepted" : "candidate"}</span>
              <span class="badge">score ${formatScore(Number(candidate.score || 0))}</span>
              <span class="badge">consistency ${formatScore(Number(candidate.consistency?.score || 0))}</span>
            </div>
            <p>${escapeHtml(shortText(candidate.text, 220))}</p>
          </article>
        `,
      )
      .join("");
  }

  document.getElementById("scene-detail").textContent = JSON.stringify(detail, null, 2);
}

async function openSceneDetail(sceneId) {
  const detail = await api(`/api/stories/${state.selectedStoryId}/scenes/${sceneId}`);
  renderSceneDetail(detail);
}

function renderArtifacts(items) {
  const list = document.getElementById("artifact-list");
  list.innerHTML = "";

  if (!items.length) {
    renderEmpty(list, "아직 export된 artifact가 없습니다.");
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "artifact-card";
    row.innerHTML = `
      <strong>${escapeHtml(item.kind)}</strong>
      <p class="muted">${escapeHtml(item.path)}</p>
      <div class="meta-row">
        <span class="badge">${escapeHtml(item.created_at || "")}</span>
        <a href="/api/artifacts/download?path=${encodeURIComponent(item.path)}" target="_blank" rel="noreferrer">다운로드</a>
      </div>
    `;
    list.appendChild(row);
  });
}

async function refreshJobs() {
  if (!state.selectedStoryId) return;
  const data = await api(`/api/jobs?story_id=${encodeURIComponent(state.selectedStoryId)}`);
  state.jobs = data.items || [];
  const container = document.getElementById("job-list");
  container.innerHTML = "";

  if (!state.jobs.length) {
    renderEmpty(container, "실행 중인 job이 없습니다.");
    return;
  }

  let hasPending = false;
  state.jobs.forEach((job) => {
    if (job.status === "queued" || job.status === "running") {
      hasPending = true;
    }
    const div = document.createElement("div");
    div.className = "job-card";
    div.innerHTML = `
      <strong>${escapeHtml(job.job_type)}</strong>
      <div class="meta-row">
        <span class="badge">${escapeHtml(job.status)}</span>
        <span class="badge">${toPercent(Number(job.progress || 0))}</span>
      </div>
      <div class="job-progress"><span style="width:${toPercent(Number(job.progress || 0))}"></span></div>
      ${(job.messages || [])
        .slice(-4)
        .map((message) => `<p class="muted">${escapeHtml(message.message)}</p>`)
        .join("")}
      ${job.error ? `<pre class="code-panel compact">${escapeHtml(job.error)}</pre>` : ""}
    `;
    container.appendChild(div);
  });

  if (hasPending) {
    window.setTimeout(refreshJobs, 1200);
    window.setTimeout(refreshStory, 1600);
  }
}

function summaryEntry(label, value) {
  return { label, value: value == null || value === "" ? "-" : String(value) };
}

function summarizeParsedJson(payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return [];
  }

  const items = [];
  if (typeof payload.ok === "boolean") {
    items.push(summaryEntry("Result", payload.ok ? "ok" : "failed"));
  }
  if (payload.story_id) {
    items.push(summaryEntry("Story", payload.story_id));
  }
  if (payload.path) {
    items.push(summaryEntry("Path", basename(payload.path)));
  }
  if (payload.paths && typeof payload.paths === "object") {
    items.push(summaryEntry("Artifacts", Object.keys(payload.paths).length));
  }
  if (payload.counts && typeof payload.counts === "object") {
    items.push(summaryEntry("Count keys", Object.keys(payload.counts).length));
  }
  if (payload.dataset_counts && typeof payload.dataset_counts === "object") {
    items.push(summaryEntry("Dataset keys", Object.keys(payload.dataset_counts).length));
  }
  if (Array.isArray(payload.story_ids)) {
    items.push(summaryEntry("Stories", payload.story_ids.length));
  }
  if (Array.isArray(payload.manifests)) {
    items.push(summaryEntry("Manifests", payload.manifests.length));
  }
  if (typeof payload.scene_count === "number") {
    items.push(summaryEntry("Scenes", payload.scene_count));
  }
  if (typeof payload.returncode === "number") {
    items.push(summaryEntry("Exit", payload.returncode));
  }

  Object.entries(payload)
    .filter(([key, value]) => {
      if (items.length >= 6) return false;
      if (["ok", "story_id", "path", "paths", "counts", "dataset_counts", "story_ids", "manifests", "scene_count", "returncode"].includes(key)) {
        return false;
      }
      return ["string", "number", "boolean"].includes(typeof value);
    })
    .slice(0, 6 - items.length)
    .forEach(([key, value]) => items.push(summaryEntry(key, value)));

  return items.slice(0, 6);
}

function renderSummaryGrid(items) {
  if (!items.length) return "";
  return `
    <div class="summary-grid">
      ${items
        .map(
          (item) => `
            <div class="summary-chip">
              <strong>${escapeHtml(item.label)}</strong>
              <span>${escapeHtml(item.value)}</span>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderSystemJobCard(job) {
  const parsed = job.result?.parsed_json || null;
  const summaryItems = [
    summaryEntry("Status", job.status),
    summaryEntry("Progress", toPercent(Number(job.progress || 0))),
    summaryEntry("Started", formatDateTime(job.created_at)),
    ...summarizeParsedJson(parsed),
  ].slice(0, 6);
  const commandPreview = Array.isArray(job.result?.command)
    ? shortText(job.result.command.join(" "), 220)
    : "";
  const outputTail = Array.isArray(job.result?.output_tail) ? job.result.output_tail.slice(-8).join("\n") : "";

  return `
    <div class="job-card">
      <strong>${escapeHtml(job.job_type)}</strong>
      <div class="meta-row">
        <span class="badge">${escapeHtml(job.status)}</span>
        <span class="badge">${toPercent(Number(job.progress || 0))}</span>
        <span class="badge">${escapeHtml(formatDateTime(job.created_at))}</span>
      </div>
      <div class="job-progress"><span style="width:${toPercent(Number(job.progress || 0))}"></span></div>
      ${renderSummaryGrid(summaryItems)}
      ${commandPreview ? `<p class="muted">cmd · ${escapeHtml(commandPreview)}</p>` : ""}
      ${(job.messages || [])
        .slice(-5)
        .map((message) => `<p class="muted">${escapeHtml(message.message)}</p>`)
        .join("")}
      ${parsed ? `<pre class="code-panel compact">${escapeHtml(JSON.stringify(parsed, null, 2))}</pre>` : ""}
      ${!parsed && outputTail ? `<pre class="code-panel compact">${escapeHtml(outputTail)}</pre>` : ""}
      ${job.error ? `<pre class="code-panel compact">${escapeHtml(job.error)}</pre>` : ""}
    </div>
  `;
}

function renderHFBrowserResults(items) {
  const container = document.getElementById("hf-browser-results");
  container.innerHTML = "";
  if (!items.length) {
    renderEmpty(container, "검색 결과가 없습니다.");
    return;
  }
  container.innerHTML = items
    .map(
      (item) => `
        <div class="job-card">
          <strong>${escapeHtml(item.repo_id)}</strong>
          <div class="meta-row">
            <span class="badge">${escapeHtml(item.repo_type || "-")}</span>
            <span class="badge">downloads ${escapeHtml(item.downloads ?? 0)}</span>
            <span class="badge">likes ${escapeHtml(item.likes ?? 0)}</span>
            <span class="badge">${escapeHtml(item.author || "-")}</span>
          </div>
          <div class="meta-row">
            <span class="badge">updated ${escapeHtml(formatDateTime(item.last_modified))}</span>
            <span class="badge">${item.private ? "private" : "public"}</span>
          </div>
          <p class="muted">${escapeHtml((item.tags || []).slice(0, 8).join(", "))}</p>
          <div class="action-stack horizontal">
            <button type="button" class="ghost-button compact" data-hf-repo-action="pull" data-repo-id="${escapeHtml(item.repo_id)}" data-repo-type="${escapeHtml(item.repo_type)}">Use In Pull</button>
            <button type="button" class="ghost-button compact" data-hf-repo-action="training" data-repo-id="${escapeHtml(item.repo_id)}" data-repo-type="${escapeHtml(item.repo_type)}">Use In Training</button>
            ${item.repo_type === "model" ? `<button type="button" class="ghost-button compact" data-hf-bind-role="writer" data-repo-id="${escapeHtml(item.repo_id)}" data-repo-type="${escapeHtml(item.repo_type)}">Writer</button>` : ""}
            ${item.repo_type === "model" ? `<button type="button" class="ghost-button compact" data-hf-bind-role="consistency_critic" data-repo-id="${escapeHtml(item.repo_id)}" data-repo-type="${escapeHtml(item.repo_type)}">Critic</button>` : ""}
            ${item.repo_type === "model" ? `<button type="button" class="ghost-button compact" data-hf-bind-role="world_model" data-repo-id="${escapeHtml(item.repo_id)}" data-repo-type="${escapeHtml(item.repo_type)}">World</button>` : ""}
          </div>
        </div>
      `,
    )
    .join("");
}

async function refreshSystemJobs() {
  const data = await api("/api/jobs");
  const jobs = (data.items || []).filter((job) =>
    ["one_click_loop", "generalist_loop", "training_run", "hf_publish", "hf_pull"].includes(job.job_type),
  );
  const container = document.getElementById("system-job-list");
  container.innerHTML = "";

  if (!jobs.length) {
    renderEmpty(container, "아직 GUI에서 실행한 자동화 job이 없습니다.");
    return;
  }

  container.innerHTML = jobs.map((job) => renderSystemJobCard(job)).join("");

  const hasPending = jobs.some((job) => job.status === "queued" || job.status === "running");
  if (hasPending) {
    window.setTimeout(refreshSystemJobs, 1500);
  }
}

async function refreshStory() {
  if (!state.selectedStoryId) return;
  const [detail, outline, scenes, bible, storyState, kg, artifacts, datasets] = await Promise.all([
    api(`/api/stories/${state.selectedStoryId}`),
    api(`/api/stories/${state.selectedStoryId}/outline`),
    api(`/api/stories/${state.selectedStoryId}/scenes`),
    api(`/api/stories/${state.selectedStoryId}/bible`),
    api(`/api/stories/${state.selectedStoryId}/state`),
    api(`/api/stories/${state.selectedStoryId}/kg`),
    api(`/api/stories/${state.selectedStoryId}/artifacts`),
    api(`/api/stories/${state.selectedStoryId}/datasets?limit=20`),
  ]);

  const outlineItems = outline.items || [];
  const sceneItems = scenes.items || [];
  const artifactItems = artifacts.items || [];

  renderOverview(detail, outlineItems, sceneItems, artifactItems, datasets.counts || {});
  renderBible(bible);
  renderScenes(sceneItems);
  renderArtifacts(artifactItems);
  document.getElementById("state-panel").textContent = JSON.stringify(storyState, null, 2);
  document.getElementById("kg-panel").textContent = JSON.stringify(kg.items || [], null, 2);

  if (state.currentSceneId) {
    const currentExists = sceneItems.some((scene) => scene.id === state.currentSceneId);
    if (currentExists) {
      await openSceneDetail(state.currentSceneId);
    }
  } else if (sceneItems.length) {
    state.currentSceneId = sceneItems[sceneItems.length - 1].id;
    await openSceneDetail(state.currentSceneId);
  } else {
    document.getElementById("scene-detail-summary").innerHTML =
      `<div class="empty-state">장면을 선택하면 원문과 후보 비교가 보입니다.</div>`;
    document.getElementById("scene-candidates").innerHTML = "";
    document.getElementById("scene-detail").textContent = "";
  }
}

async function runTask(label, task) {
  try {
    return await task();
  } catch (error) {
    logLine(`${label} 실패: ${error.message}`, "error");
    window.alert(error.message);
    throw error;
  }
}

async function submitCreateStory(event) {
  event.preventDefault();
  await runTask("스토리 생성", async () => {
    const form = event.target;
    const payload = {
      title: form.title.value,
      genre: form.genre.value,
      tone: form.tone.value,
      premise: form.premise.value,
      themes: lines(form.themes.value),
      characters: lines(form.characters.value),
      constraints: lines(form.constraints.value),
      target_scene_count: Number(form.target_scene_count.value || 6),
      target_word_count: Number(form.target_word_count.value || 8000),
    };
    const story = await api("/api/stories", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    logLine(`스토리 생성 완료: ${story.title}`);
    state.selectedStoryId = story.id;
    await loadStories();
  });
}

async function importStoryYaml(event) {
  const input = event.target;
  const file = input.files && input.files[0];
  if (!file) {
    return;
  }
  await runTask("Story YAML import", async () => {
    const yamlText = await file.text();
    const story = await api("/api/stories/import", {
      method: "POST",
      body: JSON.stringify({ yaml_text: yamlText }),
    });
    logLine(`YAML import 완료: ${story.title}`);
    state.selectedStoryId = story.id;
    await loadStories();
  });
  input.value = "";
}

async function saveBible(event) {
  event.preventDefault();
  await runTask("Bible 저장", async () => {
    ensureStorySelected();
    const form = event.target;
    const payload = {
      static_facts: lines(form.static_facts.value),
      rules: lines(form.rules.value),
      forbidden: lines(form.forbidden.value),
      motifs: lines(form.motifs.value),
    };
    await api(`/api/stories/${state.selectedStoryId}/bible`, {
      method: "PUT",
      body: JSON.stringify(payload),
    });
    logLine("Bible 저장 완료");
    await refreshStory();
  });
}

async function generateOutline() {
  await runTask("Outline 생성", async () => {
    ensureStorySelected();
    const sceneCount = Number(document.getElementById("outline-scene-count").value || 6);
    await api(`/api/stories/${state.selectedStoryId}/outline/generate`, {
      method: "POST",
      body: JSON.stringify({ scene_count: sceneCount }),
    });
    logLine(`outline 생성 완료: ${sceneCount} scenes`);
    await refreshStory();
  });
}

async function runNextScene() {
  await runTask("다음 Scene 생성", async () => {
    ensureStorySelected();
    const outline = await api(`/api/stories/${state.selectedStoryId}/outline`);
    let pending = (outline.items || []).find((item) => item.status !== "done");
    if (!pending) {
      const sceneCount = Number(document.getElementById("outline-scene-count").value || 6);
      await api(`/api/stories/${state.selectedStoryId}/outline/generate`, {
        method: "POST",
        body: JSON.stringify({ scene_count: sceneCount }),
      });
      const refreshed = await api(`/api/stories/${state.selectedStoryId}/outline`);
      pending = (refreshed.items || []).find((item) => item.status !== "done");
    }
    if (!pending) {
      throw new Error("실행 가능한 outline card를 찾지 못했습니다.");
    }
    const payload = {
      title_hint: pending.title,
      pov: pending.pov,
      location: pending.location,
      time_label: pending.time_label,
      goal: pending.goal,
      beat: pending.beat,
      foreshadowing: pending.foreshadowing || [],
      required_facts: pending.required_facts || [],
      outline_card_id: pending.id,
    };
    const job = await api(`/api/stories/${state.selectedStoryId}/jobs/run-scene`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    logLine(`scene job 시작: ${job.id}`);
    await refreshJobs();
  });
}

async function runAutoNovel() {
  await runTask("Auto Novel 실행", async () => {
    ensureStorySelected();
    const job = await api(`/api/stories/${state.selectedStoryId}/jobs/auto-novel`, {
      method: "POST",
    });
    logLine(`auto-novel job 시작: ${job.id}`);
    await refreshJobs();
  });
}

async function saveRuntime(event) {
  event.preventDefault();
  await runTask("Runtime 저장", async () => {
    const payload = collectRuntimePayload(event.target);
    const data = await api("/api/runtime-settings", {
      method: "PUT",
      body: JSON.stringify(payload),
    });
    document.getElementById("runtime-result").textContent = JSON.stringify(data, null, 2);
    logLine("runtime 설정 저장 완료");
    await refreshHealth();
  });
}

async function testRuntime() {
  await runTask("Runtime 테스트", async () => {
    const form = document.getElementById("runtime-form");
    const payload = collectRuntimePayload(form);
    const data = await api("/api/runtime-settings/test", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    document.getElementById("runtime-result").textContent = JSON.stringify(data, null, 2);
    logLine(`runtime 연결 테스트: ${data.ok ? "ok" : "failed"}`, data.ok ? "success" : "error");
  });
}

async function exportStory() {
  await runTask("원고 export", async () => {
    ensureStorySelected();
    const data = await api(`/api/stories/${state.selectedStoryId}/export`, { method: "POST" });
    logLine(`원고 export 완료: ${data.path}`);
    await refreshStory();
  });
}

async function evaluateStory() {
  await runTask("평가 생성", async () => {
    ensureStorySelected();
    const data = await api(`/api/stories/${state.selectedStoryId}/evaluate`, { method: "POST" });
    logLine(`평가 파일 생성 완료: ${data.path}`);
    await refreshStory();
  });
}

async function exportBundle() {
  await runTask("Training bundle export", async () => {
    ensureStorySelected();
    const data = await api(`/api/stories/${state.selectedStoryId}/export-datasets`, { method: "POST" });
    logLine(`training bundle 생성 완료: ${data.paths.manifest}`);
    await refreshStory();
  });
}

async function submitOneClickJob(event) {
  event.preventDefault();
  await runTask("One-click 실행", async () => {
    const payload = collectOneClickPayload(event.target);
    const job = await api("/api/system/jobs/one-click", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    logLine(`one-click job 시작: ${job.id}`);
    setSelectedTab("automation");
    await refreshSystemJobs();
  });
}

async function submitGeneralistJob(event) {
  event.preventDefault();
  await runTask("Generalist 실행", async () => {
    const payload = collectGeneralistPayload(event.target);
    const job = await api("/api/system/jobs/generalist", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    logLine(`generalist job 시작: ${job.id}`);
    setSelectedTab("automation");
    await refreshSystemJobs();
  });
}

async function submitTrainingJob(event) {
  event.preventDefault();
  await runTask("Training 실행", async () => {
    const payload = collectTrainingPayload(event.target);
    const job = await api("/api/system/jobs/train", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    logLine(`training job 시작: ${job.id}`);
    setSelectedTab("automation");
    await refreshSystemJobs();
  });
}

async function submitHFPublishJob(event) {
  event.preventDefault();
  await runTask("HF Publish 실행", async () => {
    const payload = collectHFPublishPayload(event.target);
    const job = await api("/api/system/jobs/hf-publish", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    logLine(`hf publish job 시작: ${job.id}`);
    setSelectedTab("automation");
    await refreshSystemJobs();
  });
}

async function submitHFPullJob(event) {
  event.preventDefault();
  await runTask("HF Pull 실행", async () => {
    const payload = collectHFPullPayload(event.target);
    const job = await api("/api/system/jobs/hf-pull", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    logLine(`hf pull job 시작: ${job.id}`);
    setSelectedTab("automation");
    await refreshSystemJobs();
  });
}

async function browseHFRepos(event) {
  event.preventDefault();
  await runTask("HF Browse", async () => {
    const form = event.target;
    const query = new URLSearchParams({
      repo_type: form.repo_type.value,
      search: form.search.value || "",
      author: form.author.value || "",
      limit: String(Number(form.limit.value || 12)),
    });
    const data = await api(`/api/hf/repos?${query.toString()}`);
    renderHFBrowserResults(data.items || []);
    logLine(`hf browse complete: ${(data.items || []).length} repos`);
  });
}

async function suggestHFPublishRelease() {
  await runTask("HF release suggestion", async () => {
    const form = document.getElementById("hf-publish-form");
    const query = new URLSearchParams({
      namespace: form.namespace.value || "",
      repo_type: form.repo_type.value || "model",
      project: form.project.value || "conarrative",
      role: form.role.value || "",
      base_model: form.base_model.value || "",
      stage: form.stage.value || "",
      release_prefix: form.release_prefix.value || "v",
    });
    const data = await api(`/api/hf/suggest-release?${query.toString()}`);
    form.repo_id.value = data.repo_id || form.repo_id.value;
    form.release_tag.value = data.suggested_tag || form.release_tag.value;
    if (!form.commit_message.value) {
      form.commit_message.value = `Upload ${form.role.value || "artifact"} adapter`;
    }
    if (!form.tag_message.value) {
      form.tag_message.value = `Release ${data.suggested_tag || ""} for ${data.repo_id || ""}`.trim();
    }
    logLine(`hf release suggestion ready: ${data.repo_id} / ${data.suggested_tag}`);
  });
}

function handleHFRepoResultClick(event) {
  const button = event.target.closest("button");
  if (!button) {
    return;
  }
  const repoId = button.dataset.repoId;
  const repoType = button.dataset.repoType || "model";
  if (!repoId) {
    return;
  }
  if (button.dataset.hfBindRole) {
    bindRuntimeRoleModel(button.dataset.hfBindRole, repoId);
    return;
  }
  if (button.dataset.hfRepoAction === "pull") {
    fillHFPullForm({ repo_id: repoId, repo_type: repoType, local_dir: `outputs/hf_download/${basename(repoId)}` });
    logLine(`hf pull form 채움: ${repoId}`);
    return;
  }
  if (button.dataset.hfRepoAction === "training") {
    fillTrainingForm({ ...collectTrainingPayload(document.getElementById("training-form")), model_name_or_path: repoId });
    logLine(`training form 채움: ${repoId}`);
  }
}

function bindTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => setSelectedTab(button.dataset.tab));
  });
}

function bindTemplates() {
  document.querySelectorAll("[data-story-template]").forEach((button) => {
    button.addEventListener("click", () => applyStoryTemplate(button.dataset.storyTemplate));
  });
  document.querySelectorAll("[data-runtime-preset]").forEach((button) => {
    button.addEventListener("click", () => applyRuntimePreset(button.dataset.runtimePreset));
  });
}

function bindPresetControls() {
  document.getElementById("load-runtime-preset-btn").addEventListener("click", () => runTask("runtime preset 불러오기", () => loadSavedPreset("runtime")));
  document.getElementById("save-runtime-preset-btn").addEventListener("click", () => runTask("runtime preset 저장", () => saveCurrentPreset("runtime")));
  document.getElementById("delete-runtime-preset-btn").addEventListener("click", () => runTask("runtime preset 삭제", () => deleteSavedPreset("runtime")));

  document.getElementById("load-one-click-preset-btn").addEventListener("click", () => runTask("one-click preset 불러오기", () => loadSavedPreset("one_click")));
  document.getElementById("save-one-click-preset-btn").addEventListener("click", () => runTask("one-click preset 저장", () => saveCurrentPreset("one_click")));
  document.getElementById("delete-one-click-preset-btn").addEventListener("click", () => runTask("one-click preset 삭제", () => deleteSavedPreset("one_click")));

  document.getElementById("load-generalist-preset-btn").addEventListener("click", () => runTask("generalist preset 불러오기", () => loadSavedPreset("generalist")));
  document.getElementById("save-generalist-preset-btn").addEventListener("click", () => runTask("generalist preset 저장", () => saveCurrentPreset("generalist")));
  document.getElementById("delete-generalist-preset-btn").addEventListener("click", () => runTask("generalist preset 삭제", () => deleteSavedPreset("generalist")));

  document.getElementById("load-training-preset-btn").addEventListener("click", () => runTask("training preset 불러오기", () => loadSavedPreset("training")));
  document.getElementById("save-training-preset-btn").addEventListener("click", () => runTask("training preset 저장", () => saveCurrentPreset("training")));
  document.getElementById("delete-training-preset-btn").addEventListener("click", () => runTask("training preset 삭제", () => deleteSavedPreset("training")));
}

function bindHFExtraControls() {
  document.getElementById("load-hf-publish-preset-btn").addEventListener("click", () => runTask("hf publish preset load", () => loadSavedPreset("hf_publish")));
  document.getElementById("save-hf-publish-preset-btn").addEventListener("click", () => runTask("hf publish preset save", () => saveCurrentPreset("hf_publish")));
  document.getElementById("delete-hf-publish-preset-btn").addEventListener("click", () => runTask("hf publish preset delete", () => deleteSavedPreset("hf_publish")));

  document.getElementById("load-hf-pull-preset-btn").addEventListener("click", () => runTask("hf pull preset load", () => loadSavedPreset("hf_pull")));
  document.getElementById("save-hf-pull-preset-btn").addEventListener("click", () => runTask("hf pull preset save", () => saveCurrentPreset("hf_pull")));
  document.getElementById("delete-hf-pull-preset-btn").addEventListener("click", () => runTask("hf pull preset delete", () => deleteSavedPreset("hf_pull")));

  document.getElementById("suggest-hf-publish-btn").addEventListener("click", () => suggestHFPublishRelease());
  document.getElementById("hf-browser-form").addEventListener("submit", browseHFRepos);
  document.getElementById("hf-browser-results").addEventListener("click", handleHFRepoResultClick);
}

function bindActions() {
  document.getElementById("create-story-form").addEventListener("submit", submitCreateStory);
  document.getElementById("story-yaml-input").addEventListener("change", importStoryYaml);
  document.getElementById("bible-form").addEventListener("submit", saveBible);
  document.getElementById("runtime-form").addEventListener("submit", saveRuntime);
  document.getElementById("one-click-form").addEventListener("submit", submitOneClickJob);
  document.getElementById("generalist-form").addEventListener("submit", submitGeneralistJob);
  document.getElementById("training-form").addEventListener("submit", submitTrainingJob);
  document.getElementById("hf-publish-form").addEventListener("submit", submitHFPublishJob);
  document.getElementById("hf-pull-form").addEventListener("submit", submitHFPullJob);
  document.getElementById("refresh-story-btn").addEventListener("click", () => runTask("동기화", refreshStory));
  document.getElementById("refresh-story-list-btn").addEventListener("click", () => runTask("스토리 목록 새로고침", loadStories));
  document.getElementById("generate-outline-btn").addEventListener("click", generateOutline);
  document.getElementById("run-scene-btn").addEventListener("click", runNextScene);
  document.getElementById("auto-novel-btn").addEventListener("click", runAutoNovel);
  document.getElementById("export-btn").addEventListener("click", exportStory);
  document.getElementById("evaluate-btn").addEventListener("click", evaluateStory);
  document.getElementById("bundle-btn").addEventListener("click", exportBundle);
  document.getElementById("test-runtime-btn").addEventListener("click", testRuntime);
  document.getElementById("clear-log-btn").addEventListener("click", () => {
    document.getElementById("log-panel").innerHTML = "";
    logLine("로그를 비웠습니다.");
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  bindTabs();
  bindTemplates();
  bindPresetControls();
  bindHFExtraControls();
  bindActions();
  applyStoryTemplate("moon-theater");
  await Promise.all([refreshHealth(), refreshRuntime(), refreshUiPresets()]);
  await loadStories();
  await refreshSystemJobs();
  setInterval(refreshHealth, 6000);
  setInterval(refreshJobs, 2500);
  setInterval(refreshSystemJobs, 3500);
});
