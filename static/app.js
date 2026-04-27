const CANONICAL_EXAMPLE = {
  microbes: {
    Fusobacterium: 0.18,
    Porphyromonas: 0.15,
    Prevotella: 0.10,
    Streptococcus: 0.09,
    Lactobacillus: 0.02
  },
  clinical: {
    age: 52,
    bmi: 24.5,
    smoking: 1,
    family_history: 1
  },
  metabolites: {
    bile_acids: 0.8,
    scfa: 0.3,
    tryptophan_metabolism: 0.7
  }
};

const RAW_CLINICAL_EXAMPLE = {
  sample_id: "DEMO-001",
  demographics: {
    age: 55,
    bmi: 25.3,
    sex: "female"
  },
  history: {
    smoking: "yes",
    family_history_colorectal_or_ibd: "positive",
    recent_antibiotics: "no",
    recent_probiotics: "yes"
  },
  oral_microbiome: {
    taxa: [
      { taxon: "Fusobacterium", abundance: 0.16 },
      { taxon: "Porphyromonas", abundance: 0.13 },
      { taxon: "Prevotella", abundance: 0.08 },
      { taxon: "Streptococcus", abundance: 0.07 },
      { taxon: "Lactobacillus", abundance: 0.03 }
    ]
  },
  metabolites: {
    bile_acids: 0.74,
    scfa: 0.31,
    tryptophan_metabolism: 0.68
  },
  clinical_context: {
    chief_complaint: "recurrent abdominal discomfort",
    suspected_condition: "gut_risk_screening"
  }
};

const PRESET_MICROBES = [
  "Fusobacterium",
  "Porphyromonas",
  "Prevotella",
  "Streptococcus",
  "Lactobacillus"
];

function prettyJson(payload) {
  return JSON.stringify(payload, null, 2);
}

function numberOrZero(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    const message = Array.isArray(data.errors) ? data.errors.join(" | ") : "请求失败。";
    throw new Error(message);
  }
  return data;
}

function setImportStatus(message, variant = "") {
  const node = document.getElementById("import-status");
  node.textContent = message;
  node.className = "status-box";
  if (variant) {
    node.classList.add(`status-${variant}`);
  }
}

function clearExtraMicrobeRows() {
  document.getElementById("extra-microbe-rows").innerHTML = "";
}

function createExtraMicrobeRow(name = "", value = "") {
  const wrapper = document.createElement("div");
  wrapper.className = "microbe-row";
  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.className = "microbe-name";
  nameInput.placeholder = "自定义菌种名称";
  nameInput.value = name;

  const valueInput = document.createElement("input");
  valueInput.type = "number";
  valueInput.className = "microbe-value";
  valueInput.step = "0.0001";
  valueInput.min = "0";
  valueInput.placeholder = "丰度值";
  valueInput.value = String(value);

  const removeButton = document.createElement("button");
  removeButton.type = "button";
  removeButton.className = "remove-row-button";
  removeButton.textContent = "删";
  removeButton.addEventListener("click", () => wrapper.remove());

  wrapper.appendChild(nameInput);
  wrapper.appendChild(valueInput);
  wrapper.appendChild(removeButton);
  document.getElementById("extra-microbe-rows").appendChild(wrapper);
}

function populateForm(payload) {
  const microbes = payload.microbes || {};
  const clinical = payload.clinical || {};
  const metabolites = payload.metabolites || {};

  PRESET_MICROBES.forEach((name) => {
    const input = document.getElementById(`microbe-${name}`);
    input.value = microbes[name] ?? "";
  });

  clearExtraMicrobeRows();
  Object.entries(microbes)
    .filter(([name]) => !PRESET_MICROBES.includes(name))
    .forEach(([name, value]) => createExtraMicrobeRow(name, value));

  document.getElementById("clinical-age").value = clinical.age ?? "";
  document.getElementById("clinical-bmi").value = clinical.bmi ?? "";
  document.getElementById("clinical-smoking").value = String(clinical.smoking ?? 0);
  document.getElementById("clinical-family-history").value = String(clinical.family_history ?? 0);

  document.getElementById("metabolite-bile-acids").value = metabolites.bile_acids ?? "";
  document.getElementById("metabolite-scfa").value = metabolites.scfa ?? "";
  document.getElementById("metabolite-tryptophan").value = metabolites.tryptophan_metabolism ?? "";
}

function buildCanonicalPayloadFromForm() {
  const microbes = {};
  PRESET_MICROBES.forEach((name) => {
    microbes[name] = numberOrZero(document.getElementById(`microbe-${name}`).value);
  });

  document.querySelectorAll("#extra-microbe-rows .microbe-row").forEach((row) => {
    const name = row.querySelector(".microbe-name").value.trim();
    const value = row.querySelector(".microbe-value").value;
    if (!name) {
      return;
    }
    microbes[name] = numberOrZero(value);
  });

  return {
    microbes,
    clinical: {
      age: numberOrZero(document.getElementById("clinical-age").value),
      bmi: numberOrZero(document.getElementById("clinical-bmi").value),
      smoking: numberOrZero(document.getElementById("clinical-smoking").value),
      family_history: numberOrZero(document.getElementById("clinical-family-history").value)
    },
    metabolites: {
      bile_acids: numberOrZero(document.getElementById("metabolite-bile-acids").value),
      scfa: numberOrZero(document.getElementById("metabolite-scfa").value),
      tryptophan_metabolism: numberOrZero(document.getElementById("metabolite-tryptophan").value)
    }
  };
}

function renderStandardizedPreview(payload, sourceFormat) {
  document.getElementById("standardized-preview").textContent = prettyJson({
    source_format: sourceFormat,
    standardized_payload: payload
  });
}

function renderRecommendations(items) {
  const list = document.getElementById("recommendation-list");
  list.innerHTML = "";

  items.forEach((item) => {
    const li = document.createElement("li");
    li.className = "result-item";
    const title = document.createElement("strong");
    title.textContent = item.suggestion || "未提供建议文本";
    const rationale = document.createElement("p");
    rationale.textContent = item.rationale || "未提供解释";
    const meta = document.createElement("span");
    meta.className = "result-item-meta";
    meta.textContent = `marker: ${item.marker || "-"} | priority: ${item.priority ?? "-"}`;
    li.appendChild(title);
    li.appendChild(rationale);
    li.appendChild(meta);
    list.appendChild(li);
  });
}

function renderTopMicrobes(items) {
  const list = document.getElementById("microbe-list");
  list.innerHTML = "";

  items.forEach((item) => {
    const [name, value] = item;
    const li = document.createElement("li");
    li.className = "result-item";
    const title = document.createElement("strong");
    title.textContent = name;
    const score = document.createElement("p");
    score.textContent = String(value);
    li.appendChild(title);
    li.appendChild(score);
    list.appendChild(li);
  });
}

function formatRiskLevel(level) {
  const normalized = String(level || "").toLowerCase();
  if (normalized === "low") {
    return "低风险";
  }
  if (normalized === "medium") {
    return "中风险";
  }
  if (normalized === "high") {
    return "高风险";
  }
  return "未识别";
}

function renderRiskBanner(riskResult) {
  const banner = document.getElementById("risk-banner");
  const riskScore = document.getElementById("risk-score");
  const riskLevel = document.getElementById("risk-level");
  const level = String(riskResult.risk_level || "unknown").toLowerCase();
  const percentile = riskResult.risk_percentile;

  banner.className = "risk-banner";
  if (level === "low") {
    banner.classList.add("risk-banner-low");
  } else if (level === "medium") {
    banner.classList.add("risk-banner-medium");
  } else if (level === "high") {
    banner.classList.add("risk-banner-high");
  } else {
    banner.classList.add("risk-banner-empty");
  }

  if (Number.isFinite(Number(percentile))) {
    riskScore.textContent = `${Number(percentile).toFixed(2)}%`;
  } else {
    riskScore.textContent = riskResult.risk_score ?? "--";
  }
  riskLevel.textContent = formatRiskLevel(level);
}

function renderResult(data) {
  renderRiskBanner(data.risk_result || {});
  renderRecommendations(data.recommendations || []);
  renderTopMicrobes(data.top_microbes || []);

  document.getElementById("recommendation-count").textContent = String((data.recommendations || []).length);
  document.getElementById("analysis-source").textContent =
    data.source_format === "raw_standardized"
      ? "输入来自原始临床 JSON，已自动标准化。"
      : "输入来自标准结构表单/JSON。";
  document.getElementById("saved-path").textContent = data.saved_to || "--";
  document.getElementById("result-json").textContent = prettyJson(data);
}

function loadJsonIntoTextarea(payload) {
  document.getElementById("json-payload").value = prettyJson(payload);
}

async function standardizeFromTextarea() {
  const textarea = document.getElementById("json-payload");
  let payload;
  try {
    payload = JSON.parse(textarea.value);
  } catch (error) {
    setImportStatus(`JSON 解析失败：${error.message}`, "error");
    return;
  }

  try {
    const data = await postJson("/standardize", payload);
    populateForm(data.standardized_payload);
    renderStandardizedPreview(data.standardized_payload, data.source_format);
    setImportStatus("JSON 已成功标准化并回填到表单。", "success");
  } catch (error) {
    setImportStatus(error.message, "error");
  }
}

async function analyzeFromForm() {
  const payload = buildCanonicalPayloadFromForm();
  renderStandardizedPreview(payload, "form_manual");

  try {
    const data = await postJson("/analyze", payload);
    renderResult(data);
    setImportStatus("分析完成，结果区已更新。", "success");
  } catch (error) {
    setImportStatus(error.message, "error");
  }
}

function bindEvents() {
  document.getElementById("load-canonical-example").addEventListener("click", () => {
    loadJsonIntoTextarea(CANONICAL_EXAMPLE);
    populateForm(CANONICAL_EXAMPLE);
    renderStandardizedPreview(CANONICAL_EXAMPLE, "canonical_example");
    setImportStatus("已载入标准示例。", "success");
  });

  document.getElementById("load-raw-example").addEventListener("click", () => {
    loadJsonIntoTextarea(RAW_CLINICAL_EXAMPLE);
    setImportStatus("已载入原始临床示例，点击“导入并回填表单”完成标准化。");
  });

  document.getElementById("clear-json").addEventListener("click", () => {
    document.getElementById("json-payload").value = "";
    document.getElementById("standardized-preview").textContent = "";
    setImportStatus("JSON 已清空。");
  });

  document.getElementById("standardize-json").addEventListener("click", standardizeFromTextarea);
  document.getElementById("analyze-form").addEventListener("click", analyzeFromForm);

  document.getElementById("reset-form").addEventListener("click", () => {
    populateForm(CANONICAL_EXAMPLE);
    renderStandardizedPreview(CANONICAL_EXAMPLE, "canonical_example");
    setImportStatus("表单已重置为标准示例。");
  });

  document.getElementById("add-microbe-row").addEventListener("click", () => createExtraMicrobeRow());

  document.getElementById("json-file-input").addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    try {
      const text = await file.text();
      document.getElementById("json-payload").value = text;
      await standardizeFromTextarea();
    } catch (error) {
      setImportStatus(`文件读取失败：${error.message}`, "error");
    }
  });
}

window.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  loadJsonIntoTextarea(CANONICAL_EXAMPLE);
  populateForm(CANONICAL_EXAMPLE);
  renderStandardizedPreview(CANONICAL_EXAMPLE, "canonical_example");
  document.getElementById("result-json").textContent = prettyJson({ message: "等待分析。" });
});
