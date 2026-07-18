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
  },
  metadata: {
    current_medications: [],
    drug_allergies: [],
    recent_antibiotics: 0,
    recent_probiotics: 0,
    renal_impairment: 0,
    hepatic_impairment: 0,
    pregnancy: 0,
    suspected_condition: "gut_risk_screening"
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
  medication_context: {
    current_medications: ["metformin 500 mg twice daily"],
    drug_allergies: ["penicillin: rash"],
    renal_impairment: "no",
    hepatic_impairment: "no",
    pregnancy: "no"
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

function readNumberInput(input, label, { emptyValue = null } = {}) {
  const rawValue = input.value.trim();
  if (rawValue === "") {
    return emptyValue;
  }

  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${label} 必须是有效数字。`);
  }
  if (!input.checkValidity()) {
    throw new Error(`${label} 输入非法：${input.validationMessage}`);
  }
  return parsed;
}

function readOptionalListInput(input) {
  const rawValue = input.value.trim();
  if (rawValue === "") {
    return null;
  }
  if (["无", "none", "no"].includes(rawValue.toLowerCase())) {
    return [];
  }
  return rawValue
    .split(/[,，;；\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function readOptionalBinarySelect(input) {
  if (input.value === "") {
    return null;
  }
  return Number(input.value);
}

function formatOptionalList(metadata, key) {
  if (!Object.prototype.hasOwnProperty.call(metadata, key)) {
    return "";
  }
  const values = Array.isArray(metadata[key]) ? metadata[key] : [];
  return values.length ? values.join("\n") : "无";
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
  valueInput.max = "1";
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
  const metadata = payload.metadata || {};

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
  document.getElementById("metadata-current-medications").value = formatOptionalList(metadata, "current_medications");
  document.getElementById("metadata-drug-allergies").value = formatOptionalList(metadata, "drug_allergies");
  document.getElementById("metadata-suspected-condition").value =
    metadata.suspected_condition || "gut_risk_screening";
  [
    "recent-antibiotics",
    "recent-probiotics",
    "renal-impairment",
    "hepatic-impairment",
    "pregnancy"
  ].forEach((field) => {
    const key = field.replaceAll("-", "_");
    document.getElementById(`metadata-${field}`).value =
      Object.prototype.hasOwnProperty.call(metadata, key) ? String(metadata[key]) : "";
  });
}

function buildCanonicalPayloadFromForm() {
  const microbes = {};
  PRESET_MICROBES.forEach((name) => {
    const value = readNumberInput(
      document.getElementById(`microbe-${name}`),
      `${name} 丰度`
    );
    if (value !== null) {
      microbes[name] = value;
    }
  });

  document.querySelectorAll("#extra-microbe-rows .microbe-row").forEach((row) => {
    const name = row.querySelector(".microbe-name").value.trim();
    if (!name) {
      return;
    }
    const valueInput = row.querySelector(".microbe-value");
    const abundance = readNumberInput(valueInput, `${name} 丰度`);
    if (abundance === null) {
      throw new Error(`${name} 丰度不能为空。`);
    }
    microbes[name] = abundance;
  });

  if (!Object.values(microbes).some((value) => value > 0)) {
    throw new Error("至少需要填写一个大于 0 的菌群丰度。");
  }

  const clinical = {
    smoking: readNumberInput(document.getElementById("clinical-smoking"), "吸烟状态"),
    family_history: readNumberInput(document.getElementById("clinical-family-history"), "家族史")
  };
  const age = readNumberInput(document.getElementById("clinical-age"), "年龄");
  const bmi = readNumberInput(document.getElementById("clinical-bmi"), "BMI");
  if (age !== null) {
    clinical.age = age;
  }
  if (bmi !== null) {
    clinical.bmi = bmi;
  }

  const metabolites = {};
  const bileAcids = readNumberInput(document.getElementById("metabolite-bile-acids"), "胆汁酸");
  const scfa = readNumberInput(document.getElementById("metabolite-scfa"), "短链脂肪酸（SCFA）");
  const tryptophan = readNumberInput(document.getElementById("metabolite-tryptophan"), "色氨酸代谢");
  if (bileAcids !== null) {
    metabolites.bile_acids = bileAcids;
  }
  if (scfa !== null) {
    metabolites.scfa = scfa;
  }
  if (tryptophan !== null) {
    metabolites.tryptophan_metabolism = tryptophan;
  }

  const metadata = {};
  const currentMedications = readOptionalListInput(
    document.getElementById("metadata-current-medications")
  );
  const drugAllergies = readOptionalListInput(
    document.getElementById("metadata-drug-allergies")
  );
  if (currentMedications !== null) {
    metadata.current_medications = currentMedications;
  }
  if (drugAllergies !== null) {
    metadata.drug_allergies = drugAllergies;
  }
  metadata.suspected_condition =
    document.getElementById("metadata-suspected-condition").value || "gut_risk_screening";
  [
    "recent-antibiotics",
    "recent-probiotics",
    "renal-impairment",
    "hepatic-impairment",
    "pregnancy"
  ].forEach((field) => {
    const value = readOptionalBinarySelect(document.getElementById(`metadata-${field}`));
    if (value !== null) {
      metadata[field.replaceAll("-", "_")] = value;
    }
  });

  return {
    microbes,
    clinical,
    metabolites,
    metadata
  };
}

function renderStandardizedPreview(payload, sourceFormat) {
  document.getElementById("standardized-preview").textContent = prettyJson({
    source_format: sourceFormat,
    standardized_payload: payload
  });
}

function renderPlainLanguageSummary(assessment) {
  const summary = assessment.plain_language_summary || {};
  const headline = document.getElementById("pharmacy-headline");
  const actionList = document.getElementById("pharmacy-now-list");
  const safetyNote = document.getElementById("pharmacy-safety-note");
  const actions = Array.isArray(summary.what_to_do_now) ? summary.what_to_do_now : [];

  headline.textContent = summary.headline || "结果已生成，请按优先级逐项核对";
  actionList.innerHTML = "";
  if (!actions.length) {
    const item = document.createElement("li");
    item.textContent = "当前没有可展示的行动项，请查看输入完整性或稍后重新分析。";
    actionList.appendChild(item);
  } else {
    actions.forEach((action) => {
      const item = document.createElement("li");
      item.textContent = action;
      actionList.appendChild(item);
    });
  }
  safetyNote.textContent = summary.safety_note
    || "这是给医生或药师复核的辅助清单，不是诊断或处方。";
}

function renderPharmacyStatus(assessment) {
  const card = document.getElementById("pharmacy-status-card");
  const statusNode = document.getElementById("pharmacy-status");
  const reasonNode = document.getElementById("pharmacy-status-reason");
  const status = String(assessment.status || "withheld");
  const reasons = assessment.quality?.status_reasons || [];

  if (!assessment.engine_version) {
    statusNode.textContent = "药学结果暂时不可用";
    reasonNode.textContent = "请重新分析；如仍失败，改由医生或药师人工核对。";
  } else if (status === "standard") {
    statusNode.textContent = assessment.status_label || "信息较完整，可供医生或药师参考";
    reasonNode.textContent = "当前输入未触发可靠性限制，但结果仍不能替代病历、检查和人工判断。";
  } else {
    statusNode.textContent = assessment.status_label || "信息不完整，请先补充或核对";
    const readableReasons = reasons
      .map((reason) => reason.message)
      .filter(Boolean)
      .slice(0, 3);
    reasonNode.textContent = readableReasons.length
      ? `先处理：${readableReasons.join("；")}`
      : "先按优先行动补全信息或修正异常值，然后重新分析。";
  }

  card.className = `summary-card pharmacy-status-card status-${status}`;
}

function renderDrugKnowledgeSummary(assessment) {
  const valueNode = document.getElementById("drug-knowledge-coverage");
  const detailNode = document.getElementById("drug-knowledge-detail");
  const knowledge = assessment.drug_knowledge || {};
  const medicationContext = assessment.medication_context || {};
  const normalization = knowledge.normalization || {};
  const interaction = knowledge.interaction_screening || {};

  if (!knowledge.available) {
    valueNode.textContent = "药品资料暂时无法读取";
    detailNode.textContent = "请把完整用药和过敏清单交给医生或药师人工核对。";
    return;
  }

  const inputCount = Number(normalization.input_count || 0);
  const matchedCount = Number(normalization.matched_count || 0);
  const matchCount = Number(interaction.match_count || 0);
  const labelCount = Number(knowledge.label_lookup?.record_count || 0);
  if (!inputCount && medicationContext.medication_list_available === true) {
    valueNode.textContent = "当前记录：没有在用药物";
    detailNode.textContent = "如果这是真实情况，无需补填；因为没有用药组合，所以不会进行相互作用核对。";
  } else if (!inputCount) {
    valueNode.textContent = "尚未填写当前用药";
    detailNode.textContent = "填写药盒或处方上的通用名、剂型、规格和每天用法后再核对。";
  } else if (matchCount > 0) {
    valueNode.textContent = `发现 ${matchCount} 组需要优先核对的用药组合`;
    detailNode.textContent = `${matchedCount}/${inputCount} 项药名已识别，${labelCount} 份说明书可查看。`;
  } else if (interaction.interaction_screening_performed === true) {
    valueNode.textContent = "未发现已收录的最高风险用药组合";
    detailNode.textContent = `${matchedCount}/${inputCount} 项药名已识别，${labelCount} 份说明书可查看；这不等于没有其他相互作用。`;
  } else {
    valueNode.textContent = "尚未完成用药组合核对";
    detailNode.textContent = `${matchedCount}/${inputCount} 项药名已识别，${labelCount} 份说明书可查看；请补全未识别药名或至少填写两项用药。`;
  }
}

function renderMedicationLabelEvidence(assessment) {
  const container = document.getElementById("medication-label-list");
  const records = assessment.drug_knowledge?.label_lookup?.records || [];
  container.innerHTML = "";

  if (!records.length) {
    const empty = document.createElement("p");
    empty.className = "empty-evidence-note";
    empty.textContent = "未匹配到可展示的产品说明书。请补充通用名、剂型、规格和给药途径。";
    container.appendChild(empty);
    return;
  }

  records.forEach((record) => {
    const article = document.createElement("article");
    article.className = "label-evidence-item";

    const heading = document.createElement("strong");
    heading.textContent = `当前用药：${record.input || record.display_name || record.drug_id}`;
    article.appendChild(heading);

    const matchedName = document.createElement("p");
    matchedName.className = "label-matched-name";
    matchedName.textContent = `系统识别为：${record.display_name || record.drug_id}`;
    article.appendChild(matchedName);

    const prompt = document.createElement("p");
    prompt.className = "label-review-prompt";
    prompt.textContent = record.review_prompt
      || "先核对实际药名、剂型、规格和每天用法是否与这份说明书一致。";
    article.appendChild(prompt);

    const identity = record.label_identity || {};
    const links = document.createElement("div");
    links.className = "label-source-links";
    const source = record.source || {};
    [
      ["查看 DailyMed 官方说明书", source.dailymed_url],
      ["查看 openFDA 记录", source.openfda_query_url]
    ].forEach(([label, href]) => {
      if (!href) {
        return;
      }
      const link = document.createElement("a");
      link.textContent = label;
      link.className = "evidence-link-button";
      link.href = href;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      links.appendChild(link);
    });
    if (links.childNodes.length) {
      article.appendChild(links);
    }

    Object.entries(record.sections || {}).forEach(([sectionName, section]) => {
      const details = document.createElement("details");
      details.className = "label-section";
      const summary = document.createElement("summary");
      summary.textContent = `查看英文说明书原文：${section.label_zh || sectionName}`;
      const text = document.createElement("p");
      text.textContent = section.excerpt || "";
      details.appendChild(summary);
      details.appendChild(text);
      article.appendChild(details);
    });

    const technical = document.createElement("details");
    technical.className = "label-section label-technical-details";
    const technicalSummary = document.createElement("summary");
    technicalSummary.textContent = "技术信息";
    const meta = document.createElement("p");
    meta.textContent = [
      `RxCUI: ${record.rxcui || "-"}`,
      `标签日期: ${identity.effective_time || "-"}`,
      `给药途径: ${(identity.routes || []).join(", ") || "-"}`,
      `SPL SET ID: ${identity.spl_set_id || "-"}`
    ].join(" | ");
    technical.appendChild(technicalSummary);
    technical.appendChild(meta);
    article.appendChild(technical);

    const boundary = document.createElement("p");
    boundary.className = "safety-boundary-note";
    boundary.textContent = "说明书中的一般用法不是当前患者的具体剂量或疗程，不要据此自行改药。";
    article.appendChild(boundary);
    container.appendChild(article);
  });
}

function buildRecommendationItem(item, sourceIndex) {
  const li = document.createElement("li");
  li.className = `result-item guidance-item guidance-${item.urgency || "routine"}`;

  const heading = document.createElement("div");
  heading.className = "guidance-item-heading";
  const title = document.createElement("strong");
  title.textContent = item.title || item.suggestion || "需要进一步核对";
  const badge = document.createElement("span");
  const independentGuidance = item.independent_of_model_result === true;
  badge.className = independentGuidance
    ? "urgency-badge urgency-independent"
    : `urgency-badge urgency-${item.urgency || "routine"}`;
  badge.textContent = independentGuidance
    ? "独立指南提醒"
    : item.urgency_label || (item.urgency === "priority" ? "优先处理" : "后续核对");
  heading.appendChild(title);
  heading.appendChild(badge);
  li.appendChild(heading);

  if (independentGuidance && item.decision_basis) {
    const basis = document.createElement("p");
    basis.className = "guidance-basis-note";
    basis.textContent = item.decision_basis;
    li.appendChild(basis);
  }

  const actionBlock = document.createElement("div");
  actionBlock.className = "guidance-action-block";
  const actionLabel = document.createElement("span");
  actionLabel.className = "guidance-label";
  actionLabel.textContent = "下一步";
  const actionList = document.createElement("ol");
  actionList.className = "guidance-steps";
  const actions = Array.isArray(item.action_steps) && item.action_steps.length
    ? item.action_steps
    : [item.suggestion || "请由医生或药师进一步核对。"];
  actions.forEach((action) => {
    const step = document.createElement("li");
    step.textContent = action;
    actionList.appendChild(step);
  });
  actionBlock.appendChild(actionLabel);
  actionBlock.appendChild(actionList);
  li.appendChild(actionBlock);

  const rationale = document.createElement("p");
  rationale.className = "guidance-rationale";
  const rationaleLabel = document.createElement("strong");
  rationaleLabel.textContent = "为什么：";
  rationale.appendChild(rationaleLabel);
  rationale.appendChild(document.createTextNode(item.rationale || "需要结合完整临床资料判断。"));
  li.appendChild(rationale);

  if (Array.isArray(item.probiotic_candidates) && item.probiotic_candidates.length) {
    const candidates = document.createElement("details");
    candidates.className = "recommendation-details probiotic-candidates";
    const candidateSummary = document.createElement("summary");
    candidateSummary.textContent = "查看可供临床人员核对的菌株组合";
    candidates.appendChild(candidateSummary);
    item.probiotic_candidates.forEach((candidate) => {
      const candidateNode = document.createElement("p");
      candidateNode.textContent = (candidate.strains || []).join(" + ") || candidate.candidate_id;
      candidates.appendChild(candidateNode);
    });
    li.appendChild(candidates);
  }

  const technical = document.createElement("details");
  technical.className = "recommendation-details";
  const technicalSummary = document.createElement("summary");
  technicalSummary.textContent = "查看依据与技术信息";
  technical.appendChild(technicalSummary);

  const evidence = document.createElement("div");
  evidence.className = "result-item-evidence";
  (item.evidence_source_ids || []).forEach((sourceId) => {
    const source = sourceIndex.get(sourceId);
    const node = source?.url ? document.createElement("a") : document.createElement("span");
    node.textContent = source?.organization ? `${source.organization} (${source.year})` : sourceId;
    if (source?.url) {
      node.href = source.url;
      node.target = "_blank";
      node.rel = "noopener noreferrer";
    }
    evidence.appendChild(node);
  });
  if (evidence.childNodes.length) {
    technical.appendChild(evidence);
  }

  const meta = document.createElement("p");
  meta.className = "result-item-meta";
  meta.textContent = [
    `规则: ${item.recommendation_id || "-"}`,
    `类别: ${item.category || "-"}`,
    `标志物: ${item.marker || "-"}`,
    `证据等级: ${item.evidence_level || "-"}`
  ].join(" | ");
  technical.appendChild(meta);
  li.appendChild(technical);
  return li;
}

function renderGuidanceEmptyState(list, text) {
  const item = document.createElement("li");
  item.className = "guidance-empty-state";
  item.textContent = text;
  list.appendChild(item);
}

function renderRecommendations(items, assessment = {}) {
  const priorityList = document.getElementById("priority-recommendation-list");
  const routineList = document.getElementById("routine-recommendation-list");
  priorityList.innerHTML = "";
  routineList.innerHTML = "";
  const sourceIndex = new Map(
    (assessment.evidence_sources || []).map((source) => [source.source_id, source])
  );

  items.forEach((item) => {
    const target = item.urgency === "priority" ? priorityList : routineList;
    target.appendChild(buildRecommendationItem(item, sourceIndex));
  });
  if (!priorityList.childNodes.length) {
    renderGuidanceEmptyState(priorityList, "当前没有需要优先处理的已识别事项。");
  }
  if (!routineList.childNodes.length) {
    renderGuidanceEmptyState(routineList, "当前没有后续核对事项。");
  }
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

function renderRiskBanner(riskResult, assessment = {}) {
  const banner = document.getElementById("risk-banner");
  const riskKicker = document.getElementById("risk-kicker");
  const riskScore = document.getElementById("risk-score");
  const riskLevel = document.getElementById("risk-level");
  const level = String(riskResult.risk_level || "unknown").toLowerCase();
  const percentile = riskResult.risk_percentile;

  banner.className = "risk-banner";
  if (assessment.status === "withheld") {
    banner.classList.add("risk-banner-withheld");
    riskKicker.textContent = "当前输入超出模型适用范围";
    riskScore.textContent = "--";
    riskLevel.textContent = "先核对输入，再查看风险";
    return;
  }
  riskKicker.textContent = "研究队列中的相对位置";
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
  const pharmacyAssessment = data.pharmacy_assessment || data.report?.pharmacy_assessment || {};
  const recommendations = data.recommendations || pharmacyAssessment.recommendations || [];
  renderRiskBanner(data.risk_result || {}, pharmacyAssessment);
  renderPlainLanguageSummary(pharmacyAssessment);
  renderPharmacyStatus(pharmacyAssessment);
  renderDrugKnowledgeSummary(pharmacyAssessment);
  renderRecommendations(recommendations, pharmacyAssessment);
  renderMedicationLabelEvidence(pharmacyAssessment);
  renderTopMicrobes(data.top_microbes || []);

  const actionSummary = pharmacyAssessment.plain_language_summary || {};
  const actionableCount = Number.isFinite(Number(actionSummary.urgent_count))
    && Number.isFinite(Number(actionSummary.routine_count))
    ? Number(actionSummary.urgent_count) + Number(actionSummary.routine_count)
    : recommendations.filter((item) => item.independent_of_model_result !== true).length;
  document.getElementById("recommendation-count").textContent = String(actionableCount);
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
  try {
    const payload = buildCanonicalPayloadFromForm();
    renderStandardizedPreview(payload, "form_manual");
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
