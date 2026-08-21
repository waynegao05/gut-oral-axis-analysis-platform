/** 用药评估相关渲染：白话摘要、可用性状态、药名核对摘要、说明书证据。 */

import { requiredElement, setText } from "../dom";
import type { LabelRecord, PharmacyAssessmentView } from "../types";

export function renderPlainLanguageSummary(assessment: PharmacyAssessmentView): void {
  const summary = assessment.plain_language_summary ?? {};
  const actionList = requiredElement("pharmacy-now-list");
  const actions = summary.what_to_do_now ?? [];

  setText("pharmacy-headline", summary.headline || "结果已生成，请按优先级逐项核对");

  actionList.innerHTML = "";
  if (!actions.length) {
    const item = document.createElement("li");
    item.textContent = "当前没有可展示的行动项，请查看输入完整性或稍后重新分析。";
    actionList.appendChild(item);
  } else {
    for (const action of actions) {
      const item = document.createElement("li");
      item.textContent = action;
      actionList.appendChild(item);
    }
  }

  setText(
    "pharmacy-safety-note",
    summary.safety_note || "这是给医生或药师复核的辅助清单，不是诊断或处方。",
  );
}

export function renderPharmacyStatus(assessment: PharmacyAssessmentView): void {
  const card = requiredElement("pharmacy-status-card");
  const status = assessment.status ?? "withheld";
  const reasons = assessment.quality?.status_reasons ?? [];

  if (!assessment.engine_version) {
    setText("pharmacy-status", "药学结果暂时不可用");
    setText("pharmacy-status-reason", "请重新分析；如仍失败，改由医生或药师人工核对。");
  } else if (status === "standard") {
    setText("pharmacy-status", assessment.status_label || "信息较完整，可供医生或药师参考");
    setText(
      "pharmacy-status-reason",
      "当前输入未触发可靠性限制，但结果仍不能替代病历、检查和人工判断。",
    );
  } else {
    setText("pharmacy-status", assessment.status_label || "信息不完整，请先补充或核对");
    const readableReasons = reasons
      .map((reason) => reason.message)
      .filter(Boolean)
      .slice(0, 3);
    setText(
      "pharmacy-status-reason",
      readableReasons.length
        ? `先处理：${readableReasons.join("；")}`
        : "先按优先行动补全信息或修正异常值，然后重新分析。",
    );
  }

  card.className = `summary-card pharmacy-status-card status-${status}`;
}

export function renderDrugKnowledgeSummary(assessment: PharmacyAssessmentView): void {
  const knowledge = assessment.drug_knowledge ?? {};
  const medicationContext = assessment.medication_context ?? {};
  const normalization = knowledge.normalization ?? {};
  const interaction = knowledge.interaction_screening ?? {};

  if (!knowledge.available) {
    setText("drug-knowledge-coverage", "药品资料暂时无法读取");
    setText("drug-knowledge-detail", "请把完整用药和过敏清单交给医生或药师人工核对。");
    return;
  }

  const inputCount = normalization.input_count ?? 0;
  const matchedCount = normalization.matched_count ?? 0;
  const matchCount = interaction.match_count ?? 0;
  const labelCount = knowledge.label_lookup?.record_count ?? 0;

  if (!inputCount && medicationContext.medication_list_available === true) {
    setText("drug-knowledge-coverage", "当前记录：没有在用药物");
    setText(
      "drug-knowledge-detail",
      "如果这是真实情况，无需补填；因为没有用药组合，所以不会进行相互作用核对。",
    );
  } else if (!inputCount) {
    setText("drug-knowledge-coverage", "尚未填写当前用药");
    setText("drug-knowledge-detail", "填写药盒或处方上的通用名、剂型、规格和每天用法后再核对。");
  } else if (matchCount > 0) {
    setText("drug-knowledge-coverage", `发现 ${matchCount} 组需要优先核对的用药组合`);
    setText(
      "drug-knowledge-detail",
      `${matchedCount}/${inputCount} 项药名已识别，${labelCount} 份说明书可查看。`,
    );
  } else if (interaction.interaction_screening_performed === true) {
    setText("drug-knowledge-coverage", "未发现已收录的最高风险用药组合");
    setText(
      "drug-knowledge-detail",
      `${matchedCount}/${inputCount} 项药名已识别，${labelCount} 份说明书可查看；这不等于没有其他相互作用。`,
    );
  } else {
    setText("drug-knowledge-coverage", "尚未完成用药组合核对");
    setText(
      "drug-knowledge-detail",
      `${matchedCount}/${inputCount} 项药名已识别，${labelCount} 份说明书可查看；请补全未识别药名或至少填写两项用药。`,
    );
  }
}

function buildLabelEvidenceItem(record: LabelRecord): HTMLElement {
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
  prompt.textContent =
    record.review_prompt || "先核对实际药名、剂型、规格和每天用法是否与这份说明书一致。";
  article.appendChild(prompt);

  const identity = record.label_identity ?? {};
  const source = record.source ?? {};
  const links = document.createElement("div");
  links.className = "label-source-links";
  const linkTargets: ReadonlyArray<readonly [string, string | null | undefined]> = [
    ["查看 DailyMed 官方说明书", source.dailymed_url],
    ["查看 openFDA 记录", source.openfda_query_url],
  ];
  for (const [label, href] of linkTargets) {
    if (!href) {
      continue;
    }
    const link = document.createElement("a");
    link.textContent = label;
    link.className = "evidence-link-button";
    link.href = href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    links.appendChild(link);
  }
  if (links.childNodes.length) {
    article.appendChild(links);
  }

  for (const [sectionName, section] of Object.entries(record.sections ?? {})) {
    if (!section) {
      continue;
    }
    const details = document.createElement("details");
    details.className = "label-section";
    const summary = document.createElement("summary");
    summary.textContent = `查看英文说明书原文：${section.label_zh || sectionName}`;
    const text = document.createElement("p");
    text.textContent = section.excerpt || "";
    details.appendChild(summary);
    details.appendChild(text);
    article.appendChild(details);
  }

  const technical = document.createElement("details");
  technical.className = "label-section label-technical-details";
  const technicalSummary = document.createElement("summary");
  technicalSummary.textContent = "技术信息";
  const meta = document.createElement("p");
  meta.textContent = [
    `RxCUI: ${record.rxcui || "-"}`,
    `标签日期: ${identity.effective_time || "-"}`,
    `给药途径: ${(identity.routes ?? []).join(", ") || "-"}`,
    `SPL SET ID: ${identity.spl_set_id || "-"}`,
  ].join(" | ");
  technical.appendChild(technicalSummary);
  technical.appendChild(meta);
  article.appendChild(technical);

  const boundary = document.createElement("p");
  boundary.className = "safety-boundary-note";
  boundary.textContent = "说明书中的一般用法不是当前患者的具体剂量或疗程，不要据此自行改药。";
  article.appendChild(boundary);

  return article;
}

export function renderMedicationLabelEvidence(assessment: PharmacyAssessmentView): void {
  const container = requiredElement("medication-label-list");
  const records = assessment.drug_knowledge?.label_lookup?.records ?? [];
  container.innerHTML = "";

  if (!records.length) {
    const empty = document.createElement("p");
    empty.className = "empty-evidence-note";
    empty.textContent = "未匹配到可展示的产品说明书。请补充通用名、剂型、规格和给药途径。";
    container.appendChild(empty);
    return;
  }

  for (const record of records) {
    container.appendChild(buildLabelEvidenceItem(record));
  }
}
