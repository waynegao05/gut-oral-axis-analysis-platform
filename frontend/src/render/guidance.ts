/** 建议条目：优先处理 / 后续核对两列。 */

import { requiredElement } from "../dom";
import type { EvidenceSource, PharmacyAssessmentView, RecommendationCard } from "../types";

type EvidenceIndex = ReadonlyMap<string, EvidenceSource>;

function buildEvidenceLinks(item: RecommendationCard, sourceIndex: EvidenceIndex): HTMLElement {
  const evidence = document.createElement("div");
  evidence.className = "result-item-evidence";

  for (const sourceId of item.evidence_source_ids ?? []) {
    const source = sourceIndex.get(sourceId);
    if (source?.url) {
      const anchor = document.createElement("a");
      anchor.textContent = source.organization
        ? `${source.organization} (${String(source.year)})`
        : sourceId;
      anchor.href = source.url;
      anchor.target = "_blank";
      anchor.rel = "noopener noreferrer";
      evidence.appendChild(anchor);
      continue;
    }
    const span = document.createElement("span");
    span.textContent = source?.organization
      ? `${source.organization} (${String(source.year)})`
      : sourceId;
    evidence.appendChild(span);
  }

  return evidence;
}

export function buildRecommendationItem(
  item: RecommendationCard,
  sourceIndex: EvidenceIndex,
): HTMLLIElement {
  const urgency = item.urgency || "routine";
  const li = document.createElement("li");
  li.className = `result-item guidance-item guidance-${urgency}`;

  const heading = document.createElement("div");
  heading.className = "guidance-item-heading";
  const title = document.createElement("strong");
  title.textContent = item.title || item.suggestion || "需要进一步核对";
  const badge = document.createElement("span");
  const independentGuidance = item.independent_of_model_result === true;
  badge.className = independentGuidance
    ? "urgency-badge urgency-independent"
    : `urgency-badge urgency-${urgency}`;
  badge.textContent = independentGuidance
    ? "独立指南提醒"
    : item.urgency_label || (urgency === "priority" ? "优先处理" : "后续核对");
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
  const actions = item.action_steps?.length
    ? item.action_steps
    : [item.suggestion || "请由医生或药师进一步核对。"];
  for (const action of actions) {
    const step = document.createElement("li");
    step.textContent = action;
    actionList.appendChild(step);
  }
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

  if (item.probiotic_candidates?.length) {
    const candidates = document.createElement("details");
    candidates.className = "recommendation-details probiotic-candidates";
    const candidateSummary = document.createElement("summary");
    candidateSummary.textContent = "查看可供临床人员核对的菌株组合";
    candidates.appendChild(candidateSummary);
    for (const candidate of item.probiotic_candidates) {
      const candidateNode = document.createElement("p");
      candidateNode.textContent =
        (candidate.strains ?? []).join(" + ") || candidate.candidate_id;
      candidates.appendChild(candidateNode);
    }
    li.appendChild(candidates);
  }

  const technical = document.createElement("details");
  technical.className = "recommendation-details";
  const technicalSummary = document.createElement("summary");
  technicalSummary.textContent = "查看依据与技术信息";
  technical.appendChild(technicalSummary);

  const evidence = buildEvidenceLinks(item, sourceIndex);
  if (evidence.childNodes.length) {
    technical.appendChild(evidence);
  }

  const meta = document.createElement("p");
  meta.className = "result-item-meta";
  meta.textContent = [
    `规则: ${item.recommendation_id || "-"}`,
    `类别: ${item.category || "-"}`,
    `标志物: ${item.marker || "-"}`,
    `证据等级: ${item.evidence_level || "-"}`,
  ].join(" | ");
  technical.appendChild(meta);
  li.appendChild(technical);

  return li;
}

function renderGuidanceEmptyState(list: HTMLElement, text: string): void {
  const item = document.createElement("li");
  item.className = "guidance-empty-state";
  item.textContent = text;
  list.appendChild(item);
}

export function renderRecommendations(
  items: readonly RecommendationCard[],
  assessment: PharmacyAssessmentView = {},
): void {
  const priorityList = requiredElement("priority-recommendation-list");
  const routineList = requiredElement("routine-recommendation-list");
  priorityList.innerHTML = "";
  routineList.innerHTML = "";

  const sourceIndex = new Map<string, EvidenceSource>(
    (assessment.evidence_sources ?? []).map((source) => [source.source_id, source] as const),
  );

  for (const item of items) {
    const target = item.urgency === "priority" ? priorityList : routineList;
    target.appendChild(buildRecommendationItem(item, sourceIndex));
  }

  if (!priorityList.childNodes.length) {
    renderGuidanceEmptyState(priorityList, "当前没有需要优先处理的已识别事项。");
  }
  if (!routineList.childNodes.length) {
    renderGuidanceEmptyState(routineList, "当前没有后续核对事项。");
  }
}
