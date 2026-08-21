/** 结果区总编排：决定展示哪一份风险结果，再依次调用各渲染模块。 */

import { finiteNumber, prettyJson, requiredElement, setText } from "../dom";
import type {
  MainAnalysisResponse,
  PharmacyAssessmentView,
  RecommendationCard,
  RiskResultView,
  TopMicrobe,
} from "../types";
import { renderRecommendations } from "./guidance";
import {
  renderDrugKnowledgeSummary,
  renderMedicationLabelEvidence,
  renderPharmacyStatus,
  renderPlainLanguageSummary,
} from "./pharmacy";
import { renderRiskBanner, renderV8OutcomeSummary } from "./risk";

function renderTopMicrobes(items: readonly TopMicrobe[]): void {
  const list = requiredElement("microbe-list");
  list.innerHTML = "";

  for (const [name, value] of items) {
    const li = document.createElement("li");
    li.className = "result-item";
    const title = document.createElement("strong");
    title.textContent = name;
    const score = document.createElement("p");
    score.textContent = String(value);
    li.appendChild(title);
    li.appendChild(score);
    list.appendChild(li);
  }
}

/**
 * PFS 可用时展示 PFS 结果；否则若存在研究风险指数结果则展示它；
 * 两者都没有时仍展示 PFS 结果（好让横幅说明为什么没算）。
 */
function pickDisplayedRiskResult(
  pfsRiskResult: RiskResultView,
  generalRiskResult: RiskResultView,
): RiskResultView {
  if (pfsRiskResult.prediction_available === true) {
    return pfsRiskResult;
  }
  return Object.keys(generalRiskResult).length > 0 ? generalRiskResult : pfsRiskResult;
}

export function renderResult(data: MainAnalysisResponse): void {
  const pharmacyAssessment: PharmacyAssessmentView =
    data.pharmacy_assessment ?? data.report?.pharmacy_assessment ?? {};
  const recommendations: RecommendationCard[] =
    data.recommendations ?? pharmacyAssessment.recommendations ?? [];
  const pfsRiskResult: RiskResultView = data.risk_result ?? {};
  const generalRiskResult: RiskResultView =
    data.general_risk_result ?? data.report?.general_risk_result ?? {};

  renderRiskBanner(pickDisplayedRiskResult(pfsRiskResult, generalRiskResult));
  renderV8OutcomeSummary(pfsRiskResult, generalRiskResult);
  renderPlainLanguageSummary(pharmacyAssessment);
  renderPharmacyStatus(pharmacyAssessment);
  renderDrugKnowledgeSummary(pharmacyAssessment);
  renderRecommendations(recommendations, pharmacyAssessment);
  renderMedicationLabelEvidence(pharmacyAssessment);
  renderTopMicrobes(data.top_microbes ?? []);

  const actionSummary = pharmacyAssessment.plain_language_summary ?? {};
  const urgentCount = finiteNumber(actionSummary.urgent_count);
  const routineCount = finiteNumber(actionSummary.routine_count);
  const actionableCount =
    urgentCount !== null && routineCount !== null
      ? urgentCount + routineCount
      : recommendations.filter((item) => item.independent_of_model_result !== true).length;

  setText("recommendation-count", String(actionableCount));
  setText(
    "analysis-source",
    data.source_format === "raw_standardized"
      ? "输入来自原始临床 JSON，已自动标准化。"
      : "输入来自标准结构表单/JSON。",
  );
  setText("saved-path", data.saved_to || "--");
  setText("result-json", prettyJson(data));
}
