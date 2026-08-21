/** 风险横幅、分位条与 V8 结局摘要。 */

import { finiteNumber, requiredElement, setText, toInputValue } from "../dom";
import type { RiskResultView } from "../types";

export function formatRiskLevel(level: string, isResearchIndex = false): string {
  const normalized = level.toLowerCase();
  if (normalized === "low") {
    return isResearchIndex ? "较低参考位置" : "低风险";
  }
  if (normalized === "medium") {
    return isResearchIndex ? "中等参考位置" : "中风险";
  }
  if (normalized === "high") {
    return isResearchIndex ? "较高参考位置" : "高风险";
  }
  return "未识别";
}

export function formatProbability(value: unknown): string {
  const number = finiteNumber(value);
  return number === null ? "--" : `${(number * 100).toFixed(1)}%`;
}

function renderRiskScale(riskResult: RiskResultView): void {
  const scale = requiredElement("risk-scale");
  const fill = requiredElement("risk-scale-fill");
  const marker = requiredElement("risk-scale-marker");
  const percentile = finiteNumber(riskResult.risk_percentile);
  const available = riskResult.prediction_available !== false && percentile !== null;

  scale.classList.toggle("risk-scale-unavailable", !available);
  if (!available || percentile === null) {
    fill.style.width = "0%";
    marker.style.left = "0%";
    marker.style.opacity = "0";
    scale.setAttribute("aria-label", "风险参考分位暂未生成");
    return;
  }

  const boundedPercentile = Math.max(0, Math.min(100, percentile));
  fill.style.width = `${boundedPercentile}%`;
  marker.style.left = `${boundedPercentile}%`;
  marker.style.opacity = "1";
  scale.setAttribute("aria-label", `研究参考队列风险分位 ${boundedPercentile.toFixed(2)}%`);
}

export function renderRiskBanner(riskResult: RiskResultView): void {
  const banner = requiredElement("risk-banner");
  const level = (riskResult.risk_level ?? "unknown").toLowerCase();
  const percentile = riskResult.risk_percentile;
  const reliability = riskResult.prediction_reliability ?? "unknown";
  const unavailableReason = riskResult.not_available_reason ?? "";
  const endpoint = riskResult.endpoint ?? "";
  const isResearchIndex = endpoint.toLowerCase() === "research_risk_index";

  banner.className = "risk-banner";
  renderRiskScale(riskResult);

  if (isResearchIndex && unavailableReason === "incomplete_microbiome_panel") {
    const missingCount = riskResult.missing_microbe_fields?.length ?? 0;
    banner.classList.add("risk-banner-empty");
    setText("risk-kicker", "菌群-临床研究风险指数");
    setText("risk-score", "--");
    setText(
      "risk-level",
      missingCount > 0
        ? `补齐五项核心菌群后生成（还缺 ${missingCount} 项）`
        : "补齐五项核心菌群后生成",
    );
    return;
  }

  if (isResearchIndex && unavailableReason === "invalid_microbiome_panel") {
    banner.classList.add("risk-banner-withheld");
    setText("risk-kicker", "菌群-临床研究风险指数");
    setText("risk-score", "--");
    setText("risk-level", "五项菌群总和需大于 0");
    return;
  }

  if (riskResult.prediction_available === false && unavailableReason === "missing_oncology_fields") {
    banner.classList.add("risk-banner-empty");
    setText("risk-kicker", "未提供完整肿瘤病理资料");
    setText("risk-score", "--");
    setText("risk-level", "本次不计算 PFS");
    return;
  }

  if (
    riskResult.prediction_available === false
    || reliability === "caution_out_of_training_range"
  ) {
    banner.classList.add("risk-banner-withheld");
    setText("risk-kicker", "当前输入超出模型适用范围");
    setText("risk-score", "--");
    setText("risk-level", "先核对输入，再查看风险");
    return;
  }

  setText(
    "risk-kicker",
    isResearchIndex
      ? "菌群-临床研究风险指数（参考分位）"
      : endpoint.toUpperCase() === "PFS"
        ? "AC-ICAM 队列中的 PFS 相对风险位置"
        : "研究队列中的相对位置",
  );

  if (level === "low") {
    banner.classList.add("risk-banner-low");
  } else if (level === "medium") {
    banner.classList.add("risk-banner-medium");
  } else if (level === "high") {
    banner.classList.add("risk-banner-high");
  } else {
    banner.classList.add("risk-banner-empty");
  }

  const parsedPercentile = finiteNumber(percentile);
  setText(
    "risk-score",
    parsedPercentile === null
      ? toInputValue(riskResult.risk_score) || "--"
      : `${parsedPercentile.toFixed(2)}%`,
  );
  setText("risk-level", formatRiskLevel(level, isResearchIndex));
}

const MODEL_VARIANT_LABELS: Readonly<Record<string, string>> = {
  clinical_core: "临床核心模型",
  clinical_icr: "临床 + 实测 ICR 模型",
};

export function renderV8OutcomeSummary(
  riskResult: RiskResultView,
  generalRiskResult: RiskResultView = {},
): void {
  const pfs = riskResult.pfs_probability;
  const endpoint = riskResult.endpoint ?? "";
  const predictionAvailable = riskResult.prediction_available !== false;
  const pfsAvailable = predictionAvailable && endpoint.toUpperCase() === "PFS";
  const nonV8RiskAvailable =
    predictionAvailable && !riskResult.not_available_reason && endpoint.toUpperCase() !== "PFS";
  const unavailableReason = riskResult.not_available_reason ?? "";
  const generalAvailable =
    generalRiskResult.prediction_available === true
    && (generalRiskResult.endpoint ?? "").toLowerCase() === "research_risk_index";

  setText("pfs-probability-36", pfsAvailable ? formatProbability(pfs?.["36"]) : "--");
  setText("pfs-probability-60", pfsAvailable ? formatProbability(pfs?.["60"]) : "--");

  const variantLabel = MODEL_VARIANT_LABELS[riskResult.model_variant ?? ""];
  let modelSummary = "输入超出适用范围";
  if (pfsAvailable) {
    modelSummary = variantLabel || riskResult.model_release || "非 V8 后端";
  } else if (generalAvailable) {
    modelSummary = "研究风险指数已生成；PFS 未计算";
  } else if (nonV8RiskAvailable) {
    modelSummary = riskResult.model_release || "非 V8 研究后端";
  } else if (unavailableReason === "missing_oncology_fields") {
    modelSummary = "病理资料未完整；研究指数待补充菌群";
  }

  setText("v8-model-variant", modelSummary);
  setText(
    "v8-model-scope",
    (generalAvailable ? generalRiskResult.intended_use : riskResult.intended_use)
      || "当前后端未返回 V8 PFS 适用范围。",
  );
}
