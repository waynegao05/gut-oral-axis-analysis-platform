import { getJson, postJson } from "./api";
import { errorMessage, formatPercent, requiredElement, setText } from "./dom";
import type {
  NumericMap,
  OralAdenomaAnalysisResponse,
  OralAdenomaRequest,
  OralAdenomaResult,
  OralAdenomaSchemaResponse,
} from "./types";


let schemaPromise: Promise<OralAdenomaSchemaResponse> | undefined;

function getSchema(): Promise<OralAdenomaSchemaResponse> {
  schemaPromise ??= getJson<OralAdenomaSchemaResponse>(
    "/internal/oral-adenoma/schema",
  );
  return schemaPromise;
}

function setStatus(message: string, variant = ""): void {
  const node = requiredElement<HTMLDivElement>("oral-adenoma-status");
  node.textContent = message;
  node.className = "status-box";
  if (variant) {
    node.classList.add(`status-${variant}`);
  }
}

function parseAbundanceJson(text: string): NumericMap {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    throw new Error(`口腔菌群 JSON 解析失败：${errorMessage(error)}`);
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("口腔菌群数据必须是“菌属名称: 百分比”的 JSON 对象。");
  }

  const output: NumericMap = {};
  for (const [name, rawValue] of Object.entries(parsed)) {
    if (typeof rawValue !== "number" || !Number.isFinite(rawValue)) {
      throw new Error(`${name} 必须是有限数字。`);
    }
    if (rawValue < 0 || rawValue > 100) {
      throw new Error(`${name} 必须是 0 到 100 之间的百分比。`);
    }
    output[name] = rawValue;
  }
  return output;
}

function renderResult(result: OralAdenomaResult): void {
  setText("oral-adenoma-decision", result.result_label);
  setText("oral-adenoma-probability", formatPercent(result.adenoma_probability));
  setText(
    "oral-adenoma-threshold",
    formatPercent(result.operating_threshold),
  );
  setText(
    "oral-adenoma-sensitivity",
    `${formatPercent(result.formal_internal_metrics.adenoma_sensitivity.value)} `
      + `(${result.formal_internal_metrics.adenoma_sensitivity.numerator}/`
      + `${result.formal_internal_metrics.adenoma_sensitivity.denominator})`,
  );
  setText(
    "oral-adenoma-fpr",
    `${formatPercent(result.formal_internal_metrics.false_positive_rate.value)} `
      + `(${result.formal_internal_metrics.false_positive_rate.numerator}/`
      + `${result.formal_internal_metrics.false_positive_rate.denominator})`,
  );
  setText(
    "oral-adenoma-auc",
    result.formal_internal_metrics.roc_auc.value.toFixed(4),
  );
  setText("oral-adenoma-boundary", result.claim_boundary);

  const card = requiredElement<HTMLElement>("oral-adenoma-result");
  card.classList.remove("screen-positive", "screen-negative");
  card.classList.add(result.screen_positive ? "screen-positive" : "screen-negative");
  card.hidden = false;
}

function templatePayload(schema: OralAdenomaSchemaResponse): NumericMap {
  return Object.fromEntries(schema.taxonomies.map((taxonomy) => [taxonomy, 0]));
}

function downloadTemplate(schema: OralAdenomaSchemaResponse): void {
  const blob = new Blob(
    [`${JSON.stringify(templatePayload(schema), null, 2)}\n`],
    { type: "application/json;charset=utf-8" },
  );
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "oral_adenoma_381_genus_template.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

async function runAnalysis(): Promise<void> {
  const button = requiredElement<HTMLButtonElement>("run-oral-adenoma");
  button.disabled = true;
  setStatus("正在校验完整口腔菌群并运行内部模型……");
  try {
    const schema = await getSchema();
    const textarea = requiredElement<HTMLTextAreaElement>("oral-adenoma-json");
    const abundances = parseAbundanceJson(textarea.value);
    if (Object.keys(abundances).length !== schema.feature_count) {
      throw new Error(
        `模型要求完整的 ${schema.feature_count} 个口腔菌属，`
          + `当前提供 ${Object.keys(abundances).length} 个。`,
      );
    }
    const sampleType = requiredElement<HTMLSelectElement>(
      "oral-adenoma-sample-type",
    ).value as OralAdenomaRequest["sample_type"];
    const payload: OralAdenomaRequest = {
      sample_type: sampleType,
      oral_abundances: abundances,
    };
    const response = await postJson<OralAdenomaAnalysisResponse>(
      "/internal/oral-adenoma/analyze",
      payload,
    );
    renderResult(response.oral_adenoma_result);
    setStatus("内部口腔腺瘤研究结果已生成。", "success");
  } catch (error) {
    setStatus(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function handleFileInput(event: Event): Promise<void> {
  const input = event.currentTarget;
  if (!(input instanceof HTMLInputElement) || !input.files?.[0]) {
    return;
  }
  try {
    const text = await input.files[0].text();
    parseAbundanceJson(text);
    requiredElement<HTMLTextAreaElement>("oral-adenoma-json").value = text;
    setStatus("口腔菌群 JSON 已载入，请确认样本类型后运行。", "success");
  } catch (error) {
    setStatus(errorMessage(error), "error");
  }
}

export function initializeOralAdenomaPanel(): void {
  if (!document.getElementById("oral-adenoma-panel")) {
    return;
  }

  requiredElement<HTMLButtonElement>("download-oral-template").addEventListener(
    "click",
    () => {
      void getSchema()
        .then(downloadTemplate)
        .catch((error: unknown) => setStatus(errorMessage(error), "error"));
    },
  );
  requiredElement<HTMLInputElement>("oral-adenoma-file").addEventListener(
    "change",
    (event) => void handleFileInput(event),
  );
  requiredElement<HTMLButtonElement>("run-oral-adenoma").addEventListener(
    "click",
    () => void runAnalysis(),
  );

  void getSchema()
    .then((schema) => {
      setText("oral-adenoma-feature-count", String(schema.feature_count));
      setText("oral-adenoma-release", schema.model_release);
      setStatus("模型已就绪。请上传完整的381菌属百分比JSON。", "success");
    })
    .catch((error: unknown) => setStatus(errorMessage(error), "error"));
}
