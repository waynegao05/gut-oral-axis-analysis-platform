/**
 * 主分析面板的编排层：事件绑定、导入/分析流程、启动引导。
 *
 * 这里保留了原 legacy-app.js 的注册时机：模块求值时就注册 DOMContentLoaded，
 * 因此仍然早于 main.ts 内的初始化逻辑执行。
 */

import { postJson } from "./api";
import { isDesktopHost, saveStructuredReport } from "./desktop-host";
import { errorMessage, prettyJson, requiredButton, requiredInput, requiredTextArea, setText } from "./dom";
import { CANONICAL_EXAMPLE, RAW_CLINICAL_EXAMPLE } from "./examples";
import { createExtraMicrobeRow, populateForm } from "./form/fill";
import { initializeFormNavigation, revealFirstInvalidFormField } from "./form/navigation";
import { buildCanonicalPayloadFromForm } from "./form/read";
import { renderResult } from "./render/result";
import { renderStandardizedPreview, setImportStatus } from "./render/status";
import type { MainAnalysisResponse, StandardizeResponse } from "./types";

function loadJsonIntoTextarea(payload: unknown): void {
  requiredTextArea("json-payload").value = prettyJson(payload);
}

async function standardizeFromTextarea(): Promise<void> {
  const textarea = requiredTextArea("json-payload");
  let payload: unknown;
  try {
    payload = JSON.parse(textarea.value);
  } catch (error: unknown) {
    setImportStatus(`JSON 解析失败：${errorMessage(error)}`, "error");
    return;
  }

  try {
    const data = await postJson<StandardizeResponse>("standardize", payload);
    populateForm(data.standardized_payload);
    renderStandardizedPreview(data.standardized_payload, data.source_format);
    setImportStatus("JSON 已成功标准化并回填到表单。", "success");
  } catch (error: unknown) {
    setImportStatus(errorMessage(error), "error");
  }
}

/** main.ts 在桌面模式下接管文件选择后，用这个入口把文本喂回主面板。 */
export async function importMainJsonText(text: string): Promise<void> {
  requiredTextArea("json-payload").value = text;
  await standardizeFromTextarea();
}

async function analyzeFromForm(): Promise<void> {
  try {
    const payload = buildCanonicalPayloadFromForm();
    renderStandardizedPreview(payload, "form_manual");
    const data = await postJson<MainAnalysisResponse>("analyze", payload);

    if (isDesktopHost()) {
      try {
        const stored = await saveStructuredReport(
          data.report,
          `gut-oral-axis-report-${new Date().toISOString().slice(0, 10)}.json`,
        );
        data.saved_to = stored.display_location;
      } catch (saveError: unknown) {
        renderResult(data);
        setImportStatus(`分析完成，但本地报告保存失败：${errorMessage(saveError)}`, "error");
        return;
      }
    }

    renderResult(data);
    setImportStatus("分析完成，结果区已更新。", "success");
  } catch (error: unknown) {
    revealFirstInvalidFormField();
    setImportStatus(errorMessage(error), "error");
  }
}

function bindEvents(): void {
  requiredButton("load-canonical-example").addEventListener("click", () => {
    loadJsonIntoTextarea(CANONICAL_EXAMPLE);
    populateForm(CANONICAL_EXAMPLE);
    renderStandardizedPreview(CANONICAL_EXAMPLE, "canonical_example");
    setImportStatus("已载入标准示例。", "success");
  });

  requiredButton("load-raw-example").addEventListener("click", () => {
    loadJsonIntoTextarea(RAW_CLINICAL_EXAMPLE);
    setImportStatus("已载入原始临床示例，点击“导入并回填表单”完成标准化。");
  });

  requiredButton("clear-json").addEventListener("click", () => {
    requiredTextArea("json-payload").value = "";
    setText("standardized-preview", "");
    setImportStatus("JSON 已清空。");
  });

  requiredButton("standardize-json").addEventListener("click", () => {
    void standardizeFromTextarea();
  });

  requiredButton("analyze-form").addEventListener("click", () => {
    void analyzeFromForm();
  });

  requiredButton("reset-form").addEventListener("click", () => {
    populateForm(CANONICAL_EXAMPLE);
    renderStandardizedPreview(CANONICAL_EXAMPLE, "canonical_example");
    setImportStatus("表单已重置为标准示例。");
  });

  requiredButton("add-microbe-row").addEventListener("click", () => {
    createExtraMicrobeRow();
  });

  requiredInput("json-file-input").addEventListener("change", (event: Event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) {
      return;
    }
    const file = target.files?.[0];
    if (!file) {
      return;
    }

    void file
      .text()
      .then((text) => importMainJsonText(text))
      .catch((error: unknown) => {
        setImportStatus(`文件读取失败：${errorMessage(error)}`, "error");
      });
  });
}

window.addEventListener("DOMContentLoaded", () => {
  initializeFormNavigation();
  bindEvents();
  loadJsonIntoTextarea(CANONICAL_EXAMPLE);
  populateForm(CANONICAL_EXAMPLE);
  renderStandardizedPreview(CANONICAL_EXAMPLE, "canonical_example");
  setText("result-json", prettyJson({ message: "等待分析。" }));
});
