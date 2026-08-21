/** 导入状态条与标准化预览。 */

import { prettyJson, requiredElement } from "../dom";

export type ImportStatusVariant = "" | "error" | "success";

export function setImportStatus(message: string, variant: ImportStatusVariant = ""): void {
  const node = requiredElement("import-status");
  node.textContent = message;
  node.className = "status-box";
  if (variant) {
    node.classList.add(`status-${variant}`);
  }
}

export function renderStandardizedPreview(payload: unknown, sourceFormat: string): void {
  requiredElement("standardized-preview").textContent = prettyJson({
    source_format: sourceFormat,
    standardized_payload: payload,
  });
}
