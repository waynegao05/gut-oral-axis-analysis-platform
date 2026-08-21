/** payload → 表单：回填、自定义菌种行的增删。 */

import { requiredElement, requiredInput, requiredSelect, requiredTextArea, toInputValue } from "../dom";
import { BINARY_METADATA_FIELDS, PRESET_MICROBES } from "../examples";
import type { StandardizedPayload } from "../types";

function extraMicrobeContainer(): HTMLElement {
  return requiredElement("extra-microbe-rows");
}

export function clearExtraMicrobeRows(): void {
  extraMicrobeContainer().innerHTML = "";
}

export function createExtraMicrobeRow(name = "", value: string | number = ""): void {
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
  removeButton.addEventListener("click", () => {
    wrapper.remove();
  });

  wrapper.appendChild(nameInput);
  wrapper.appendChild(valueInput);
  wrapper.appendChild(removeButton);
  extraMicrobeContainer().appendChild(wrapper);
}

/**
 * 未提供该键 → 空串（表单留空）；
 * 提供了但为空数组 → 「无」（明确没有用药/过敏）。
 * 这个区分会一路传到后端的 medication_context，不能简化。
 */
function formatOptionalList(metadata: Record<string, unknown>, key: string): string {
  if (!Object.prototype.hasOwnProperty.call(metadata, key)) {
    return "";
  }
  const raw = metadata[key];
  const values: unknown[] = Array.isArray(raw) ? raw : [];
  return values.length ? values.join("\n") : "无";
}

export function populateForm(payload: StandardizedPayload): void {
  const microbes: Record<string, unknown> = payload.microbes ?? {};
  const clinical: Record<string, unknown> = payload.clinical ?? {};
  const metabolites: Record<string, unknown> = payload.metabolites ?? {};
  const metadata: Record<string, unknown> = payload.metadata ?? {};

  for (const name of PRESET_MICROBES) {
    requiredInput(`microbe-${name}`).value = toInputValue(microbes[name]);
  }

  clearExtraMicrobeRows();
  for (const [name, value] of Object.entries(microbes)) {
    if (PRESET_MICROBES.some((preset) => preset === name)) {
      continue;
    }
    createExtraMicrobeRow(name, toInputValue(value));
  }

  requiredInput("clinical-age").value = toInputValue(clinical["age"]);
  requiredSelect("clinical-sex").value = toInputValue(clinical["sex"]);
  requiredSelect("clinical-stage").value = toInputValue(clinical["stage"]);
  requiredSelect("clinical-path-t").value = toInputValue(clinical["path_t"]);
  requiredSelect("clinical-path-n").value = toInputValue(clinical["path_n"]);
  requiredSelect("clinical-path-m").value = toInputValue(clinical["path_m"]);
  requiredSelect("clinical-tumor-location").value = toInputValue(clinical["tumor_location"]);
  requiredSelect("clinical-tumor-morphology").value = toInputValue(clinical["tumor_morphology"]);
  requiredInput("clinical-icr-score").value = toInputValue(clinical["icr_score"]);
  requiredInput("clinical-bmi").value = toInputValue(clinical["bmi"]);
  requiredSelect("clinical-smoking").value = String(clinical["smoking"] ?? 0);
  requiredSelect("clinical-family-history").value = String(clinical["family_history"] ?? 0);

  requiredInput("metabolite-bile-acids").value = toInputValue(metabolites["bile_acids"]);
  requiredInput("metabolite-scfa").value = toInputValue(metabolites["scfa"]);
  requiredInput("metabolite-tryptophan").value = toInputValue(metabolites["tryptophan_metabolism"]);

  requiredTextArea("metadata-current-medications").value = formatOptionalList(
    metadata,
    "current_medications",
  );
  requiredTextArea("metadata-drug-allergies").value = formatOptionalList(metadata, "drug_allergies");
  requiredSelect("metadata-suspected-condition").value =
    toInputValue(metadata["suspected_condition"]) || "gut_risk_screening";

  for (const { id, key } of BINARY_METADATA_FIELDS) {
    requiredSelect(`metadata-${id}`).value = Object.prototype.hasOwnProperty.call(metadata, key)
      ? String(metadata[key])
      : "";
  }
}
