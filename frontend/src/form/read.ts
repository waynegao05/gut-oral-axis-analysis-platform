/** 表单 → CanonicalPayload。留空的可选字段一律不写入载荷，绝不补默认值。 */

import { requiredElement, requiredInput, requiredSelect, requiredTextArea } from "../dom";
import { BINARY_METADATA_FIELDS, PRESET_MICROBES } from "../examples";
import type { CanonicalPayload, NumericMap } from "../types";
import {
  readNumberInput,
  readOptionalBinarySelect,
  readOptionalListInput,
  readOptionalSelect,
  readRequiredNumberInput,
  readRequiredSelect,
} from "./fields";

function readMicrobes(): NumericMap {
  const microbes: NumericMap = {};

  for (const name of PRESET_MICROBES) {
    const value = readNumberInput(requiredInput(`microbe-${name}`), `${name} 丰度`);
    if (value !== null) {
      microbes[name] = value;
    }
  }

  const rows = requiredElement("extra-microbe-rows").querySelectorAll(".microbe-row");
  for (const row of rows) {
    const nameField = row.querySelector(".microbe-name");
    const valueField = row.querySelector(".microbe-value");
    if (!(nameField instanceof HTMLInputElement) || !(valueField instanceof HTMLInputElement)) {
      throw new Error("自定义菌种行结构异常，请重新添加。");
    }

    const name = nameField.value.trim();
    if (!name) {
      continue;
    }
    const abundance = readNumberInput(valueField, `${name} 丰度`);
    if (abundance === null) {
      throw new Error(`${name} 丰度不能为空。`);
    }
    microbes[name] = abundance;
  }

  return microbes;
}

function readClinical(): CanonicalPayload["clinical"] {
  // 吸烟与家族史沿用旧实现：无条件写入，留空即写入 null。
  const clinical: CanonicalPayload["clinical"] = {
    age: readRequiredNumberInput(requiredInput("clinical-age"), "年龄"),
    sex: readRequiredSelect(requiredSelect("clinical-sex"), "生物学性别"),
    smoking: readNumberInput(requiredSelect("clinical-smoking"), "吸烟状态"),
    family_history: readNumberInput(requiredSelect("clinical-family-history"), "家族史"),
  };

  const optionalNumbers: ReadonlyArray<readonly [string, string, string]> = [
    ["stage", "clinical-stage", "AJCC 分期"],
    ["path_t", "clinical-path-t", "病理 T 分期"],
    ["path_n", "clinical-path-n", "病理 N 分期"],
  ];
  for (const [field, inputId, label] of optionalNumbers) {
    const value = readNumberInput(requiredSelect(inputId), label);
    if (value !== null) {
      clinical[field] = value;
    }
  }

  const tumorLocation = readOptionalSelect(requiredSelect("clinical-tumor-location"), "肿瘤部位");
  const tumorMorphology = readOptionalSelect(
    requiredSelect("clinical-tumor-morphology"),
    "肿瘤形态学",
  );
  if (tumorLocation !== null) {
    clinical["tumor_location"] = tumorLocation;
  }
  if (tumorMorphology !== null) {
    clinical["tumor_morphology"] = tumorMorphology;
  }

  const pathM = readNumberInput(requiredSelect("clinical-path-m"), "病理 M 分期");
  const icrScore = readNumberInput(requiredInput("clinical-icr-score"), "肿瘤 RNA ICR 评分");
  const bmi = readNumberInput(requiredInput("clinical-bmi"), "BMI");
  if (pathM !== null) {
    clinical["path_m"] = pathM;
  }
  if (icrScore !== null) {
    clinical["icr_score"] = icrScore;
  }
  if (bmi !== null) {
    clinical["bmi"] = bmi;
  }

  return clinical;
}

function readMetabolites(): NumericMap {
  const metabolites: NumericMap = {};
  const bileAcids = readNumberInput(requiredInput("metabolite-bile-acids"), "胆汁酸");
  const scfa = readNumberInput(requiredInput("metabolite-scfa"), "短链脂肪酸（SCFA）");
  const tryptophan = readNumberInput(requiredInput("metabolite-tryptophan"), "色氨酸代谢");
  if (bileAcids !== null) {
    metabolites["bile_acids"] = bileAcids;
  }
  if (scfa !== null) {
    metabolites["scfa"] = scfa;
  }
  if (tryptophan !== null) {
    metabolites["tryptophan_metabolism"] = tryptophan;
  }
  return metabolites;
}

function readMetadata(): Record<string, unknown> {
  const metadata: Record<string, unknown> = {};

  const currentMedications = readOptionalListInput(requiredTextArea("metadata-current-medications"));
  const drugAllergies = readOptionalListInput(requiredTextArea("metadata-drug-allergies"));
  if (currentMedications !== null) {
    metadata["current_medications"] = currentMedications;
  }
  if (drugAllergies !== null) {
    metadata["drug_allergies"] = drugAllergies;
  }

  metadata["suspected_condition"] =
    requiredSelect("metadata-suspected-condition").value || "gut_risk_screening";

  for (const { id, key } of BINARY_METADATA_FIELDS) {
    const value = readOptionalBinarySelect(requiredSelect(`metadata-${id}`));
    if (value !== null) {
      metadata[key] = value;
    }
  }

  return metadata;
}

export function buildCanonicalPayloadFromForm(): CanonicalPayload {
  return {
    microbes: readMicrobes(),
    clinical: readClinical(),
    metabolites: readMetabolites(),
    metadata: readMetadata(),
  };
}
