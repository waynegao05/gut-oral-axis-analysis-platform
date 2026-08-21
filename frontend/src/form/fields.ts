/**
 * 表单读取原语。
 * 每个函数都保留旧实现的报错文案与空值语义，改动仅限于加类型。
 */

import type { FormControl } from "../dom";

/** 空值返回 null；非有限数或未通过 HTML 约束校验时抛出中文错误。 */
export function readNumberInput(input: FormControl, label: string): number | null {
  const rawValue = input.value.trim();
  if (rawValue === "") {
    return null;
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

export function readRequiredNumberInput(input: FormControl, label: string): number {
  const value = readNumberInput(input, label);
  if (value === null) {
    throw new Error(`${label} 不能为空。`);
  }
  return value;
}

/**
 * 空值返回 null（视为「未提供」）；
 * 明确填写「无 / none / no」返回空数组（视为「确认没有」）；
 * 其余按逗号、分号、换行切分。
 */
export function readOptionalListInput(input: FormControl): string[] | null {
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

/** 「未提供」选项的 value 是空串，返回 null；否则转数字。 */
export function readOptionalBinarySelect(input: FormControl): number | null {
  if (input.value === "") {
    return null;
  }
  return Number(input.value);
}

export function readRequiredSelect(input: FormControl, label: string): string {
  const value = input.value.trim();
  if (!value) {
    throw new Error(`${label} 不能为空。`);
  }
  if (!input.checkValidity()) {
    throw new Error(`${label} 输入非法：${input.validationMessage}`);
  }
  return value;
}

export function readOptionalSelect(input: FormControl, label: string): string | null {
  const value = input.value.trim();
  if (!value) {
    return null;
  }
  if (!input.checkValidity()) {
    throw new Error(`${label} 输入非法：${input.validationMessage}`);
  }
  return value;
}
