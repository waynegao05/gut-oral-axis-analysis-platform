/** DOM 访问与格式化的共用原语。所有取元素的函数都做真实的运行时窄化，不使用类型断言。 */

export function requiredElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!(element instanceof HTMLElement)) {
    throw new Error(`页面缺少必要元素：#${id}`);
  }
  return element as T;
}

export function requiredInput(id: string): HTMLInputElement {
  const element = document.getElementById(id);
  if (!(element instanceof HTMLInputElement)) {
    throw new Error(`页面缺少必要的输入框：#${id}`);
  }
  return element;
}

export function requiredSelect(id: string): HTMLSelectElement {
  const element = document.getElementById(id);
  if (!(element instanceof HTMLSelectElement)) {
    throw new Error(`页面缺少必要的下拉框：#${id}`);
  }
  return element;
}

export function requiredTextArea(id: string): HTMLTextAreaElement {
  const element = document.getElementById(id);
  if (!(element instanceof HTMLTextAreaElement)) {
    throw new Error(`页面缺少必要的文本域：#${id}`);
  }
  return element;
}

export function requiredButton(id: string): HTMLButtonElement {
  const element = document.getElementById(id);
  if (!(element instanceof HTMLButtonElement)) {
    throw new Error(`页面缺少必要的按钮：#${id}`);
  }
  return element;
}

/** 表单控件既可能是 input 也可能是 select，读取逻辑共用这一层。 */
export type FormControl = HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement;

export function setText(id: string, value: string): void {
  requiredElement(id).textContent = value;
}

export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "发生未知错误。";
}

/** 与旧实现的 `value ?? ""` 等价：null / undefined 归空串，其余走 String()。 */
export function toInputValue(value: unknown): string {
  return value === null || value === undefined ? "" : String(value);
}

export function prettyJson(payload: unknown): string {
  return JSON.stringify(payload, null, 2);
}

/** Number(value) 的有限性收敛，非有限值返回 null。 */
export function finiteNumber(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}
