export function requiredElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!(element instanceof HTMLElement)) {
    throw new Error(`页面缺少必要元素：#${id}`);
  }
  return element as T;
}

export function setText(id: string, value: string): void {
  requiredElement(id).textContent = value;
}

export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "发生未知错误。";
}
