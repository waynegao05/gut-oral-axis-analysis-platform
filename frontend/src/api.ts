import type { ApiEnvelope } from "./types";

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function decodeResponse<T extends ApiEnvelope>(response: Response): Promise<T> {
  let payload: T;
  try {
    payload = (await response.json()) as T;
  } catch {
    throw new ApiError("服务器返回了无法解析的响应。", response.status);
  }
  if (!response.ok || !payload.ok) {
    const message = payload.errors?.filter(Boolean).join(" | ") || "请求失败。";
    throw new ApiError(message, response.status);
  }
  return payload;
}

export async function getJson<T extends ApiEnvelope>(url: string): Promise<T> {
  return decodeResponse<T>(
    await fetch(url, {
      headers: { Accept: "application/json" },
      credentials: "same-origin",
    }),
  );
}

export async function postJson<T extends ApiEnvelope>(
  url: string,
  payload: unknown,
): Promise<T> {
  return decodeResponse<T>(
    await fetch(url, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      credentials: "same-origin",
      body: JSON.stringify(payload),
    }),
  );
}
