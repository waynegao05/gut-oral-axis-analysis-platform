import {
  getTransport,
  type ApiOperation,
  type BridgeOperation,
  type TransportResult,
} from "./transport";
import type { ApiEnvelope } from "./types";

export { setTransportForTesting } from "./transport";

export class ApiError extends Error {
  readonly status: number;
  readonly errorCode: string | undefined;
  readonly requestId: string | undefined;

  constructor(
    message: string,
    status: number,
    errorCode?: string,
    requestId?: string,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.errorCode = errorCode;
    this.requestId = requestId;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function decodeResponse<T extends ApiEnvelope>(result: TransportResult): T {
  if (!isRecord(result.payload)) {
    throw new ApiError("服务器返回了无法解析的响应。", result.status);
  }
  const payload = result.payload as T;
  const status = payload.status;
  const success = payload.ok === true || status === "success" || status === "ok";
  if (result.status < 200 || result.status >= 300 || !success) {
    const message =
      payload.errors?.filter(Boolean).join(" | ")
      || payload.message
      || "请求失败。";
    throw new ApiError(message, result.status, payload.error_code, payload.request_id);
  }
  return payload;
}

export async function getJson<T extends ApiEnvelope>(operation: ApiOperation): Promise<T> {
  return decodeResponse<T>(await getTransport().request(operation));
}

export async function postJson<T extends ApiEnvelope>(
  operation: ApiOperation,
  payload: unknown,
): Promise<T> {
  return decodeResponse<T>(await getTransport().request(operation, payload));
}

export async function requestBridgeJson<T extends ApiEnvelope>(
  operation: BridgeOperation,
  payload?: unknown,
): Promise<T> {
  return decodeResponse<T>(await getTransport().request(operation, payload));
}
