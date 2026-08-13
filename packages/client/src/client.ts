import type {
  AnalysisInput,
  MainAnalysisResponse,
  OralAdenomaAnalysisResponse,
  OralAdenomaRequest,
  OralAdenomaSchemaResponse,
  StandardizeResponse,
  SuccessEnvelope,
} from "./types.js";

export interface GutOralAxisClientOptions {
  baseUrl?: string;
  fetch?: typeof globalThis.fetch;
  credentials?: RequestCredentials;
  headers?: HeadersInit;
  timeoutMs?: number;
}

export interface GutOralAxisRequestOptions {
  signal?: AbortSignal;
  headers?: HeadersInit;
  timeoutMs?: number;
}

export class GutOralAxisApiError extends Error {
  readonly status: number;
  readonly payload: unknown;

  constructor(message: string, status: number, payload: unknown) {
    super(message);
    this.name = "GutOralAxisApiError";
    this.status = status;
    this.payload = payload;
  }
}

export class GutOralAxisRequestError extends Error {
  readonly causeValue: unknown;

  constructor(message: string, causeValue: unknown) {
    super(message);
    this.name = "GutOralAxisRequestError";
    this.causeValue = causeValue;
  }
}

interface RequestDefinition {
  method: "GET" | "POST";
  body?: unknown;
}

const DEFAULT_TIMEOUT_MS = 30_000;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function errorMessage(payload: unknown, status: number): string {
  if (isRecord(payload) && Array.isArray(payload["errors"])) {
    const errors = payload["errors"].filter(
      (entry): entry is string => typeof entry === "string" && entry.length > 0,
    );
    if (errors.length > 0) {
      return errors.join("; ");
    }
  }
  return `Gut-Oral Axis API request failed with HTTP ${status}.`;
}

function assertTimeout(timeoutMs: number): void {
  if (!Number.isFinite(timeoutMs) || timeoutMs < 0) {
    throw new TypeError("timeoutMs must be a finite non-negative number.");
  }
}

export class GutOralAxisClient {
  private readonly baseUrl: string;
  private readonly fetchImpl: typeof globalThis.fetch;
  private readonly credentials: RequestCredentials;
  private readonly defaultHeaders: Headers;
  private readonly timeoutMs: number;

  constructor(options: GutOralAxisClientOptions = {}) {
    const fetchImpl = options.fetch ?? globalThis.fetch;
    if (typeof fetchImpl !== "function") {
      throw new TypeError(
        "No Fetch API implementation is available. Use Node.js 20+, a modern browser, or pass options.fetch.",
      );
    }

    const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    assertTimeout(timeoutMs);

    this.baseUrl = (options.baseUrl ?? "").replace(/\/+$/, "");
    this.fetchImpl = fetchImpl;
    this.credentials = options.credentials ?? "same-origin";
    this.defaultHeaders = new Headers(options.headers);
    this.timeoutMs = timeoutMs;
  }

  standardize(
    payload: AnalysisInput,
    options: GutOralAxisRequestOptions = {},
  ): Promise<StandardizeResponse> {
    return this.request("/standardize", { method: "POST", body: payload }, options);
  }

  analyze(
    payload: AnalysisInput,
    options: GutOralAxisRequestOptions = {},
  ): Promise<MainAnalysisResponse> {
    return this.request("/analyze", { method: "POST", body: payload }, options);
  }

  getOralAdenomaSchema(
    options: GutOralAxisRequestOptions = {},
  ): Promise<OralAdenomaSchemaResponse> {
    return this.request(
      "/internal/oral-adenoma/schema",
      { method: "GET" },
      options,
    );
  }

  analyzeOralAdenoma(
    payload: OralAdenomaRequest,
    options: GutOralAxisRequestOptions = {},
  ): Promise<OralAdenomaAnalysisResponse> {
    return this.request(
      "/internal/oral-adenoma/analyze",
      { method: "POST", body: payload },
      options,
    );
  }

  private async request<T extends SuccessEnvelope>(
    path: string,
    definition: RequestDefinition,
    options: GutOralAxisRequestOptions,
  ): Promise<T> {
    const timeoutMs = options.timeoutMs ?? this.timeoutMs;
    assertTimeout(timeoutMs);

    const headers = new Headers(this.defaultHeaders);
    headers.set("accept", "application/json");
    if (definition.body !== undefined) {
      headers.set("content-type", "application/json");
    }
    new Headers(options.headers).forEach((value, key) => headers.set(key, value));

    const controller = new AbortController();
    let timedOut = false;
    let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
    const relayAbort = (): void => controller.abort();

    if (options.signal?.aborted) {
      controller.abort();
    } else {
      options.signal?.addEventListener("abort", relayAbort, { once: true });
    }

    if (timeoutMs > 0) {
      timeoutHandle = setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, timeoutMs);
    }

    const requestInit: RequestInit = {
      method: definition.method,
      headers,
      credentials: this.credentials,
      signal: controller.signal,
    };
    if (definition.body !== undefined) {
      requestInit.body = JSON.stringify(definition.body);
    }

    let response: Response;
    try {
      response = await this.fetchImpl(`${this.baseUrl}${path}`, requestInit);
    } catch (error) {
      if (timedOut) {
        throw new GutOralAxisRequestError(
          `Gut-Oral Axis API request timed out after ${timeoutMs} ms.`,
          error,
        );
      }
      if (options.signal?.aborted) {
        throw new GutOralAxisRequestError("Gut-Oral Axis API request was aborted.", error);
      }
      throw new GutOralAxisRequestError("Gut-Oral Axis API network request failed.", error);
    } finally {
      if (timeoutHandle !== undefined) {
        clearTimeout(timeoutHandle);
      }
      options.signal?.removeEventListener("abort", relayAbort);
    }

    let payload: unknown;
    try {
      payload = await response.json();
    } catch (error) {
      throw new GutOralAxisRequestError(
        `Gut-Oral Axis API returned a non-JSON response with HTTP ${response.status}.`,
        error,
      );
    }

    if (!response.ok || !isRecord(payload) || payload["ok"] !== true) {
      throw new GutOralAxisApiError(
        errorMessage(payload, response.status),
        response.status,
        payload,
      );
    }

    return payload as T;
  }
}
