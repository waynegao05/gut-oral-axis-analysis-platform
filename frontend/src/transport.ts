export type ApiOperation =
  | "standardize"
  | "analyze"
  | "oralAdenoma.schema"
  | "oralAdenoma.analyze";

export type HostOperation =
  | "file.openJson"
  | "file.saveJson"
  | "report.save"
  | "report.list"
  | "report.exportPdf"
  | "report.print"
  | "app.getVersion"
  | "device.discover";

export type BridgeOperation = ApiOperation | HostOperation;

export interface OperationDefinition {
  readonly method: "GET" | "POST";
  readonly browserPath: string;
  readonly enginePath: string;
}

export const API_OPERATIONS: Readonly<Record<ApiOperation, OperationDefinition>> = {
  standardize: {
    method: "POST",
    browserPath: "/standardize",
    enginePath: "/api/v1/standardize",
  },
  analyze: {
    method: "POST",
    browserPath: "/analyze",
    enginePath: "/api/v1/analyze",
  },
  "oralAdenoma.schema": {
    method: "GET",
    browserPath: "/internal/oral-adenoma/schema",
    enginePath: "/api/v1/oral-adenoma/schema",
  },
  "oralAdenoma.analyze": {
    method: "POST",
    browserPath: "/internal/oral-adenoma/analyze",
    enginePath: "/api/v1/oral-adenoma/analyze",
  },
};

export interface TransportResult {
  readonly status: number;
  readonly payload: unknown;
}

export interface ApiTransport {
  request(operation: BridgeOperation, payload?: unknown): Promise<TransportResult>;
  dispose?(): void;
}

export interface WebViewMessageEvent {
  readonly data: unknown;
}

export interface WebViewBridge {
  postMessage(message: unknown): void;
  addEventListener(
    type: "message",
    listener: (event: WebViewMessageEvent) => void,
  ): void;
  removeEventListener(
    type: "message",
    listener: (event: WebViewMessageEvent) => void,
  ): void;
}

declare global {
  interface Window {
    chrome?: {
      webview?: WebViewBridge;
    };
  }
}

interface DesktopRequestMessage {
  readonly type: "goa.request";
  readonly version: 1;
  readonly requestId: string;
  readonly operation: BridgeOperation;
  readonly payload?: unknown;
}

interface DesktopResponseMessage {
  readonly type: "goa.response";
  readonly version: 1;
  readonly requestId: string;
  readonly status: number;
  readonly payload: unknown;
}

interface PendingRequest {
  readonly resolve: (result: TransportResult) => void;
  readonly reject: (error: Error) => void;
  readonly timeoutId: ReturnType<typeof setTimeout>;
}

export const DEFAULT_DESKTOP_TIMEOUT_MS = 120_000;
export const MAX_DESKTOP_MESSAGE_BYTES = 2 * 1024 * 1024;

function byteLength(value: unknown): number {
  return new TextEncoder().encode(JSON.stringify(value)).byteLength;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isDesktopResponse(value: unknown): value is DesktopResponseMessage {
  if (!isRecord(value)) {
    return false;
  }
  return (
    value.type === "goa.response"
    && value.version === 1
    && typeof value.requestId === "string"
    && typeof value.status === "number"
    && Number.isInteger(value.status)
    && value.status >= 100
    && value.status <= 599
    && Object.prototype.hasOwnProperty.call(value, "payload")
  );
}

function newRequestId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

export class HttpTransport implements ApiTransport {
  constructor(
    private readonly fetchImplementation: typeof fetch = globalThis.fetch.bind(globalThis),
  ) {}

  async request(operation: BridgeOperation, payload?: unknown): Promise<TransportResult> {
    if (!isApiOperation(operation)) {
      throw new Error("该系统操作仅在 Windows 桌面版中可用。");
    }
    const definition = API_OPERATIONS[operation];
    const init: RequestInit = {
      method: definition.method,
      headers: { Accept: "application/json" },
      credentials: "same-origin",
    };
    if (definition.method === "POST") {
      init.headers = {
        Accept: "application/json",
        "Content-Type": "application/json",
      };
      init.body = JSON.stringify(payload ?? {});
    }

    const response = await this.fetchImplementation(definition.browserPath, init);
    let responsePayload: unknown;
    try {
      responsePayload = await response.json();
    } catch {
      responsePayload = {
        status: "error",
        error_code: "INVALID_RESPONSE",
        message: "服务器返回了无法解析的响应。",
      };
    }
    return { status: response.status, payload: responsePayload };
  }
}

export class WebViewTransport implements ApiTransport {
  private readonly pending = new Map<string, PendingRequest>();
  private readonly handleMessageBound = (event: WebViewMessageEvent): void => {
    this.handleMessage(event);
  };

  constructor(
    private readonly bridge: WebViewBridge,
    private readonly timeoutMs = DEFAULT_DESKTOP_TIMEOUT_MS,
    private readonly maxMessageBytes = MAX_DESKTOP_MESSAGE_BYTES,
  ) {
    if (timeoutMs <= 0) {
      throw new Error("WebView2 请求超时必须大于 0。");
    }
    if (maxMessageBytes <= 0) {
      throw new Error("WebView2 消息大小限制必须大于 0。");
    }
    bridge.addEventListener("message", this.handleMessageBound);
  }

  request(operation: BridgeOperation, payload?: unknown): Promise<TransportResult> {
    const requestId = newRequestId();
    const message: DesktopRequestMessage = {
      type: "goa.request",
      version: 1,
      requestId,
      operation,
      ...(payload === undefined ? {} : { payload }),
    };
    if (byteLength(message) > this.maxMessageBytes) {
      return Promise.reject(new Error("提交的数据过大，无法发送到本地分析引擎。"));
    }

    return new Promise<TransportResult>((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        this.pending.delete(requestId);
        reject(new Error("本地分析请求超时，请检查分析引擎状态后重试。"));
      }, this.timeoutMs);
      this.pending.set(requestId, { resolve, reject, timeoutId });
      try {
        this.bridge.postMessage(message);
      } catch (error) {
        clearTimeout(timeoutId);
        this.pending.delete(requestId);
        reject(error instanceof Error ? error : new Error("无法连接 Windows 宿主。"));
      }
    });
  }

  dispose(): void {
    this.bridge.removeEventListener("message", this.handleMessageBound);
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timeoutId);
      pending.reject(new Error("Windows 宿主连接已关闭。"));
    }
    this.pending.clear();
  }

  private handleMessage(event: WebViewMessageEvent): void {
    if (!isDesktopResponse(event.data)) {
      return;
    }
    const pending = this.pending.get(event.data.requestId);
    if (!pending) {
      return;
    }
    clearTimeout(pending.timeoutId);
    this.pending.delete(event.data.requestId);
    pending.resolve({ status: event.data.status, payload: event.data.payload });
  }
}

let activeTransport: ApiTransport | undefined;

export function createDefaultTransport(): ApiTransport {
  const bridge = typeof window !== "undefined" ? window.chrome?.webview : undefined;
  return bridge ? new WebViewTransport(bridge) : new HttpTransport();
}

export function getTransport(): ApiTransport {
  activeTransport ??= createDefaultTransport();
  return activeTransport;
}

export function setTransportForTesting(transport: ApiTransport | undefined): void {
  activeTransport?.dispose?.();
  activeTransport = transport;
}

function isApiOperation(operation: BridgeOperation): operation is ApiOperation {
  return Object.prototype.hasOwnProperty.call(API_OPERATIONS, operation);
}
