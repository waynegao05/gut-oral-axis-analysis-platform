import { requestBridgeJson } from "./api";
import type { ApiEnvelope } from "./types";

export interface OpenJsonResult extends ApiEnvelope {
  file_name: string;
  content: string;
}

export interface SaveJsonResult extends ApiEnvelope {
  file_name: string;
}

export interface SaveReportResult extends ApiEnvelope {
  report_id: string;
  display_name: string;
  display_location: string;
  sha256: string;
}

export interface ExportPdfResult extends ApiEnvelope {
  file_name: string;
  exported: boolean;
}

export function isDesktopHost(): boolean {
  return typeof window !== "undefined" && window.chrome?.webview !== undefined;
}

export function openJsonFile(): Promise<OpenJsonResult> {
  return requestBridgeJson<OpenJsonResult>("file.openJson");
}

export function saveJsonFile(suggestedName: string, content: unknown): Promise<SaveJsonResult> {
  return requestBridgeJson<SaveJsonResult>("file.saveJson", {
    suggested_name: suggestedName,
    content,
  });
}

export function saveStructuredReport(
  report: unknown,
  suggestedName: string,
): Promise<SaveReportResult> {
  return requestBridgeJson<SaveReportResult>("report.save", {
    report,
    suggested_name: suggestedName,
  });
}

export function exportCurrentPagePdf(): Promise<ExportPdfResult> {
  return requestBridgeJson<ExportPdfResult>("report.exportPdf");
}

export function printCurrentPage(): Promise<ApiEnvelope> {
  return requestBridgeJson<ApiEnvelope>("report.print");
}
