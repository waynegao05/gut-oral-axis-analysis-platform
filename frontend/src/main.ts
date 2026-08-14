import { importMainJsonText } from "./legacy-app.js";
import {
  exportCurrentPagePdf,
  isDesktopHost,
  openJsonFile,
  printCurrentPage,
} from "./desktop-host";
import {
  importOralAdenomaJsonText,
  initializeOralAdenomaPanel,
} from "./oral-adenoma";
import { errorMessage, requiredElement } from "./dom";


function initializeDesktopFeatures(): void {
  if (!isDesktopHost()) {
    return;
  }

  const mainFileInput = requiredElement<HTMLInputElement>("json-file-input");
  mainFileInput.addEventListener(
    "click",
    (event) => {
      event.preventDefault();
      void openJsonFile()
        .then((result) => importMainJsonText(result.content))
        .catch((error: unknown) => window.alert(errorMessage(error)));
    },
    { capture: true },
  );

  const oralFileInput = document.getElementById("oral-adenoma-file");
  oralFileInput?.addEventListener(
    "click",
    (event) => {
      event.preventDefault();
      void openJsonFile()
        .then((result) => importOralAdenomaJsonText(result.content))
        .catch((error: unknown) => window.alert(errorMessage(error)));
    },
    { capture: true },
  );

  const actions = requiredElement<HTMLDivElement>("desktop-report-actions");
  actions.hidden = false;
  requiredElement<HTMLButtonElement>("desktop-export-pdf").addEventListener(
    "click",
    () => void exportCurrentPagePdf().catch((error: unknown) => window.alert(errorMessage(error))),
  );
  requiredElement<HTMLButtonElement>("desktop-print").addEventListener(
    "click",
    () => void printCurrentPage().catch((error: unknown) => window.alert(errorMessage(error))),
  );
}


window.addEventListener("DOMContentLoaded", () => {
  initializeOralAdenomaPanel();
  initializeDesktopFeatures();
});
