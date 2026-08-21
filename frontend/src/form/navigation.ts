const TAB_SELECTOR = "[data-form-tab]";
const PANEL_SELECTOR = ".form-tab-panel[role='tabpanel']";

export function getNextFormTabIndex(
  currentIndex: number,
  key: string,
  tabCount: number,
): number | null {
  if (tabCount <= 0 || currentIndex < 0 || currentIndex >= tabCount) {
    return null;
  }

  if (key === "Home") {
    return 0;
  }
  if (key === "End") {
    return tabCount - 1;
  }
  if (key === "ArrowRight") {
    return (currentIndex + 1) % tabCount;
  }
  if (key === "ArrowLeft") {
    return (currentIndex - 1 + tabCount) % tabCount;
  }
  return null;
}

function controlledPanel(root: HTMLElement, tab: HTMLButtonElement): HTMLElement {
  const panelId = tab.getAttribute("aria-controls");
  const panel = panelId ? document.getElementById(panelId) : null;
  if (!(panel instanceof HTMLElement) || !root.contains(panel)) {
    throw new Error(`表单页签 ${tab.id || "未命名"} 缺少对应内容。`);
  }
  return panel;
}

export function initializeFormNavigation(): void {
  const root = document.getElementById("analysis-form-workspace");
  const tabList = document.getElementById("analysis-form-tabs");
  if (!(root instanceof HTMLElement) || !(tabList instanceof HTMLElement)) {
    throw new Error("页面缺少输入表单分组导航。");
  }

  const tabs = Array.from(tabList.querySelectorAll<HTMLButtonElement>(TAB_SELECTOR));
  const panels = Array.from(root.querySelectorAll<HTMLElement>(PANEL_SELECTOR));
  if (tabs.length === 0 || tabs.length !== panels.length) {
    throw new Error("输入表单页签与内容分组数量不一致。");
  }

  const tabByPanel = new Map<HTMLElement, HTMLButtonElement>();
  for (const tab of tabs) {
    tabByPanel.set(controlledPanel(root, tab), tab);
  }

  const activate = (activeTab: HTMLButtonElement, focusTab: boolean): void => {
    const activePanel = controlledPanel(root, activeTab);
    for (const tab of tabs) {
      const selected = tab === activeTab;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      tab.classList.toggle("is-active", selected);
    }
    for (const panel of panels) {
      panel.hidden = panel !== activePanel;
      panel.classList.toggle("is-active", panel === activePanel);
    }

    if (focusTab) {
      activeTab.focus();
      activeTab.scrollIntoView({ block: "nearest", inline: "nearest" });
    }
  };

  for (const tab of tabs) {
    tab.addEventListener("click", () => activate(tab, false));
  }

  tabList.addEventListener("keydown", (event: KeyboardEvent) => {
    const target = event.target;
    const currentTab = target instanceof Element ? target.closest<HTMLButtonElement>(TAB_SELECTOR) : null;
    if (!currentTab || !tabList.contains(currentTab)) {
      return;
    }

    const nextIndex = getNextFormTabIndex(tabs.indexOf(currentTab), event.key, tabs.length);
    if (nextIndex === null) {
      return;
    }
    const nextTab = tabs[nextIndex];
    if (!nextTab) {
      return;
    }

    event.preventDefault();
    activate(nextTab, true);
  });

  root.addEventListener(
    "invalid",
    (event: Event) => {
      const field = event.target;
      const panel = field instanceof Element ? field.closest<HTMLElement>(PANEL_SELECTOR) : null;
      if (!panel || !panel.hidden) {
        return;
      }
      const tab = tabByPanel.get(panel);
      if (tab) {
        activate(tab, false);
      }
    },
    true,
  );

  const initialTab = tabs.find((tab) => tab.getAttribute("aria-selected") === "true") ?? tabs[0];
  if (!initialTab) {
    throw new Error("输入表单没有可用页签。");
  }
  activate(initialTab, false);
}

export function revealFirstInvalidFormField(): void {
  const root = document.getElementById("analysis-form-workspace");
  if (!(root instanceof HTMLElement)) {
    return;
  }

  const controls = root.querySelectorAll<
    HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement
  >("input, select, textarea");
  for (const control of controls) {
    if (control.checkValidity()) {
      continue;
    }
    control.focus();
    control.reportValidity();
    return;
  }
}
