import { app } from "../../scripts/app.js";

const NODE_CLASS = "LongCat_Video_SM_Model";
const WIDGET_NAME = "inference_weight_mode";
const DOM_WIDGET_NAME = "longcat_inference_weight_mode_selector";
const STYLE_ID = "longcat-avatar-model-selector-style";

const MODES = {
  single_file_safetensors: {
    title: "Single Safetensors",
    badge: "LOCAL",
    detail: "Converted DiT file",
    path: "ComfyUI/models/diffusion_models/LongCat-Video-Avatar-1.5-int8.safetensors",
    icon: "folder",
  },
  official_sharded: {
    title: "Official Sharded",
    badge: "AUTO",
    detail: "Avatar 1.5 BF16 sharded DiT",
    path: "ComfyUI/models/longcat/LongCat-Video-Avatar-1.5/base_model/",
    icon: "cloud",
  },
  official_int8_sharded: {
    title: "Official INT8",
    badge: "LOW VRAM",
    detail: "Avatar 1.5 INT8 sharded DiT",
    path: "ComfyUI/models/longcat/LongCat-Video-Avatar-1.5/base_model_int8/",
    icon: "chip",
  },
};

const ICONS = {
  folder:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 7.5A2.5 2.5 0 0 1 5.5 5h4l2 2h7A2.5 2.5 0 0 1 21 9.5v7A2.5 2.5 0 0 1 18.5 19h-13A2.5 2.5 0 0 1 3 16.5z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>',
  cloud:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 18h10.5a4 4 0 0 0 .3-8A6 6 0 0 0 6.2 8.4 4.8 4.8 0 0 0 7 18z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 11v6m0 0-2.2-2.2M12 17l2.2-2.2" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  chip:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="2" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M4 9h3m-3 6h3m10-6h3m-3 6h3M9 4v3m6-3v3M9 17v3m6-3v3" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
  box:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 8.5 12 4l8 4.5v7L12 20l-8-4.5z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M4 8.5 12 13l8-4.5M12 13v7" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>',
};

const CSS = `
.longcat-model-selector {
  position: relative;
  width: 100%;
  box-sizing: border-box;
  padding: 2px 0;
  --longcat-accent: #00ff9d;
  --longcat-accent-dim: rgba(0, 255, 157, 0.15);
  --longcat-border: rgba(0, 255, 157, 0.34);
  --longcat-bg: rgba(8, 12, 17, 0.96);
  --longcat-bg-hover: rgba(0, 255, 157, 0.08);
  --longcat-text: #e7ecea;
  --longcat-muted: rgba(231, 236, 234, 0.58);
  --longcat-font: "JetBrains Mono", "Fira Code", "Cascadia Code", Consolas, monospace;
}
.longcat-model-selector__row {
  display: flex;
  align-items: center;
  gap: 4px;
  width: 100%;
}
.longcat-model-selector__display,
.longcat-model-selector__folder {
  min-height: 28px;
  border: 1px solid var(--longcat-border);
  border-radius: 2px;
  background: var(--longcat-bg);
  color: var(--longcat-text);
  font-family: var(--longcat-font);
  box-sizing: border-box;
}
.longcat-model-selector__display {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 4px 8px;
  cursor: pointer;
  overflow: hidden;
}
.longcat-model-selector__display:hover,
.longcat-model-selector__folder:hover {
  border-color: var(--longcat-accent);
  box-shadow: 0 0 8px rgba(0, 255, 157, 0.38);
}
.longcat-model-selector__icon,
.longcat-model-selector__folder {
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--longcat-accent);
  flex-shrink: 0;
}
.longcat-model-selector__icon svg,
.longcat-model-selector__folder svg {
  width: 16px;
  height: 16px;
}
.longcat-model-selector__label {
  flex: 1;
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: 12px;
  line-height: 1.25;
}
.longcat-model-selector__arrow {
  color: var(--longcat-accent);
  font-size: 9px;
  opacity: 0.55;
}
.longcat-model-selector__folder {
  all: unset;
  display: flex;
  width: 32px;
  min-height: 28px;
  border: 1px solid var(--longcat-border);
  border-radius: 2px;
  background: var(--longcat-bg);
  color: var(--longcat-accent);
  font-family: var(--longcat-font);
  box-sizing: border-box;
  cursor: pointer;
}
.longcat-model-selector__path {
  display: none;
  margin-top: 3px;
  color: var(--longcat-muted);
  font-family: var(--longcat-font);
  font-size: 10px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.longcat-model-selector__path--visible {
  display: block;
}
.longcat-model-dropdown {
  position: relative;
  z-index: 20;
  width: 100%;
  margin-top: 4px;
  box-sizing: border-box;
  background: var(--longcat-bg, rgba(8, 12, 17, 0.96));
  border: 1px solid var(--longcat-accent, #00ff9d);
  box-shadow: 0 0 16px rgba(0, 255, 157, 0.32), inset 0 0 12px rgba(0, 255, 157, 0.07);
  color: var(--longcat-text, #e7ecea);
  font-family: var(--longcat-font, Consolas, monospace);
  overflow: hidden;
}
.longcat-model-dropdown::before {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background: repeating-linear-gradient(0deg, rgba(0, 255, 157, 0.035), rgba(0, 255, 157, 0.035) 1px, transparent 1px, transparent 5px);
}
.longcat-model-dropdown__header,
.longcat-model-dropdown__item {
  position: relative;
  display: flex;
  align-items: center;
}
.longcat-model-dropdown__header {
  gap: 8px;
  min-height: 24px;
  padding: 4px 8px;
  color: var(--longcat-accent, #00ff9d);
  border-bottom: 1px solid rgba(0, 255, 157, 0.24);
  font-size: 10px;
  letter-spacing: 1.5px;
}
.longcat-model-dropdown__header svg {
  width: 14px;
  height: 14px;
  flex-shrink: 0;
}
.longcat-model-dropdown__item {
  width: 100%;
  min-height: 34px;
  padding: 5px 8px;
  gap: 7px;
  border: 0;
  border-bottom: 1px solid rgba(0, 255, 157, 0.13);
  background: transparent;
  color: inherit;
  cursor: pointer;
  text-align: left;
  box-sizing: border-box;
}
.longcat-model-dropdown__item:hover,
.longcat-model-dropdown__item--active {
  background: rgba(0, 255, 157, 0.09);
}
.longcat-model-dropdown__text {
  min-width: 0;
  flex: 1;
}
.longcat-model-dropdown__title {
  font-size: 11px;
  line-height: 1.25;
}
.longcat-model-dropdown__detail {
  margin-top: 2px;
  color: rgba(231, 236, 234, 0.55);
  font-size: 10px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.longcat-model-dropdown__badge {
  color: var(--longcat-accent, #00ff9d);
  border: 1px solid rgba(0, 255, 157, 0.45);
  background: rgba(0, 255, 157, 0.12);
  padding: 1px 5px;
  font-size: 9px;
  flex-shrink: 0;
}
.longcat-model-dropdown__item--folder .longcat-model-dropdown__detail {
  color: rgba(231, 236, 234, 0.72);
}
`;

let openDropdown = null;
const enhancedNodes = new WeakSet();

function injectStyles() {
  if (document.getElementById(STYLE_ID)) {
    return;
  }
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = CSS;
  document.head.appendChild(style);
}

function getWidgetValues(widget) {
  const values = widget?.options?.values ?? widget?.options;
  return Array.isArray(values) ? values : Object.keys(MODES);
}

function modeInfo(value) {
  return MODES[value] ?? {
    title: value || "Select model mode",
    badge: "MODE",
    detail: "LongCat inference weight mode",
    path: "",
    icon: "box",
  };
}

function updateDisplay(widget, label, icon, path) {
  const info = modeInfo(widget.value);
  label.textContent = info.title;
  icon.innerHTML = ICONS[info.icon] ?? ICONS.box;
  path.textContent = info.path;
}

function hideOriginalWidget(widget) {
  if (!widget || widget.__longcatAvatarWrapped) {
    return;
  }
  // IMPORTANT: the original Combo remains serialized; this only hides canvas drawing.
  widget.__longcatAvatarWrapped = true;
  widget.type = "converted-widget";
  widget.computeSize = () => [0, -4];
  widget.draw = () => {};
  widget.options ??= {};
  widget.options.canvasOnly = true;
}

function closeDropdown() {
  if (!openDropdown) {
    return;
  }
  openDropdown.root.__longcatDropdownOpen = false;
  openDropdown.dropdown.remove();
  openDropdown.node.setDirtyCanvas?.(true, true);
  openDropdown = null;
}

function setWidgetValue(node, widget, value, label, icon, path) {
  widget.value = value;
  updateDisplay(widget, label, icon, path);
  node.setDirtyCanvas?.(true, true);
  widget.callback?.(value, app.canvas, node, widget);
}

function showDropdown(node, widget, root, label, icon, path) {
  closeDropdown();
  const values = getWidgetValues(widget).filter((value) => Object.prototype.hasOwnProperty.call(MODES, value));
  const dropdown = document.createElement("div");
  dropdown.className = "longcat-model-dropdown longcat-model-selector";
  dropdown.setAttribute("role", "listbox");

  const header = document.createElement("div");
  header.className = "longcat-model-dropdown__header";
  header.innerHTML = `${ICONS.box}<span>SELECT MODEL</span>`;
  dropdown.append(header);

  for (const value of values) {
    const info = modeInfo(value);
    const item = document.createElement("button");
    item.type = "button";
    item.className = `longcat-model-dropdown__item${value === widget.value ? " longcat-model-dropdown__item--active" : ""}`;
    item.innerHTML = `
      <span class="longcat-model-selector__icon">${ICONS[info.icon] ?? ICONS.box}</span>
      <span class="longcat-model-dropdown__text">
        <span class="longcat-model-dropdown__title">${info.title}</span>
        <span class="longcat-model-dropdown__detail">${info.detail}</span>
      </span>
      <span class="longcat-model-dropdown__badge">${info.badge}</span>
    `;
    item.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      setWidgetValue(node, widget, value, label, icon, path);
      closeDropdown();
    });
    dropdown.append(item);
  }

  dropdown.addEventListener("pointerdown", (event) => event.stopPropagation());
  root.append(dropdown);
  root.__longcatDropdownOpen = true;
  openDropdown = { dropdown, root, node };
  setTimeout(() => document.addEventListener("pointerdown", closeDropdown, { once: true }), 0);
  node.setDirtyCanvas?.(true, true);
}

function showFolderPanel(node, widget, root, label, icon, path) {
  closeDropdown();
  const values = getWidgetValues(widget).filter((value) => Object.prototype.hasOwnProperty.call(MODES, value));
  const dropdown = document.createElement("div");
  dropdown.className = "longcat-model-dropdown longcat-model-selector";
  dropdown.setAttribute("role", "listbox");

  const header = document.createElement("div");
  header.className = "longcat-model-dropdown__header";
  header.innerHTML = `${ICONS.folder}<span>MODEL FOLDERS</span>`;
  dropdown.append(header);

  for (const value of values) {
    const info = modeInfo(value);
    const item = document.createElement("button");
    item.type = "button";
    item.className = `longcat-model-dropdown__item longcat-model-dropdown__item--folder${
      value === widget.value ? " longcat-model-dropdown__item--active" : ""
    }`;
    item.innerHTML = `
      <span class="longcat-model-selector__icon">${ICONS[info.icon] ?? ICONS.box}</span>
      <span class="longcat-model-dropdown__text">
        <span class="longcat-model-dropdown__title">${info.title}</span>
        <span class="longcat-model-dropdown__detail">${info.path}</span>
      </span>
      <span class="longcat-model-dropdown__badge">${info.badge}</span>
    `;
    item.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      setWidgetValue(node, widget, value, label, icon, path);
      path.classList.add("longcat-model-selector__path--visible");
      root.__longcatPathVisible = true;
      closeDropdown();
    });
    dropdown.append(item);
  }

  dropdown.addEventListener("pointerdown", (event) => event.stopPropagation());
  root.append(dropdown);
  root.__longcatDropdownOpen = true;
  openDropdown = { dropdown, root, node };
  setTimeout(() => document.addEventListener("pointerdown", closeDropdown, { once: true }), 0);
  node.setDirtyCanvas?.(true, true);
}

function createSelector(node, widget) {
  const root = document.createElement("div");
  root.className = "longcat-model-selector";
  root.addEventListener("pointerdown", (event) => event.stopPropagation());
  root.addEventListener("click", (event) => event.stopPropagation());

  const row = document.createElement("div");
  row.className = "longcat-model-selector__row";

  const display = document.createElement("div");
  display.className = "longcat-model-selector__display";
  display.tabIndex = 0;
  display.setAttribute("role", "combobox");
  display.setAttribute("aria-haspopup", "listbox");
  display.title = "Select LongCat inference weight mode";

  const icon = document.createElement("span");
  icon.className = "longcat-model-selector__icon";

  const label = document.createElement("span");
  label.className = "longcat-model-selector__label";

  const arrow = document.createElement("span");
  arrow.className = "longcat-model-selector__arrow";
  arrow.textContent = "▼";
  display.append(icon, label, arrow);

  const folder = document.createElement("button");
  folder.type = "button";
  folder.className = "longcat-model-selector__folder";
  folder.innerHTML = ICONS.folder;
  folder.title = "Select LongCat model folder mode";

  const path = document.createElement("div");
  path.className = "longcat-model-selector__path";

  updateDisplay(widget, label, icon, path);

  const open = (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (openDropdown?.root === root) {
      closeDropdown();
      return;
    }
    showDropdown(node, widget, root, label, icon, path);
  };
  display.addEventListener("click", open);
  display.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      open(event);
    } else if (event.key === "Escape") {
      closeDropdown();
    }
  });
  folder.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (openDropdown?.root === root) {
      closeDropdown();
      return;
    }
    showFolderPanel(node, widget, root, label, icon, path);
  });

  row.append(display, folder);
  root.append(row, path);

  const domWidget = node.addDOMWidget(DOM_WIDGET_NAME, "custom", root, {
    serialize: false,
    hideOnZoom: false,
    getValue: () => widget.value,
    setValue: (value) => setWidgetValue(node, widget, value, label, icon, path),
  });
  domWidget.tooltip = "Select LongCat Avatar inference weight source";
  domWidget.computeSize = () => {
    if (root.__longcatDropdownOpen) {
      return [0, 166];
    }
    return [0, root.__longcatPathVisible ? 62 : 44];
  };

  const index = node.widgets?.indexOf(domWidget) ?? -1;
  if (index > 0) {
    node.widgets.splice(index, 1);
    const originalIndex = node.widgets.findIndex((item) => item.name === WIDGET_NAME);
    node.widgets.splice(Math.max(originalIndex, 0), 0, domWidget);
  }
}

function enhanceNode(node) {
  if (!node || node.comfyClass !== NODE_CLASS || enhancedNodes.has(node) || typeof node.addDOMWidget !== "function") {
    return;
  }
  const widget = node.widgets?.find((item) => item.name === WIDGET_NAME);
  if (!widget) {
    return;
  }
  injectStyles();
  hideOriginalWidget(widget);
  createSelector(node, widget);
  enhancedNodes.add(node);
}

app.registerExtension({
  name: "longcat-avatar.model-selector-ui",
  nodeCreated(node) {
    enhanceNode(node);
  },
  loadedGraphNode(node) {
    setTimeout(() => enhanceNode(node), 50);
  },
});
