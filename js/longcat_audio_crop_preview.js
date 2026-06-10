import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";

const AUDIO_CROP_NODE_CLASS = "LongCat_Video_SM_AudioCrop";
const AUDIO_WIDGET_NAME = "longcat_audio_crop_preview";
const EXECUTE_WIDGET_NAME = "longcat_audio_crop_execute";
const EMPTY_AUDIO_CLASS = "longcat-audio-crop-preview--empty";
const STYLE_ID = "longcat-audio-crop-preview-style";

function injectStyles() {
  if (document.getElementById(STYLE_ID)) {
    return;
  }

  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
.longcat-audio-crop-preview {
  width: 100%;
  min-width: 220px;
  height: 38px;
  box-sizing: border-box;
  padding: 0 2px;
}
.longcat-audio-crop-preview audio {
  width: 100%;
  height: 34px;
  display: block;
}
.longcat-audio-crop-preview--empty {
  display: none;
}
`;
  document.head.appendChild(style);
}

function audioResultURL(audio) {
  const params = new URLSearchParams();
  params.set("filename", audio?.filename ?? "");
  params.set("subfolder", audio?.subfolder ?? "");
  params.set("type", audio?.type ?? "temp");
  return api.apiURL(`/view?${params.toString()}`);
}

function updatePreview(container, audioElement, output) {
  const audio = output?.audio?.[0];
  if (!audio?.filename) {
    audioElement.removeAttribute("src");
    container.classList.add(EMPTY_AUDIO_CLASS);
    return;
  }

  audioElement.src = audioResultURL(audio);
  container.classList.remove(EMPTY_AUDIO_CLASS);
}

function addAudioCropPreview(node) {
  if (
    !node ||
    node.comfyClass !== AUDIO_CROP_NODE_CLASS ||
    typeof node.addDOMWidget !== "function" ||
    node.widgets?.some((widget) => widget.name === AUDIO_WIDGET_NAME)
  ) {
    return;
  }

  injectStyles();

  const container = document.createElement("div");
  container.className = `longcat-audio-crop-preview ${EMPTY_AUDIO_CLASS}`;

  const audio = document.createElement("audio");
  audio.controls = true;
  audio.preload = "metadata";
  audio.setAttribute("name", "media");
  container.append(audio);

  const widget = node.addDOMWidget(AUDIO_WIDGET_NAME, "longcatAudioPreview", container);
  widget.serialize = false;
  widget.computeSize = () => [node.size?.[0] ?? 220, 42];

  const previousOnExecuted = node.onExecuted;
  node.onExecuted = function (output) {
    previousOnExecuted?.call(this, output);
    updatePreview(container, audio, output);
  };
}

function addCropPreviewButton(node) {
  if (
    !node ||
    node.comfyClass !== AUDIO_CROP_NODE_CLASS ||
    typeof node.addWidget !== "function" ||
    node.widgets?.some((widget) => widget.name === EXECUTE_WIDGET_NAME)
  ) {
    return;
  }

  const widget = node.addWidget("button", "Crop Preview", "", async () => {
    if (node.id === null || node.id === undefined) {
      console.warn("LongCat Audio Crop preview cannot run before the node has an id.");
      return;
    }

    // IMPORTANT: partial execution targets only this output node, so downstream sampler/video nodes are not queued.
    await app.queuePrompt(0, 1, [String(node.id)]);
  });
  widget.name = EXECUTE_WIDGET_NAME;
  widget.serialize = false;
}

app.registerExtension({
  name: "longcat-avatar.audio-crop-preview",
  nodeCreated(node) {
    addAudioCropPreview(node);
    addCropPreviewButton(node);
  },
  loadedGraphNode(node) {
    setTimeout(() => {
      addAudioCropPreview(node);
      addCropPreviewButton(node);
    }, 50);
  },
});
