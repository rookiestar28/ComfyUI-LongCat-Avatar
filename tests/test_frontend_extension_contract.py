from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INIT_FILE = REPO_ROOT / "__init__.py"
MODEL_SELECTOR_JS = REPO_ROOT / "js" / "longcat_model_selector.js"
AUDIO_CROP_PREVIEW_JS = REPO_ROOT / "js" / "longcat_audio_crop_preview.js"


class FrontendExtensionContractTests(unittest.TestCase):
    def test_package_exports_comfyui_web_directory(self):
        init_source = INIT_FILE.read_text(encoding="utf-8")

        self.assertRegex(init_source, r'WEB_DIRECTORY\s*=\s*["\']\./js["\']')
        self.assertRegex(init_source, r'__all__\s*=\s*\[[^\]]*["\']WEB_DIRECTORY["\']')

    def test_model_selector_targets_only_autoload_model_weight_mode(self):
        source = MODEL_SELECTOR_JS.read_text(encoding="utf-8")

        self.assertIn('const NODE_CLASS = "LongCat_Video_SM_Model";', source)
        self.assertIn('const WIDGET_NAME = "inference_weight_mode";', source)
        self.assertIn('node.comfyClass !== NODE_CLASS', source)
        self.assertNotIn("LongCat_Video_SM_Sampler", source)
        self.assertNotIn("LongCat_Video_SM_WhisperModel", source)

    def test_model_selector_preserves_serialized_combo_widget(self):
        source = MODEL_SELECTOR_JS.read_text(encoding="utf-8")

        self.assertIn("node.addDOMWidget", source)
        self.assertIn("serialize: false", source)
        self.assertIn('widget.type = "converted-widget"', source)
        self.assertIn("widget.options.canvasOnly = true", source)
        self.assertIn("widget.value = value", source)
        self.assertIn("widget.callback?.(value", source)

    def test_model_selector_exposes_all_supported_modes_with_location_affordance(self):
        source = MODEL_SELECTOR_JS.read_text(encoding="utf-8")

        for mode in (
            "single_file_safetensors",
            "official_sharded",
            "official_int8_sharded",
        ):
            self.assertIn(mode, source)

        self.assertIn("SELECT MODEL", source)
        self.assertIn("MODEL FOLDERS", source)
        self.assertIn("showFolderPanel", source)
        self.assertIn("longcat-model-selector__folder", source)
        self.assertIn("longcat-model-selector__path--visible", source)
        self.assertIn("ComfyUI/models/diffusion_models", source)
        self.assertIn("ComfyUI/models/longcat/LongCat-Video-Avatar-1.5/base_model/", source)
        self.assertIn("ComfyUI/models/longcat/LongCat-Video-Avatar-1.5/base_model_int8/", source)

    def test_model_selector_does_not_add_unsupported_browser_path_api(self):
        source = MODEL_SELECTOR_JS.read_text(encoding="utf-8")

        forbidden_patterns = (
            r"showDirectoryPicker",
            r"webkitdirectory",
            r"fetchApi",
            r"/longcat/models",
            r"custom_model_path",
        )
        for pattern in forbidden_patterns:
            self.assertIsNone(re.search(pattern, source), pattern)

    def test_model_selector_dropdown_stays_inside_node_widget(self):
        source = MODEL_SELECTOR_JS.read_text(encoding="utf-8")

        self.assertIn("root.append(dropdown)", source)
        self.assertIn('dropdown.setAttribute("role", "listbox")', source)
        self.assertIn("root.__longcatDropdownOpen", source)
        self.assertIn("return [0, 166]", source)
        self.assertIn(".longcat-model-dropdown__header svg", source)
        self.assertNotIn("document.body.append(dropdown)", source)
        self.assertNotRegex(source, r"position:\s*fixed")
        self.assertNotRegex(source, r"min-width:\s*300px")
        self.assertNotRegex(source, r"Math\.max\(rect\.width,\s*300\)")

    def test_audio_crop_preview_extension_targets_only_crop_node(self):
        source = AUDIO_CROP_PREVIEW_JS.read_text(encoding="utf-8")

        self.assertIn('const AUDIO_CROP_NODE_CLASS = "LongCat_Video_SM_AudioCrop";', source)
        self.assertIn('node.comfyClass !== AUDIO_CROP_NODE_CLASS', source)
        self.assertIn("node.addDOMWidget", source)
        self.assertIn("widget.serialize = false", source)
        self.assertIn('document.createElement("audio")', source)
        self.assertIn("audio.controls = true", source)
        self.assertIn("node.onExecuted", source)
        self.assertIn("output?.audio?.[0]", source)
        self.assertIn("api.apiURL(`/view?${params.toString()}`)", source)
        self.assertNotIn("LongCat_Video_SM_Sampler", source)
        self.assertNotIn("LongCat_Video_SM_Model", source)

    def test_audio_crop_preview_extension_adds_partial_execute_button(self):
        source = AUDIO_CROP_PREVIEW_JS.read_text(encoding="utf-8")

        self.assertIn('const EXECUTE_WIDGET_NAME = "longcat_audio_crop_execute";', source)
        self.assertIn('node.addWidget("button", "Crop Preview"', source)
        self.assertIn("widget.serialize = false", source)
        self.assertIn("app.queuePrompt(0, 1, [String(node.id)])", source)
        self.assertIn("partial execution targets only this output node", source)
        self.assertIn("downstream sampler/video nodes are not queued", source)

    def test_audio_crop_preview_extension_does_not_depend_on_comfyui_audio_ui_whitelist(self):
        source = AUDIO_CROP_PREVIEW_JS.read_text(encoding="utf-8")

        self.assertNotIn("AUDIO_UI", source)
        self.assertNotIn("PreviewAudio", source)
        self.assertNotIn("SaveAudio", source)


if __name__ == "__main__":
    unittest.main()
