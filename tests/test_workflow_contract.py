import json
import unittest
from pathlib import Path


WORKFLOW_PATH = Path("example_workflows/longcat-avatar1.5.json")
NODE_SOURCE_PATH = Path("LongCat_Video_node.py")


def load_workflow():
    return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))


def nodes_by_type(workflow):
    return {node["type"]: node for node in workflow["nodes"]}


def link_ids(workflow):
    return {link[0] for link in workflow["links"]}


def input_by_name(node, name):
    for input_item in node["inputs"]:
        if input_item["name"] == name:
            return input_item
    raise AssertionError(f"Missing input: {name}")


class WorkflowContractTests(unittest.TestCase):
    def test_example_workflow_uses_supported_longcat_nodes(self):
        workflow_nodes = nodes_by_type(load_workflow())

        for node_type in (
            "LongCat_Video_SM_Model",
            "LongCat_Video_SM_WhisperModel",
            "LongCat_Video_SM_Encode",
            "LongCat_Video_SM_Audio",
            "LongCat_Video_SM_Sampler",
            "LongCat_Video_SM_VocalModel",
            "LongCat_Video_SM_Vocal",
        ):
            self.assertIn(node_type, workflow_nodes)

        self.assertNotIn("LongCat_Video_SM_MLXGenerate", workflow_nodes)
        self.assertNotIn("AudioEncoderLoader", workflow_nodes)

    def test_model_workflow_uses_current_official_sharded_schema(self):
        model_node = nodes_by_type(load_workflow())["LongCat_Video_SM_Model"]

        self.assertEqual(model_node["widgets_values"], [
            "official_sharded",
            "sdpa",
            True,
            "LongCat-Video-Avatar-vae.safetensors",
            "longcat-avatar-dmd_lora.safetensors",
        ])

    def test_sampler_workflow_declares_video_defaults(self):
        sampler_node = nodes_by_type(load_workflow())["LongCat_Video_SM_Sampler"]

        self.assertEqual(sampler_node["widgets_values"][0], "ai2v")
        self.assertEqual(sampler_node["widgets_values"][10:], ["", "cpu", False])
        self.assertEqual(sampler_node["outputs"][0]["name"], "image")
        self.assertEqual(sampler_node["outputs"][1]["name"], "video_path")
        self.assertEqual(sampler_node["outputs"][1]["type"], "STRING")

    def test_text_encode_workflow_uses_current_mps_schema_defaults(self):
        text_node = nodes_by_type(load_workflow())["LongCat_Video_SM_Encode"]

        self.assertEqual(text_node["widgets_values"][:3], ["LongCat-Video", True, "cpu"])

    def test_node_source_marks_gguf_as_unsupported(self):
        source = NODE_SOURCE_PATH.read_text(encoding="utf-8")

        self.assertNotIn("gguf_experimental_unsupported", source)
        self.assertNotIn('io.Combo.Input("gguf",', source)
        self.assertNotIn('add_model_folder_path("gguf"', source)

    def test_unvalidated_gguf_loader_helpers_are_not_present(self):
        utils_source = Path("LongCat_Video/utils.py").read_text(encoding="utf-8")
        node_source = NODE_SOURCE_PATH.read_text(encoding="utf-8")

        for token in (
            "GGUFQuantizationConfig",
            "GGUFQuantizer",
            "GGUFReader",
            "load_gguf_checkpoint",
            "set_gguf2meta_model",
            "apply_loras_gguf",
        ):
            self.assertNotIn(token, utils_source)
            self.assertNotIn(token, node_source)
        self.assertIn("GGUF DiT loading is not supported", utils_source)

    def test_sampler_source_exposes_bounded_distill_values(self):
        source = NODE_SOURCE_PATH.read_text(encoding="utf-8")

        self.assertIn('io.Int.Input("steps", default=OFFICIAL_V15_DISTILL_STEPS', source)
        self.assertIn("min=AVATAR_MIN_INFERENCE_STEPS", source)
        self.assertIn("max=AVATAR_MAX_INFERENCE_STEPS", source)
        self.assertIn('io.Float.Input("text_guidance_scale", default=OFFICIAL_V15_DISTILL_TEXT_CFG', source)
        self.assertIn('io.Float.Input("audio_guidance_scale", default=OFFICIAL_V15_DISTILL_AUDIO_CFG', source)
        self.assertIn("max=AVATAR_MAX_GUIDANCE_SCALE", source)
        self.assertNotIn('io.Boolean.Input("save_video"', source)
        self.assertNotIn('io.String.Input("video_prefix"', source)

    def test_public_workflow_exists_and_parses(self):
        self.assertTrue(WORKFLOW_PATH.is_file(), WORKFLOW_PATH)
        workflow = load_workflow()
        self.assertIn("nodes", workflow)
        self.assertIn("links", workflow)

    def test_current_ai2v_workflow_contract(self):
        workflow = load_workflow()
        workflow_nodes = nodes_by_type(workflow)
        audio_node = workflow_nodes["LongCat_Video_SM_Audio"]
        whisper_node = workflow_nodes["LongCat_Video_SM_WhisperModel"]
        vocal_model_node = workflow_nodes["LongCat_Video_SM_VocalModel"]
        vocal_node = workflow_nodes["LongCat_Video_SM_Vocal"]

        self.assertIsNone(input_by_name(audio_node, "left_audio")["link"])
        self.assertNotIn("num_segments", [item["name"] for item in audio_node["inputs"]])
        self.assertEqual(audio_node["widgets_values"], [25, "para", ""])
        self.assertEqual(whisper_node["widgets_values"], ["whisper-large-v3.safetensors"])
        self.assertEqual(vocal_model_node["widgets_values"], ["Kim_Vocal_2.onnx"])
        self.assertIn(input_by_name(vocal_node, "audio")["link"], link_ids(workflow))

    def test_public_workflow_does_not_embed_internal_paths(self):
        forbidden = (".planning", "reference/", "/mnt/", "/home/")

        text = WORKFLOW_PATH.read_text(encoding="utf-8")
        for token in forbidden:
            self.assertNotIn(token, text)

    def test_mps_workflow_does_not_store_cuda_or_unsupported_attention_widgets(self):
        text = WORKFLOW_PATH.read_text(encoding="utf-8")

        for token in ('"cuda"', '"flash_attn_2"', '"flash_attn_3"', '"xformers"', '"sageattn"', '"sageattn_3"'):
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
