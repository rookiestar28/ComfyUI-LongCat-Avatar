from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .LongCat_Video_node import LongCat_Video_SM_Model, LongCat_Video_SM_WhisperModel, LongCat_Video_SM_Sampler,LongCat_Video_SM_Encode,LongCat_Video_SM_Audio,LongCat_Video_SM_AudioWindow,LongCat_Video_SM_AudioCrop,LongCat_Video_SM_Vocal,LongCat_Video_SM_VocalModel

WEB_DIRECTORY = "./js"

class LongCat_Video_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LongCat_Video_SM_Model,
            LongCat_Video_SM_Sampler,
            LongCat_Video_SM_Encode,
            LongCat_Video_SM_Audio,
            LongCat_Video_SM_AudioWindow,
            LongCat_Video_SM_AudioCrop,
            LongCat_Video_SM_Vocal,
            LongCat_Video_SM_VocalModel,
            LongCat_Video_SM_WhisperModel,
        ]

async def comfy_entrypoint() -> LongCat_Video_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return LongCat_Video_SM_Extension()

__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
