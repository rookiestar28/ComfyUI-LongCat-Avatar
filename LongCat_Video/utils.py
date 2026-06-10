import gc

import torch


UNSUPPORTED_GGUF_MESSAGE = "GGUF DiT loading is not supported by this ComfyUI node yet."


def cleanup_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def raise_unsupported_gguf() -> None:
    raise ValueError(UNSUPPORTED_GGUF_MESSAGE)
