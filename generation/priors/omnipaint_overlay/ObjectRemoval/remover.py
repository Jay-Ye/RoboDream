from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers.pipelines import FluxPipeline
from pathlib import Path

def _import_omnipaint_src():
    """
    Import OmniPaint's internal `src/*` modules reliably.

    In some environments only the OmniPaint checkout is on `PYTHONPATH` (or only the
    `ObjectRemoval` package is installed). We ensure the OmniPaint root directory is
    on `sys.path` so `import src.*` resolves to *this* OmniPaint checkout.
    """

    import sys
    from pathlib import Path

    omnipaint_root = Path(__file__).resolve().parents[1]  # OmniPaint checkout
    if str(omnipaint_root) not in sys.path:
        sys.path.insert(0, str(omnipaint_root))

    try:
        from src.condition import Condition  # type: ignore
        from src.generate import generate, seed_everything  # type: ignore
        from src.embedding_loader import load_npz_embeddings  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Failed to import OmniPaint internals (`OmniPaint/src`). "
            "If you are using editable install, run `bash generation/setup_generation_env.sh`. "
            "Otherwise ensure the OmniPaint checkout is on PYTHONPATH."
        ) from exc

    return Condition, generate, seed_everything, load_npz_embeddings


@dataclass
class ObjectRemovalConfig:
    # Model + weights (same defaults as `main_omnipaint.py`)
    model_id: str = "black-forest-labs/FLUX.1-dev"
    lora_weights: str = "weights/omnipaint_remove.safetensors"
    embed_path: str = "demo_assets/embeddings/remove.npz"

    # Inference
    steps: int = 28
    seed: int = 42
    device: str = "cuda:0"

    # Preprocess
    max_length: int = 1024  # resize down if max(w,h) exceeds this
    input_is_bgr: bool = False  # True if you pass OpenCV BGR arrays

    # Generation uses multiples-of-8 sizes; output is resized back to input size.

def _omnipaint_root() -> Path:
    # OmniPaint checkout
    return Path(__file__).resolve().parents[1]


def _resolve_path_maybe_under_root(path_str: str) -> str:
    """
    Resolve a potentially-relative path to be under OmniPaint root.

    This makes defaults like `weights/...` work even when the
    current working directory is not the repo root.
    """

    p = Path(path_str)
    if p.is_absolute():
        return str(p)

    # First try relative to OmniPaint root.
    candidate = _omnipaint_root() / p
    if candidate.exists():
        return str(candidate)

    # If user provided ".../OmniPaint/..." already, strip leading parts and retry.
    parts = list(p.parts)
    try:
        idx = parts.index("OmniPaint")
        stripped = Path(*parts[idx + 1 :])  # path after ".../OmniPaint/"
        candidate2 = _omnipaint_root() / stripped
        if candidate2.exists():
            return str(candidate2)
    except ValueError:
        pass

    # Fall back to original string (could be HF repo id like "user/repo").
    return path_str


def _to_pil_rgb(image: Union[np.ndarray, Image.Image], *, input_is_bgr: bool) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(image)}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape={image.shape}")
    img = image
    if input_is_bgr:
        img = img[..., ::-1]
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return Image.fromarray(img, mode="RGB")


def _to_pil_mask(mask: Union[np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(mask, Image.Image):
        return mask.convert("L")
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Unsupported mask type: {type(mask)}")
    if mask.ndim == 3:
        # If HxWx3, collapse to grayscale
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"Expected HxW mask, got shape={mask.shape}")
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return Image.fromarray(mask, mode="L")


def _resize_with_aspect_ratio(img: Image.Image, max_length: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_length:
        return img
    scale = max_length / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def _build_composite(raw_img: Image.Image, mask_img: Image.Image) -> Image.Image:
    """
    Matches `main_omnipaint.py`:
    - mask white => region to remove (becomes black in composite)
    - mask black => keep original pixels
    """

    mask = mask_img.convert("L")
    inverted_mask = ImageOps.invert(mask)
    black_background = Image.new("RGB", raw_img.size, (0, 0, 0))
    return Image.composite(raw_img.convert("RGB"), black_background, inverted_mask)


class ObjectRemover:
    """
    Object removal wrapper around OmniPaint.

    - Call `remove(image, mask)` where mask is white for pixels to remove.
    - Returns a PIL image.
    """

    def __init__(self, config: Optional[ObjectRemovalConfig] = None) -> None:
        self.config = config or ObjectRemovalConfig()
        self._pipe: Optional[FluxPipeline] = None
        self._prompt_embeds: Optional[torch.Tensor] = None
        self._pooled_prompt_embeds: Optional[torch.Tensor] = None
        self._text_ids: Optional[torch.Tensor] = None

    def _load_pipe(self) -> FluxPipeline:
        if self._pipe is not None:
            return self._pipe
        pipe = FluxPipeline.from_pretrained(self.config.model_id, torch_dtype=torch.bfloat16).to(self.config.device)
        lora_path = _resolve_path_maybe_under_root(self.config.lora_weights)
        if Path(lora_path).suffix in {".safetensors", ".bin", ".pt"} and not Path(lora_path).exists():
            raise FileNotFoundError(
                f"LoRA weights not found: {lora_path}. "
                f"If you have the weights locally, put them under `{_omnipaint_root() / 'weights'}` "
                "or set ObjectRemovalConfig(lora_weights=...) to an absolute path."
            )
        pipe.load_lora_weights(lora_path, adapter_name="removal")
        pipe.set_adapters(["removal"])
        self._pipe = pipe
        return pipe

    def _load_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._prompt_embeds is not None:
            return self._prompt_embeds, self._pooled_prompt_embeds, self._text_ids
        _, _, _, load_npz_embeddings = _import_omnipaint_src()
        pipe = self._load_pipe()
        device = getattr(pipe, "_execution_device", getattr(pipe, "device", torch.device(self.config.device)))
        dtype = pipe.transformer.dtype
        embed_path = _resolve_path_maybe_under_root(self.config.embed_path)
        if not Path(embed_path).exists():
            raise FileNotFoundError(
                f"Embedding NPZ not found: {embed_path}. "
                f"Expected under `{_omnipaint_root() / 'demo_assets' / 'embeddings'}` "
                "or set ObjectRemovalConfig(embed_path=...) to an absolute path."
            )
        prompt_embeds, pooled_prompt_embeds, text_ids = load_npz_embeddings(embed_path, device=device, dtype=dtype)
        self._prompt_embeds = prompt_embeds
        self._pooled_prompt_embeds = pooled_prompt_embeds
        self._text_ids = text_ids
        return prompt_embeds, pooled_prompt_embeds, text_ids

    @torch.inference_mode()
    def remove(self, image: Union[np.ndarray, Image.Image], mask: Union[np.ndarray, Image.Image]) -> Image.Image:
        Condition, generate, seed_everything, _ = _import_omnipaint_src()
        pipe = self._load_pipe()
        prompt_embeds, pooled_prompt_embeds, text_ids = self._load_embeddings()

        raw_img = _to_pil_rgb(image, input_is_bgr=self.config.input_is_bgr)
        raw_img = _resize_with_aspect_ratio(raw_img, self.config.max_length)

        mask_img = _to_pil_mask(mask).resize(raw_img.size, Image.Resampling.LANCZOS)
        composite_img = _build_composite(raw_img, mask_img)

        # Match batch runner behavior
        width = (raw_img.size[0] // 8) * 8
        height = (raw_img.size[1] // 8) * 8

        seed_everything(self.config.seed)
        condition = Condition("removal", condition=[composite_img])
        result = generate(
            pipe,
            conditions=[condition],
            width=width,
            height=height,
            num_inference_steps=self.config.steps,
            prompt=None,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
        ).images

        # `generate(...).images` is a list of PIL images
        out = result[0]
        return out.resize(raw_img.size, Image.Resampling.LANCZOS)

