"""Bounded, opt-in property detection using pinned local Grounding DINO weights.

No network download occurs during requests. A process-local lock serializes CPU
inference and lazy model loading. Deployment must install the visual extra and
prewarm the pinned snapshot; normal worker startup does not require PyTorch.
"""

import math
import threading
from functools import lru_cache

MODEL = "IDEA-Research/grounding-dino-base"
REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
POLICY = "home-tour-visual-1"
LABELS = (
    "sofa",
    "chair",
    "coffee table",
    "dining table",
    "window",
    "door",
    "kitchen cabinet",
    "kitchen island",
    "bed",
    "wardrobe",
    "window seat",
    "fireplace",
    "person",
    "statue",
    "television console",
    "mirror",
    "bathroom vanity",
    "sink",
    "faucet",
    "toilet",
    "shower",
    "pendant light",
    "shelf",
    "plant",
    "framed picture",
    "staircase",
)
_LOCK = threading.Lock()


def available(enabled: bool) -> bool:
    """Check optional libraries and the local pinned snapshot without inference."""
    if not enabled:
        return False
    try:
        import importlib.util

        from huggingface_hub import try_to_load_from_cache

        if any(importlib.util.find_spec(name) is None for name in ("torch", "transformers")):
            return False
        return all(
            isinstance(try_to_load_from_cache(MODEL, name, revision=REVISION), str)
            for name in (
                "model.safetensors",
                "config.json",
                "preprocessor_config.json",
                "tokenizer.json",
            )
        )
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _load():
    """Load immutable CPU weights once; caller holds the inference lock."""
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL,
        revision=REVISION,
        local_files_only=True,
        use_safetensors=True,
    ).eval()
    return processor, model


def normalize(box, width, height):
    """Reject invalid raw geometry rather than repairing or inventing bounds."""
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        return None
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return None
    return dict(x=x1 / width, y=y1 / height, width=(x2 - x1) / width, height=(y2 - y1) / height)


def detect(bgr, cancel_check):
    """Return at most 64 score-ordered raw regions in normalized image geometry.

    Input is the worker's orientation-corrected analysis image. Uniform scaling
    preserves normalized canonical geometry. Cancellation is checked before and
    after the native model call; a running CPU kernel is not interruptible here.
    """
    import torch
    from PIL import Image

    while not _LOCK.acquire(timeout=0.1):
        cancel_check()
    try:
        cancel_check()
        processor, model = _load()
        image = Image.fromarray(bgr[:, :, ::-1].copy())
        width, height = image.size
        inputs = processor(images=image, text=". ".join(LABELS) + ".", return_tensors="pt")
        with torch.inference_mode():
            outputs = model(**inputs)
        cancel_check()
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.30,
            text_threshold=0.25,
            target_sizes=[(height, width)],
        )[0]
        labels = result.get("text_labels", result.get("labels"))
        regions = []
        for box, score, label in zip(result["boxes"].tolist(), result["scores"].tolist(), labels):
            geometry = normalize(box, width, height)
            if geometry and isinstance(label, str) and 0 <= score <= 1:
                regions.append(dict(label=label, score=score, box=geometry))
        regions.sort(key=lambda r: r["score"], reverse=True)
        return dict(
            policyVersion=POLICY,
            modelRevision=REVISION,
            regions=[dict(id=f"region-{i}", **r) for i, r in enumerate(regions[:64])],
        )
    finally:
        _LOCK.release()
