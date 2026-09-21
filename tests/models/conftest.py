"""
A fake ``transformers`` namespace so the wrappers can be tested without weights or network.

Every fake mirrors only the parts of the real API the wrappers touch, and returns
tensors of realistic shape. ``from_pretrained`` calls are recorded so tests can check
that device, dtype and cache options are forwarded correctly.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from anytraverse.models import _base

# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------

FROM_PRETRAINED_CALLS: list[tuple[str, str, dict[str, Any]]] = []


class FakeBatch(dict):
    """Mimics ``transformers.BatchFeature``: a dict with ``.to(device)``."""

    def to(self, device) -> "FakeBatch":
        return FakeBatch(
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in self.items()}
        )


class FakeModule(torch.nn.Module):
    """Base for fake models: records ``from_pretrained`` and supports ``.to().eval()``."""

    config = SimpleNamespace()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.param = torch.nn.Parameter(torch.zeros(1))

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> "FakeModule":
        FROM_PRETRAINED_CALLS.append((cls.__name__, model_id, kwargs))
        return cls(**kwargs)

    @property
    def dtype(self) -> torch.dtype:
        return self.param.dtype


class FakeProcessor:
    """Base for fake processors: records ``from_pretrained``."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> "FakeProcessor":
        FROM_PRETRAINED_CALLS.append((cls.__name__, model_id, kwargs))
        return cls(**kwargs)


def _pixel_values(images, size: int) -> torch.Tensor:
    images = images if isinstance(images, list) else [images]
    return torch.zeros(len(images), 3, size, size)


class FakeImageProcessor(FakeProcessor):
    size = 32

    def __call__(self, images, return_tensors="pt", **_: Any) -> FakeBatch:
        return FakeBatch(pixel_values=_pixel_values(images, self.size))


class FakeTokenizer:
    """Maps each whitespace token to a fixed id; ``.`` is the separator (id 1012)."""

    cls_id, sep_id, dot_id = 101, 102, 1012
    all_special_ids = [cls_id, sep_id]

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "."
        return self.dot_id

    def encode_words(self, text: str) -> list[int]:
        # Deterministic ids (sum of bytes, not hash(), which is salted per process).
        ids = [self.cls_id]
        for word in text.replace(".", " . ").split():
            ids.append(self.dot_id if word == "." else 2000 + (sum(word.encode()) % 1000))
        return [*ids, self.sep_id]

    def __call__(self, text, padding=True, return_tensors="pt") -> FakeBatch:
        texts = [text] if isinstance(text, str) else list(text)
        encoded = [self.encode_words(t) for t in texts]
        length = max(len(e) for e in encoded)
        ids = torch.zeros(len(encoded), length, dtype=torch.long)
        mask = torch.zeros(len(encoded), length, dtype=torch.long)
        for i, e in enumerate(encoded):
            ids[i, : len(e)] = torch.tensor(e)
            mask[i, : len(e)] = 1
        return FakeBatch(input_ids=ids, attention_mask=mask)


# --------------------------------------------------------------------------------------
# CLIPSeg
# --------------------------------------------------------------------------------------

CLIPSEG_PROJ, CLIPSEG_TOKENS, CLIPSEG_HIDDEN, CLIPSEG_OUT = 8, 5, 6, 16


class FakeCLIPSegProcessor(FakeProcessor):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tokenizer = FakeTokenizer()
        self.image_processor = FakeImageProcessor()


class FakeCLIPSegModel(FakeModule):
    extract_layers = [0, 1]
    vision_calls = 0
    cond_calls = 0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.clip = SimpleNamespace(vision_model=self._vision_model)
        self.decoder = self._decoder

    def _vision_model(self, pixel_values, output_hidden_states=False):
        type(self).vision_calls += 1
        batch = pixel_values.shape[0]
        hidden = tuple(
            torch.full((batch, CLIPSEG_TOKENS, CLIPSEG_HIDDEN), float(i), dtype=pixel_values.dtype)
            for i in range(3)
        )
        return SimpleNamespace(hidden_states=hidden)

    def get_conditional_embeddings(self, batch_size, input_ids, attention_mask):
        type(self).cond_calls += 1
        # Embed each prompt as its first real token id so different prompts differ.
        return input_ids[:, 1].float().reshape(batch_size, 1).repeat(1, CLIPSEG_PROJ)

    def _decoder(self, activations, conditional):
        n = conditional.shape[0]
        assert all(a.shape[0] == n for a in activations)
        # Logit encodes the prompt id so each map is distinguishable.
        logits = conditional[:, :1, None] / 3000.0 * torch.ones(n, CLIPSEG_OUT, CLIPSEG_OUT)
        return SimpleNamespace(logits=logits.to(activations[0].dtype))


# --------------------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------------------


class FakeCLIPVision(FakeModule):
    config = SimpleNamespace(projection_dim=8)

    def forward(self, pixel_values):
        return SimpleNamespace(image_embeds=torch.ones(pixel_values.shape[0], 8) * 3)


class FakeSiglip2Model(FakeModule):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.vision_model = SimpleNamespace(config=SimpleNamespace(hidden_size=6))

    def get_image_features(self, pixel_values):
        return SimpleNamespace(pooler_output=torch.ones(pixel_values.shape[0], 6))


class FakeDinov2(FakeModule):
    config = SimpleNamespace(hidden_size=4)

    def forward(self, pixel_values):
        hidden = torch.zeros(pixel_values.shape[0], 5, 4)
        hidden[:, 0] = torch.tensor([3.0, 4.0, 0.0, 0.0])
        return SimpleNamespace(last_hidden_state=hidden)


# --------------------------------------------------------------------------------------
# SAM 3
# --------------------------------------------------------------------------------------

SAM3_QUERIES, SAM3_MASK = 4, 8


class FakeSam3Config:
    calls: list[str] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> "FakeSam3Config":
        cls.calls.append(model_id)
        return cls()


class FakeSam3Processor(FakeProcessor):
    def __call__(self, images=None, text=None, return_tensors="pt") -> FakeBatch:
        batch = FakeBatch()
        if images is not None:
            size = self.kwargs.get("size", {"height": 1008})["height"]
            batch["pixel_values"] = _pixel_values(images, size)
        if text is not None:
            batch.update(FakeTokenizer()(text))
        return batch


class FakeSam3Model(FakeModule):
    vision_calls = 0
    text_calls = 0
    forward_calls = 0

    def get_vision_features(self, pixel_values):
        type(self).vision_calls += 1
        return SimpleNamespace(fpn_hidden_states=[pixel_values])

    def get_text_features(self, input_ids, attention_mask):
        type(self).text_calls += 1
        return SimpleNamespace(pooler_output=input_ids[:, 1:2].float())

    def forward(self, vision_embeds, text_embeds, attention_mask):
        type(self).forward_calls += 1
        prompt_id = text_embeds.pooler_output[0, 0].item()
        # Semantic logits: positive in the bottom half.
        semantic = torch.full((1, 1, SAM3_MASK, SAM3_MASK), -4.0)
        semantic[..., SAM3_MASK // 2 :, :] = 4.0
        # Instance masks: query 0 covers the left half (high score), others low score.
        pred_masks = torch.full((1, SAM3_QUERIES, SAM3_MASK, SAM3_MASK), -6.0)
        pred_masks[0, 0, :, : SAM3_MASK // 2] = 6.0
        pred_masks[0, 1, : SAM3_MASK // 2, :] = 6.0
        pred_logits = torch.tensor([[4.0, -4.0, -4.0, -4.0]])
        if prompt_id % 2 == 0:  # "even" prompts have no confident instances
            pred_logits = torch.full((1, SAM3_QUERIES), -4.0)
        return SimpleNamespace(
            semantic_seg=semantic, pred_masks=pred_masks, pred_logits=pred_logits
        )


# --------------------------------------------------------------------------------------
# SAM 2 + detectors
# --------------------------------------------------------------------------------------

SAM2_MASK = 8


class FakeSam2Processor(FakeProcessor):
    def __call__(self, images, input_boxes, return_tensors="pt") -> FakeBatch:
        boxes = torch.tensor(input_boxes, dtype=torch.float32)
        return FakeBatch(pixel_values=_pixel_values(images, 16), input_boxes=boxes)


class FakeSam2Model(FakeModule):
    forward_calls = 0

    def forward(self, pixel_values, input_boxes, multimask_output=True):
        type(self).forward_calls += 1
        assert multimask_output is False
        n = input_boxes.shape[1]
        masks = torch.full((1, n, 1, SAM2_MASK, SAM2_MASK), -6.0)
        for i in range(n):
            # Box i fills the i-th row band of the low-res mask.
            masks[0, i, 0, i % SAM2_MASK] = 6.0
        return SimpleNamespace(pred_masks=masks)


class FakeGroundingDinoProcessor(FakeProcessor):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tokenizer = FakeTokenizer()

    def __call__(self, images, text, return_tensors="pt") -> FakeBatch:
        batch = FakeBatch(pixel_values=_pixel_values(images, 16))
        batch.update(self.tokenizer(text))
        return batch


class FakeGroundingDinoModel(FakeModule):
    """Emits 3 queries: one for the first prompt, one for the second, one below threshold."""

    def forward(self, pixel_values, input_ids, attention_mask):
        length = input_ids.shape[1]
        logits = torch.full((1, 3, length), -6.0)
        ids = input_ids[0].tolist()
        first_tok = 1  # position of first prompt's first token
        second_tok = ids.index(FakeTokenizer.dot_id) + 1 if FakeTokenizer.dot_id in ids else None
        logits[0, 0, first_tok] = 3.0
        if second_tok is not None and ids[second_tok] not in FakeTokenizer.all_special_ids:
            logits[0, 1, second_tok] = 2.0
        logits[0, 2, first_tok] = -1.0  # sigmoid ~0.27 < 0.3 threshold
        boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1]]])
        return SimpleNamespace(logits=logits, pred_boxes=boxes)


class FakeOwlv2Processor(FakeProcessor):
    def __call__(self, images, text, return_tensors="pt") -> FakeBatch:
        batch = FakeBatch(pixel_values=_pixel_values(images, 16))
        batch.update(FakeTokenizer()([t for group in text for t in group]))
        return batch

    def post_process_grounded_object_detection(self, outputs, threshold, target_sizes):
        side = target_sizes[0][0]
        keep = outputs.scores > threshold
        return [
            {
                "boxes": outputs.boxes[keep] * side,
                "labels": outputs.labels[keep],
                "scores": outputs.scores[keep],
            }
        ]


class FakeOwlv2Model(FakeModule):
    def forward(self, pixel_values, input_ids, attention_mask):
        n_prompts = input_ids.shape[0]
        # Two boxes for prompt 0 (one far outside the image), one weak box for the last prompt.
        return SimpleNamespace(
            boxes=torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.5, 1.5], [0.0, 0.0, 0.1, 0.1]]),
            labels=torch.tensor([0, 0, n_prompts - 1]),
            scores=torch.tensor([0.9, 0.8, 0.05]),
        )


# --------------------------------------------------------------------------------------
# Fixture wiring
# --------------------------------------------------------------------------------------

FAKE_TRANSFORMERS = SimpleNamespace(
    AutoProcessor=FakeCLIPSegProcessor,
    AutoImageProcessor=FakeImageProcessor,
    CLIPSegForImageSegmentation=FakeCLIPSegModel,
    CLIPVisionModelWithProjection=FakeCLIPVision,
    Siglip2Model=FakeSiglip2Model,
    Dinov2Model=FakeDinov2,
    Sam3Config=FakeSam3Config,
    Sam3Processor=FakeSam3Processor,
    Sam3Model=FakeSam3Model,
    Sam2Processor=FakeSam2Processor,
    Sam2Model=FakeSam2Model,
    GroundingDinoProcessor=FakeGroundingDinoProcessor,
    GroundingDinoForObjectDetection=FakeGroundingDinoModel,
    Owlv2Processor=FakeOwlv2Processor,
    Owlv2ForObjectDetection=FakeOwlv2Model,
)


@pytest.fixture(autouse=True)
def fake_transformers(monkeypatch: pytest.MonkeyPatch):
    """Routes every wrapper to the fakes and resets the recorded calls and counters."""
    monkeypatch.setattr(_base, "require_transformers", lambda: FAKE_TRANSFORMERS)
    for module in (
        "clipseg",
        "clip",
        "siglip2",
        "dinov2",
        "sam3",
        "_sam2",
        "grounded_sam2",
        "owlv2_sam2",
    ):
        monkeypatch.setattr(
            f"anytraverse.models.{module}.require_transformers", lambda: FAKE_TRANSFORMERS
        )
    FROM_PRETRAINED_CALLS.clear()
    FakeSam3Config.calls.clear()
    for cls in (FakeCLIPSegModel, FakeSam3Model, FakeSam2Model):
        for attr in ("vision_calls", "cond_calls", "text_calls", "forward_calls"):
            if hasattr(cls, attr):
                setattr(cls, attr, 0)
    return FAKE_TRANSFORMERS
