import logging
import math
import re
from dataclasses import dataclass
from collections.abc import Iterable as IterableABC
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch


log = logging.getLogger(__name__)


LATENT_FRAME_STRIDE = 4


def enforce_4t_plus_1(n: int) -> int:
    """Clamp frame counts to the closest 4t+1 form used by Wan latents."""
    t = round((n - 1) / LATENT_FRAME_STRIDE)
    return LATENT_FRAME_STRIDE * t + 1


def frames_to_latents(frame_idx: int) -> int:
    """Convert pixel frame index to latent index."""
    if frame_idx <= 0:
        return 0
    return (frame_idx - 1) // LATENT_FRAME_STRIDE + 1


def build_shot_indices(num_latent_frames: int, shot_cut_frames: Sequence[int]) -> torch.Tensor:
    """
    Construct a shot index tensor of shape (1, num_latent_frames).

    Args:
        num_latent_frames: Number of latent frames in the generated sequence.
        shot_cut_frames: Frame indices (pixel space) where a new shot begins.
    """
    if num_latent_frames <= 0:
        raise ValueError("num_latent_frames must be positive")

    latent_cuts = [0]
    for frame_idx in sorted(set(shot_cut_frames)):
        latent_idx = frames_to_latents(frame_idx)
        if 0 < latent_idx < num_latent_frames:
            latent_cuts.append(latent_idx)
    latent_cuts = sorted(set(latent_cuts)) + [num_latent_frames]

    shot_indices = torch.zeros(num_latent_frames, dtype=torch.long)
    for shot_id, (start, end) in enumerate(zip(latent_cuts[:-1], latent_cuts[1:])):
        shot_indices[start:end] = shot_id

    return shot_indices.unsqueeze(0)


def labels_to_cuts(batch_labels: torch.Tensor) -> List[List[int]]:
    """Convert per-token shot labels into cut offsets for varlen attention."""
    if batch_labels.dim() != 2:
        raise ValueError("batch_labels must have shape [batch, seq]")

    bsz, seq = batch_labels.shape
    diffs = torch.zeros((bsz, seq), dtype=torch.bool, device=batch_labels.device)
    diffs[:, 1:] = batch_labels[:, 1:] != batch_labels[:, :-1]

    cuts: List[List[int]] = []
    for row in diffs:
        change_pos = torch.nonzero(row, as_tuple=False).flatten()
        indices = [0]
        indices.extend(change_pos.tolist())
        if indices[-1] != seq:
            indices.append(seq)
        cuts.append(indices)
    return cuts


SHOT_GLOBAL_TAG = "[global caption]"
SHOT_PER_TAG = "[per shot caption]"
SHOT_CUT_TAG = "[shot cut]"


def _find_shot_char_spans(prompt: str) -> Dict[str, Optional[List[Tuple[int, int]]]]:
    global_idx = prompt.find(SHOT_GLOBAL_TAG)
    per_idx = prompt.find(SHOT_PER_TAG)

    spans: Dict[str, Optional[List[Tuple[int, int]]]] = {
        "global": None,
        "shots": None,
    }

    if global_idx != -1:
        global_start = global_idx + len(SHOT_GLOBAL_TAG)
        global_end = per_idx if per_idx != -1 else len(prompt)
        spans["global"] = [(global_start, global_end)]

    if per_idx != -1:
        shots: List[Tuple[int, int]] = []
        cursor = per_idx + len(SHOT_PER_TAG)
        while cursor < len(prompt):
            next_cut = prompt.find(SHOT_CUT_TAG, cursor)
            if next_cut == -1:
                shots.append((cursor, len(prompt)))
                break
            shots.append((cursor, next_cut))
            cursor = next_cut + len(SHOT_CUT_TAG)
        spans["shots"] = shots

    return spans


def parse_structured_prompt(
    prompt: str,
    tokenizer,
) -> Optional[Dict[str, List[List[int]]]]:
    """Return token ranges for global and per-shot text conditioned by Wan format.

    Args:
        prompt: Prompt string containing shot tags.
        tokenizer: HuggingfaceTokenizer instance used for Wan text encoding.
    """
    spans = _find_shot_char_spans(prompt)
    shot_span_count = len(spans["shots"]) if spans["shots"] is not None else None
    log.warning(
        "parse_structured_prompt spans detected: global=%s, shots=%s",
        spans["global"],
        shot_span_count,
    )
    if spans["global"] is None and spans["shots"] is None:
        return None

    tokenized = tokenizer(
        prompt,
        return_mask=True,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )

    offsets_raw = tokenized["offset_mapping"]

    if isinstance(offsets_raw, dict):
        offsets_iterable = offsets_raw.values()
    elif hasattr(offsets_raw, "tolist"):
        offsets_iterable = offsets_raw.tolist()
    else:
        offsets_iterable = offsets_raw

    def _ensure_iterable(obj: Any) -> Iterable:
        if isinstance(obj, str):
            return []
        if isinstance(obj, IterableABC):
            return obj
        if hasattr(obj, "__iter__"):
            return obj
        return []

    try:
        offsets_list = list(offsets_iterable)
    except TypeError:
        offsets_list = [offsets_iterable]

    if not offsets_list:
        raise ValueError("Tokenizer offsets mapping is empty; ensure prompt was tokenized correctly.")

    first_item = offsets_list[0]
    log.warning(
        "offset mapping container=%s sample=%s repr=%r",
        type(offsets_iterable),
        type(first_item),
        first_item,
    )

    def _normalize_offset_item(item: Any) -> Tuple[int, int]:
        if item is None:
            raise TypeError("Offset item is None")
        if isinstance(item, dict):
            if "start" in item and "end" in item:
                return int(item["start"]), int(item["end"])
            if hasattr(item, "get") and item.get("start") is not None and item.get("end") is not None:
                return int(item.get("start")), int(item.get("end"))
            unexpected_keys = list(item.keys())[:4]
            raise ValueError(f"Unexpected dict keys in offsets mapping: {unexpected_keys}")
        if hasattr(item, "start") and hasattr(item, "end"):
            return int(getattr(item, "start")), int(getattr(item, "end"))
        if isinstance(item, (list, tuple)) and len(item) == 2:
            return int(item[0]), int(item[1])
        raise TypeError(f"Unsupported offset item type: {type(item)} -> {item}")

    def _iter_offsets(node: Any, depth: int = 0) -> Iterator[Tuple[int, int]]:
        if node is None:
            return
        try:
            yield _normalize_offset_item(node)
            return
        except TypeError:
            pass
        except ValueError:
            pass

        if isinstance(node, dict):
            for key, value in node.items():
                log.warning("descending into dict key=%s depth=%d", key, depth)
                yield from _iter_offsets(value, depth + 1)
            return

        if isinstance(node, (list, tuple)):
            for item in node:
                yield from _iter_offsets(item, depth + 1)
            return

        if hasattr(node, "__iter__") and not isinstance(node, str):
            for item in node:
                yield from _iter_offsets(item, depth + 1)
            return

        raise TypeError(f"Unexpected offsets structure at depth {depth}: {type(node)} -> {node}")

    try:
        offset_pairs = list(_iter_offsets(offsets_list))
    except Exception:
        log.exception(
            "Failed to flatten tokenizer offsets. raw_type=%s sample_types=%s",
            type(offsets_raw),
            {type(x) for x in offsets_list[:4]},
        )
        raise

    if not offset_pairs:
        raise ValueError("Tokenizer offsets mapping could not be flattened; got empty result.")

    log.warning(
        "Normalized %d token offsets (first 4 shown): %s",
        len(offset_pairs),
        offset_pairs[:4],
    )

    token_shot_ids = torch.full((len(offset_pairs),), fill_value=-2, dtype=torch.long)

    def assign(indices: Sequence[Tuple[int, int]], label: int):
        for span_start, span_end in indices:
            for i, (tok_start, tok_end) in enumerate(offset_pairs):
                if tok_start == tok_end:
                    continue  # special tokens
                if tok_end <= span_start or tok_start >= span_end:
                    continue
                if token_shot_ids[i] == -2:
                    token_shot_ids[i] = label

    if spans["global"]:
        assign(spans["global"], -1)
    if spans["shots"]:
        for shot_id, span in enumerate(spans["shots"]):
            assign([span], shot_id)

    positions = {"global": None, "shots": []}
    global_tokens = torch.where(token_shot_ids == -1)[0]
    if len(global_tokens) > 0:
        positions["global"] = [int(global_tokens.min().item()), int(global_tokens.max().item()) + 1]

    max_shot = token_shot_ids.max().item()
    for shot_id in range(max(0, max_shot) + 1):
        shot_tokens = torch.where(token_shot_ids == shot_id)[0]
        if len(shot_tokens) == 0:
            continue
        positions["shots"].append([int(shot_tokens.min().item()), int(shot_tokens.max().item()) + 1])

    return positions


def build_cross_attention_mask(
    shot_indices: torch.Tensor,
    positions: Dict[str, List[List[int]]],
    context_length: int,
    spatial_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    num_heads: int = 1,
    block_value: float = -1e4,
) -> Optional[torch.Tensor]:
    if positions is None or positions.get("global") is None:
        return None

    shot_ranges = positions.get("shots", [])
    if len(shot_ranges) == 0 and shot_indices.numel() > 0 and shot_indices.max().item() > 0:
        return None

    batch, latent_frames = shot_indices.shape
    vid_shot = shot_indices.repeat_interleave(spatial_tokens, dim=1)

    global_mask = torch.zeros(context_length, dtype=torch.bool, device=device)
    g0, g1 = positions["global"]
    g0 = max(0, min(context_length, g0))
    g1 = max(0, min(context_length, g1))
    if g0 < g1:
        global_mask[g0:g1] = True

    if len(shot_ranges) > 0:
        shot_table = torch.zeros(len(shot_ranges), context_length, dtype=torch.bool, device=device)
        for sid, (s0, s1) in enumerate(shot_ranges):
            s0 = max(0, min(context_length, s0))
            s1 = max(0, min(context_length, s1))
            if s0 < s1:
                shot_table[sid, s0:s1] = True
        allow = shot_table[vid_shot]
        allow = allow | global_mask.view(1, 1, context_length)
    else:
        allow = global_mask.view(1, 1, context_length)

    max_end = max([positions["global"][1]] + [end for _, end in shot_ranges])
    pad_mask = torch.zeros(context_length, dtype=torch.bool, device=device)
    if max_end < context_length:
        pad_mask[max_end:] = True
    allow = allow | pad_mask.view(1, 1, context_length)

    bias = torch.zeros(batch, latent_frames * spatial_tokens, context_length, dtype=dtype, device=device)
    bias = bias.masked_fill(~allow, block_value)
    attn_mask = bias.view(batch, 1, latent_frames * spatial_tokens, context_length)
    return attn_mask


@dataclass
class ShotAttentionConfig:
    enabled: bool
    global_tokens: int
    mode: str = "firstk"
    mask_type: Optional[str] = None


def parse_shot_cut_string(cut_string: str) -> List[int]:
    if not cut_string.strip():
        return []
    parts = re.split(r"[;,\s]+", cut_string.strip())
    values: List[int] = []
    for part in parts:
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid frame index '{part}' in shot cut string") from exc
    return values
