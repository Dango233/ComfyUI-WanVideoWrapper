import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

    offset_pairs: List[Tuple[int, int]] = []
    offset_source = "encodings"

    encodings = getattr(tokenized, "encodings", None)
    if encodings:
        for enc_idx, enc in enumerate(encodings):
            offsets = getattr(enc, "offsets", None)
            if offsets is None:
                print(f"[ShotUtils] encodings[{enc_idx}] missing offsets attribute, type={type(enc)}")
                continue
            for pair_idx, pair in enumerate(offsets):
                start: Optional[int] = None
                end: Optional[int] = None
                try:
                    if isinstance(pair, dict):
                        start = pair.get("start")
                        end = pair.get("end")
                    elif hasattr(pair, "start") and hasattr(pair, "end"):
                        start = getattr(pair, "start")
                        end = getattr(pair, "end")
                    elif isinstance(pair, (list, tuple)) and len(pair) == 2:
                        start, end = pair
                    if start is None or end is None:
                        raise TypeError("offset pair missing start/end")
                    offset_pairs.append((int(start), int(end)))
                except Exception as exc:
                    print(
                        f"[ShotUtils] encodings[{enc_idx}].offsets[{pair_idx}] failed to normalize: type={type(pair)} repr={pair} err={exc}"
                    )
        if offset_pairs:
            print(f"[ShotUtils] offsets extracted via encodings: count={len(offset_pairs)} sample={offset_pairs[:4]}")

    if not offset_pairs:
        offset_source = "offset_mapping"
        if isinstance(tokenized, dict):
            offsets_raw = tokenized.get("offset_mapping")
        else:
            offsets_raw = getattr(tokenized, "offset_mapping", None)

        sample_repr = offsets_raw if isinstance(offsets_raw, (list, tuple, dict)) else type(offsets_raw)
        print(f"[ShotUtils] raw offset_mapping type={type(offsets_raw)} sample={sample_repr}")

        def _collect(node: Any, depth: int = 0) -> None:
            prefix = f"[ShotUtils] flatten depth={depth}"
            if node is None:
                print(f"{prefix} encountered None")
                return
            if hasattr(node, "tolist"):
                node = node.tolist()
                print(f"{prefix} tolist() -> {type(node)}")
            if isinstance(node, dict):
                if "start" in node and "end" in node:
                    offset_pairs.append((int(node["start"]), int(node["end"])) )
                    return
                for key, value in node.items():
                    print(f"{prefix} visiting key={key} type={type(value)}")
                    _collect(value, depth + 1)
                return
            if hasattr(node, "start") and hasattr(node, "end"):
                offset_pairs.append((int(getattr(node, "start")), int(getattr(node, "end"))))
                return
            if isinstance(node, (list, tuple)):
                if len(node) == 2 and all(isinstance(v, (int, float)) for v in node):
                    offset_pairs.append((int(node[0]), int(node[1])))
                    return
                head_type = type(node[0]) if len(node) > 0 else None
                print(f"{prefix} iterating {len(node)} items, head_type={head_type}")
                for item in node:
                    _collect(item, depth + 1)
                return
            print(f"{prefix} unsupported type={type(node)} repr={node}")

        _collect(offsets_raw)

    if not offset_pairs:
        raise ValueError("Tokenizer offsets mapping could not be flattened; got empty result.")

    print(
        f"[ShotUtils] normalized offset count={len(offset_pairs)} source={offset_source} sample={offset_pairs[:4]}"
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
