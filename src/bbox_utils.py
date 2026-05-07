"""Bbox → image-token index mapping for Qwen2.5-VL (dynamic resolution).

Spatial targets for LVR1 follow the same convention as the packed LVR
box-to-token mapping:
``bbox_to_token_idxs`` (floor/ceil on the fine grid + merge stride 2).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch


def _bbox_xyxy_normalized(bbox_norm: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Clip normalized corners to ``[0, 1]`` (same assumption as ``lvr_sft_dataset_packed``)."""
    x0, y0, x1, y1 = float(bbox_norm[0]), float(bbox_norm[1]), float(bbox_norm[2]), float(bbox_norm[3])
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(1.0, x1), min(1.0, y1)
    return x0, y0, x1, y1


def _first_image_grid_thw(image_grid_thw: torch.Tensor) -> Tuple[int, int, int]:
    """Return (t, h, w) for the first image / frame group."""
    if image_grid_thw.dim() == 1 and image_grid_thw.numel() >= 3:
        return int(image_grid_thw[0]), int(image_grid_thw[1]), int(image_grid_thw[2])
    if image_grid_thw.dim() == 2 and image_grid_thw.shape[0] >= 1:
        return int(image_grid_thw[0, 0]), int(image_grid_thw[0, 1]), int(image_grid_thw[0, 2])
    raise ValueError(f"unexpected image_grid_thw shape {tuple(image_grid_thw.shape)}")


def bbox_to_token_idxs_lvr_packed(
    bbox_norm: Tuple[float, float, float, float],
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
) -> List[int]:
    """Same token indices as ``lvr_sft_dataset_packed.bbox_to_token_idxs``.

    ``image_grid_thw`` rows are ``(t, h, w)`` as returned by Qwen2.5-VL processor — i.e. the fine
    spatial grid **before** pooling to LM tokens (see transformers ``get_vision_position_ids``).
    """
    _, h, w = _first_image_grid_thw(image_grid_thw)
    x0, y0, x1, y1 = _bbox_xyxy_normalized(bbox_norm)

    x0_grid = max(0, min(int(math.floor(x0 * w)), w - 1))
    x1_grid = max(0, min(int(math.ceil(x1 * w)), w))
    y0_grid = max(0, min(int(math.floor(y0 * h)), h - 1))
    y1_grid = max(0, min(int(math.ceil(y1 * h)), h))

    m = spatial_merge_size
    x0_token = x0_grid // m
    x1_token = (x1_grid + 1) // m
    y0_token = y0_grid // m
    y1_token = (y1_grid + 1) // m
    w_m = w // m  # Packed uses ``yy * W2 + xx`` with ``W2 = w // 2``
    idxs = [
        int(yy * w_m + xx)
        for yy in range(y0_token, y1_token)
        for xx in range(x0_token, x1_token)
    ]
    return idxs


def bbox_to_token_idxs_dynamic(
    bbox_norm: Tuple[float, float, float, float],
    h_tokens: int,
    w_tokens: int,
) -> torch.LongTensor:
    """Fallback: normalized bbox → merged-token grid assuming ``h_tokens × w_tokens`` is LM layout."""
    x0, y0, x1, y1 = _bbox_xyxy_normalized(bbox_norm)
    cx0 = int(math.floor(x0 * w_tokens))
    cy0 = int(math.floor(y0 * h_tokens))
    cx1 = min(int(math.ceil(x1 * w_tokens)), w_tokens)
    cy1 = min(int(math.ceil(y1 * h_tokens)), h_tokens)
    if cx1 <= cx0:
        cx1 = cx0 + 1
    if cy1 <= cy0:
        cy1 = cy0 + 1

    idxs: List[int] = []
    for r in range(cy0, cy1):
        for c in range(cx0, cx1):
            idxs.append(r * w_tokens + c)
    return torch.tensor(idxs, dtype=torch.long)


def build_spatial_target(
    bbox_norm: Tuple[float, float, float, float],
    n_image_tokens: int,
    h_tokens: int,
    w_tokens: int,
    image_grid_thw: Optional[torch.Tensor] = None,
    spatial_merge_size: int = 2,
) -> torch.FloatTensor:
    """Uniform distribution over LM image tokens covered by bbox (LVR Packed convention when possible).

    If ``image_grid_thw`` is set, indices come from ``bbox_to_token_idxs_lvr_packed`` (same as
    ``~/nips/code/lvr``). Otherwise fall back to a legacy grid counted as ``h_tokens × w_tokens``
    (only for missing-grid edge cases).
    """
    target = torch.zeros(n_image_tokens, dtype=torch.float32)
    idxs: Union[List[int], torch.Tensor]

    if image_grid_thw is not None and isinstance(image_grid_thw, torch.Tensor) and image_grid_thw.numel() >= 3:
        idxs = bbox_to_token_idxs_lvr_packed(
            bbox_norm, image_grid_thw, spatial_merge_size=spatial_merge_size,
        )
    else:
        idxs = bbox_to_token_idxs_dynamic(bbox_norm, h_tokens, w_tokens).tolist()

    seen = []
    if isinstance(idxs, list):
        for i in idxs:
            if 0 <= int(i) < n_image_tokens:
                seen.append(int(i))
    else:
        for i in idxs.tolist():
            if 0 <= i < n_image_tokens:
                seen.append(i)

    # dedupe preserving order for stability
    seen_u: List[int] = []
    for i in seen:
        if i not in seen_u:
            seen_u.append(i)

    if not seen_u:
        target.fill_(1.0 / max(n_image_tokens, 1))
        return target

    for i in seen_u:
        target[i] = 1.0
    target = target / float(target.sum())
    return target
