"""PinReason v2 forward — 4-step latent visual reasoning on Qwen2.5-VL.

Sequence layout (assistant turn):
    <plan_lvr_start>  <lvr>×K1 <plan_lvr_end>       ← CoT text embeddings
    <bbox_lvr_start>  <lvr>×3  <bbox_lvr_end>       ← coord embeddings ((cx, cy), w, h)
    <patch_lvr_start> <lvr>×K3 <patch_lvr_end>      ← image patch embeddings (cos-sim weighted)
    <full_lvr_start>  <lvr>×K4 <full_lvr_end>       ← full-answer text embeddings
    <answer> short_answer </answer>

Training losses
    loss_ce    : CE on all tokens except <lvr> placeholders
    loss_plan  : MSE(hidden[pos-1], plan_embed[t])   weight = loss_plan_lambda   (low)
    loss_bbox  : MSE(hidden[pos-1], coord_embed[t])  weight = loss_bbox_lambda   (high)
    loss_patch : weighted MSE, weight_t ∝ cos_sim_t  weight = loss_patch_lambda  (high)
    loss_answer: MSE(hidden[pos-1], ans_embed[t])    weight = loss_answer_lambda (medium)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.compiler import is_dynamo_compiling as is_torchdynamo_compiling
from transformers.modeling_outputs import ModelOutput

from src.bbox_utils import bbox_to_token_idxs_lvr_packed
from src.constants import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    BBOX_LVR_START,
    FULL_LVR_START,
    IGNORE_INDEX,
    PATCH_LVR_START,
    PLAN_LVR_START,
)

_orig_forward = None


@dataclass
class PinReasonV2Output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_ce: Optional[torch.FloatTensor] = None
    loss_plan: Optional[torch.FloatTensor] = None
    loss_bbox: Optional[torch.FloatTensor] = None
    loss_patch: Optional[torch.FloatTensor] = None
    loss_answer: Optional[torch.FloatTensor] = None
    loss_answer_open: Optional[torch.FloatTensor] = None
    loss_short_answer: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Any = None
    hidden_states: Any = None
    attentions: Any = None
    rope_deltas: Optional[torch.LongTensor] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shifted_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    lf = logits.float()
    sh_logits = lf[..., :-1, :].contiguous().view(-1, lf.size(-1))
    sh_labels = labels[..., 1:].contiguous().view(-1)
    if not bool(sh_labels.ne(IGNORE_INDEX).any().item()):
        return lf.new_tensor(0.0, dtype=torch.float32)
    return F.cross_entropy(sh_logits, sh_labels, ignore_index=IGNORE_INDEX)


def _find_subseq(seq: torch.Tensor, pattern: List[int], start: int = 0) -> int:
    if not pattern or seq.numel() < len(pattern):
        return -1
    pat = torch.tensor(pattern, device=seq.device, dtype=seq.dtype)
    end = int(seq.numel()) - len(pattern) + 1
    for i in range(max(0, start), end):
        if bool(torch.equal(seq[i:i + len(pattern)], pat)):
            return i
    return -1


def _short_answer_ce(logits: torch.Tensor, labels: torch.Tensor,
                     input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    if tokenizer is None or labels is None:
        return logits.new_tensor(0.0)
    open_ids = tokenizer(ANSWER_OPEN, add_special_tokens=False)["input_ids"]
    close_ids = tokenizer(ANSWER_CLOSE, add_special_tokens=False)["input_ids"]
    if not open_ids or not close_ids:
        return logits.new_tensor(0.0)

    short_labels = torch.full_like(labels, IGNORE_INDEX)
    for b in range(input_ids.shape[0]):
        open_pos = _find_subseq(input_ids[b], open_ids)
        if open_pos < 0:
            continue
        ans_start = open_pos + len(open_ids)
        close_pos = _find_subseq(input_ids[b], close_ids, start=ans_start)
        if close_pos < 0:
            continue
        while ans_start < close_pos and int(input_ids[b, ans_start].item()) == getattr(tokenizer, "pad_token_id", -1):
            ans_start += 1
        short_labels[b, ans_start:close_pos] = labels[b, ans_start:close_pos]
    return _shifted_ce(logits, short_labels)


def _answer_open_ce(logits: torch.Tensor, labels: torch.Tensor,
                    input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    if tokenizer is None or labels is None:
        return logits.new_tensor(0.0)
    open_ids = tokenizer(ANSWER_OPEN, add_special_tokens=False)["input_ids"]
    if not open_ids:
        return logits.new_tensor(0.0)

    open_labels = torch.full_like(labels, IGNORE_INDEX)
    for b in range(input_ids.shape[0]):
        open_pos = _find_subseq(input_ids[b], open_ids)
        if open_pos < 0:
            continue
        open_end = open_pos + len(open_ids)
        open_labels[b, open_pos:open_end] = labels[b, open_pos:open_end]
    return _shifted_ce(logits, open_labels)


def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0:
        return pred.new_tensor(0.0, dtype=torch.float32)
    return F.mse_loss(pred.float(), target.float())


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor,
                  weights: torch.Tensor) -> torch.Tensor:
    """MSE loss with per-sample weights. pred/target: [N, H], weights: [N]."""
    if pred.numel() == 0:
        return pred.new_tensor(0.0, dtype=torch.float32)
    diff_sq = ((pred.float() - target.float()) ** 2).mean(-1)   # [N]
    w = weights.float().clamp(min=0.0)
    denom = w.sum().clamp(min=1e-8)
    return (diff_sq * w).sum() / denom


def _coord_embed(val: float, w_fn, tokenizer, device, dtype) -> torch.Tensor:
    """Embed a single coordinate value via mean of its word-piece embeddings."""
    text = f"{val:.3f}"
    ids  = tokenizer(text, add_special_tokens=False)["input_ids"] or [0]
    t    = torch.tensor(ids, device=device, dtype=torch.long)
    return w_fn(t).mean(0).to(dtype)   # [H]


def _coord_pair_embed(x: float, y: float, w_fn, tokenizer, device, dtype) -> torch.Tensor:
    """Embed a center coordinate pair as one bbox latent target."""
    text = f"({x:.3f},{y:.3f})"
    ids = tokenizer(text, add_special_tokens=False)["input_ids"] or [0]
    t = torch.tensor(ids, device=device, dtype=torch.long)
    return w_fn(t).mean(0).to(dtype)


def _build_embeds(model, input_ids, pixel_values, image_grid_thw, lvr_id):
    """Build inputs_embeds with image tokens scattered in. Returns (embeds, image_features)."""
    core   = model.model
    config = model.config

    tc      = getattr(config, "text_config", config)
    voc     = int(getattr(tc, "vocab_size", 152064))
    safe_ids = input_ids.clone()
    # <lvr> may be out-of-vocab range — replace with 0 for embedding lookup
    if lvr_id >= voc:
        safe_ids[input_ids == lvr_id] = 0

    inputs_embeds = core.get_input_embeddings()(safe_ids)

    image_features = None
    if pixel_values is not None:
        raw = core.get_image_features(pixel_values, image_grid_thw)
        if hasattr(raw, "pooler_output"):
            feats = raw.pooler_output
        elif isinstance(raw, (list, tuple)):
            feats = raw
        else:
            feats = (raw,)
        image_features = torch.cat(feats, dim=0) if isinstance(feats, (list, tuple)) else feats

    if image_features is not None:
        image_mask = input_ids == config.image_token_id
        n_img_tok  = image_mask.sum().item()
        n_img_feat = image_features.shape[0]
        if not is_torchdynamo_compiling() and n_img_tok != n_img_feat:
            raise ValueError(f"Image tokens ({n_img_tok}) != image features ({n_img_feat})")
        mask_exp = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            mask_exp, image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        )

    return inputs_embeds, image_features


def _compute_position_ids(model, input_ids, mm_token_type_ids, image_grid_thw, attention_mask):
    position_ids, rope_deltas = model.model.get_rope_index(
        input_ids, mm_token_type_ids, image_grid_thw, None,
        second_per_grid_ts=None, attention_mask=attention_mask,
    )
    model.model.rope_deltas = rope_deltas
    return position_ids


def _group_lvr_positions(input_ids: torch.Tensor,
                          stage_start_ids: List[int], lvr_id: int,
                          B: int) -> List[List[List[int]]]:
    """
    For each batch item, split <lvr> token positions into 4 groups by the most
    recent typed stage-start token that precedes them.

    Returns: groups[b][g] = list of sequence positions (ints), g in 0..3
    """
    groups: List[List[List[int]]] = []
    for b in range(B):
        lvr_mask    = (input_ids[b] == lvr_id)
        lvr_pos     = lvr_mask.nonzero(as_tuple=True)[0]      # positions of <lvr>
        typed_starts: List[Tuple[int, int]] = []
        for g, sid in enumerate(stage_start_ids):
            for pos in (input_ids[b] == sid).nonzero(as_tuple=True)[0].tolist():
                typed_starts.append((int(pos), g))
        typed_starts.sort()

        g_b: List[List[int]] = [[], [], [], []]
        for pos in lvr_pos.tolist():
            g = -1
            for start_pos, start_group in typed_starts:
                if start_pos >= pos:
                    break
                g = start_group
            if 0 <= g <= 3:
                g_b[g].append(pos)
        groups.append(g_b)
    return groups


# ---------------------------------------------------------------------------
# Main forward
# ---------------------------------------------------------------------------

def _forward_pinreason_v2(
    self,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Any = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: bool = True,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    mm_token_type_ids: Optional[torch.Tensor] = None,
    *,
    # Dataset-supplied supervision tensors
    cot_token_ids: torch.LongTensor,      # [B, max_plan_tokens]
    cot_token_num: torch.LongTensor,      # [B]
    answer_token_ids: torch.LongTensor,   # [B, max_full_answer_tokens]
    answer_token_num: torch.LongTensor,   # [B]
    bbox_coords: torch.Tensor,            # [B, 4]: cx, cy, w, h  (normalised)
    bbox_norm: torch.Tensor,              # [B, 4]: x0, y0, x1, y1 (normalised)
    **kwargs: Any,
) -> Union[PinReasonV2Output, Tuple]:

    if inputs_embeds is not None:
        raise ValueError("PinReason v2 expects input_ids, not inputs_embeds.")

    cfg = self.config
    tc  = getattr(cfg, "text_config", cfg)
    H   = int(getattr(tc, "hidden_size", 3584))
    B, S = input_ids.shape

    lvr_id       = int(cfg.lvr_id)
    w_fn         = self.model.get_input_embeddings()
    tokenizer_   = getattr(self, "_tokenizer", None)  # injected during init if available
    if tokenizer_ is None:
        raise ValueError("PinReason v2 typed LVR forward requires tokenizer to be installed.")
    stage_start_ids = [
        int(getattr(cfg, "plan_lvr_start_id", tokenizer_.convert_tokens_to_ids(PLAN_LVR_START))),
        int(getattr(cfg, "bbox_lvr_start_id", tokenizer_.convert_tokens_to_ids(BBOX_LVR_START))),
        int(getattr(cfg, "patch_lvr_start_id", tokenizer_.convert_tokens_to_ids(PATCH_LVR_START))),
        int(getattr(cfg, "full_lvr_start_id", tokenizer_.convert_tokens_to_ids(FULL_LVR_START))),
    ]

    # Position IDs (M-RoPE)
    if position_ids is None:
        position_ids = _compute_position_ids(
            self, input_ids, mm_token_type_ids, image_grid_thw, attention_mask
        )
    if attention_mask is not None:
        attention_mask = attention_mask.to(input_ids.device)

    # Build base embeddings (image tokens scattered in, <lvr> → zero-embed placeholder)
    emb, image_features = _build_embeds(self, input_ids, pixel_values, image_grid_thw, lvr_id)

    # DDP dummy vision pass to keep all-reduce happy when no images
    if pixel_values is None:
        dummy_pv = torch.zeros(784, 1176, device=self.model.visual.device,
                               dtype=self.model.visual.dtype)
        dummy_grid = torch.tensor([[1, 28, 28]], device=self.model.visual.device)
        dummy_out  = self.model.visual(dummy_pv, grid_thw=dummy_grid)
        dummy_emb  = getattr(dummy_out, "pooler_output", dummy_out)
        if isinstance(dummy_emb, (list, tuple)):
            dummy_emb = dummy_emb[0]
        emb = emb + dummy_emb.mean() * 0

    # Identify which <lvr> tokens belong to each of the 4 groups
    groups = _group_lvr_positions(input_ids, stage_start_ids, lvr_id, B)

    # ------------------------------------------------------------
    # Inject target embeddings + collect targets for MSE
    # ------------------------------------------------------------
    # plan_targets[b]   : [K1, H]
    # bbox_targets[b]   : [3,  H]
    # patch_targets[b]  : [K3, H]
    # patch_weights[b]  : [K3]   cosine-sim based
    # answer_targets[b] : [K4, H]
    _dtype = emb.dtype
    plan_targets, bbox_targets, patch_targets, patch_weights, answer_targets = \
        [], [], [], [], []

    for b in range(B):
        dev = emb.device

        # ── Group 1: plan (CoT text embeddings) ──────────────────
        n_plan = int(cot_token_num[b].item())
        g1_pos = groups[b][0]
        K1     = len(g1_pos)
        if K1 > 0 and n_plan > 0:
            ids_ = cot_token_ids[b, :n_plan].to(dev, dtype=torch.long)
            plan_emb = w_fn(ids_).to(_dtype)   # [n_plan, H]
            # align length: if K1 > n_plan, tile; if K1 < n_plan, truncate
            if plan_emb.shape[0] < K1:
                plan_emb = plan_emb[
                    torch.arange(K1, device=dev) % plan_emb.shape[0]
                ]
            else:
                plan_emb = plan_emb[:K1]
            for i, pos in enumerate(g1_pos):
                emb[b, pos] = plan_emb[i]
            plan_targets.append(plan_emb.detach())
        else:
            plan_targets.append(None)

        # ── Group 2: bbox (coordinate embeddings) ─────────────────
        g2_pos = groups[b][1]
        if len(g2_pos) >= 3 and tokenizer_ is not None:
            cx_, cy_, bw_, bh_ = bbox_coords[b].tolist()
            center_e = _coord_pair_embed(cx_, cy_, w_fn, tokenizer_, dev, _dtype)
            w_e  = _coord_embed(bw_, w_fn, tokenizer_, dev, _dtype)
            h_e  = _coord_embed(bh_, w_fn, tokenizer_, dev, _dtype)
            bbox_embs = torch.stack([center_e, w_e, h_e], dim=0)   # [3, H]
            for i, pos in enumerate(g2_pos[:3]):
                emb[b, pos] = bbox_embs[i]
            bbox_targets.append(bbox_embs.detach())
        else:
            bbox_targets.append(None)

        # ── Group 3: image patches (cos-sim selected) ─────────────
        g3_pos = groups[b][2]
        K3     = len(g3_pos)
        if K3 > 0 and image_features is not None and image_grid_thw is not None:
            # Get image patch indices for this batch item's bbox
            n_img_b = (input_ids[b] == cfg.image_token_id).sum().item()
            img_offset = sum(
                (input_ids[bb] == cfg.image_token_id).sum().item()
                for bb in range(b)
            )
            img_f_b = image_features[img_offset : img_offset + n_img_b]  # [N_img, H]

            # Select bbox patches
            grid_b = image_grid_thw[b] if image_grid_thw.dim() == 2 else image_grid_thw
            try:
                bbox_idxs = bbox_to_token_idxs_lvr_packed(
                    tuple(bbox_norm[b].tolist()), grid_b
                )
            except Exception:
                bbox_idxs = list(range(min(K3, img_f_b.shape[0])))

            # Clamp to valid range
            bbox_idxs = [i for i in bbox_idxs if 0 <= i < img_f_b.shape[0]]
            if not bbox_idxs:
                bbox_idxs = list(range(min(K3, img_f_b.shape[0])))

            bbox_patches = img_f_b[
                torch.tensor(bbox_idxs, device=dev, dtype=torch.long)
            ].to(_dtype)  # [N_bbox, H]

            # Answer embedding (mean of full-answer word embeddings)
            n_ans = int(answer_token_num[b].item())
            if n_ans > 0:
                ans_ids_ = answer_token_ids[b, :n_ans].to(dev, dtype=torch.long)
                ans_emb  = w_fn(ans_ids_).mean(0).to(_dtype)   # [H]
            else:
                ans_emb = img_f_b.mean(0).to(_dtype)

            # Cosine similarity between each bbox patch and answer embedding
            cos_sim = F.cosine_similarity(
                bbox_patches.float(),
                ans_emb.float().unsqueeze(0).expand(bbox_patches.shape[0], -1),
                dim=-1,
            )  # [N_bbox]

            # Training K3 is sample-dependent. Prefer patches above the cosine
            # threshold; if there are too few, fall back to the top-K3 ranking.
            threshold = float(getattr(cfg, "patch_cos_threshold", 0.5))
            sorted_scores, sorted_idx = cos_sim.sort(descending=True)
            threshold_idx = sorted_idx[sorted_scores >= threshold]
            if threshold_idx.numel() >= K3:
                top_idx = threshold_idx[:K3]
            elif bbox_patches.shape[0] >= K3:
                top_idx = sorted_idx[:K3]
            else:
                top_idx = sorted_idx[torch.arange(K3, device=dev) % bbox_patches.shape[0]]
            selected_patches = bbox_patches[top_idx]      # [K3, H]
            selected_scores  = cos_sim[top_idx]           # [K3]
            if bool((selected_scores <= 0).all().item()):
                selected_weights = torch.ones_like(selected_scores)
            else:
                selected_weights = selected_scores.clamp(min=0.0)

            for i, pos in enumerate(g3_pos):
                emb[b, pos] = selected_patches[i]
            patch_targets.append(selected_patches.detach())
            patch_weights.append(selected_weights.detach())
        else:
            patch_targets.append(None)
            patch_weights.append(None)

        # ── Group 4: full answer (text embeddings) ────────────────
        g4_pos = groups[b][3]
        K4     = len(g4_pos)
        n_ans2 = int(answer_token_num[b].item())
        if K4 > 0 and n_ans2 > 0:
            ids_ = answer_token_ids[b, :n_ans2].to(dev, dtype=torch.long)
            ans_emb2 = w_fn(ids_).to(_dtype)   # [n_ans2, H]
            if ans_emb2.shape[0] < K4:
                ans_emb2 = ans_emb2[torch.arange(K4, device=dev) % ans_emb2.shape[0]]
            else:
                ans_emb2 = ans_emb2[:K4]
            for i, pos in enumerate(g4_pos):
                emb[b, pos] = ans_emb2[i]
            answer_targets.append(ans_emb2.detach())
        else:
            answer_targets.append(None)

    # ------------------------------------------------------------
    # Single forward pass
    # ------------------------------------------------------------
    fwd_kw = dict(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
        cache_position=cache_position,
    )
    out = self.model.language_model(inputs_embeds=emb, **fwd_kw)
    hidden = out.hidden_states[-1] if out.hidden_states else out.last_hidden_state
    logits = self.lm_head(hidden).float()

    # ------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------
    loss_ce = _shifted_ce(logits, labels) if labels is not None else logits.new_tensor(0.0)
    loss_answer_open = (
        _answer_open_ce(logits, labels, input_ids, tokenizer_)
        if labels is not None else logits.new_tensor(0.0)
    )
    loss_short_answer = (
        _short_answer_ce(logits, labels, input_ids, tokenizer_)
        if labels is not None else logits.new_tensor(0.0)
    )

    loss_plan = loss_bbox = loss_patch = loss_answer = logits.new_tensor(0.0)

    for b in range(B):
        # ── Plan MSE ─────────────────────────────────────────────
        g1_pos = groups[b][0]
        if plan_targets[b] is not None and len(g1_pos) > 0:
            src_pos = torch.tensor([p - 1 for p in g1_pos], device=hidden.device)
            src_pos = src_pos.clamp(min=0)
            pred = hidden[b, src_pos].to(torch.float32)
            tgt  = plan_targets[b].to(hidden.device, torch.float32)
            loss_plan = loss_plan + _mse(pred, tgt)

        # ── BBox MSE ──────────────────────────────────────────────
        g2_pos = groups[b][1]
        if bbox_targets[b] is not None and len(g2_pos) >= 3:
            src_pos = torch.tensor([p - 1 for p in g2_pos[:3]], device=hidden.device)
            src_pos = src_pos.clamp(min=0)
            pred = hidden[b, src_pos].to(torch.float32)
            tgt  = bbox_targets[b].to(hidden.device, torch.float32)
            loss_bbox = loss_bbox + _mse(pred, tgt)

        # ── Patch MSE (cos-sim weighted) ──────────────────────────
        g3_pos = groups[b][2]
        if patch_targets[b] is not None and len(g3_pos) > 0:
            src_pos = torch.tensor([p - 1 for p in g3_pos], device=hidden.device)
            src_pos = src_pos.clamp(min=0)
            pred = hidden[b, src_pos].to(torch.float32)
            tgt  = patch_targets[b].to(hidden.device, torch.float32)
            w    = patch_weights[b].to(hidden.device, torch.float32)
            loss_patch = loss_patch + _weighted_mse(pred, tgt, w)

        # ── Answer MSE ────────────────────────────────────────────
        g4_pos = groups[b][3]
        if answer_targets[b] is not None and len(g4_pos) > 0:
            src_pos = torch.tensor([p - 1 for p in g4_pos], device=hidden.device)
            src_pos = src_pos.clamp(min=0)
            pred = hidden[b, src_pos].to(torch.float32)
            tgt  = answer_targets[b].to(hidden.device, torch.float32)
            loss_answer = loss_answer + _mse(pred, tgt)

    if B > 1:
        loss_plan   = loss_plan   / B
        loss_bbox   = loss_bbox   / B
        loss_patch  = loss_patch  / B
        loss_answer = loss_answer / B

    return PinReasonV2Output(
        loss=None,     # combined in trainer
        loss_ce=loss_ce,
        loss_plan=loss_plan,
        loss_bbox=loss_bbox,
        loss_patch=loss_patch,
        loss_answer=loss_answer,
        loss_answer_open=loss_answer_open,
        loss_short_answer=loss_short_answer,
        logits=logits,
        past_key_values=getattr(out, "past_key_values", None),
        hidden_states=getattr(out, "hidden_states", None),
        attentions=getattr(out, "attentions", None),
        rope_deltas=getattr(self.model, "rope_deltas", None),
    )


# ---------------------------------------------------------------------------
# Dispatch + monkey-patch
# ---------------------------------------------------------------------------

def _dispatch_forward(self, *args, **kwargs):
    global _orig_forward
    kwargs.pop("num_items_in_batch", None)
    has_v2 = (
        "cot_token_ids"    in kwargs and kwargs.get("cot_token_ids")    is not None
        and "answer_token_ids" in kwargs and kwargs.get("answer_token_ids") is not None
        and "bbox_coords"  in kwargs and kwargs.get("bbox_coords")      is not None
    )
    if has_v2:
        return _forward_pinreason_v2(self, *args, **kwargs)
    return _orig_forward(self, *args, **kwargs)


def install_pinreason_v2_patches(tokenizer=None):
    """Monkey-patch Qwen2_5_VLForConditionalGeneration.forward (idempotent).

    Pass tokenizer so the coordinate embedding helper can tokenise float strings.
    The tokenizer is stashed on the model instance via model._tokenizer.
    """
    global _orig_forward
    from transformers import Qwen2_5_VLForConditionalGeneration
    if _orig_forward is None:
        _orig_forward = Qwen2_5_VLForConditionalGeneration.forward
        Qwen2_5_VLForConditionalGeneration.forward = _dispatch_forward

    if tokenizer is not None:
        # stash for coordinate embedding helper — set at load time
        _stash_tokenizer(tokenizer)


def _stash_tokenizer(tokenizer):
    """Attach tokenizer to all future model instances via __init_subclass__ hook."""
    from transformers import Qwen2_5_VLForConditionalGeneration

    _orig_init = Qwen2_5_VLForConditionalGeneration.__init__

    def _new_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self._tokenizer = tokenizer

    Qwen2_5_VLForConditionalGeneration.__init__ = _new_init
