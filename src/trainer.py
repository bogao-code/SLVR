"""PinReason v2 Trainer — 4-step latent reasoning losses."""

from __future__ import annotations
from typing import Any, Dict, Optional

import torch
from transformers import Trainer


class PinReasonV2Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_components: Dict[str, float] = {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        out = model(**inputs)
        ta  = self.args

        plan_lam   = getattr(ta, "loss_plan_lambda",   0.1)
        bbox_lam   = getattr(ta, "loss_bbox_lambda",   1.0)
        patch_lam  = getattr(ta, "loss_patch_lambda",  1.0)
        answer_lam = getattr(ta, "loss_answer_lambda", 0.5)
        answer_open_lam = getattr(ta, "loss_answer_open_lambda", 2.0)
        short_answer_lam = getattr(ta, "loss_short_answer_lambda", 3.0)

        loss = (
            out.loss_ce
            + plan_lam   * out.loss_plan
            + bbox_lam   * out.loss_bbox
            + patch_lam  * out.loss_patch
            + answer_lam * out.loss_answer
            + answer_open_lam * out.loss_answer_open
            + short_answer_lam * out.loss_short_answer
        )

        self._loss_components = {
            "ce":     float(out.loss_ce.detach()),
            "plan":   float(out.loss_plan.detach()),
            "bbox":   float(out.loss_bbox.detach()),
            "patch":  float(out.loss_patch.detach()),
            "answer": float(out.loss_answer.detach()),
            "answer_open": float(out.loss_answer_open.detach()),
            "short_answer": float(out.loss_short_answer.detach()),
            "total":  float(loss.detach()),
        }
        return (loss, out) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    def log(self, logs: Dict[str, Any], *args, **kwargs):
        if self._loss_components:
            logs.update(self._loss_components)
            self._loss_components = {}
        super().log(logs, *args, **kwargs)
