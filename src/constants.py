"""Special tokens and constants for PinReason v2 on Qwen2.5-VL."""

IGNORE_INDEX = -100

# Typed structural special tokens + one latent placeholder
PLAN_LVR_START = "<plan_lvr_start>"
PLAN_LVR_END = "<plan_lvr_end>"
BBOX_LVR_START = "<bbox_lvr_start>"
BBOX_LVR_END = "<bbox_lvr_end>"
PATCH_LVR_START = "<patch_lvr_start>"
PATCH_LVR_END = "<patch_lvr_end>"
FULL_LVR_START = "<full_lvr_start>"
FULL_LVR_END = "<full_lvr_end>"

# Backward-compatible aliases for old checkpoints/scripts.
LVR_START = PLAN_LVR_START
LVR_END = PLAN_LVR_END
LVR = "<lvr>"

ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

STAGE_LVR_TOKENS = (
    PLAN_LVR_START, PLAN_LVR_END,
    BBOX_LVR_START, BBOX_LVR_END,
    PATCH_LVR_START, PATCH_LVR_END,
    FULL_LVR_START, FULL_LVR_END,
)
SPECIAL_TOKENS = (*STAGE_LVR_TOKENS, LVR, ANSWER_OPEN, ANSWER_CLOSE)

SYSTEM_MESSAGE = (
    "Look at the image and answer the question. "
    "Reason step by step using latent reasoning before answering."
)
