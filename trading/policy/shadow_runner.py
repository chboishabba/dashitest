from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShadowPolicyStep:
    intent_live: object
    intent_shadow: object
    beam_summary: object | None = None
    beam_diagnostics: object | None = None

    def to_log_fields(self) -> dict[str, object]:
        row = {
            "live_direction": getattr(self.intent_live, "direction", 0),
            "live_target_exposure": getattr(self.intent_live, "target_exposure", 0.0),
            "live_hold": int(bool(getattr(self.intent_live, "hold", False))),
            "live_reason": getattr(self.intent_live, "reason", ""),
            "shadow_direction": getattr(self.intent_shadow, "direction", 0),
            "shadow_target_exposure": getattr(self.intent_shadow, "target_exposure", 0.0),
            "shadow_hold": int(bool(getattr(self.intent_shadow, "hold", False))),
            "shadow_reason": getattr(self.intent_shadow, "reason", ""),
        }
        if self.beam_summary is not None and hasattr(self.beam_summary, "to_log_fields"):
            row.update(self.beam_summary.to_log_fields())
        if self.beam_diagnostics is not None and hasattr(self.beam_diagnostics, "to_log_fields"):
            row.update(self.beam_diagnostics.to_log_fields())
        return row


class ShadowPolicyRunner:
    """
    Run the live/base policy and an observe-only futures policy side by side.
    """

    def __init__(self, base_policy, futures_policy):
        self.base = base_policy
        self.futures = futures_policy

    def step(self, *args, **kwargs) -> ShadowPolicyStep:
        intent_live = self.base.step(*args, **kwargs)
        shadow_result = self.futures.step(*args, **kwargs)
        if hasattr(shadow_result, "intent") and hasattr(shadow_result, "summary"):
            intent_shadow = shadow_result.intent
            beam_summary = shadow_result.summary
            beam_diagnostics = getattr(shadow_result, "diagnostics", None)
        elif isinstance(shadow_result, tuple) and len(shadow_result) == 2:
            intent_shadow, beam_summary = shadow_result
            beam_diagnostics = None
        else:
            intent_shadow = shadow_result
            beam_summary = None
            beam_diagnostics = None
        return ShadowPolicyStep(
            intent_live=intent_live,
            intent_shadow=intent_shadow,
            beam_summary=beam_summary,
            beam_diagnostics=beam_diagnostics,
        )
