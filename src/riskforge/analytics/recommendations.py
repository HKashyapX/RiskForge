"""Deterministic preventive-recommendation mapping.

Recommendations are derived only from detected hazards, matched Life-Saving
Rules, and failed barriers, and are phrased as preventive priorities that
SUPPORT established HSE procedures.  They never replace professional safety
judgement; every response carrying recommendations also carries that
disclaimer.
"""

from __future__ import annotations

from riskforge.core.contracts import LifeSavingRule, ModelInferenceResult

DISCLAIMER = (
    "Preventive priorities are decision support only and do not replace "
    "Oil India Limited HSE procedures or professional judgement."
)

_RECOMMENDATIONS: dict[LifeSavingRule, tuple[str, ...]] = {
    LifeSavingRule.WORK_AT_HEIGHT: (
        "Verify 100% fall protection (harness with double lanyard) for work above 1.8 m.",
        "Inspect and close edge-protection and guardrail gaps before resuming work.",
    ),
    LifeSavingRule.ENERGY_ISOLATION: (
        "Confirm Lockout/Tagout isolation with zero-energy verification before line breaking.",
        "Audit isolation permits and bleed/vent procedures for the affected system.",
    ),
    LifeSavingRule.CONFINED_SPACE: (
        "Enforce entry permits with atmospheric gas testing before and during entry.",
        "Verify standby attendant and rescue arrangements at the confined space.",
    ),
    LifeSavingRule.TOXIC_GAS: (
        "Verify fixed and portable H2S/gas detectors are calibrated and alarmed.",
        "Confirm breathing-apparatus availability and muster drills for affected areas.",
    ),
    LifeSavingRule.LINE_OF_FIRE: (
        "Establish exclusion zones and spotter control around suspended loads and pinch points.",
        "Review task planning to remove personnel from the line of fire.",
    ),
    LifeSavingRule.BYPASSING_SAFETY_CONTROLS: (
        "Restore disabled alarms/interlocks and raise a management-of-change record for any bypass.",
        "Investigate why the safeguard was bypassed and address the incentive or pressure.",
    ),
    LifeSavingRule.SAFE_MECHANICAL_LIFTING: (
        "Verify lift plan, rigging inspection, and SWL ratings before the next lift.",
        "Confirm exclusion zone and tag-line use for suspended loads.",
    ),
    LifeSavingRule.HOT_WORK: (
        "Enforce hot-work permits with fire watch and fire extinguishment readiness.",
        "Remove or shield flammable material from the work area.",
    ),
    LifeSavingRule.DRIVING: (
        "Reinforce journey management, seat-belt, and speed compliance checks.",
        "Require spotter control for reversing vehicles in operational areas.",
    ),
}

_BARRIER_RECOMMENDATIONS: dict[str, str] = {
    "pressure_relief_valve": "Function-test the pressure relief/safety valve and verify set pressure.",
    "energy_isolation_loto": "Re-verify the isolation certificate and physical lock placement.",
    "well_control_line": "Pressure-test well-control lines and verify valve alignment.",
    "blowout_preventer": "Schedule BOP function and pressure testing; verify annular and ram condition.",
}


def build_recommendations(result: ModelInferenceResult) -> tuple[str, ...]:
    """Map detected hazards and failed barriers to preventive priorities."""
    recommendations: list[str] = []
    for rule in result.matched_iogp_rules:
        recommendations.extend(_RECOMMENDATIONS.get(rule, ()))
    barrier = result.triad.failed_barrier
    if barrier is not None:
        barrier_text = _BARRIER_RECOMMENDATIONS.get(barrier.canonical_form)
        if barrier_text:
            recommendations.append(barrier_text)
    if result.triad.failed_barrier is None and not result.matched_iogp_rules:
        return ()
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for item in recommendations:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return tuple(unique)
