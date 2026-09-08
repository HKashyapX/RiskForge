import re
from riskforge.core.contracts import IncidentNormalizedRecord
from riskforge.supervision.engine import (
    ABSTAIN,
    NON_SIF,
    SIF_P,
    LabelingFunction,
    labeling_function,
)

NEGATION_WINDOW = 6
NEGATION_SET: set[str] = {
    "no", "not", "zero", "none", "neither", "never", "without",
    "intact", "passed", "normal", "tested", "secure", "prevented"
}

PRESSURE_HAZARD = re.compile(r"\b(\d{3,5}\s*(?:psi|bar|kg/cm2)|high\s*pressure|well\s*kick|surge)\b", re.IGNORECASE)
WELL_CONTROL_BARRIERS = re.compile(r"\b(blowout_preventer|well_control_line|pressure_relief_valve|flange|gasket|seal|choke)\b", re.IGNORECASE)
BARRIER_LOSS = re.compile(r"\b(leak|blown|crack|rupture|bypass|damage|malfunction|passing|blown\s*out)\b", re.IGNORECASE)
RIG_CREW = re.compile(r"\b(derrickman|driller|roustabout|roughneck|operator|crew|personnel|worker)\b", re.IGNORECASE)

HEIGHT_LOCATION = re.compile(r"\b(monkey_board|derrick|mast|scaffold|ladder|elevated)\b", re.IGNORECASE)
FALL_PROTECTION_LOSS = re.compile(r"\b(harness|lanyard|lifeline|safety\s*belt|inertia\s*reel)\s+(?:missing|unhooked|not\s*anchored|damaged|open|failed)\b", re.IGNORECASE)

LIFTING_EQUIPMENT = re.compile(r"\b(crane|winch_line|mast\s*raising|suspended\s*load|hoist|cathead)\b", re.IGNORECASE)
RIGGING_FAILURE = re.compile(r"\b(snapped|parted|unhooked|brake\s*failure|dropped|slipped)\b", re.IGNORECASE)
LINE_OF_FIRE = re.compile(r"\b(line\s*of\s*fire|underneath|below|cellar|rotary_table)\b", re.IGNORECASE)

TOXIC_GAS = re.compile(r"\b(hydrogen_sulfide|h2s|sour\s*gas)\b", re.IGNORECASE)
GAS_LEAK_ACTION = re.compile(r"\b(leak|detected|release|alarm|emission)\b", re.IGNORECASE)
VENTILATION_CONTAINMENT_LOSS = re.compile(r"\b(breathing\s*apparatus|cascade|detector|scba)\s+(?:fail|missing|empty|malfunction|bypassed)\b", re.IGNORECASE)

LOTO_VIOLATION = re.compile(r"\b(energy_isolation_loto)\s+(?:bypassed|missing|not\s*applied|energized|live)\b", re.IGNORECASE)
ELECTRICAL_MECHANICAL_HAZARD = re.compile(r"\b(breaker|panel|motor|switchgear|pressurized\s*line|generator)\b", re.IGNORECASE)

BENIGN_HOUSEKEEPING = re.compile(r"\b(housekeeping|litter|dust|water\s*puddle|bruise|office|mess\s*room|first\s*aid|tea|canteen)\b", re.IGNORECASE)
PPE_NON_CRITICAL = re.compile(r"\b(?:not\s*wearing|missing)\s*(?:glasses|goggles|helmet|hard\s*hat|ear\s*plug|gloves)\b", re.IGNORECASE)

def is_negated(text: str, index: int) -> bool:
    tokens = text[:index].lower().split()
    preceding = tokens[-NEGATION_WINDOW:] if len(tokens) >= NEGATION_WINDOW else tokens
    return any(t in NEGATION_SET for t in preceding)

@labeling_function()
def lf_well_control_breach(record: IncidentNormalizedRecord) -> int:
    text = record.raw_narrative
    canonical_tokens = " ".join(s.canonical_form for s in record.spans)
    p = PRESSURE_HAZARD.search(text)
    b = WELL_CONTROL_BARRIERS.search(text) or WELL_CONTROL_BARRIERS.search(canonical_tokens)
    l = BARRIER_LOSS.search(text)
    w = RIG_CREW.search(text)
    if p and b and l and w:
        if is_negated(text, l.start()):
            return NON_SIF
        return SIF_P
    return ABSTAIN

@labeling_function()
def lf_height_fall_hazard(record: IncidentNormalizedRecord) -> int:
    text = record.raw_narrative
    canonical_tokens = " ".join(s.canonical_form for s in record.spans)
    h = HEIGHT_LOCATION.search(text) or HEIGHT_LOCATION.search(canonical_tokens)
    f = FALL_PROTECTION_LOSS.search(text)
    if h and f:
        if is_negated(text, f.start()):
            return NON_SIF
        return SIF_P
    return ABSTAIN

@labeling_function()
def lf_dropped_object_exposure(record: IncidentNormalizedRecord) -> int:
    text = record.raw_narrative
    l = LIFTING_EQUIPMENT.search(text)
    r = RIGGING_FAILURE.search(text)
    w = LINE_OF_FIRE.search(text)
    if l and r and w:
        if is_negated(text, r.start()):
            return NON_SIF
        return SIF_P
    return ABSTAIN

@labeling_function()
def lf_h2s_confinement_breach(record: IncidentNormalizedRecord) -> int:
    text = record.raw_narrative
    canonical_tokens = " ".join(s.canonical_form for s in record.spans)
    g = TOXIC_GAS.search(text) or TOXIC_GAS.search(canonical_tokens)
    a = GAS_LEAK_ACTION.search(text)
    b = VENTILATION_CONTAINMENT_LOSS.search(text)
    if g and a and b:
        if is_negated(text, b.start()):
            return NON_SIF
        return SIF_P
    return ABSTAIN

@labeling_function()
def lf_isolation_loto_breach(record: IncidentNormalizedRecord) -> int:
    text = record.raw_narrative
    canonical_tokens = " ".join(s.canonical_form for s in record.spans)
    l = LOTO_VIOLATION.search(text) or LOTO_VIOLATION.search(canonical_tokens)
    e = ELECTRICAL_MECHANICAL_HAZARD.search(text)
    if l and e:
        return SIF_P
    return ABSTAIN

@labeling_function()
def lf_low_energy_noise(record: IncidentNormalizedRecord) -> int:
    text = record.raw_narrative
    has_benign = bool(BENIGN_HOUSEKEEPING.search(text) or PPE_NON_CRITICAL.search(text))
    has_critical = bool(
        PRESSURE_HAZARD.search(text)
        or HEIGHT_LOCATION.search(text)
        or LIFTING_EQUIPMENT.search(text)
        or TOXIC_GAS.search(text)
    )
    if has_benign and not has_critical:
        return NON_SIF
    return ABSTAIN

DEFAULT_LFS: list[LabelingFunction] = [
    lf_well_control_breach,
    lf_height_fall_hazard,
    lf_dropped_object_exposure,
    lf_h2s_confinement_breach,
    lf_isolation_loto_breach,
    lf_low_energy_noise,
]
