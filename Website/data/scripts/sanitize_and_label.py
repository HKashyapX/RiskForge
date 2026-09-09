#!/usr/bin/env python3
"""
Sanitize and weak-label OSHA incident data using the actual schemas observed in:
  - January2015toNovember2025.zip -> January2015toNovember2025.csv (28 columns)
  - tagged1000.xlsx -> Sheet1, labeled examples
  - osha.xlsx -> legacy 5-column positional workbook

Outputs:
  processed/clean_incidents.parquet
  processed/sif_training.jsonl
  processed/sif_validation.jsonl
  processed/sif_test.jsonl
  processed/sif_eval.jsonl
  processed/label_report.csv

The Parquet output requires pyarrow. Other outputs use the Python standard library.
"""

from __future__ import annotations
import argparse, csv, hashlib, io, json, math, os, re, unicodedata, zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from xml.etree import ElementTree as ET

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

TAG_CANON = {
    "falls": "falls",
    "collapse of object": "collapse_of_object",
    "struck by moving objects": "struck_by_moving_object",
    "electrocution": "electrocution",
    "caught in/between objects": "caught_in_between",
    "traffic": "traffic",
    "fires and explosion": "fire_explosion",
    "others": "other",
    "struck by falling object": "struck_by_falling_object",
    "exposure to chemical substances": "chemical_exposure",
    "exposure to extreme temperatures": "extreme_temperature",
}
SIF_HIGH = {
    "falls", "collapse_of_object", "struck_by_moving_object",
    "electrocution", "caught_in_between", "traffic",
    "fire_explosion", "struck_by_falling_object",
}

HAZARD_RULES = [
    ("electrocution", ["electrocution","electrocuted","arc flash","live wire","energized"]),
    ("traffic", ["traffic","highway","vehicle struck","struck by vehicle","backing truck","pedestrian"]),
    ("fire_explosion", ["explosion","exploded","flash fire","ignition","combustible"," fire "]),
    ("caught_in_between", ["caught in","caught between","pinned","crushed","pinch point","entangled","in between"]),
    ("struck_by_falling_object", ["struck by falling","falling object","falling material","dropped load","object fell","load fell"]),
    ("struck_by_moving_object", ["struck by","struck against","flying object","projectile","moving object"]),
    ("collapse_of_object", ["collapse","cave-in","cave in","trench collapse","unstable load","overturn","tipped over","tipped back"]),
    ("falls", ["fall from","fell from","fell to the ground","fell to lower","falling","ladder","scaffold","roof fall"]),
    ("chemical_exposure", ["chemical","solvent","toxic vapor","toxic gas","toxic fume","corrosive","acid","chlorine","caustic"]),
    ("extreme_temperature", ["second degree burn","third degree burn","heat stress","heat illness","hypothermia","thermal","hot pressurized"]),
]
ACTIVITY_RULES = [
    ("maintenance_repair",["maintenance","repairing","repair","servicing","service"]),
    ("cleaning",["cleaning","cleaned","cleanup"]),
    ("installation",["installing","installed","installation","constructing","construction"]),
    ("driving_transport",["truck driver","driver","driving","vehicle","transport"]),
    ("material_handling",["forklift","pallet jack","moving","moved","loading","loaded","unloading","unloaded"]),
    ("lifting_rigging",["crane","hoist","rigging","lifting","lifted","load line"]),
    ("hot_work",["welding","welded","torch","cutting","hot work"]),
    ("excavation",["trench","excavation","excavated","shoring"]),
    ("confined_space",["confined space","tank car","manhole"]),
    ("roof_ladder_scaffold",["roof","ladder","scaffold"]),
    ("process_line_break",["valve","coupler","pressurized","pressure","line break"]),
]
BARRIER_RULES = [
    ("fall_protection",["fall protection","safety harness","guardrail","guard rail","tie-off","lifeline"]),
    ("machine_guarding",["unguarded","guarding","point of operation","safety guard"]),
    ("lockout_tagout",["lockout","tagout","de-energize","deenergize","energized"]),
    ("traffic_control",["flagger","traffic control","work zone","vehicle barrier"]),
    ("lifting_load_control",["dunnage","tag line","rigging","stable load","load secure"]),
    ("process_isolation",["shut off","isolated","isolate","depressurize","depressurized","pressurized","valve"]),
    ("confined_space_control",["permit-required","permit required","air monitor","atmospheric","confined space"]),
    ("ppe",["respirator","gloves","safety glasses","face shield","hearing protection"]),
]
BARRIER_GAP_RULES = [
    ("fall_protection_gap",["no fall protection","without fall protection","not tied off","no guardrail","unprotected edge","unprotected opening"]),
    ("machine_guarding_gap",["unguarded","guard removed","guarding removed","breached guard","breached machine"]),
    ("lockout_tagout_gap",["not locked out","without lockout","failed to de-energize","worked on energized"]),
    ("traffic_control_gap",["no flagger","no traffic control","without traffic control"]),
    ("load_control_gap",["unstable load","improperly secured","not secured"]),
]

PII_PATTERNS = [
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), "[REDACTED_EMAIL]"),
    (re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]\d{4}(?!\d)"), "[REDACTED_PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),

    # Consume the COMPLETE serial/equipment identifier, including identifiers
    # containing hyphens, so fragments such as 0-[REDACTED_SSN] cannot remain.
    (re.compile(
        r"\b(?:serial\s*(?:number|no\.?)|s/n|serial)\s*[:#-]?\s*"
        r"[A-Z0-9]+(?:[-./][A-Z0-9]+){1,}\b", re.I
    ), "serial number [REDACTED_ID]"),

    # Model identifiers may contain letters, digits and separators.
    (re.compile(
        r"\b(?:model\s*(?:number|no\.?)|model)\s*[:#-]?\s*"
        r"[A-Z0-9]+(?:[-./][A-Z0-9]+){1,}\b", re.I
    ), "model [REDACTED_ID]"),

    # Equipment/asset/rental identifiers.
    (re.compile(
        r"\b(?:id\s*(?:number|no\.?)|equipment\s*(?:id|number)|"
        r"asset\s*(?:id|number))\s*[:#-]?\s*"
        r"[A-Z0-9]+(?:[-./][A-Z0-9]+){1,}\b", re.I
    ), "ID Number [REDACTED_ID]"),
]

OUTPUT_FIELDS = [
    "incident_id","source_datasets","source_file_ids","event_date","employer","state","naics",
    "title_raw","narrative_raw","narrative_sanitized","title_normalized","text_normalized",
    "keywords_raw","nature","nature_title","body","body_title","event","event_title",
    "source_title","secondary_source_title","federal_state","hospitalized","amputation","loss_of_eye",
    "legacy_meta","cause_raw","tagged_label_raw","hazard_class","hazard_label_source",
    "hazard_label_confidence","hazard_evidence","actual_severity","actual_severity_confidence",
    "sif_potential","sif_confidence","sif_rationale","activity_tags","activity_evidence",
    "barrier_labels","barrier_evidence","barrier_gap_signals","barrier_gap_evidence",
    "quality_flags","transformation_log","pipeline_version","processed_at_utc"
]

def clean_space(value: object) -> str:
    if value is None:
        return ""
    s = unicodedata.normalize("NFKC", str(value)).replace("\x00", " ")
    s = re.sub(r"[\r\n\t]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def norm_text(value: object) -> str:
    s = clean_space(value).lower()
    s = re.sub(r"\bemployee\s*#\s*\d+\b", "employee", s)
    return s

def sanitize_text(value: object) -> str:
    s = clean_space(value)
    for pattern, replacement in PII_PATTERNS:
        s = pattern.sub(replacement, s)
    s = re.sub(r"\b(Employee\s*#\s*)\d+\b", r"\1[ID]", s, flags=re.I)
    return s

def boolish(value: object) -> bool:
    return clean_space(value).lower() in {"1","1.0","1.00","true","yes","y"}

def first_date(value: object) -> str:
    s = clean_space(value)
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    return ""

NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RELNS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

def xlsx_rows(path: str):
    with zipfile.ZipFile(path) as archive:
        wb_xml = ET.fromstring(archive.read("xl/workbook.xml"))
        rel_xml = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rels = {r.attrib["Id"]: r.attrib["Target"] for r in rel_xml}
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            ss = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for si in ss.findall("main:si", NS):
                shared.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["main"])))
        for sheet in wb_xml.find("main:sheets", NS):
            rid = sheet.attrib["{%s}id" % RELNS]
            target = rels[rid]
            if not target.startswith("xl/"):
                target = "xl/" + target
            root = ET.fromstring(archive.read(target))
            for row in root.findall(".//main:sheetData/main:row", NS):
                vals = {}
                for cell in row.findall("main:c", NS):
                    ref = cell.attrib.get("r", "")
                    match = re.match(r"([A-Z]+)", ref)
                    if not match:
                        continue
                    col = match.group(1)
                    v = cell.find("main:v", NS)
                    cell_type = cell.attrib.get("t")
                    if cell_type == "s" and v is not None:
                        value = shared[int(v.text)]
                    elif cell_type == "inlineStr":
                        isel = cell.find("main:is", NS)
                        value = "".join(t.text or "" for t in isel.iter("{%s}t" % NS["main"])) if isel is not None else ""
                    elif v is not None:
                        value = v.text
                    else:
                        value = ""
                    vals[col] = value
                if vals:
                    yield sheet.attrib["name"], vals

def header_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_space(s).lower())

def validate_and_read_tagged(path: str):
    header = None
    rows = []
    for _, row in xlsx_rows(path):
        if header is None:
            header = row
            continue
        rows.append(row)
    header_map = {header_key(v): k for k, v in header.items() if v}
    required = {"id","title","summary2","cause","newkeys","titlenew","summarynew","tagged2"}
    missing = [x for x in required if x not in header_map]
    if missing:
        raise ValueError(f"tagged1000.xlsx schema mismatch; missing normalized columns: {missing}; actual headers: {list(header.values())}")
    out = []
    for row in rows:
        rec = {}
        for normalized, col in header_map.items():
            if col in row:
                rec[normalized] = row[col]
        out.append(rec)
    return out

def validate_and_read_legacy(path: str):
    rows = []
    for _, row in xlsx_rows(path):
        rows.append({
            "id": row.get("A",""),
            "title": row.get("B",""),
            "narrative": row.get("C",""),
            "keywords": row.get("D",""),
            "meta": row.get("E",""),
        })
    return rows

def validate_csv_headers(headers):
    required = [
        "ID","UPA","EventDate","Employer","Address1","Address2","City","State","Zip",
        "Latitude","Longitude","Primary NAICS","Hospitalized","Amputation","Loss of Eye",
        "Inspection","Final Narrative","Nature","NatureTitle","Part of Body",
        "Part of Body Title","Event","EventTitle","Source","SourceTitle","Secondary Source",
        "Secondary Source Title","FederalState"
    ]
    missing = [h for h in required if h not in headers]
    if missing:
        raise ValueError(f"2015-2025 CSV schema mismatch; missing columns: {missing}; actual headers: {headers}")
    return {h:h for h in headers}

def infer_severity(r: dict):
    meta = norm_text(r.get("legacy_meta",""))
    text = norm_text(" ".join([r.get("narrative_raw",""), r.get("title_raw","")]))
    if "fatality" in meta or re.search(r"\b(?:died|dies|dead|killed|fatality|deceased|pronounced dead)\b", text):
        return "fatality", "high" if "fatality" in meta else "medium"
    if boolish(r.get("amputation")) or re.search(r"\bamputat(?:ed|ion)\b", text):
        return "amputation", "high" if boolish(r.get("amputation")) else "medium"
    if boolish(r.get("loss_of_eye")) or re.search(r"\bloss of (?:an? )?eye\b|\benucleat", text):
        return "loss_of_eye", "high" if boolish(r.get("loss_of_eye")) else "medium"
    if boolish(r.get("hospitalized")) or "hospitalized injury" in meta or re.search(r"\bhospitali[sz](?:ed|ation)\b|\badmitted\b|\bkept overnight\b", text):
        return "hospitalization", "high" if (boolish(r.get("hospitalized")) or "hospitalized injury" in meta) else "medium"
    if r.get("narrative_raw","").strip():
        return "injury", "medium"
    return "unknown", "low"

def infer_hazard(text: str, gold: str | None):
    if gold:
        return gold, "tagged1000", "gold", gold
    t = norm_text(text)
    for label, needles in HAZARD_RULES:
        for needle in needles:
            if needle in t:
                conf = "high" if label in {"electrocution","fire_explosion","traffic","caught_in_between","collapse_of_object","falls"} else "medium"
                return label, "weak_rule", conf, needle
    return "other", "weak_rule", "low", "no canonical hazard cue matched"

def infer_sif(hazard: str, text: str):
    """
    Conservative, mechanism-based SIF potential inference.

    SIF is intentionally assessed independently from the hazard taxonomy.
    A hazard class alone does NOT automatically imply SIF=yes.

    Decision hierarchy:
      1. Explicit, credible high-energy mechanism -> yes
      2. Credible but incomplete high-energy signal -> possible
      3. No meaningful high-energy evidence -> no

    This is still a weak/rule-derived label and is not human adjudication.
    """
    t = norm_text(text)

    # Explicit high-energy mechanisms with reasonably specific wording.
    # These are stronger than merely seeing a broad hazard keyword.
    explicit_yes_rules = [
        (
            "fall_from_elevated_surface",
            [
                "fell from roof",
                "fell from a roof",
                "fell from ladder",
                "fell from scaffold",
                "fell from scaffolding",
                "fell from elevated",
                "fell from platform",
                "fell from mezzanine",
                "fell from a height",
                "fell from height",
                "fell through roof",
                "fell through skylight",
                "fell through opening",
                "fell down shaft",
            ],
        ),
        (
            "electrical_energy_exposure",
            [
                "electrocuted",
                "electrocution",
                "arc flash",
                "arc-flash",
                "contacted energized",
                "contacted an energized",
                "contact with energized",
                "struck an energized",
                "energized conductor",
                "energized electrical",
                "high voltage contact",
            ],
        ),
        (
            "caught_in_crush_entanglement",
            [
                "caught between",
                "caught in",
                "pinned between",
                "pinned against",
                "crushed between",
                "crushed by",
                "entangled in",
                "drawn into",
                "pulled into machinery",
                "caught in machinery",
                "caught in machine",
                "body was pulled into",
            ],
        ),
        (
            "structural_or_load_collapse",
            [
                "trench collapse",
                "trench caved in",
                "cave-in",
                "cave in",
                "wall collapsed",
                "roof collapsed",
                "structure collapsed",
                "building collapsed",
                "scaffold collapsed",
                "collapsed onto",
                "collapsed on",
                "load collapsed",
                "unstable load fell",
                "load overturned",
            ],
        ),
        (
            "vehicle_pedestrian_impact",
            [
                "vehicle struck",
                "truck struck",
                "forklift struck",
                "vehicle ran over",
                "truck ran over",
                "forklift ran over",
                "backed over",
                "run over by vehicle",
                "struck by a vehicle",
                "struck by a truck",
                "struck by forklift",
            ],
        ),
        (
            "high_energy_moving_object",
            [
                "struck by falling",
                "struck by a falling",
                "struck by moving",
                "struck by a moving",
                "object struck worker",
                "object hit worker",
                "projectile struck",
                "projectile hit",
                "flying object struck",
                "flying object hit",
            ],
        ),
        (
            "explosion_or_flash_fire",
            [
                "explosion occurred",
                "an explosion occurred",
                "exploded",
                "flash fire",
                "flash-fire",
                "fireball",
                "pressure vessel exploded",
                "tank exploded",
                "vessel exploded",
                "blast injured",
                "blast struck",
            ],
        ),
        (
            "high_pressure_release",
            [
                "pressurized line ruptured",
                "pressurized line broke",
                "pressure line ruptured",
                "pressure vessel ruptured",
                "pressure vessel failed",
                "hose ruptured",
                "hose burst",
                "line ruptured",
                "line burst",
                "high pressure release",
                "unexpected pressure release",
            ],
        ),
    ]

    explicit_possible_rules = [
        (
            "fall_hazard_signal",
            [
                "fall from",
                "fell from",
                "fell to lower level",
                "fell to lower",
                "fell",
                "ladder",
                "scaffold",
                "roof",
                "unprotected edge",
                "unprotected opening",
            ],
        ),
        (
            "electrical_hazard_signal",
            [
                "energized",
                "live wire",
                "live electrical",
                "electrical panel",
                "electrical source",
                "electrical contact",
            ],
        ),
        (
            "caught_in_crush_signal",
            [
                "caught in",
                "caught between",
                "pinned",
                "crushed",
                "pinch point",
                "entangled",
                "in between",
            ],
        ),
        (
            "collapse_load_signal",
            [
                "collapse",
                "cave in",
                "cave-in",
                "unstable load",
                "overturn",
                "tipped over",
                "tipped back",
            ],
        ),
        (
            "traffic_signal",
            [
                "vehicle struck",
                "struck by vehicle",
                "traffic",
                "backing truck",
                "truck",
                "forklift",
            ],
        ),
        (
            "moving_object_signal",
            [
                "struck by",
                "struck against",
                "flying object",
                "projectile",
                "moving object",
                "falling object",
                "falling material",
                "dropped load",
            ],
        ),
        (
            "pressure_signal",
            [
                "pressurized",
                "pressure",
                "compressed gas",
                "pressure release",
                "ruptured line",
                "burst hose",
            ],
        ),
        (
            "explosion_fire_signal",
            [
                "explosion",
                "exploded",
                "flash fire",
                "fireball",
                "blast",
            ],
        ),
    ]

    # Strong, specific evidence gets SIF=yes.
    yes_hits = []

    for rule_name, needles in explicit_yes_rules:
        hits = [
            needle
            for needle in needles
            if needle in t
        ]
        if hits:
            yes_hits.append((rule_name, hits))

    if yes_hits:
        evidence = "; ".join(
            f"{name}: {', '.join(hits[:2])}"
            for name, hits in yes_hits[:2]
        )
        return (
            "yes",
            "high",
            f"explicit high-energy mechanism: {evidence}",
        )

    # Chemical exposure by itself is not enough for SIF.
    # Require an additional severe-energy/exposure cue.
    chemical_signals = [
        x
        for x in [
            "toxic",
            "corrosive",
            "toxic vapor",
            "toxic gas",
            "toxic fume",
            "chemical release",
            "chemical spill",
            "chlorine release",
        ]
        if x in t
    ]

    chemical_high_consequence = [
        x
        for x in [
            "confined space",
            "unconscious",
            "collapsed",
            "respiratory failure",
            "severe burn",
            "third degree burn",
            "second degree burn",
            "overcome by fumes",
            "overcome by vapors",
        ]
        if x in t
    ]

    if chemical_signals and chemical_high_consequence:
        return (
            "possible",
            "medium",
            "chemical exposure with additional high-consequence/exposure signal",
        )

    # Thermal injuries alone are not automatically SIF.
    thermal_severe = [
        x
        for x in [
            "third degree burn",
            "second degree burn",
            "severe burn",
            "thermal burn",
        ]
        if x in t
    ]

    if thermal_severe and any(
        x in t
        for x in [
            "pressurized",
            "steam",
            "molten",
            "fireball",
            "explosion",
            "flash fire",
        ]
    ):
        return (
            "possible",
            "medium",
            "severe thermal injury with high-energy source",
        )

    # Other incomplete high-energy signals become possible.
    possible_hits = []

    for rule_name, needles in explicit_possible_rules:
        hits = [
            needle
            for needle in needles
            if needle in t
        ]
        if hits:
            possible_hits.append((rule_name, hits))

    # Avoid making generic "fire" or generic "vehicle" alone sufficient.
    # Generic hazard wording is treated as insufficient evidence.
    if possible_hits:
        # Require either one moderately strong signal or two distinct
        # mechanism signals before assigning "possible".
        distinct_rules = len(possible_hits)

        strong_possible_rule_names = {
            "fall_hazard_signal",
            "electrical_hazard_signal",
            "caught_in_crush_signal",
            "collapse_load_signal",
            "pressure_signal",
            "explosion_fire_signal",
        }

        has_strong_possible = any(
            name in strong_possible_rule_names
            for name, _ in possible_hits
        )

        if has_strong_possible or distinct_rules >= 2:
            evidence = "; ".join(
                f"{name}: {', '.join(hits[:2])}"
                for name, hits in possible_hits[:2]
            )
            return (
                "possible",
                "medium",
                f"incomplete high-energy mechanism signal: {evidence}",
            )

    return (
        "no",
        "low",
        "no sufficiently specific high-energy SIF mechanism identified",
    )


def fast_list_labels(text: str, rules):
    t = norm_text(text)
    labels, evidence = [], {}
    for label, needles in rules:
        hits = [x for x in needles if x in t]
        if hits:
            labels.append(label); evidence[label] = "; ".join(hits[:3])
    return labels, evidence

def upsert(store: dict, rec: dict):
    iid = rec["incident_id"]
    if iid not in store:
        store[iid] = rec
        return
    current = store[iid]
    for k,v in rec.items():
        if k in {"source_datasets","source_file_ids","quality_flags","transformation_log"}:
            continue
        if (not current.get(k)) and v not in ("",None,[]):
            current[k] = v
    for key in ("source_datasets","source_file_ids"):
        current[key] = sorted(set(current.get(key,[]) + rec.get(key,[])))
    for key in ("quality_flags","transformation_log"):
        current[key] = sorted(set(current.get(key,[]) + rec.get(key,[])))
    current["transformation_log"].append("duplicate_incident_id_merged")

def build_records(tagged_path: str, legacy_path: str, main_zip_path: str):
    tagged_rows = validate_and_read_tagged(tagged_path)
    legacy_rows = validate_and_read_legacy(legacy_path)
    tag_by_id = {}
    for r in tagged_rows:
        iid = clean_space(r.get("id",""))
        if not iid:
            continue
        raw = clean_space(r.get("tagged2",""))
        tag_by_id[iid] = {
            "tagged_label_raw": raw,
            "hazard_class_gold": TAG_CANON.get(raw.lower(), "other"),
            "tagged_cause": clean_space(r.get("cause","")),
            "tagged_keywords": clean_space(r.get("newkeys","")),
            "tagged_title_new": clean_space(r.get("titlenew","")),
            "tagged_summary_new": clean_space(r.get("summarynew","")),
        }

    store = {}
    for r in legacy_rows:
        iid = clean_space(r["id"])
        if not iid:
            continue
        td = tag_by_id.get(iid,{})
        narrative = clean_space(r["narrative"])
        upsert(store,{
            "incident_id":iid,"source_datasets":["osha_legacy"],"source_file_ids":[iid],
            "event_date":first_date(narrative),"employer":"","state":"","naics":"",
            "title_raw":clean_space(r["title"]),"narrative_raw":narrative,
            "narrative_sanitized":sanitize_text(narrative),
            "title_normalized":norm_text(r["title"]),"text_normalized":"",
            "keywords_raw":clean_space(r["keywords"]),
            "nature":"","nature_title":"","body":"","body_title":"","event":"","event_title":"",
            "source_title":"","secondary_source_title":"","federal_state":"",
            "hospitalized":"","amputation":"","loss_of_eye":"",
            "legacy_meta":clean_space(r["meta"]),
            "cause_raw":td.get("tagged_cause",clean_space(r["meta"])),
            "tagged_label_raw":td.get("tagged_label_raw",""),
            "hazard_class_gold":td.get("hazard_class_gold",""),
            "quality_flags":[],"transformation_log":["whitespace_normalized","pii_identifier_redaction_applied"]
        })

    for tid, td in tag_by_id.items():
        if tid in store:
            store[tid]["tagged_label_raw"] = td["tagged_label_raw"]
            store[tid]["hazard_class_gold"] = td["hazard_class_gold"]
            store[tid]["cause_raw"] = td["tagged_cause"]
            store[tid]["transformation_log"].append("tagged1000_label_mapped_to_canonical_taxonomy")

    with zipfile.ZipFile(main_zip_path) as archive:
        csv_name = next((n for n in archive.namelist() if n.lower().endswith(".csv")), None)
        if not csv_name:
            raise ValueError("No CSV found inside the 2015-2025 ZIP archive.")
        with archive.open(csv_name) as raw:
            txt = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
            reader = csv.DictReader(txt)
            validate_csv_headers(reader.fieldnames or [])
            for row in reader:
                iid = clean_space(row.get("ID",""))
                if not iid:
                    continue
                narrative = clean_space(row.get("Final Narrative",""))
                title = clean_space(row.get("EventTitle",""))
                upsert(store,{
                    "incident_id":iid,"source_datasets":["osha_2015_2025"],"source_file_ids":[iid],
                    "event_date":first_date(row.get("EventDate","")),"employer":clean_space(row.get("Employer","")),
                    "state":clean_space(row.get("State","")),"naics":clean_space(row.get("Primary NAICS","")),
                    "title_raw":title,"narrative_raw":narrative,
                    "narrative_sanitized":sanitize_text(narrative),
                    "title_normalized":norm_text(title),"text_normalized":"",
                    "keywords_raw":" | ".join([clean_space(row.get("NatureTitle","")),clean_space(row.get("Part of Body Title","")),clean_space(row.get("SourceTitle",""))]),
                    "nature":clean_space(row.get("Nature","")),"nature_title":clean_space(row.get("NatureTitle","")),
                    "body":clean_space(row.get("Part of Body","")),"body_title":clean_space(row.get("Part of Body Title","")),
                    "event":clean_space(row.get("Event","")),"event_title":title,
                    "source_title":clean_space(row.get("SourceTitle","")),
                    "secondary_source_title":clean_space(row.get("Secondary Source Title","")),
                    "federal_state":clean_space(row.get("FederalState","")),
                    "hospitalized":clean_space(row.get("Hospitalized","")),
                    "amputation":clean_space(row.get("Amputation","")),
                    "loss_of_eye":clean_space(row.get("Loss of Eye","")),
                    "legacy_meta":"","cause_raw":clean_space(row.get("EventTitle","")),
                    "tagged_label_raw":"","hazard_class_gold":"",
                    "quality_flags":[],"transformation_log":["whitespace_normalized","pii_identifier_redaction_applied"]
                })
    return store

def classify_records(store: dict):
    records = list(store.values())
    for r in records:
        r["actual_severity"], r["actual_severity_confidence"] = infer_severity(r)

        # Build all ML-facing text from sanitized fields.
        # Raw narrative/title remain available separately for audit/provenance.
        sanitized_title = sanitize_text(r.get("title_raw", ""))
        sanitized_narrative = r.get("narrative_sanitized", "")

        sanitized_keywords = sanitize_text(r.get("keywords_raw", ""))
        sanitized_event_title = sanitize_text(r.get("event_title", ""))
        sanitized_cause = sanitize_text(r.get("cause_raw", ""))

        text = " ".join([
            sanitized_title,
            sanitized_narrative,
            sanitized_keywords,
            sanitized_event_title,
            sanitized_cause
        ])

        hazard, source, conf, evidence = infer_hazard(
            text, r.get("hazard_class_gold") or None
        )
        r["hazard_class"] = hazard
        r["hazard_label_source"] = source
        r["hazard_label_confidence"] = conf
        r["hazard_evidence"] = evidence
        r["sif_potential"], r["sif_confidence"], r["sif_rationale"] = infer_sif(
            hazard, text
        )
        r["activity_tags"], r["activity_evidence"] = fast_list_labels(
            text, ACTIVITY_RULES
        )
        r["barrier_labels"], r["barrier_evidence"] = fast_list_labels(
            text, BARRIER_RULES
        )
        r["barrier_gap_signals"], r["barrier_gap_evidence"] = fast_list_labels(
            text, BARRIER_GAP_RULES
        )

        # ML-facing normalized fields must be derived from sanitized content.
        r["title_normalized"] = norm_text(sanitized_title)
        r["text_normalized"] = norm_text(text)

        q = set(r.get("quality_flags", []))
        if not r["narrative_raw"] and not r["title_raw"]:
            q.add("empty_text")
        if len(r["narrative_sanitized"]) < 20:
            q.add("very_short_narrative")
        r["quality_flags"] = sorted(q)

        r["transformation_log"] = sorted(set(
            r.get("transformation_log", []) + [
                "canonical_hazard_inferred",
                "severity_classified_separately_from_sif",
                "activity_and_barrier_weak_labels_added",
                "ml_text_derived_from_sanitized_fields"
            ]
        ))
    return records

def dedupe(records):
    # Exact duplicate content
    exact = defaultdict(list)
    for i,r in enumerate(records):
        key = hashlib.sha1((r.get("event_date","")+"|"+r.get("title_normalized","")+"|"+norm_text(r.get("narrative_raw",""))).encode()).hexdigest()
        exact[key].append(i)
    drop=set()
    exact_groups=0
    for inds in exact.values():
        if len(inds)>1:
            exact_groups += 1
            best=max(inds,key=lambda i:(len(records[i].get("narrative_raw","")),len(records[i].get("source_datasets",[]))))
            drop.update(i for i in inds if i!=best)

    # Conservative near-duplicate check
    buckets=defaultdict(list)
    for i,r in enumerate(records):
        buckets[(r.get("event_date",""),r.get("title_normalized","")[:80])].append(i)
    near_pairs=0
    near_drop=set()
    for inds in buckets.values():
        if len(inds)<2 or len(inds)>50:
            continue
        kept=[]
        for i in inds:
            if i in drop: continue
            a=norm_text(records[i].get("narrative_raw",""))
            if len(a)<80:
                kept.append(i); continue
            found=None
            for j in kept:
                b=norm_text(records[j].get("narrative_raw",""))
                if len(b)<80: continue
                ratio = fuzz.ratio(a,b) if fuzz else 100*SequenceMatcher(None,a,b).ratio()
                if ratio>=98:
                    found=j; break
            if found is None:
                kept.append(i)
            else:
                near_pairs += 1
                if len(a)>len(norm_text(records[found].get("narrative_raw",""))):
                    near_drop.add(found)
                    kept[-1]=i
                else:
                    near_drop.add(i)
    drop |= near_drop
    clean = [r for i,r in enumerate(records) if i not in drop]
    return clean, len([v for v in exact.values() if len(v)>1]), len(drop), near_pairs

def split_gold(clean):
    gold=[r for r in clean if r.get("hazard_label_source")=="tagged1000"]
    by=defaultdict(list)
    for r in gold: by[r["hazard_class"]].append(r)
    train=[]; val=[]; test=[]
    for cls,items in by.items():
        items=sorted(items,key=lambda r:hashlib.sha256((r["incident_id"]+"|"+cls).encode()).hexdigest())
        n=len(items); ntr=max(1,int(math.floor(n*0.8))); nval=max(1,int(math.floor(n*0.1)))
        if ntr+nval>=n: nval=max(0,n-ntr-1)
        train+=items[:ntr]; val+=items[ntr:ntr+nval]; test+=items[ntr+nval:]
    return train,val,test

def jsonl_payload(r):
    """
    Strict LLM-facing payload.

    Only sanitized/public fields are allowed here.
    Raw OSHA fields such as title_raw, narrative_raw,
    employer, addresses, coordinates, etc. must never
    enter the LLM dataset.

    Supervision provenance is made explicit:
      - hazard labels may be reference-labeled (tagged1000) or weak-rule
      - SIF labels are currently rule-derived for ALL records
    """

    is_gold = (
        r.get("hazard_label_source") == "tagged1000"
    )

    supervision_tier = (
        "gold" if is_gold else "weak"
    )

    # SIF potential is currently inferred by rules for every record.
    # The tagged1000 dataset provides reference hazard labels, but it does
    # not contain independently adjudicated SIF labels.
    sif_label_source = "rule_derived"

    payload = {
        "id": r["incident_id"],

        "input": {
            "title": r["title_normalized"],
            "narrative": r["narrative_sanitized"],
            "event": sanitize_text(
                r.get("event_title", "")
            ),
            "nature": sanitize_text(
                r.get("nature_title", "")
            ),
            "source": sanitize_text(
                r.get("source_title", "")
            ),
            "activity_tags": r["activity_tags"],
        },

        "labels": {
            "hazard_class": r["hazard_class"],
            "hazard_label_source": r[
                "hazard_label_source"
            ],
            "hazard_confidence": r[
                "hazard_label_confidence"
            ],

            "actual_severity": r[
                "actual_severity"
            ],
            "actual_severity_confidence": r[
                "actual_severity_confidence"
            ],

            "sif_potential": r[
                "sif_potential"
            ],
            "sif_confidence": r[
                "sif_confidence"
            ],
            "sif_label_source": sif_label_source,

            "supervision_tier": supervision_tier,

            "activity_tags": r["activity_tags"],
            "barrier_labels": r["barrier_labels"],
            "barrier_gap_signals": r[
                "barrier_gap_signals"
            ],
        },

        "evidence": {
            "hazard": r[
                "hazard_evidence"
            ],
            "sif": r[
                "sif_rationale"
            ],
            "barrier_gap": r[
                "barrier_gap_evidence"
            ],
        },

        "provenance": {
            "source_datasets": r[
                "source_datasets"
            ],
            "tagged_label_raw": r[
                "tagged_label_raw"
            ],
            "pipeline_version": r[
                "pipeline_version"
            ],
        },
    }

    # Safety check:
    # Inspect JSON FIELD NAMES only.
    # Do not scan text values, because legitimate narrative text
    # can naturally contain words such as "city", "source", etc.
    forbidden_field_names = {
        "title_raw",
        "narrative_raw",
        "employer",
        "address",
        "address1",
        "address2",
        "city",
        "zip",
        "latitude",
        "longitude",
        "phone",
        "email",
        "ssn",
        "employee_id",
        "serial_number",
        "model_number",
        "equipment_id",
        "asset_id",
    }

    def collect_keys(value):
        keys = []

        if isinstance(value, dict):
            for key, child in value.items():
                keys.append(str(key).lower())
                keys.extend(collect_keys(child))

        elif isinstance(value, list):
            for child in value:
                keys.extend(collect_keys(child))

        return keys

    payload_keys = set(
        collect_keys(payload)
    )

    forbidden_present = sorted(
        payload_keys.intersection(
            forbidden_field_names
        )
    )

    if forbidden_present:
        raise ValueError(
            "LLM safety check failed. "
            "Forbidden raw field names detected: "
            f"{forbidden_present}"
        )

    # Supervision invariants:
    # supervision_tier describes hazard-label provenance.
    if is_gold:
        if (
            r.get("hazard_label_source") != "tagged1000"
            or supervision_tier != "gold"
        ):
            raise ValueError(
                "Supervision provenance inconsistency: "
                "gold hazard record is not tagged1000."
            )
    else:
        if (
            r.get("hazard_label_source") == "tagged1000"
            or supervision_tier != "weak"
        ):
            raise ValueError(
                "Supervision provenance inconsistency: "
                "weak hazard record is incorrectly marked as gold."
            )

    # SIF labels are rule-derived for every record.
    if sif_label_source != "rule_derived":
        raise ValueError(
            "SIF provenance inconsistency: "
            "sif_label_source must be rule_derived."
        )

    return payload


def write_jsonl(path, records):
    with open(path,"w",encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(jsonl_payload(r),ensure_ascii=False,separators=(",",":"))+"\n")

def write_report(path, clean, dropped, near_pairs):
    n=len(clean); rows=[]
    def add(metric,category,count,notes=""):
        rows.append([metric,category,count,round(100*count/n,3) if n else 0,notes])
    add("dataset","clean_incidents",n)
    add("dataset","dropped_duplicate_rows",dropped)
    add("dataset","near_duplicate_pairs",near_pairs,"Conservative same-date/same-title bucket with similarity >=98%")
    for cat,cnt in Counter(r["hazard_class"] for r in clean).most_common():
        add("hazard_class",cat,cnt)
    for cat,cnt in Counter(r["actual_severity"] for r in clean).most_common():
        add("actual_severity",cat,cnt,"Outcome severity; not a SIF proxy")
    for cat,cnt in Counter(r["sif_potential"] for r in clean).most_common():
        add("sif_potential",cat,cnt,"Mechanism-based weak potential label; separate from outcome severity")
    for cat,cnt in Counter(d for r in clean for d in r["barrier_gap_signals"]).most_common():
        add("barrier_gap_signal",cat,cnt,"Rule-triggered signal, not causal adjudication")
    for cat,cnt in Counter(s for r in clean for s in r["source_datasets"]).most_common():
        add("source",cat,cnt)
    for flag,cnt in Counter(flag for r in clean for flag in r["quality_flags"]).most_common():
        add("quality_flag",flag,cnt)
    with open(path,"w",newline="",encoding="utf-8") as fh:
        w=csv.writer(fh); w.writerow(["metric","category","count","percent_of_clean","notes"]); w.writerows(rows)

def write_parquet(path, clean):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("Parquet output requires pyarrow. Install with: pip install pyarrow") from exc
    rows=[]
    for r in clean:
        rows.append({k:r.get(k,[]) if k in {"source_datasets","source_file_ids","activity_tags","barrier_labels","barrier_gap_signals","quality_flags","transformation_log"} else r.get(k,"") for k in OUTPUT_FIELDS})
    table=pa.Table.from_pylist(rows)
    pq.write_table(table,path,compression="zstd")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tagged",required=True)
    ap.add_argument("--legacy",required=True)
    ap.add_argument("--main-zip",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    os.makedirs(args.out,exist_ok=True)

    store=build_records(args.tagged,args.legacy,args.main_zip)
    raw_count=len(store)
    records=classify_records(store)
    clean,dedup_groups,dropped,near_pairs=dedupe(records)

    timestamp=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")
    for r in clean:
        r["pipeline_version"]="1.0.2"
        r["processed_at_utc"]=timestamp
        r["transformation_log"]=sorted(set(r.get("transformation_log",[])+["exact_and_conservative_near_duplicate_check_applied"]))

    gold_train, validation, test=split_gold(clean)
    weak=[r for r in clean if r.get("hazard_label_source")!="tagged1000"]
    training=sorted(gold_train+weak,key=lambda r:hashlib.sha256(r["incident_id"].encode()).hexdigest())
    validation=sorted(validation,key=lambda r:r["incident_id"])
    test=sorted(test,key=lambda r:r["incident_id"])

    write_parquet(os.path.join(args.out,"clean_incidents.parquet"),clean)
    write_jsonl(os.path.join(args.out,"sif_training.jsonl"),training)
    write_jsonl(os.path.join(args.out,"sif_validation.jsonl"),validation)
    write_jsonl(os.path.join(args.out,"sif_test.jsonl"),test)
    write_jsonl(os.path.join(args.out,"sif_eval.jsonl"),validation+test)
    write_report(os.path.join(args.out,"label_report.csv"),clean,dropped,near_pairs)

    print(json.dumps({
        "raw_unique_incident_ids":raw_count,
        "clean_incidents":len(clean),
        "dedupe_groups":dedup_groups,
        "dropped_rows":dropped,
        "near_duplicate_pairs":near_pairs,
        "gold_train":len(gold_train),
        "validation":len(validation),
        "test":len(test),
        "outputs":sorted(os.listdir(args.out))
    },indent=2))

if __name__=="__main__":
    main()
