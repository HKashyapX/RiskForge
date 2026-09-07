import json
import random
from pathlib import Path

DRILLING_ACTIVITIES = [
    "tripping pipe",
    "making connection",
    "casing lowering",
    "pressure testing",
    "swabbing operation",
]
LOCATIONS = [
    "drill floor",
    "monkey board",
    "cellar pit",
    "GGS manifold",
    "OCS pump house",
    "mud tank",
]
CRITICAL_ENERGIES = [
    ("well kick with pressure surge to 3200 psi", "high_pressure"),
    ("heavy drill collar lifted by winch line", "suspended_load"),
    ("H2S concentration rising above 15 ppm", "toxic_gas"),
    ("derrickman working on elevated beam at 12 meters", "fall_hazard"),
    ("high-voltage generator breaker maintenance", "electrical"),
]
FAILED_BARRIERS = [
    ("BOP blind rams failed to close properly", "blowout_preventer"),
    ("winch cable wire rope parted suddenly", "winch_line"),
    ("harness safety lanyard was unhooked", "safety_harness"),
    ("pressure relief valve stuck in closed position", "pressure_relief_valve"),
    ("LOTO not applied on energized circuit", "energy_isolation_loto"),
]
INTACT_BARRIERS = [
    ("BOP pressure tested to 5000 psi with zero drop", "blowout_preventer"),
    ("safety harness anchored securely to inertia reel", "safety_harness"),
    ("PRV popped at set limit and relieved safely", "pressure_relief_valve"),
    ("LOTO verified and zero voltage confirmed", "energy_isolation_loto"),
]
LOW_ENERGY_EVENTS = [
    "Operator dropped wrench into empty mud pit, no injuries.",
    "Housekeeping required near tool rack due to rain water collection.",
    "Worker observed not wearing safety glasses in yard area.",
    "Mess room sink drain clogged, cleared by maintenance team.",
    "Minor thumb abrasion while carrying grease pail, first aid applied.",
]

def generate_record(log_id: int):
    if random.random() < 0.20:
        activity = random.choice(DRILLING_ACTIVITIES)
        location = random.choice(LOCATIONS)
        energy_desc, energy_type = random.choice(CRITICAL_ENERGIES)
        barrier_desc, _ = random.choice(FAILED_BARRIERS)
        narrative = f"During {activity} at {location}, observed {energy_desc}. Critical barrier failure: {barrier_desc}. Crew on scene intervened."
        sif_label = 1
        rule_map = {"high_pressure": 0, "electrical": 3, "suspended_load": 6, "toxic_gas": 7, "fall_hazard": 8}
        rules = [rule_map[energy_type]]
    else:
        sif_label = 0
        rules = []
        if random.random() < 0.5:
            activity = random.choice(DRILLING_ACTIVITIES)
            location = random.choice(LOCATIONS)
            energy_desc, _ = random.choice(CRITICAL_ENERGIES)
            intact_desc, _ = random.choice(INTACT_BARRIERS)
            narrative = f"Routine {activity} conducted at {location}. {energy_desc} present, but {intact_desc}. Safe operations maintained."
        else:
            narrative = random.choice(LOW_ENERGY_EVENTS)
    return {
        "log_id": f"OIL_SYNTH_{log_id:05d}",
        "raw_narrative": narrative,
        "sif_label": sif_label,
        "iogp_labels": rules,
    }

def main(num_samples: int = 5000):
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [generate_record(i) for i in range(num_samples)]
    train_size = int(num_samples * 0.8)

    with open(output_dir / "synthetic_train.jsonl", "w", encoding="utf-8") as f:
        for r in records[:train_size]:
            f.write(json.dumps(r) + "\n")

    with open(output_dir / "synthetic_val.jsonl", "w", encoding="utf-8") as f:
        for r in records[train_size:]:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()
