import json
import random
import os

input_file = r"C:\Users\Lenovo\OneDrive\Desktop\Github Projects\RiskForge\data\processed\sif_eval.jsonl"
output_file = r"C:\Users\Lenovo\OneDrive\Desktop\Github Projects\RiskForge\frontend\public\demo-data.json"

data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# Sample 100 random incidents to keep it lightweight but dynamic
random.seed()
sampled = random.sample(data, min(100, len(data)))

demo_data = []

def map_routing(sif_potential, hazard):
    if sif_potential == "yes":
        return "critical_escalation"
    if hazard in ["fire_explosion", "electrocution", "fall_from_height"]:
        return "hitl_review"
    return "standard_log"

def map_score(sif_potential):
    if sif_potential == "yes":
        return random.uniform(0.7, 0.99)
    return random.uniform(0.01, 0.3)

for item in sampled:
    log_id = f"LOG-{item.get('id', item.get('incident_id', random.randint(1000000, 9999999)))}"
    
    narrative = ""
    if "input" in item and "narrative" in item["input"]:
        narrative = item["input"]["narrative"]
    elif "narrative_normalized" in item:
        narrative = item["narrative_normalized"]
    elif "narrative_raw" in item:
        narrative = item["narrative_raw"]

    labels = item.get("labels", {})
    if not labels and "hazard_class" in item:
        labels = item

    hazard = labels.get("hazard_class", "unknown")
    sif_potential = labels.get("sif_potential", "no")

    sif_score = map_score(sif_potential)
    
    import datetime
    
    # Generate random timestamp in the last 90 days
    days_ago = random.randint(0, 90)
    random_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_ago, hours=random.randint(0,23), minutes=random.randint(0,59))
    timestamp = random_time.isoformat()

    demo_item = {
        "incident": {
            "log_id": log_id,
            "timestamp": timestamp,
            "asset_id": "Not available",
            "asset_type": "Not available",
            "raw_narrative": narrative,
            "spans": []
        },
        "inference": {
            "log_id": log_id,
            "raw_sif_p_score": sif_score,
            "calibrated_sif_p_score": sif_score,
            "deterministic_override": False,
            "routing": map_routing(sif_potential, hazard),
            "matched_iogp_rules": [hazard] if hazard and hazard != "unknown" else [],
            "triad": {
                "activity": None,
                "asset_location": None,
                "failed_barrier": None
            },
            "latency_ms": random.randint(120, 450)
        }
    }
    demo_data.append(demo_item)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(demo_data, f, indent=2)

print(f"Successfully generated {len(demo_data)} records to {output_file}")
