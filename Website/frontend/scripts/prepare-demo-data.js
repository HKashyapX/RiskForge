import fs from 'fs';
import path from 'path';
import readline from 'readline';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const inputPath = path.resolve(__dirname, '../../data/processed/sif_validation.jsonl');
const outputPath = path.resolve(__dirname, '../public/demo-data.json');

async function processData() {
  const fileStream = fs.createReadStream(inputPath);
  const rl = readline.createInterface({
    input: fileStream,
    crlfDelay: Infinity
  });

  const incidents = [];

  for await (const line of rl) {
    if (!line.trim()) continue;
    const record = JSON.parse(line);
    
    // Map to IncidentNormalizedRecord format as best as possible
    const incident = {
      log_id: `LOG-${record.id}`,
      timestamp: "Not available", // Do not invent timestamps
      asset_id: "Not available",
      asset_type: "Not available",
      raw_narrative: record.input.narrative || "",
      spans: [] // Not provided in simple JSONL
    };
    
    // Map to ModelInferenceResult
    const sifPotential = record.labels?.sif_potential === "yes";
    const inference = {
      log_id: incident.log_id,
      raw_sif_p_score: null, // Do not invent model scores
      calibrated_sif_p_score: null,
      deterministic_override: false,
      routing: sifPotential ? 'critical_escalation' : 'auto_dismiss', // We map the human label to the routing bucket
      matched_iogp_rules: record.labels?.hazard_class ? [record.labels.hazard_class] : [],
      triad: {
        activity: null,
        asset_location: null,
        failed_barrier: null
      },
      latency_ms: null // Do not invent latency
    };

    incidents.push({ incident, inference });
  }

  fs.writeFileSync(outputPath, JSON.stringify(incidents, null, 2));
  console.log(`Successfully generated demo-data.json with ${incidents.length} records.`);
}

processData().catch(console.error);
