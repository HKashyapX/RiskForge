export type RoutingBucket = 'auto_dismiss' | 'hitl_review' | 'critical_escalation';

export type AssetType = 
  | 'drilling_rig' 
  | 'workover_rig' 
  | 'gas_gathering_station' 
  | 'oil_collecting_station' 
  | 'pipeline_network';

export type LifeSavingRule = 
  | 'bypassing_safety_controls' 
  | 'confined_space' 
  | 'driving' 
  | 'energy_isolation' 
  | 'hot_work' 
  | 'line_of_fire' 
  | 'safe_mechanical_lifting' 
  | 'toxic_gas' 
  | 'work_at_height';

export interface EntitySpan {
  text: string;
  canonical_form: string;
  start_char: number;
  end_char: number;
  entity_type: string;
}

export interface IncidentNormalizedRecord {
  log_id: string;
  timestamp: string;
  asset_id: string;
  asset_type: AssetType;
  raw_narrative: string;
  spans: EntitySpan[];
}

export interface OperationalTriad {
  activity: EntitySpan | null;
  asset_location: EntitySpan | null;
  failed_barrier: EntitySpan | null;
}

export interface ModelInferenceResult {
  log_id: string;
  raw_sif_p_score: number;
  calibrated_sif_p_score: number;
  deterministic_override: boolean;
  routing: RoutingBucket;
  matched_iogp_rules: LifeSavingRule[];
  triad: OperationalTriad;
  latency_ms: number;
}

export interface AssetRiskSummary {
  asset_id: string;
  asset_type: AssetType;
  total_logs: number;
  sif_precursor_count: number;
  spd_score: number;
  recurrent_failed_barriers: string[];
}
