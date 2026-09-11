import type {
  AssetType,
  EntitySpan,
  IncidentNormalizedRecord,
  LifeSavingRule,
  ModelInferenceResult,
  OperationalTriad,
  RoutingBucket,
} from '../../../domain/types';

const ROUTINGS: RoutingBucket[] = ['auto_dismiss', 'hitl_review', 'critical_escalation'];

function isRouting(value: unknown): value is RoutingBucket {
  return typeof value === 'string' && (ROUTINGS as string[]).includes(value);
}

function str(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function num(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function asSpan(value: unknown): EntitySpan | null {
  if (typeof value !== 'object' || value === null) return null;
  const record = value as Record<string, unknown>;
  return {
    text: str(record.text),
    canonical_form: str(record.canonical_form),
    start_char: num(record.start_char),
    end_char: num(record.end_char),
    entity_type: str(record.entity_type),
  };
}

function asTriad(value: unknown): OperationalTriad {
  const record = (typeof value === 'object' && value !== null ? value : {}) as Record<string, unknown>;
  const activity = asSpan(record.activity);
  const assetLocation = asSpan(record.asset_location);
  const failedBarrier = asSpan(record.failed_barrier);
  return {
    activity,
    asset_location: assetLocation,
    failed_barrier: failedBarrier,
  };
}

export interface IncidentViewPayload {
  incident: Record<string, unknown>;
  result: Record<string, unknown>;
}

export function mapIncidentView(view: IncidentViewPayload): {
  incident: IncidentNormalizedRecord;
  inference: ModelInferenceResult;
} {
  const inc = view.incident;
  const res = view.result;
  const routing = res.routing;
  const rules = Array.isArray(res.matched_iogp_rules) ? res.matched_iogp_rules : [];
  const incident: IncidentNormalizedRecord = {
    log_id: str(inc.log_id),
    timestamp: str(inc.timestamp),
    asset_id: str(inc.asset_id),
    asset_type: str(inc.asset_type, 'pipeline_network') as AssetType,
    raw_narrative: str(inc.raw_narrative),
    spans: Array.isArray(inc.spans)
      ? inc.spans.map(asSpan).filter((s): s is EntitySpan => s !== null)
      : [],
  };
  const inference: ModelInferenceResult = {
    log_id: str(res.log_id, str(inc.log_id)),
    raw_sif_p_score: num(res.raw_sif_p_score),
    calibrated_sif_p_score: num(res.calibrated_sif_p_score),
    deterministic_override: res.deterministic_override === true,
    routing: isRouting(routing) ? routing : 'auto_dismiss',
    matched_iogp_rules: rules.filter((r): r is LifeSavingRule => typeof r === 'string'),
    triad: asTriad(res.triad),
    latency_ms: num(res.latency_ms),
  };
  return { incident, inference };
}

export function mapInferenceResult(value: unknown): ModelInferenceResult {
  const res = (typeof value === 'object' && value !== null ? value : {}) as Record<string, unknown>;
  const routing = res.routing;
  const rules = Array.isArray(res.matched_iogp_rules) ? res.matched_iogp_rules : [];
  return {
    log_id: str(res.log_id),
    raw_sif_p_score: num(res.raw_sif_p_score),
    calibrated_sif_p_score: num(res.calibrated_sif_p_score),
    deterministic_override: res.deterministic_override === true,
    routing: isRouting(routing) ? routing : 'auto_dismiss',
    matched_iogp_rules: rules.filter((r): r is LifeSavingRule => typeof r === 'string'),
    triad: asTriad(res.triad),
    latency_ms: num(res.latency_ms),
  };
}
