/**
 * Canonical domain constants to prevent duplicating magic strings
 * throughout the frontend application.
 */

export const ASSET_TYPES = [
  'drilling_rig',
  'workover_rig',
  'gas_gathering_station',
  'oil_collecting_station',
  'pipeline_network'
] as const;

export const ROUTING_BUCKETS = [
  'auto_dismiss',
  'hitl_review',
  'critical_escalation'
] as const;

export const IOGP_RULES = [
  'bypassing_safety_controls',
  'confined_space',
  'driving',
  'energy_isolation',
  'hot_work',
  'line_of_fire',
  'safe_mechanical_lifting',
  'toxic_gas',
  'work_at_height'
] as const;

export const IOGP_LABELS: Record<string, string> = {
  bypassing_safety_controls: 'Bypassing Safety Controls',
  confined_space: 'Confined Space',
  driving: 'Driving',
  energy_isolation: 'Energy Isolation',
  hot_work: 'Hot Work',
  line_of_fire: 'Line of Fire',
  safe_mechanical_lifting: 'Safe Mechanical Lifting',
  toxic_gas: 'Toxic Gas',
  work_at_height: 'Work at Height'
};
