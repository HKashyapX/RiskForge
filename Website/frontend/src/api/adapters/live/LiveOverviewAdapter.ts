import type { OverviewData, SPDTrendPoint, RecentEscalationRow, PatternRow } from '../../../domain/overviewTypes';
import { apiGet } from './apiClient';
import { mapIncidentView, type IncidentViewPayload } from './mappers';
import type { LifeSavingRule, RoutingBucket } from '../../../domain/types';

interface AnalyticsAsset {
  asset_id: string;
  asset_type: string;
  total_reports: number;
  sif_precursors: number;
  spd: number;
}

interface AnalyticsSummary {
  total_reports: number;
  sif_precursors: number;
  sif_precursor_density: number;
  routing_counts: Record<string, number>;
  matched_rule_counts: Record<string, number>;
  assets: AnalyticsAsset[];
  weekly_trend: { week_start: string; reports: number; sif_precursors: number; density: number }[];
  emerging_risks: { week_start: string; sif_precursors: number; prior_mean: number; threshold: number; note: string }[];
  patterns: { rule_combination: string[]; count: number; assets: string[]; example_narratives: string[] }[];
  failed_barriers: { barrier: string; failures: number }[];
}

interface IncidentViewResponse {
  page: { items: IncidentViewPayload[]; total: number; offset: number; limit: number };
}

function ruleLabel(rule: string): string {
  return rule.replace(/_/g, ' ');
}

function trendDirection(series: number[]): 'up' | 'down' | 'stable' {
  if (series.length < 2) return 'stable';
  const half = Math.floor(series.length / 2);
  const early = series.slice(0, half).reduce((a, b) => a + b, 0) / Math.max(1, half);
  const late = series.slice(-half).reduce((a, b) => a + b, 0) / Math.max(1, half);
  if (late > early * 1.15) return 'up';
  if (late < early * 0.85) return 'down';
  return 'stable';
}

export async function fetchLiveOverviewData(_timePeriod: string): Promise<OverviewData> {
  const summary = await apiGet<{ summary: AnalyticsSummary }>('/v1/analytics/summary');
  const s = summary.summary;

  const kpis = {
    periodLabel: 'Live Backend Data',
    dataFreshnessTs: new Date().toISOString(),
    totalReports: s.total_reports ?? 0,
    sifPotentialCount: s.sif_precursors ?? 0,
    sifPrecursorDensity: (s.sif_precursor_density ?? 0) * 100,
    humanReviewBacklog: s.routing_counts?.hitl_review ?? 0,
    criticalEscalations: s.routing_counts?.critical_escalation ?? 0,
  };

  const routingDistribution = (['auto_dismiss', 'hitl_review', 'critical_escalation'] as const).map(
    (bucket: RoutingBucket) => ({
      bucket,
      label:
        bucket === 'auto_dismiss'
          ? 'Auto Dismissed'
          : bucket === 'hitl_review'
            ? 'Human Review'
            : 'Critical Escalation',
      count: s.routing_counts?.[bucket] ?? 0,
      color:
        bucket === 'auto_dismiss' ? '#10b981' : bucket === 'hitl_review' ? '#f59e0b' : '#ef4444',
    }),
  );

  const iogpRuleDistribution = Object.entries(s.matched_rule_counts ?? {})
    .map(([rule, count]) => ({ rule, label: ruleLabel(rule), count }))
    .sort((a, b) => b.count - a.count);

  const spdTrend: SPDTrendPoint[] = (s.weekly_trend ?? []).map((point) => ({
    week: point.week_start,
    totalReports: point.reports,
    sifPotential: point.sif_precursors,
    spd: point.density * 100,
  }));

  const topAssets = (s.assets ?? []).slice(0, 8).map((asset) => ({
    assetId: asset.asset_id,
    assetLabel: asset.asset_id,
    assetType: asset.asset_type,
    sifPrecursorCount: asset.sif_precursors,
    spdScore: asset.spd * 100,
    trend: trendDirection(
      (s.weekly_trend ?? []).map((p) => p.sif_precursors),
    ) as 'up' | 'down' | 'stable',
  }));

  const topBarriers = (s.failed_barriers ?? []).slice(0, 8).map((barrier) => ({
    canonicalForm: barrier.barrier,
    label: ruleLabel(barrier.barrier),
    iogpRule: barrier.barrier,
    count: barrier.failures,
    trend: 'stable' as const,
  }));

  const patterns: PatternRow[] = (s.patterns ?? []).slice(0, 10).map((pattern, index) => ({
    id: `pattern-${index}`,
    components: pattern.rule_combination,
    count: pattern.count,
    location: pattern.assets.length > 0 ? pattern.assets.join(', ') : 'Multiple assets',
    sifRelevance: pattern.count >= 3 ? 'High' : pattern.count >= 2 ? 'Medium' : 'Low',
  }));

  let recentEscalations: RecentEscalationRow[] = [];
  try {
    const page = await apiGet<IncidentViewResponse>('/v1/incidents/critical', {
      offset: '0',
      limit: '5',
    });
    recentEscalations = page.page.items
      .map((item) => {
        const mapped = mapIncidentView(item);
        const score = mapped.inference.calibrated_sif_p_score;
        const rules = mapped.inference.matched_iogp_rules as LifeSavingRule[];
        return {
          logId: mapped.incident.log_id,
          timestamp: mapped.incident.timestamp,
          assetId: mapped.incident.asset_id,
          assetType: mapped.incident.asset_type as string,
          narrativeSnippet: `${mapped.incident.raw_narrative.substring(0, 100)}...`,
          calibratedScore: score,
          deterministicOverride: mapped.inference.deterministic_override,
          primaryIogpRule: rules[0] ?? 'Unknown',
        };
      })
      .filter((row) => row.timestamp && row.timestamp !== 'Not available');
  } catch {
    // Critical queue unavailable (e.g. no data yet); surface an empty table.
    recentEscalations = [];
  }

  return {
    kpis,
    spdTrend,
    topAssets,
    topBarriers,
    recentEscalations,
    routingDistribution,
    iogpRuleDistribution,
    patterns,
  };
}
