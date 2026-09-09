import type { OverviewData, SPDTrendPoint, RecentEscalationRow } from '../../../domain/overviewTypes';
import { fetchDemoData } from './DevIncidentAdapter';

export async function fetchOverviewData(_timePeriod: string): Promise<OverviewData> {
  const allData = await fetchDemoData();
  const totalLogs = allData.length;
  const sifPrecursors = allData.filter(d => d.inference.routing === 'critical_escalation' || d.inference.routing === 'hitl_review').length;
  const spdScore = totalLogs > 0 ? (sifPrecursors / totalLogs) * 100 : 0;
  const criticalEscalations = allData.filter(d => d.inference.routing === 'critical_escalation').length;
  const humanReviewCount = allData.filter(d => d.inference.routing === 'hitl_review').length;

  const kpis = {
    periodLabel: 'Available Dataset',
    dataFreshnessTs: new Date().toISOString(),
    totalReports: totalLogs,
    sifPotentialCount: sifPrecursors,
    sifPrecursorDensity: spdScore,
    humanReviewBacklog: humanReviewCount,
    criticalEscalations: criticalEscalations,
  };

  // Group by routing
  let autoDismiss = 0, hitl = 0, critical = 0;
  allData.forEach(d => {
    if (d.inference.routing === 'critical_escalation') critical++;
    else if (d.inference.routing === 'hitl_review') hitl++;
    else autoDismiss++;
  });
  const routingDistribution = [
    { bucket: 'auto_dismiss' as const, label: 'Auto Dismissed', count: autoDismiss, color: '#10b981' },
    { bucket: 'hitl_review' as const, label: 'Human Review', count: hitl, color: '#f59e0b' },
    { bucket: 'critical_escalation' as const, label: 'Critical Escalation', count: critical, color: '#ef4444' },
  ];

  // Group by matched rules
  const ruleCounts: Record<string, number> = {};
  allData.forEach(d => {
    d.inference.matched_iogp_rules.forEach(rule => {
      ruleCounts[rule] = (ruleCounts[rule] || 0) + 1;
    });
  });
  const iogpRuleDistribution = Object.entries(ruleCounts).map(([rule, count]) => ({
    rule,
    label: rule.replace(/_/g, ' '),
    count
  })).sort((a, b) => b.count - a.count);

  // Generate spdTrend from timestamps
  const now = Date.now();
  const weekBuckets = new Array(12).fill(0).map((_, i) => ({
    weekOffset: 11 - i,
    label: `W-${11 - i}`,
    total: 0,
    sif: 0
  }));

  allData.forEach(d => {
    if (d.incident.timestamp && d.incident.timestamp !== 'Not available') {
      const dTime = new Date(d.incident.timestamp).getTime();
      const diffDays = (now - dTime) / (1000 * 60 * 60 * 24);
      const weekIndex = 11 - Math.floor(diffDays / 7);
      if (weekIndex >= 0 && weekIndex < 12) {
        weekBuckets[weekIndex].total += 1;
        if (d.inference.routing === 'critical_escalation' || d.inference.routing === 'hitl_review') {
          weekBuckets[weekIndex].sif += 1;
        }
      }
    }
  });

  const spdTrend: SPDTrendPoint[] = weekBuckets.map(b => ({
    week: b.label,
    totalReports: b.total,
    sifPotential: b.sif,
    spd: b.total > 0 ? (b.sif / b.total) * 100 : 0
  }));

  // Generate recentEscalations
  const recentEscalations: RecentEscalationRow[] = allData
    .filter(d => d.inference.routing === 'critical_escalation' && d.incident.timestamp && d.incident.timestamp !== 'Not available')
    .sort((a, b) => new Date(b.incident.timestamp).getTime() - new Date(a.incident.timestamp).getTime())
    .slice(0, 5)
    .map(d => ({
      logId: d.incident.log_id,
      timestamp: d.incident.timestamp,
      assetId: d.incident.asset_id,
      assetType: d.incident.asset_type,
      narrativeSnippet: d.incident.raw_narrative.substring(0, 100) + '...',
      calibratedScore: d.inference.calibrated_sif_p_score || 0,
      deterministicOverride: d.inference.deterministic_override,
      primaryIogpRule: d.inference.matched_iogp_rules[0] || 'Unknown'
    }));

  return {
    kpis,
    spdTrend,
    topAssets: [], // Not available in current dataset
    topBarriers: [], // Not available in current dataset
    recentEscalations,
    routingDistribution,
    iogpRuleDistribution,
    patterns: [], // Not available
  };
}
