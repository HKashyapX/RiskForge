// ─── Overview KPIs ────────────────────────────────────────────────────────────

export interface OverviewKPIs {
  periodLabel: string;
  dataFreshnessTs: string;
  totalReports: number;
  sifPotentialCount: number;
  sifPrecursorDensity: number;  // ratio: sifPotential / totalReports, expressed as %
  humanReviewBacklog: number;
  criticalEscalations: number;
}

// ─── SPD Trend ────────────────────────────────────────────────────────────────

export interface SPDTrendPoint {
  week: string;          // e.g. "W32"
  spd: number;           // SIF Precursor Density %
  totalReports: number;
  sifPotential: number;
}

// ─── Asset Risk ───────────────────────────────────────────────────────────────

export interface AssetRiskRow {
  assetId: string;
  assetLabel: string;
  assetType: string;
  sifPrecursorCount: number;
  spdScore: number;          // 0-100%
  trend: 'up' | 'down' | 'stable';
}

// ─── Barrier Failures ─────────────────────────────────────────────────────────

export interface BarrierFailureRow {
  canonicalForm: string;
  label: string;
  iogpRule: string;
  count: number;
  trend: 'up' | 'down' | 'stable';
}

// ─── Recent Critical Escalation ───────────────────────────────────────────────

export interface RecentEscalationRow {
  logId: string;
  timestamp: string;
  assetId: string;
  assetType: string;
  narrativeSnippet: string;
  calibratedScore: number;
  deterministicOverride: boolean;
  primaryIogpRule: string;
}

// ─── Routing Distribution ────────────────────────────────────────────────────

export interface RoutingDistributionRow {
  bucket: 'critical_escalation' | 'hitl_review' | 'auto_dismiss';
  label: string;
  count: number;
  color: string;
}

// ─── IOGP Rule Distribution ──────────────────────────────────────────────────

export interface IOGPRuleRow {
  rule: string;
  label: string;
  count: number;
}

// ─── Combined Overview Response ──────────────────────────────────────────────

export interface PatternRow {
  id: string;
  components: string[];
  count: number;
  location: string;
  sifRelevance: 'High' | 'Medium' | 'Low';
}

export interface OverviewData {
  kpis: OverviewKPIs;
  spdTrend: SPDTrendPoint[];
  topAssets: AssetRiskRow[];
  topBarriers: BarrierFailureRow[];
  recentEscalations: RecentEscalationRow[];
  routingDistribution: RoutingDistributionRow[];
  iogpRuleDistribution: IOGPRuleRow[];
  patterns: PatternRow[];
}
