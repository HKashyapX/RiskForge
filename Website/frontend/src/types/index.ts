export interface SafetyFacts {
  activity: string;
  energySource: string;
  workerExposure: string;
  criticalBarrier: string;
  barrierState: string;
  potentialConsequence: string;
}

export interface SIFAssessment {
  potential: boolean;
  reasoning: string;
  evidence: string;
  mappedRules: string[];
}

export interface SafetyReport {
  id: string;
  type: 'Unsafe Act' | 'Unsafe Condition' | 'Near Miss' | 'Incident';
  date: string;
  site: string;
  department: string;
  contractor: string;
  text: string;
  facts: SafetyFacts;
  sif: SIFAssessment;
}

export interface Pattern {
  id: string;
  components: string[];
  count: number;
  location: string;
  sifRelevance: 'High' | 'Medium' | 'Low';
}

export interface SiteMetric {
  name: string;
  value: number; // Density percentage
  reports: number;
  sifReports: number;
}

export interface TrendMetric {
  period: string; // e.g. Jan
  density: number;
}

export interface DistributionMetric {
  name: string;
  value: number;
}

export interface DashboardMetrics {
  totalReports: string;
  sifPotentialReports: string;
  sifPrecursorDensity: string;
  highPrecursorLocations: string;
  densityBySite: SiteMetric[];
  monthlyTrend: TrendMetric[];
  typeDistribution: DistributionMetric[];
  patterns: Pattern[];
}

export interface AnalyticsData {
  densityByDepartment: { name: string; value: number }[];
  densityByActivity: { name: string; value: number }[];
  densityByContractor: { name: string; value: number }[];
}

export interface DemoData {
  dashboard: DashboardMetrics;
  analytics: AnalyticsData;
  reports: SafetyReport[];
  sifDemoReport: SafetyReport;
}
