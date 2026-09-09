import type {
  IncidentNormalizedRecord,
  ModelInferenceResult,
  AssetRiskSummary
} from '../../domain/types';

export interface PaginatedResult<T> {
  data: T[];
  total: number;
  page: number;
  pageSize: number;
}

export interface IncidentFilter {
  assetId?: string;
  assetType?: string;
  routing?: string;
  deterministicOverride?: boolean;
  dateRange?: { start: string; end: string };
}

export interface IIncidentRepository {
  getIncidents(filter: IncidentFilter, page: number, pageSize: number): Promise<PaginatedResult<IncidentNormalizedRecord>>;
  getIncidentById(logId: string): Promise<IncidentNormalizedRecord | null>;
  getInferenceResult(logId: string): Promise<ModelInferenceResult | null>;
}

export interface IAssetRiskRepository {
  getAssetRiskSummary(assetId: string): Promise<AssetRiskSummary | null>;
  getAllAssetRisks(): Promise<AssetRiskSummary[]>;
}
