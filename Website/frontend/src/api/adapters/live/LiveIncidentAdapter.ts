import type {
  IIncidentRepository,
  IncidentFilter,
  PaginatedResult,
} from '../../repositories/interfaces';
import type {
  IncidentNormalizedRecord,
  ModelInferenceResult,
} from '../../../domain/types';
import { apiGet } from './apiClient';
import { mapIncidentView, type IncidentViewPayload } from './mappers';

interface PagePayload {
  items: IncidentViewPayload[];
  total: number;
  offset: number;
  limit: number;
}

function routingBucket(routing: IncidentFilter['routing']): string | undefined {
  if (routing === 'auto_dismiss' || routing === 'hitl_review' || routing === 'critical_escalation') {
    return routing;
  }
  return undefined;
}

export class LiveIncidentAdapter implements IIncidentRepository {
  async getIncidents(
    filter: IncidentFilter,
    page: number,
    pageSize: number,
  ): Promise<PaginatedResult<IncidentNormalizedRecord>> {
    const offset = Math.max(0, (page - 1) * pageSize);
    const payload = await apiGet<{ page: PagePayload }>('/v1/incidents', {
      offset: String(offset),
      limit: String(pageSize),
      asset_id: filter.assetId,
      asset_type: filter.assetType,
      routing: routingBucket(filter.routing),
      timestamp_from: filter.dateRange?.start,
      timestamp_to: filter.dateRange?.end,
    });
    const mapped = payload.page.items.map(mapIncidentView);
    let data = mapped.map((entry) => entry.incident);
    if (filter.deterministicOverride !== undefined) {
      // Backend has no override filter; apply it client-side over the page.
      // (The dev adapter had the same semantics.)
      const overrides = await Promise.all(
        payload.page.items.map(async (item) => {
          const view = mapped.find((entry) => entry.incident.log_id === item.incident.log_id);
          return view ? view.inference.deterministic_override : false;
        }),
      );
      data = data.filter((_, index) => overrides[index] === filter.deterministicOverride);
    }
    return {
      data,
      total: payload.page.total,
      page,
      pageSize,
    };
  }

  async getIncidentById(logId: string): Promise<IncidentNormalizedRecord | null> {
    try {
      const payload = await apiGet<{ incident: IncidentViewPayload }>(
        `/v1/incidents/${encodeURIComponent(logId)}`,
      );
      return mapIncidentView(payload.incident).incident;
    } catch (error) {
      if (error instanceof Error && 'status' in error && (error as { status: number }).status === 404) {
        return null;
      }
      throw error;
    }
  }

  async getInferenceResult(logId: string): Promise<ModelInferenceResult | null> {
    try {
      const payload = await apiGet<{ incident: IncidentViewPayload }>(
        `/v1/incidents/${encodeURIComponent(logId)}`,
      );
      return mapIncidentView(payload.incident).inference;
    } catch (error) {
      if (error instanceof Error && 'status' in error && (error as { status: number }).status === 404) {
        return null;
      }
      throw error;
    }
  }
}

export const liveIncidentRepository = new LiveIncidentAdapter();
