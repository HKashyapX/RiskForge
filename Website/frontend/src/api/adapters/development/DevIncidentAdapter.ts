import type { IIncidentRepository, IncidentFilter, PaginatedResult } from '../../repositories/interfaces';
import type { IncidentNormalizedRecord, ModelInferenceResult } from '../../../domain/types';

let cachedData: { incident: IncidentNormalizedRecord, inference: ModelInferenceResult }[] | null = null;

export async function fetchDemoData(): Promise<{ incident: IncidentNormalizedRecord, inference: ModelInferenceResult }[]> {
  if (cachedData) return cachedData;
  try {
    const res = await fetch('/demo-data.json');
    if (!res.ok) throw new Error('Failed to load demo data');
    cachedData = await res.json() as { incident: IncidentNormalizedRecord, inference: ModelInferenceResult }[];
    return cachedData || [];
  } catch (err) {
    console.error(err);
    return [];
  }
}

export class DevIncidentAdapter implements IIncidentRepository {
  async getIncidents(_filter: IncidentFilter, page: number, pageSize: number): Promise<PaginatedResult<IncidentNormalizedRecord>> {
    const allData = await fetchDemoData();
    const incidents = allData?.map(d => d.incident) || [];
    
    // Simple pagination
    const start = (page - 1) * pageSize;
    const paginated = incidents.slice(start, start + pageSize);

    return {
      data: paginated,
      total: incidents.length,
      page,
      pageSize
    };
  }

  async getIncidentById(logId: string): Promise<IncidentNormalizedRecord | null> {
    const allData = await fetchDemoData();
    const record = allData?.find(d => d.incident.log_id === logId);
    return record ? record.incident : null;
  }

  async getInferenceResult(logId: string): Promise<ModelInferenceResult | null> {
    const allData = await fetchDemoData();
    const record = allData?.find(d => d.inference.log_id === logId);
    return record ? record.inference : null;
  }
}

export const incidentRepository = new DevIncidentAdapter();
