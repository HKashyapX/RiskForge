// Data-source selector: pick the demo (baked JSONL) or live (FastAPI) adapters
// at module load based on VITE_API_MODE. Defaults to live when an API base URL
// is configured, otherwise demo — so `npm run dev` with no env keeps working.
import type { IIncidentRepository } from '../repositories/interfaces';
import type { OverviewData } from '../../domain/overviewTypes';
import { incidentRepository as devIncidentRepository, fetchDemoData } from './development/DevIncidentAdapter';
import { fetchOverviewData as fetchDevOverviewData } from './development/DevOverviewAdapter';
import { liveIncidentRepository } from './live/LiveIncidentAdapter';
import { fetchLiveOverviewData } from './live/LiveOverviewAdapter';

const mode: 'demo' | 'live' =
  (import.meta.env?.VITE_API_MODE as 'demo' | 'live' | undefined) ??
  (import.meta.env?.VITE_API_BASE_URL ? 'live' : 'demo');

export interface OverviewFetcher {
  (timePeriod: string): Promise<OverviewData>;
}

export const incidentRepository: IIncidentRepository =
  mode === 'live' ? liveIncidentRepository : devIncidentRepository;

export const fetchOverviewData: OverviewFetcher =
  mode === 'live' ? fetchLiveOverviewData : fetchDevOverviewData;

export { fetchDemoData };
export const dataSourceMode = mode;
