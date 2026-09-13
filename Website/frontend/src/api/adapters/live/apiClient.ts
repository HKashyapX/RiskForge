const BASE_URL: string =
  (import.meta.env?.VITE_API_BASE_URL as string | undefined) ?? 'http://127.0.0.1:8000';

/**
 * Bearer token for authenticated deployments (pilot/production).  Set via
 * VITE_AUTH_TOKEN for browser-held tokens, or resolved by the host app and
 * registered through `setAuthToken`.  Requests without a token against an
 * authenticated API surface a 401 to the user — never silently degrade.
 */
let authToken: string | undefined = import.meta.env?.VITE_AUTH_TOKEN as string | undefined;

export function setAuthToken(token: string | undefined): void {
  authToken = token;
}

let correlationCounter = 0;

export function nextCorrelationId(): string {
  correlationCounter += 1;
  const unique = `${Date.now().toString(36)}-${correlationCounter.toString(36)}`;
  // FastAPI header pattern: ^[A-Za-z0-9._:-]+$
  return `web-${unique}`;
}

export class ApiRequestError extends Error {
  readonly status: number;
  readonly code: string;
  readonly correlationId: string;

  constructor(status: number, code: string, message: string, correlationId: string) {
    super(message);
    this.name = 'ApiRequestError';
    this.status = status;
    this.code = code;
    this.correlationId = correlationId;
  }
}

interface ErrorBody {
  error?: { code?: string; message?: string };
}

export async function apiGet<T>(path: string, params?: Record<string, string | undefined>): Promise<T> {
  const correlationId = nextCorrelationId();
  const url = new URL(path, BASE_URL);
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== '') url.searchParams.set(key, value);
    }
  }
  const headers: Record<string, string> = {
    'X-Correlation-ID': correlationId,
    Accept: 'application/json',
  };
  if (authToken) headers.Authorization = `Bearer ${authToken}`;
  const response = await fetch(url.toString(), { headers });
  if (!response.ok) {
    let code = `http_${response.status}`;
    let message = `Request failed with status ${response.status}`;
    try {
      const body = (await response.json()) as ErrorBody;
      code = body.error?.code ?? code;
      message = body.error?.message ?? message;
    } catch {
      // Non-JSON error body; keep defaults.
    }
    throw new ApiRequestError(response.status, code, message, correlationId);
  }
  return (await response.json()) as T;
}
