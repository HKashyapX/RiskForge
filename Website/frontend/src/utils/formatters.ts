/**
 * Safe formatting utilities to ensure the UI never crashes or displays
 * 'Invalid Date', 'NaN', or 'undefined' when rendering data.
 */

export function formatNumber(value: number | null | undefined, fractionDigits: number = 0): string {
  if (value === null || value === undefined || Number.isNaN(value) || !Number.isFinite(value)) {
    return 'Not available';
  }
  return value.toFixed(fractionDigits);
}

export function formatPercentage(value: number | null | undefined, fractionDigits: number = 1): string {
  if (value === null || value === undefined || Number.isNaN(value) || !Number.isFinite(value)) {
    return 'Not available';
  }
  return `${value.toFixed(fractionDigits)}%`;
}

export function formatDate(timestamp: string | null | undefined, format: 'date' | 'time' | 'datetime' = 'datetime'): string {
  if (!timestamp) return 'Date unavailable';
  
  const d = new Date(timestamp);
  if (Number.isNaN(d.getTime())) {
    return 'Date unavailable';
  }

  if (format === 'date') {
    return d.toLocaleDateString('en-US', { day: 'numeric', month: 'short', year: 'numeric' });
  }
  if (format === 'time') {
    return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  }
  
  // datetime default
  return d.toLocaleString('en-US', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export function formatLabel(snakeCaseString: string | null | undefined): string {
  if (!snakeCaseString) return 'Not available';
  return snakeCaseString
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}
