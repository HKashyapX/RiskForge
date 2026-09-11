import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchOverviewData } from '../../api/adapters/index';
import type { BarrierFailureRow } from '../../domain/overviewTypes';
import { ShieldAlert, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { formatNumber, formatLabel } from '../../utils/formatters';

const SafetyBarriersPage: React.FC = () => {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['overview', '30d'],
    queryFn: () => fetchOverviewData('30d'),
  });

  const card: React.CSSProperties = {
    backgroundColor: 'var(--color-surface)',
    border: '1px solid var(--color-border)',
    borderRadius: '8px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
    overflow: 'hidden',
  };

  if (isLoading) return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '64px', color: 'var(--color-text-muted)' }}>
      <div style={{ width: 32, height: 32, border: '3px solid var(--color-border)', borderTopColor: 'var(--color-brand)', borderRadius: '50%', animation: 'spin 0.8s linear infinite', marginBottom: 16 }} />
      <p style={{ margin: 0, fontWeight: 500 }}>Loading safety barriers…</p>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );

  if (isError || !data) return (
    <div style={{ padding: '20px 24px', backgroundColor: 'var(--color-critical-bg)', border: '1px solid var(--color-critical-border)', borderRadius: '8px', color: 'var(--color-critical)' }}>
      <strong>System Error</strong> — Failed to retrieve safety barriers.
    </div>
  );

  const barriers = data.topBarriers ?? [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ borderBottom: '1px solid var(--color-border)', paddingBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '22px', fontWeight: 700, color: 'var(--color-text-primary)' }}>Safety Barriers</h2>
          <p style={{ margin: '4px 0 0', fontSize: '14px', color: 'var(--color-text-muted)' }}>Analyze recurring failures in physical and administrative controls.</p>
        </div>
      </div>

      <div style={{ ...card }}>
        <div style={{ backgroundColor: 'var(--color-surface-hover)', borderBottom: '1px solid var(--color-border)', padding: '16px 20px', fontWeight: 600, color: 'var(--color-text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <ShieldAlert size={16} color="var(--color-warning)" />
          Recurrent Failed Barriers
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
              <th style={{ padding: '10px 20px', textAlign: 'left', fontSize: '11px', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>Barrier Type</th>
              <th style={{ padding: '10px 20px', textAlign: 'left', fontSize: '11px', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>IOGP Rule</th>
              <th style={{ padding: '10px 20px', textAlign: 'right', fontSize: '11px', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>Failure Count</th>
              <th style={{ padding: '10px 20px', textAlign: 'center', fontSize: '11px', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>Trend</th>
            </tr>
          </thead>
          <tbody>
            {barriers.length === 0 ? (
               <tr>
               <td colSpan={4} style={{ padding: '48px', textAlign: 'center', color: 'var(--color-text-muted)', fontSize: '14px' }}>
                 Not available in current dataset.
               </td>
             </tr>
            ) : barriers.map((b: BarrierFailureRow, i: number) => (
              <tr 
                key={i} 
                style={{ borderBottom: '1px solid var(--color-border)' }}
              >
                <td style={{ padding: '14px 20px', color: 'var(--color-text-primary)', fontWeight: 500 }}>{formatLabel(b.label)}</td>
                <td style={{ padding: '14px 20px', color: 'var(--color-text-secondary)' }}>{formatLabel(b.iogpRule)}</td>
                <td style={{ padding: '14px 20px', textAlign: 'right', color: 'var(--color-warning)', fontWeight: 700 }}>{formatNumber(b.count)}</td>
                <td style={{ padding: '14px 20px', textAlign: 'center' }}>
                  {b.trend === 'up' && <TrendingUp size={16} color="var(--color-critical)" style={{ margin: '0 auto' }} />}
                  {b.trend === 'down' && <TrendingDown size={16} color="var(--color-safe)" style={{ margin: '0 auto' }} />}
                  {b.trend === 'stable' && <Minus size={16} color="var(--color-text-muted)" style={{ margin: '0 auto' }} />}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default SafetyBarriersPage;
