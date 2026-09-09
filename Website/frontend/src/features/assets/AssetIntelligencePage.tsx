import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { fetchOverviewData } from '../../api/adapters/development/DevOverviewAdapter';
import type { AssetRiskRow } from '../../domain/overviewTypes';
import { Database, ChevronRight, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { formatPercentage, formatNumber, formatLabel } from '../../utils/formatters';

const AssetIntelligencePage: React.FC = () => {
  const navigate = useNavigate();

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
      <p style={{ margin: 0, fontWeight: 500 }}>Loading asset risk data…</p>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );

  if (isError || !data) return (
    <div style={{ padding: '20px 24px', backgroundColor: 'var(--color-critical-bg)', border: '1px solid var(--color-critical-border)', borderRadius: '8px', color: 'var(--color-critical)' }}>
      <strong>System Error</strong> — Failed to retrieve asset intelligence.
    </div>
  );

  const assets = data.topAssets ?? [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ borderBottom: '1px solid var(--color-border)', paddingBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '22px', fontWeight: 700, color: 'var(--color-text-primary)' }}>Assets</h2>
          <p style={{ margin: '4px 0 0', fontSize: '14px', color: 'var(--color-text-muted)' }}>Identify which locations and equipment have concentrated risk.</p>
        </div>
      </div>

      <div style={{ ...card }}>
        <div style={{ backgroundColor: 'var(--color-surface-hover)', borderBottom: '1px solid var(--color-border)', padding: '16px 20px', fontWeight: 600, color: 'var(--color-text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Database size={16} color="var(--color-brand)" />
          Highest-Risk Assets
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
              <th style={{ padding: '10px 20px', textAlign: 'left', fontSize: '11px', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>Asset ID</th>
              <th style={{ padding: '10px 20px', textAlign: 'left', fontSize: '11px', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>Type</th>
              <th style={{ padding: '10px 20px', textAlign: 'right', fontSize: '11px', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>SIF Precursors</th>
              <th style={{ padding: '10px 20px', textAlign: 'right', fontSize: '11px', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>SPD Score</th>
              <th style={{ padding: '10px 20px', textAlign: 'center', fontSize: '11px', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>Trend</th>
              <th style={{ padding: '10px 20px' }}></th>
            </tr>
          </thead>
          <tbody>
            {assets.length === 0 ? (
               <tr>
               <td colSpan={6} style={{ padding: '48px', textAlign: 'center', color: 'var(--color-text-muted)', fontSize: '14px' }}>
                 Not available in current dataset.
               </td>
             </tr>
            ) : assets.map((a: AssetRiskRow) => (
              <tr 
                key={a.assetId} 
                style={{ borderBottom: '1px solid var(--color-border)', cursor: 'pointer' }}
                onClick={() => navigate(`/assets/${a.assetId}`)}
                onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'var(--color-surface-hover)')}
                onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'transparent')}
              >
                <td style={{ padding: '14px 20px', color: 'var(--color-text-primary)', fontWeight: 500 }}>{formatLabel(a.assetLabel)}</td>
                <td style={{ padding: '14px 20px', color: 'var(--color-text-secondary)' }}>{formatLabel(a.assetType)}</td>
                <td style={{ padding: '14px 20px', textAlign: 'right', color: 'var(--color-warning)', fontWeight: 700 }}>{formatNumber(a.sifPrecursorCount)}</td>
                <td style={{ padding: '14px 20px', textAlign: 'right', color: 'var(--color-text-primary)', fontWeight: 600 }}>{formatPercentage(a.spdScore, 1)}</td>
                <td style={{ padding: '14px 20px', textAlign: 'center' }}>
                  {a.trend === 'up' && <TrendingUp size={16} color="var(--color-critical)" style={{ margin: '0 auto' }} />}
                  {a.trend === 'down' && <TrendingDown size={16} color="var(--color-safe)" style={{ margin: '0 auto' }} />}
                  {a.trend === 'stable' && <Minus size={16} color="var(--color-text-muted)" style={{ margin: '0 auto' }} />}
                </td>
                <td style={{ padding: '14px 20px', textAlign: 'right' }}>
                  <ChevronRight size={16} color="var(--color-text-muted)" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default AssetIntelligencePage;
