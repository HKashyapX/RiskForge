import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { fetchOverviewData } from '../../api/adapters/index';
import { ArrowLeft, Database, AlertTriangle } from 'lucide-react';
import { formatPercentage, formatNumber, formatLabel } from '../../utils/formatters';

const AssetDetailPage: React.FC = () => {
  const { assetId } = useParams<{ assetId: string }>();
  const navigate = useNavigate();

  const { data, isLoading } = useQuery({
    queryKey: ['overview', '30d'],
    queryFn: () => fetchOverviewData('30d'),
  });

  if (isLoading) return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '64px', color: 'var(--color-text-muted)' }}>
      <div style={{ width: 32, height: 32, border: '3px solid var(--color-border)', borderTopColor: 'var(--color-brand)', borderRadius: '50%', animation: 'spin 0.8s linear infinite', marginBottom: 16 }} />
      <p style={{ margin: 0 }}>Loading asset details…</p>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );

  const asset = data?.topAssets.find(a => a.assetId === assetId);

  if (!asset) return (
    <div style={{ padding: '20px 24px', backgroundColor: 'var(--color-critical-bg)', border: '1px solid var(--color-critical-border)', borderRadius: '8px', color: 'var(--color-critical)' }}>
      <strong>Record Not Found</strong> — The requested asset could not be located in the development dataset.
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', fontFamily: 'var(--font-family-sans)' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', borderBottom: '1px solid var(--color-border)', paddingBottom: '20px' }}>
        <button
          onClick={() => navigate('/assets')}
          aria-label="Back to Assets"
          style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: 36, height: 36, border: '1px solid var(--color-border)', borderRadius: '6px', backgroundColor: 'var(--color-surface)', cursor: 'pointer', flexShrink: 0 }}
        >
          <ArrowLeft size={16} color="var(--color-text-secondary)" />
        </button>
        <div>
          <h2 style={{ margin: 0, fontSize: '24px', fontWeight: 700, color: 'var(--color-text-primary)' }}>{formatLabel(asset.assetLabel)}</h2>
          <p style={{ margin: '4px 0 0', fontSize: '14px', color: 'var(--color-text-muted)', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Database size={14} /> {formatLabel(asset.assetType)}
          </p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
        <div style={{ padding: '20px', backgroundColor: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: '8px' }}>
          <div style={{ fontSize: '13px', color: 'var(--color-text-muted)', marginBottom: '8px', fontWeight: 600, textTransform: 'uppercase' }}>SIF Precursors</div>
          <div style={{ fontSize: '28px', fontWeight: 700, color: 'var(--color-warning)' }}>{formatNumber(asset.sifPrecursorCount)}</div>
        </div>
        <div style={{ padding: '20px', backgroundColor: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: '8px' }}>
          <div style={{ fontSize: '13px', color: 'var(--color-text-muted)', marginBottom: '8px', fontWeight: 600, textTransform: 'uppercase' }}>SPD Score</div>
          <div style={{ fontSize: '28px', fontWeight: 700, color: 'var(--color-text-primary)' }}>{formatPercentage(asset.spdScore, 1)}</div>
        </div>
      </div>

      <div style={{ padding: '32px', textAlign: 'center', backgroundColor: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: '8px', color: 'var(--color-text-muted)' }}>
        <AlertTriangle size={32} style={{ opacity: 0.5, marginBottom: '12px' }} />
        <p style={{ margin: 0, fontSize: '14px' }}>Deeper analytics and incident history for this specific asset are not available in the current development dataset snapshot.</p>
      </div>

    </div>
  );
};

export default AssetDetailPage;
