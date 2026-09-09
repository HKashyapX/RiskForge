import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { fetchOverviewData } from '../../api/adapters/development/DevOverviewAdapter';
import type { RecentEscalationRow } from '../../domain/overviewTypes';
import { AlertTriangle, ChevronRight, CheckCircle } from 'lucide-react';

const EscalationsPage: React.FC = () => {
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
      <p style={{ margin: 0, fontWeight: 500 }}>Loading escalations…</p>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );

  if (isError || !data) return (
    <div style={{ padding: '20px 24px', backgroundColor: 'var(--color-critical-bg)', border: '1px solid var(--color-critical-border)', borderRadius: '8px', color: 'var(--color-critical)' }}>
      <strong>System Error</strong> — Failed to retrieve escalations.
    </div>
  );

  const escalations = data.recentEscalations ?? [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ borderBottom: '1px solid var(--color-border)', paddingBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '22px', fontWeight: 700, color: 'var(--color-text-primary)' }}>Escalations</h2>
          <p style={{ margin: '4px 0 0', fontSize: '14px', color: 'var(--color-text-muted)' }}>Review critical escalations requiring immediate human attention.</p>
        </div>
        <span style={{ fontSize: '13px', color: 'var(--color-text-muted)', fontWeight: 500 }}>{escalations.length} records</span>
      </div>

      <div style={{ ...card }}>
        <div style={{ backgroundColor: 'var(--color-surface-hover)', borderBottom: '1px solid var(--color-border)', padding: '16px 20px', fontWeight: 600, color: 'var(--color-text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <AlertTriangle size={16} color="var(--color-critical)" />
          Critical Escalations Queue
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
              <th style={{ padding: '10px 20px', textAlign: 'left', fontSize: '11px', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>Log ID</th>
              <th style={{ padding: '10px 20px', textAlign: 'left', fontSize: '11px', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>Asset</th>
              <th style={{ padding: '10px 20px', textAlign: 'left', fontSize: '11px', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>Primary IOGP Rule</th>
              <th style={{ padding: '10px 20px', textAlign: 'left', fontSize: '11px', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>Action Required</th>
              <th style={{ padding: '10px 20px' }}></th>
            </tr>
          </thead>
          <tbody>
            {escalations.length === 0 ? (
               <tr>
               <td colSpan={5} style={{ padding: '48px', textAlign: 'center', color: 'var(--color-text-muted)', fontSize: '14px' }}>
                 No escalation data is available in the current development dataset.
               </td>
             </tr>
            ) : escalations.map((esc: RecentEscalationRow) => (
              <tr 
                key={esc.logId} 
                style={{ borderBottom: '1px solid var(--color-border)', cursor: 'pointer' }}
                onClick={() => navigate(`/incidents/${esc.logId}`)}
                onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'var(--color-surface-hover)')}
                onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'transparent')}
              >
                <td style={{ padding: '14px 20px', fontFamily: 'var(--font-family-mono)', fontSize: '13px', color: 'var(--color-brand)', fontWeight: 600 }}>{esc.logId}</td>
                <td style={{ padding: '14px 20px', color: 'var(--color-text-primary)' }}>{esc.assetId}</td>
                <td style={{ padding: '14px 20px', color: 'var(--color-text-secondary)' }}>{esc.primaryIogpRule.replace(/_/g, ' ').toUpperCase()}</td>
                <td style={{ padding: '14px 20px', color: 'var(--color-critical)', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '6px' }}>
                   {esc.deterministicOverride ? <AlertTriangle size={14} /> : <CheckCircle size={14} />}
                   {esc.deterministicOverride ? 'Safety Override' : 'Model Flagged'}
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

export default EscalationsPage;
