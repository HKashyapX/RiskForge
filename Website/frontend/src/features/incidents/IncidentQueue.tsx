import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { incidentRepository } from '../../api/adapters/development/DevIncidentAdapter';
import { Search, ChevronRight, Filter } from 'lucide-react';
import { formatDate, formatLabel } from '../../utils/formatters';

const IncidentQueue: React.FC = () => {
  const navigate = useNavigate();

  const [search, setSearch] = useState('');
  const [routing, setRouting] = useState('');

  const { data, isLoading, isError } = useQuery({
    queryKey: ['incidents', search, routing],
    queryFn: () => incidentRepository.getIncidents({ routing: routing || undefined }, 1, 50),
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
      <p style={{ margin: 0, fontWeight: 500 }}>Loading incident queue…</p>
    </div>
  );

  if (isError) return (
    <div style={{ padding: '20px 24px', backgroundColor: 'var(--color-critical-bg)', border: '1px solid var(--color-critical-border)', borderRadius: '8px', color: 'var(--color-critical)' }}>
      <strong>System Error</strong> — Failed to retrieve the incident queue. Please try again later.
    </div>
  );

  const incidents = data?.data ?? [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>

      {/* Header */}
      <div style={{ borderBottom: '1px solid var(--color-border)', paddingBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '22px', fontWeight: 700, color: 'var(--color-text-primary)' }}>Incident Queue</h2>
          <p style={{ margin: '4px 0 0', fontSize: '14px', color: 'var(--color-text-muted)' }}>Review SIF precursors and structured safety intelligence.</p>
        </div>
        <span style={{ fontSize: '13px', color: 'var(--color-text-muted)', fontWeight: 500 }}>{data?.total ?? 0} records</span>
      </div>

      {/* Toolbar */}
      <div style={{ ...card, borderRadius: '8px 8px 0 0', padding: '12px 16px', display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
        <div style={{ position: 'relative', width: 260 }}>
          <Search size={15} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--color-text-muted)' }} />
          <input
            type="text"
            placeholder="Search log ID or narrative…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{
              width: '100%', paddingLeft: 32, paddingRight: 12, paddingTop: 8, paddingBottom: 8,
              fontSize: '13px', border: '1px solid var(--color-border)', borderRadius: '6px',
              outline: 'none', color: 'var(--color-text-primary)', backgroundColor: 'var(--color-background)',
              fontFamily: 'inherit',
            }}
          />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Filter size={15} color="var(--color-text-muted)" />
          <select
            value={routing}
            onChange={(e) => setRouting(e.target.value)}
            style={{
              padding: '7px 12px', fontSize: '13px', border: '1px solid var(--color-border)',
              borderRadius: '6px', outline: 'none', color: 'var(--color-text-primary)',
              backgroundColor: 'var(--color-background)', fontFamily: 'inherit', cursor: 'pointer'
            }}
          >
            <option value="">All Routing Priorities</option>
            <option value="critical_escalation">Critical Escalation</option>
            <option value="hitl_review">Human Review</option>
            <option value="auto_dismiss">Auto Dismissed</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div style={{ ...card, borderRadius: '0 0 8px 8px', marginTop: '-24px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
          <thead>
            <tr style={{ backgroundColor: 'var(--color-surface-hover)', borderBottom: '1px solid var(--color-border)' }}>
              {['Log ID', 'Timestamp', 'Asset Type', 'Asset ID', ''].map(h => (
                <th key={h} style={{ padding: '10px 16px', textAlign: 'left', fontSize: '11px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--color-text-muted)' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {incidents.length === 0 ? (
              <tr>
                <td colSpan={5} style={{ padding: '48px', textAlign: 'center', color: 'var(--color-text-muted)', fontSize: '14px' }}>
                  No incidents found.
                </td>
              </tr>
            ) : incidents.map(inc => (
              <tr
                key={inc.log_id}
                onClick={() => navigate(`/incidents/${inc.log_id}`)}
                style={{ borderBottom: '1px solid var(--color-border)', cursor: 'pointer', transition: 'background 0.1s' }}
                onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'var(--color-surface-hover)')}
                onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'transparent')}
              >
                <td style={{ padding: '12px 16px', fontFamily: 'var(--font-family-mono)', fontSize: '13px', color: 'var(--color-brand)', fontWeight: 600 }}>{inc.log_id}</td>
                <td style={{ padding: '12px 16px', color: 'var(--color-text-secondary)' }}>{formatDate(inc.timestamp)}</td>
                <td style={{ padding: '12px 16px', color: 'var(--color-text-primary)', fontWeight: 500 }}>{formatLabel(inc.asset_type)}</td>
                <td style={{ padding: '12px 16px', color: 'var(--color-text-secondary)', fontFamily: 'var(--font-family-mono)', fontSize: '13px' }}>{inc.asset_id}</td>
                <td style={{ padding: '12px 16px', textAlign: 'right' }}>
                  <ChevronRight size={16} color="var(--color-text-muted)" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
};

export default IncidentQueue;
