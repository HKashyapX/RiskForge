import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchOverviewData } from '../../api/adapters/development/DevOverviewAdapter';
import { BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { formatPercentage } from '../../utils/formatters';
import { AlertTriangle, TrendingUp, Compass, Target, Grid, Beaker } from 'lucide-react';

type Tab = 'density' | 'patterns' | 'iogp' | 'activities';

const AnalyticsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<Tab>('density');

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
      <p style={{ margin: 0, fontWeight: 500 }}>Loading analytics engine…</p>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );

  if (isError || !data) return (
    <div style={{ padding: '20px 24px', backgroundColor: 'var(--color-critical-bg)', border: '1px solid var(--color-critical-border)', borderRadius: '8px', color: 'var(--color-critical)' }}>
      <strong>System Error</strong> — Failed to retrieve safety analytics.
    </div>
  );

  const renderTabs = () => (
    <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid var(--color-border)', paddingBottom: '16px', marginBottom: '24px' }}>
      {[
        { id: 'density', label: 'Density Metrics', icon: <Target size={16} /> },
        { id: 'patterns', label: 'Risk Patterns', icon: <TrendingUp size={16} /> },
        { id: 'iogp', label: 'IOGP Rules', icon: <Compass size={16} /> },
        { id: 'activities', label: 'Activity Correlation', icon: <Beaker size={16} /> }
      ].map(t => (
        <button
          key={t.id}
          onClick={() => setActiveTab(t.id as Tab)}
          style={{
            display: 'flex', alignItems: 'center', gap: '6px',
            padding: '8px 16px', fontSize: '13px', fontWeight: 600,
            borderRadius: '6px', border: 'none', cursor: 'pointer', fontFamily: 'inherit',
            backgroundColor: activeTab === t.id ? 'var(--color-brand)' : 'transparent',
            color: activeTab === t.id ? '#fff' : 'var(--color-text-secondary)',
            transition: 'all 0.2s ease'
          }}
        >
          {t.icon}
          {t.label}
        </button>
      ))}
    </div>
  );

  const renderDensity = () => (
    <div style={{ ...card, padding: '24px' }}>
      <h3 style={{ margin: '0 0 16px 0', color: 'var(--color-text-primary)' }}>SIF Precursor Density</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px' }}>
        <div style={{ padding: '20px', border: '1px solid var(--color-border)', borderRadius: '8px', backgroundColor: 'var(--color-surface-hover)' }}>
          <div style={{ fontSize: '12px', color: 'var(--color-text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: '8px' }}>Current SPD</div>
          <div style={{ fontSize: '32px', fontWeight: 700, color: 'var(--color-brand)' }}>
            {formatPercentage(data.kpis.sifPrecursorDensity, 1)}
          </div>
        </div>
      </div>
      <p style={{ fontSize: '14px', color: 'var(--color-text-secondary)', marginTop: '24px', lineHeight: 1.6 }}>
        SPD is calculated as <strong>(SIF Precursors / Total Incident Logs) × 100</strong>. 
        Further dimensional breakdowns (e.g., by Location or Shift) are pending deeper dataset processing.
      </p>
    </div>
  );

  const renderPatterns = () => (
    <div style={{ ...card, padding: '24px' }}>
      <h3 style={{ margin: '0 0 16px 0', color: 'var(--color-text-primary)' }}>Discovered Patterns</h3>
      {data.patterns && data.patterns.length > 0 ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {data.patterns.map((p) => (
            <div key={p.id} style={{ display: 'flex', alignItems: 'center', gap: '16px', padding: '16px', backgroundColor: 'var(--color-surface-hover)', borderRadius: '8px', border: '1px solid var(--color-border)' }}>
               <div style={{ flex: 1, display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {p.components.map((c, j) => (
                  <span key={j} style={{ backgroundColor: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: '4px', padding: '4px 8px', fontSize: '13px', fontWeight: 500, color: 'var(--color-text-secondary)' }}>
                    {c}
                  </span>
                ))}
              </div>
              <div style={{ fontSize: '13px', color: 'var(--color-text-muted)' }}>{p.location}</div>
              <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--color-text-primary)' }}>{p.count} <span style={{ fontSize: '11px', fontWeight: 500, color: 'var(--color-text-muted)' }}>events</span></div>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ padding: '40px', textAlign: 'center', color: 'var(--color-text-muted)', border: '1px dashed var(--color-border)', borderRadius: '8px' }}>
          <Grid size={32} style={{ opacity: 0.5, marginBottom: '12px' }} />
          <p style={{ margin: 0 }}>Pattern analysis requires additional processed data. Not available in current dataset.</p>
        </div>
      )}
    </div>
  );

  const renderIogp = () => (
    <div style={{ ...card, padding: '24px' }}>
      <h3 style={{ margin: '0 0 16px 0', color: 'var(--color-text-primary)' }}>IOGP Rule Distribution</h3>
      {data.iogpRuleDistribution && data.iogpRuleDistribution.length > 0 ? (
        <div style={{ height: 300, marginTop: '24px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.iogpRuleDistribution} layout="vertical" margin={{ left: 80, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="var(--color-border)" />
              <XAxis type="number" axisLine={false} tickLine={false} tick={{ fill: 'var(--color-text-muted)', fontSize: 12 }} />
              <YAxis dataKey="label" type="category" axisLine={false} tickLine={false} tick={{ fill: 'var(--color-text-secondary)', fontSize: 12 }} width={140} />
              <Tooltip cursor={{ fill: 'var(--color-surface-hover)' }} contentStyle={{ borderRadius: 8, border: '1px solid var(--color-border)' }} />
              <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                {data.iogpRuleDistribution.map((_, i) => (
                   <Cell key={i} fill={i === 0 ? 'var(--color-critical)' : i === 1 ? 'var(--color-warning)' : 'var(--color-brand)'} fillOpacity={0.8} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <p style={{ color: 'var(--color-text-muted)' }}>No IOGP rule data available.</p>
      )}
    </div>
  );

  const renderActivities = () => (
    <div style={{ ...card, padding: '48px', textAlign: 'center' }}>
      <AlertTriangle size={32} color="var(--color-text-muted)" style={{ opacity: 0.5, marginBottom: '12px' }} />
      <h3 style={{ margin: '0 0 8px 0', color: 'var(--color-text-primary)' }}>Activity Correlation Data Unavailable</h3>
      <p style={{ margin: 0, color: 'var(--color-text-secondary)', maxWidth: '400px', marginLeft: 'auto', marginRight: 'auto', lineHeight: 1.5 }}>
        Deep activity correlation and causality tracking is currently pending integration with the production NLP model. 
        This prototype dataset does not contain sufficient activity matrices to render valid charts.
      </p>
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <div>
        <h2 style={{ margin: 0, fontSize: '24px', fontWeight: 700, color: 'var(--color-text-primary)' }}>Analytics</h2>
        <p style={{ margin: '4px 0 24px', fontSize: '14px', color: 'var(--color-text-muted)' }}>Deep-dive analysis of safety metadata, rules, and discovered risk associations.</p>
      </div>

      {renderTabs()}

      <div>
        {activeTab === 'density' && renderDensity()}
        {activeTab === 'patterns' && renderPatterns()}
        {activeTab === 'iogp' && renderIogp()}
        {activeTab === 'activities' && renderActivities()}
      </div>
    </div>
  );
};

export default AnalyticsPage;
