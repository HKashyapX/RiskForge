import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { useNavigate } from 'react-router-dom';
import {
  FileText, AlertTriangle, Users, TrendingUp, TrendingDown,
  Minus, ChevronRight, RefreshCw, Clock, ShieldAlert, Activity,
  ArrowUpRight, CheckCircle2,
} from 'lucide-react';
import { fetchOverviewData } from '../../api/adapters/development/DevOverviewAdapter';
import { formatDate, formatPercentage } from '../../utils/formatters';

// ─── Shared Style Tokens ──────────────────────────────────────────────────────

const S = {
  card: {
    backgroundColor: 'var(--color-surface)',
    border: '1px solid var(--color-border)',
    borderRadius: '8px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
  } satisfies React.CSSProperties,
  cardHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '14px 20px',
    borderBottom: '1px solid var(--color-border)',
  } satisfies React.CSSProperties,
  cardTitle: {
    margin: 0,
    fontSize: '13px',
    fontWeight: 600,
    color: 'var(--color-text-secondary)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.05em',
  } satisfies React.CSSProperties,
  label: {
    fontSize: '11px',
    fontWeight: 600,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.07em',
    color: 'var(--color-text-muted)',
  } satisfies React.CSSProperties,
};

// ─── Sub-components ───────────────────────────────────────────────────────────

const TrendIcon: React.FC<{ trend: 'up' | 'down' | 'stable' }> = ({ trend }) => {
  if (trend === 'up')     return <TrendingUp  size={14} color="var(--color-critical)" />;
  if (trend === 'down')   return <TrendingDown size={14} color="var(--color-safe)"    />;
  return <Minus size={14} color="var(--color-text-muted)" />;
};

const ScoreBadge: React.FC<{ score: number }> = ({ score }) => {
  const pct = Math.round(score * 100);
  const color = pct >= 90 ? 'var(--color-critical)' : pct >= 75 ? 'var(--color-warning)' : 'var(--color-safe)';
  const bg   = pct >= 90 ? 'var(--color-critical-bg)' : pct >= 75 ? 'var(--color-warning-bg)' : 'var(--color-safe-bg)';
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center',
      backgroundColor: bg, color, border: `1px solid ${color}33`,
      borderRadius: '20px', padding: '2px 9px',
      fontSize: '12px', fontWeight: 700, fontFamily: 'var(--font-family-mono)',
    }}>
      {pct}%
    </span>
  );
};

type Period = '7d' | '30d' | '90d';

// ─── Custom Tooltip for SPD Chart ─────────────────────────────────────────────

const SPDTooltip: React.FC<{ active?: boolean; payload?: Array<{ value: number; name: string }>; label?: string }> = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ backgroundColor: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: '6px', padding: '10px 14px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
      <div style={{ fontWeight: 600, fontSize: '13px', marginBottom: '6px', color: 'var(--color-text-primary)' }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ fontSize: '12px', color: 'var(--color-text-secondary)', display: 'flex', gap: '8px', justifyContent: 'space-between' }}>
          <span>{p.name}</span>
          <span style={{ fontWeight: 600, color: 'var(--color-text-primary)' }}>{typeof p.value === 'number' && p.name.includes('%') ? `${p.value}%` : p.value}</span>
        </div>
      ))}
    </div>
  );
};

// ─── Main Overview Page ────────────────────────────────────────────────────────

const OverviewPage: React.FC = () => {
  const navigate = useNavigate();
  const [period, setPeriod] = useState<Period>('30d');

  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ['overview', period],
    queryFn: () => fetchOverviewData(period),
    staleTime: 5 * 60 * 1000,
  });

  if (isLoading) return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '60vh', gap: '16px' }}>
      <div style={{ width: 36, height: 36, border: '3px solid var(--color-border)', borderTopColor: 'var(--color-brand)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
      <p style={{ margin: 0, color: 'var(--color-text-muted)', fontSize: '14px' }}>Loading safety intelligence…</p>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );

  if (isError || !data) return (
    <div style={{ padding: '20px 24px', backgroundColor: 'var(--color-critical-bg)', border: '1px solid var(--color-critical-border)', borderRadius: '8px', color: 'var(--color-critical)' }}>
      <strong>Data Unavailable</strong> — Could not load overview data. Please try again.
    </div>
  );

  const { kpis, spdTrend, topAssets, topBarriers, recentEscalations, routingDistribution, iogpRuleDistribution } = data;

  let freshnessAge: number | 'Unknown' = 'Unknown';
  if (kpis.dataFreshnessTs) {
    const d = new Date(kpis.dataFreshnessTs);
    if (!Number.isNaN(d.getTime())) {
      freshnessAge = Math.round((Date.now() - d.getTime()) / 60000);
    }
  }

  // ── KPI card definitions
  const kpiCards = [
    {
      label: 'Total Safety Reports',
      value: kpis.totalReports,
      icon: <FileText size={18} color="var(--color-info)" />,
      bg: 'var(--color-info-bg)',
      suffix: '',
    },
    {
      label: 'SIF-Potential Reports',
      value: kpis.sifPotentialCount,
      icon: <AlertTriangle size={18} color="var(--color-warning)" />,
      bg: 'var(--color-warning-bg)',
      suffix: '',
    },
    {
      label: 'SIF Precursor Density',
      value: kpis.sifPrecursorDensity,
      icon: <TrendingUp size={18} color="var(--color-brand)" />,
      bg: 'var(--color-brand-light)',
      suffix: '%',
    },
    {
      label: 'Human Review Backlog',
      value: kpis.humanReviewBacklog,
      icon: <Users size={18} color="var(--color-warning)" />,
      bg: 'var(--color-warning-bg)',
      suffix: '',
    },
    {
      label: 'Critical Escalations',
      value: kpis.criticalEscalations,
      icon: <ShieldAlert size={18} color="var(--color-critical)" />,
      bg: 'var(--color-critical-bg)',
      suffix: '',
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px', fontFamily: 'var(--font-family-sans)' }}>

      {/* ── Page Header ── */}
      <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between' }}>
        <div>
          <h1 style={{ margin: 0, fontSize: '24px', fontWeight: 700, color: 'var(--color-text-primary)', letterSpacing: '-0.3px' }}>
            Safety Intelligence Overview
          </h1>
          <p style={{ margin: '4px 0 0', fontSize: '14px', color: 'var(--color-text-secondary)' }}>
            AI/NLP-extracted SIF precursor analysis across all upstream assets · Oil India Limited
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {/* Data freshness */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--color-text-muted)' }}>
            <Clock size={13} />
            Updated {freshnessAge === 'Unknown' ? 'Unknown' : `${freshnessAge}m ago`}
            <button
              onClick={() => refetch()}
              title="Refresh"
              style={{ display: 'flex', alignItems: 'center', background: 'none', border: 'none', cursor: 'pointer', padding: '2px', color: 'var(--color-text-muted)' }}
            >
              <RefreshCw size={13} />
            </button>
          </div>

          {/* DEMO DATA badge */}
          <span style={{
            backgroundColor: '#fef9c3', color: '#854d0e', border: '1px solid #fde047',
            borderRadius: '4px', padding: '3px 8px', fontSize: '11px', fontWeight: 600,
          }}>
            DEMO DATA
          </span>

          {/* Period selector */}
          <div style={{ display: 'flex', border: '1px solid var(--color-border)', borderRadius: '6px', overflow: 'hidden' }}>
            {(['7d', '30d', '90d'] as Period[]).map(p => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                style={{
                  padding: '6px 14px', fontSize: '13px', fontWeight: 500,
                  border: 'none', cursor: 'pointer', fontFamily: 'inherit',
                  backgroundColor: period === p ? 'var(--color-brand)' : 'var(--color-surface)',
                  color: period === p ? '#fff' : 'var(--color-text-secondary)',
                  transition: 'background 0.15s, color 0.15s',
                }}
              >
                {p === '7d' ? '7 days' : p === '30d' ? '30 days' : '90 days'}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── KPI Row ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '16px' }}>
        {kpiCards.map(({ label, value, icon, bg, suffix }) => (
          <div key={label} style={S.card}>
            <div style={{ padding: '20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                <span style={S.label}>{label}</span>
                <div style={{ width: 36, height: 36, borderRadius: '8px', backgroundColor: bg, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {icon}
                </div>
              </div>
              <div style={{ fontSize: '32px', fontWeight: 700, color: 'var(--color-text-primary)', lineHeight: 1, letterSpacing: '-1px' }}>
                {value}{suffix}
              </div>
              <div style={{ marginTop: '6px', fontSize: '12px', color: 'var(--color-text-muted)' }}>
                {kpis.periodLabel}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* ── Row 2: SPD Trend + Routing Distribution ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px' }}>

        {/* SPD Trend Chart */}
        <div style={S.card}>
          <div style={S.cardHeader}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Activity size={15} color="var(--color-brand)" />
              <span style={S.cardTitle}>SIF Precursor Density Trend</span>
            </div>
            <span style={{ fontSize: '12px', color: 'var(--color-text-muted)' }}>Weekly · 12 weeks</span>
          </div>
          <div style={{ padding: '20px 20px 8px' }}>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={spdTrend} margin={{ top: 4, right: 4, left: -10, bottom: 0 }}>
                <defs>
                  <linearGradient id="spdGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#ea580c" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="#ea580c" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
                <XAxis dataKey="week" tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} unit="%" />
                <Tooltip content={<SPDTooltip />} />
                <Area
                  type="monotone"
                  dataKey="spd"
                  name="SPD %"
                  stroke="var(--color-brand)"
                  strokeWidth={2.5}
                  fill="url(#spdGrad)"
                  dot={false}
                  activeDot={{ r: 5, fill: 'var(--color-brand)', stroke: '#fff', strokeWidth: 2 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div style={{ padding: '8px 20px 16px', borderTop: '1px solid var(--color-border)', display: 'flex', gap: '24px' }}>
            <div>
              <div style={S.label}>Current SPD</div>
              <div style={{ fontWeight: 700, fontSize: '20px', color: 'var(--color-brand)', marginTop: 2 }}>{formatPercentage(kpis.sifPrecursorDensity, 1)}</div>
            </div>
            <div>
              <div style={S.label}>12-Week Avg</div>
              <div style={{ fontWeight: 600, fontSize: '20px', color: 'var(--color-text-primary)', marginTop: 2 }}>
                {spdTrend.length > 0 ? formatPercentage(spdTrend.reduce((a, b) => a + b.spd, 0) / spdTrend.length, 1) : 'Not available'}
              </div>
            </div>
            <div>
              <div style={S.label}>Peak Week</div>
              <div style={{ fontWeight: 600, fontSize: '20px', color: 'var(--color-critical)', marginTop: 2 }}>
                {spdTrend.length > 0 ? spdTrend.reduce((a, b) => a.spd > b.spd ? a : b).spd : 'N/A'}%
              </div>
            </div>
          </div>
        </div>

        {/* Routing Distribution Donut */}
        <div style={S.card}>
          <div style={S.cardHeader}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <CheckCircle2 size={15} color="var(--color-safe)" />
              <span style={S.cardTitle}>Routing Distribution</span>
            </div>
          </div>
          <div style={{ padding: '12px 20px' }}>
            <ResponsiveContainer width="100%" height={160}>
              <PieChart>
                <Pie
                  data={routingDistribution}
                  dataKey="count"
                  nameKey="label"
                  cx="50%" cy="50%"
                  innerRadius={50}
                  outerRadius={70}
                  paddingAngle={3}
                >
                  {routingDistribution.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(v: any, name: any) => [v, name]}
                  contentStyle={{ borderRadius: 6, border: '1px solid var(--color-border)', fontSize: 12 }}
                />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '8px' }}>
              {routingDistribution.map(r => (
                <div key={r.bucket} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '13px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: r.color, flexShrink: 0, display: 'inline-block' }} />
                    <span style={{ color: 'var(--color-text-secondary)' }}>{r.label}</span>
                  </div>
                  <span style={{ fontWeight: 700, color: 'var(--color-text-primary)' }}>{r.count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── Row 3: Top Assets + Top Failed Barriers ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>

        {/* Highest-Risk Assets */}
        <div style={S.card}>
          <div style={S.cardHeader}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <AlertTriangle size={15} color="var(--color-warning)" />
              <span style={S.cardTitle}>Highest-Risk Assets</span>
            </div>
            <span style={{ fontSize: '11px', color: 'var(--color-text-muted)' }}>Ranked by SPD</span>
          </div>
          <div>
            {topAssets.map((asset, i) => (
              <div
                key={asset.assetId}
                style={{
                  display: 'flex', alignItems: 'center', gap: '14px',
                  padding: '12px 20px',
                  borderBottom: i < topAssets.length - 1 ? '1px solid var(--color-border)' : 'none',
                }}
              >
                <span style={{
                  width: 24, height: 24, borderRadius: '50%',
                  backgroundColor: i === 0 ? 'var(--color-critical-bg)' : 'var(--color-surface-hover)',
                  color: i === 0 ? 'var(--color-critical)' : 'var(--color-text-muted)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '11px', fontWeight: 700, flexShrink: 0,
                }}>
                  {i + 1}
                </span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: 600, fontSize: '14px', color: 'var(--color-text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {asset.assetLabel}
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--color-text-muted)', marginTop: '2px' }}>
                    {asset.assetType} · {asset.sifPrecursorCount} SIF precursors
                  </div>
                </div>
                <div style={{ textAlign: 'right', flexShrink: 0 }}>
                  <div style={{ fontWeight: 700, fontSize: '16px', color: asset.spdScore >= 20 ? 'var(--color-critical)' : asset.spdScore >= 15 ? 'var(--color-warning)' : 'var(--color-text-primary)' }}>
                    {asset.spdScore}%
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '4px', marginTop: '2px' }}>
                    <TrendIcon trend={asset.trend} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recurrent Failed Safety Barriers */}
        <div style={S.card}>
          <div style={S.cardHeader}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <ShieldAlert size={15} color="var(--color-critical)" />
              <span style={S.cardTitle}>Recurrent Failed Barriers</span>
            </div>
            <span style={{ fontSize: '11px', color: 'var(--color-text-muted)' }}>By frequency</span>
          </div>
          <div>
            {topBarriers.map((b, i) => {
              const maxCount = topBarriers[0].count;
              const barWidth = Math.round((b.count / maxCount) * 100);
              return (
                <div
                  key={b.canonicalForm}
                  style={{
                    padding: '11px 20px',
                    borderBottom: i < topBarriers.length - 1 ? '1px solid var(--color-border)' : 'none',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                    <div>
                      <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--color-text-primary)' }}>{b.label}</span>
                      <span style={{ fontSize: '11px', color: 'var(--color-text-muted)', marginLeft: '8px' }}>{b.iogpRule}</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <TrendIcon trend={b.trend} />
                      <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--color-text-primary)' }}>{b.count}</span>
                    </div>
                  </div>
                  <div style={{ height: 4, borderRadius: 2, backgroundColor: 'var(--color-surface-hover)', overflow: 'hidden' }}>
                    <div style={{
                      height: '100%', width: `${barWidth}%`,
                      backgroundColor: i === 0 ? 'var(--color-critical)' : i === 1 ? 'var(--color-warning)' : 'var(--color-brand)',
                      borderRadius: 2, transition: 'width 0.8s ease',
                    }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* ── Row 4: IOGP Rule Distribution ── */}
      <div style={S.card}>
        <div style={S.cardHeader}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <ArrowUpRight size={15} color="var(--color-info)" />
            <span style={S.cardTitle}>IOGP Life-Saving Rule Violation Frequency</span>
          </div>
          <span style={{ fontSize: '11px', color: 'var(--color-text-muted)' }}>SIF-potential reports only</span>
        </div>
        <div style={{ padding: '16px 20px' }}>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={iogpRuleDistribution} margin={{ top: 4, right: 4, left: -16, bottom: 0 }} barSize={28}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
              <XAxis dataKey="label" tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} interval={0} />
              <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
              <Tooltip
                formatter={(v: any) => [v, 'Reports']}
                contentStyle={{ borderRadius: 6, border: '1px solid var(--color-border)', fontSize: 12 }}
              />
              <Bar dataKey="count" name="Reports" radius={[4, 4, 0, 0]}>
                {iogpRuleDistribution.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? '#dc2626' : i === 1 ? '#d97706' : '#ea580c'} fillOpacity={1 - i * 0.05} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Row 4b: Risk Patterns ── */}
      <div style={S.card}>
        <div style={S.cardHeader}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Activity size={15} color="var(--color-warning)" />
            <span style={S.cardTitle}>Emerging Risk Patterns</span>
          </div>
          <button
            onClick={() => navigate('/analytics')}
            style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '12px', color: 'var(--color-brand)', fontWeight: 600, border: 'none', background: 'none', cursor: 'pointer', fontFamily: 'inherit' }}
          >
            View Patterns <ChevronRight size={13} />
          </button>
        </div>
        <div style={{ padding: '0 20px' }}>
          {data.patterns?.map((p, i) => (
            <div key={p.id} style={{ display: 'flex', alignItems: 'center', gap: '16px', padding: '14px 0', borderBottom: i < data.patterns.length - 1 ? '1px solid var(--color-border)' : 'none' }}>
              <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', flex: 1 }}>
                {p.components.map((c, j) => (
                  <span key={j} style={{ backgroundColor: 'var(--color-surface-hover)', padding: '4px 8px', borderRadius: '4px', fontSize: '12px', fontWeight: 500, color: 'var(--color-text-secondary)', border: '1px solid var(--color-border)' }}>
                    {c}
                  </span>
                ))}
              </div>
              <div style={{ fontSize: '12px', color: 'var(--color-text-muted)', width: '120px' }}>
                {p.location}
              </div>
              <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--color-text-primary)', width: '60px', textAlign: 'right' }}>
                {p.count} <span style={{ fontSize: '11px', color: 'var(--color-text-muted)', fontWeight: 500 }}>events</span>
              </div>
            </div>
          ))}
        </div>
      </div>


      {/* ── Row 5: Recent Critical Escalations ── */}
      <div style={S.card}>
        <div style={S.cardHeader}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <ShieldAlert size={15} color="var(--color-critical)" />
            <span style={S.cardTitle}>Recent Critical Escalations</span>
          </div>
          <button
            onClick={() => navigate('/incidents')}
            style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '12px', color: 'var(--color-brand)', fontWeight: 600, border: 'none', background: 'none', cursor: 'pointer', fontFamily: 'inherit' }}
          >
            View All <ChevronRight size={13} />
          </button>
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
          <thead>
            <tr style={{ backgroundColor: 'var(--color-surface-hover)' }}>
              {['Log ID', 'Time', 'Asset', 'Narrative Excerpt', 'AI Score', 'Primary Rule', ''].map(h => (
                <th key={h} style={{ padding: '9px 16px', textAlign: 'left', fontSize: '11px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--color-text-muted)', borderBottom: '1px solid var(--color-border)', whiteSpace: 'nowrap' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {recentEscalations.map((esc) => (
              <tr
                key={esc.logId}
                onClick={() => navigate(`/incidents/${esc.logId}`)}
                style={{ borderBottom: '1px solid var(--color-border)', cursor: 'pointer', transition: 'background 0.1s' }}
                onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'var(--color-surface-hover)')}
                onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'transparent')}
              >
                <td style={{ padding: '13px 16px', fontFamily: 'var(--font-family-mono)', fontSize: '12px', color: 'var(--color-brand)', fontWeight: 700, whiteSpace: 'nowrap' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    {esc.deterministicOverride && <span title="Safety Rule Override" style={{ color: 'var(--color-critical)' }}><ShieldAlert size={13} /></span>}
                    {esc.logId}
                  </div>
                </td>
                <td style={{ padding: '13px 16px', color: 'var(--color-text-muted)', fontSize: '12px', whiteSpace: 'nowrap' }}>
                  {formatDate(esc.timestamp, 'date')}
                  {' '}
                  {formatDate(esc.timestamp, 'time')}
                </td>
                <td style={{ padding: '13px 16px', whiteSpace: 'nowrap' }}>
                  <div style={{ fontWeight: 600, fontSize: '13px', color: 'var(--color-text-primary)' }}>{esc.assetId}</div>
                  <div style={{ fontSize: '11px', color: 'var(--color-text-muted)' }}>{esc.assetType}</div>
                </td>
                <td style={{ padding: '13px 16px', maxWidth: '320px' }}>
                  <p style={{ margin: 0, fontSize: '13px', color: 'var(--color-text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {esc.narrativeSnippet}
                  </p>
                </td>
                <td style={{ padding: '13px 16px', whiteSpace: 'nowrap' }}>
                  <ScoreBadge score={esc.calibratedScore} />
                </td>
                <td style={{ padding: '13px 16px' }}>
                  <span style={{ fontSize: '12px', fontWeight: 500, color: 'var(--color-text-secondary)', whiteSpace: 'nowrap' }}>
                    {esc.primaryIogpRule}
                  </span>
                </td>
                <td style={{ padding: '13px 16px', textAlign: 'right' }}>
                  <ChevronRight size={15} color="var(--color-text-muted)" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

    </div>
  );
};

export default OverviewPage;
