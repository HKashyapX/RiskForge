import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { incidentRepository } from '../../api/adapters/development/DevIncidentAdapter';
import { segmentTextBySpans } from '../../utils/spanParser';
import { ArrowLeft, AlertTriangle, ShieldAlert, Clock, Activity } from 'lucide-react';
import { formatDate, formatNumber, formatLabel } from '../../utils/formatters';
import { IOGP_LABELS } from '../../domain/constants';

const card: React.CSSProperties = {
  backgroundColor: 'var(--color-surface)',
  border: '1px solid var(--color-border)',
  borderRadius: '8px',
  boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
  overflow: 'hidden',
};

const cardHeader: React.CSSProperties = {
  backgroundColor: 'var(--color-surface-hover)',
  borderBottom: '1px solid var(--color-border)',
  padding: '12px 20px',
  fontSize: '13px',
  fontWeight: 600,
  color: 'var(--color-text-secondary)',
  textTransform: 'uppercase' as const,
  letterSpacing: '0.05em',
};

const routingConfig: Record<string, { label: string; bg: string; color: string; border: string }> = {
  critical_escalation: { label: 'Critical Escalation', bg: 'var(--color-critical-bg)', color: 'var(--color-critical)', border: 'var(--color-critical-border)' },
  hitl_review:         { label: 'Human Review',        bg: 'var(--color-warning-bg)', color: 'var(--color-warning)', border: 'var(--color-warning-border)' },
  auto_dismiss:        { label: 'Lower Priority',      bg: 'var(--color-safe-bg)',    color: 'var(--color-safe)',    border: 'var(--color-safe-border)' },
};

const IncidentDetail: React.FC = () => {
  const { logId } = useParams<{ logId: string }>();
  const navigate = useNavigate();

  const { data: incident, isLoading: iLoading } = useQuery({
    queryKey: ['incident', logId],
    queryFn: () => incidentRepository.getIncidentById(logId!),
  });
  const { data: inference, isLoading: rLoading } = useQuery({
    queryKey: ['inference', logId],
    queryFn: () => incidentRepository.getInferenceResult(logId!),
  });

  if (iLoading || rLoading) return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '64px', color: 'var(--color-text-muted)' }}>
      <div style={{ width: 32, height: 32, border: '3px solid var(--color-border)', borderTopColor: 'var(--color-brand)', borderRadius: '50%', animation: 'spin 0.8s linear infinite', marginBottom: 16 }} />
      <p style={{ margin: 0 }}>Loading incident record…</p>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );

  if (!incident || !inference) return (
    <div style={{ padding: '20px 24px', backgroundColor: 'var(--color-critical-bg)', border: '1px solid var(--color-critical-border)', borderRadius: '8px', color: 'var(--color-critical)' }}>
      <strong>Record Not Found</strong> — The requested incident log could not be located.
    </div>
  );

  const segments = segmentTextBySpans(incident.raw_narrative, incident.spans);
  const routing = routingConfig[inference.routing] ?? routingConfig['auto_dismiss'];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', fontFamily: 'var(--font-family-sans)' }}>

      {/* Page Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', borderBottom: '1px solid var(--color-border)', paddingBottom: '20px' }}>
        <button
          onClick={() => navigate('/incidents')}
          aria-label="Back to Incidents"
          style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: 36, height: 36, border: '1px solid var(--color-border)', borderRadius: '6px', backgroundColor: 'var(--color-surface)', cursor: 'pointer', flexShrink: 0 }}
        >
          <ArrowLeft size={16} color="var(--color-text-secondary)" />
        </button>
        <div>
          <h2 style={{ margin: 0, fontSize: '20px', fontWeight: 700, color: 'var(--color-text-primary)', display: 'flex', alignItems: 'baseline', gap: '10px' }}>
            Incident Report
            <span style={{ fontFamily: 'var(--font-family-mono)', fontSize: '15px', fontWeight: 500, color: 'var(--color-text-muted)' }}>#{incident.log_id}</span>
          </h2>
          <p style={{ margin: '4px 0 0', fontSize: '13px', color: 'var(--color-text-muted)', display: 'flex', alignItems: 'center', gap: '16px' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Clock size={13} /> {formatDate(incident.timestamp)}
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Activity size={13} /> {formatLabel(incident.asset_type)}
            </span>
          </p>
        </div>
      </div>

      {/* Two-column layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: '24px', alignItems: 'start' }}>

        {/* ── Left Column ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

          {/* Original Report */}
          <div style={card}>
            <div style={cardHeader}>Original Safety Report</div>
            <div style={{ padding: '20px 24px', fontSize: '15px', lineHeight: 1.75, color: 'var(--color-text-primary)' }}>
              {segments.map((seg, i) => (
                <span
                  key={i}
                  style={seg.isEvidence ? {
                    backgroundColor: 'var(--color-warning-bg)',
                    borderBottom: '2px solid var(--color-warning)',
                    padding: '1px 3px',
                    margin: '0 1px',
                    borderRadius: '2px',
                    fontWeight: 500,
                    cursor: 'help',
                    position: 'relative',
                  } : undefined}
                  title={seg.isEvidence ? `${seg.spanContext?.entity_type?.replace(/_/g, ' ')}: ${seg.spanContext?.canonical_form}` : undefined}
                >
                  {seg.text}
                </span>
              ))}
            </div>
            <div style={{ padding: '10px 24px', borderTop: '1px solid var(--color-border)', backgroundColor: 'var(--color-surface-hover)', fontSize: '12px', color: 'var(--color-text-muted)' }}>
              ⚠ Highlighted text indicates extracted safety factors. Hover for entity type.
            </div>
          </div>

          {/* Operational Triad */}
          <div style={card}>
            <div style={cardHeader}>Operational Triad</div>
            <div style={{ padding: '20px 24px', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px' }}>
              {[
                { label: 'Activity', value: inference.triad.activity?.canonical_form, color: 'var(--color-text-primary)' },
                { label: 'Asset / Location', value: inference.triad.asset_location?.canonical_form, color: 'var(--color-text-primary)' },
                { label: 'Failed Barrier', value: inference.triad.failed_barrier?.canonical_form, color: 'var(--color-critical)' },
              ].map(({ label, value, color }) => (
                <div key={label}>
                  <div style={{ fontSize: '11px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--color-text-muted)', marginBottom: '6px' }}>{label}</div>
                  <div style={{ fontSize: '14px', fontWeight: 600, color: value ? color : 'var(--color-text-muted)', fontStyle: value ? 'normal' : 'italic' }}>
                    {value ? formatLabel(value) : 'Not identified'}
                  </div>
                </div>
              ))}
            </div>
          </div>

        </div>

        {/* ── Right Column: SIF Assessment ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div style={card}>
            <div style={cardHeader}>Safety Assessment</div>
            <div style={{ padding: '20px' }}>

              {/* Deterministic Override Banner */}
              {inference.deterministic_override && (
                <div style={{ display: 'flex', gap: '12px', backgroundColor: 'var(--color-critical-bg)', border: '1px solid var(--color-critical-border)', borderRadius: '6px', padding: '12px 14px', marginBottom: '20px' }}>
                  <AlertTriangle size={18} color="var(--color-critical)" style={{ flexShrink: 0, marginTop: 1 }} />
                  <div>
                    <div style={{ fontWeight: 700, fontSize: '13px', color: 'var(--color-critical)', marginBottom: '2px' }}>Safety Rule Override Active</div>
                    <div style={{ fontSize: '12px', color: 'var(--color-text-secondary)', lineHeight: 1.5 }}>
                      A deterministic safety rule triggered this escalation based on extracted evidence.
                    </div>
                  </div>
                </div>
              )}

              {/* Score Rows */}
              {[
                { label: 'Raw Model Score', value: formatNumber(inference.raw_sif_p_score, 3) || 'Not processed', mono: true },
                { label: 'Calibrated Score', value: formatNumber(inference.calibrated_sif_p_score, 3) || 'Not processed', mono: true, bold: true },
              ].map(({ label, value, mono, bold }) => (
                <div key={label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 0', borderBottom: '1px solid var(--color-border)' }}>
                  <span style={{ fontSize: '13px', color: 'var(--color-text-secondary)' }}>{label}</span>
                  <span style={{ fontSize: bold ? '16px' : '14px', fontWeight: bold ? 700 : 400, fontFamily: mono ? 'var(--font-family-mono)' : 'inherit', color: 'var(--color-text-primary)' }}>{value}</span>
                </div>
              ))}

              {/* Routing */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 0', borderBottom: '1px solid var(--color-border)' }}>
                <span style={{ fontSize: '13px', color: 'var(--color-text-secondary)' }}>Routing Action</span>
                <span style={{ backgroundColor: routing.bg, color: routing.color, border: `1px solid ${routing.border}`, borderRadius: '20px', padding: '3px 10px', fontSize: '12px', fontWeight: 600 }}>
                  {routing.label}
                </span>
              </div>

              {/* IOGP Rules */}
              <div style={{ paddingTop: '14px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--color-text-secondary)', marginBottom: '10px' }}>
                  <ShieldAlert size={14} /> IOGP Life-Saving Rules
                </div>
                {inference.matched_iogp_rules.length === 0 ? (
                  <span style={{ fontSize: '13px', color: 'var(--color-text-muted)', fontStyle: 'italic' }}>No rules matched</span>
                ) : (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                    {inference.matched_iogp_rules.map(rule => (
                      <span key={rule} style={{ backgroundColor: 'var(--color-surface-hover)', border: '1px solid var(--color-border)', borderRadius: '4px', padding: '4px 10px', fontSize: '12px', color: 'var(--color-text-primary)', fontWeight: 500 }}>
                        {IOGP_LABELS[rule] || formatLabel(rule)}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {/* Latency */}
              <div style={{ marginTop: '20px', paddingTop: '12px', borderTop: '1px solid var(--color-border)', display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--color-text-muted)' }}>
                <span>Model Inference Latency</span>
                <span style={{ fontFamily: 'var(--font-family-mono)' }}>{inference.latency_ms}ms</span>
              </div>

            </div>
          </div>
        </div>

      </div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
};

export default IncidentDetail;
