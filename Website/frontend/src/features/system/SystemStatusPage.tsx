import React from 'react';

const SystemStatusPage: React.FC = () => {
  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '32px' }}>
      <header style={{ marginBottom: '32px' }}>
        <h1 style={{ margin: '0 0 8px', fontSize: '24px', fontWeight: 700, color: 'var(--color-text-primary)' }}>
          System Status
        </h1>
        <p style={{ margin: 0, fontSize: '14px', color: 'var(--color-text-secondary)' }}>
          Current operational state of the RiskForge prototype.
        </p>
      </header>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        
        <div style={{ padding: '24px', backgroundColor: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: '8px' }}>
          <h2 style={{ margin: '0 0 16px', fontSize: '16px', fontWeight: 600, color: 'var(--color-text-primary)' }}>Frontend Application</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: 'var(--color-success)' }}></div>
            <span style={{ fontSize: '14px', color: 'var(--color-text-primary)', fontWeight: 500 }}>Online</span>
          </div>
          <p style={{ margin: '8px 0 0', fontSize: '14px', color: 'var(--color-text-secondary)' }}>
            Version 0.1.0-alpha. Professional light theme applied.
          </p>
        </div>

        <div style={{ padding: '24px', backgroundColor: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: '8px' }}>
          <h2 style={{ margin: '0 0 16px', fontSize: '16px', fontWeight: 600, color: 'var(--color-text-primary)' }}>Development Data</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: 'var(--color-warning)' }}></div>
            <span style={{ fontSize: '14px', color: 'var(--color-text-primary)', fontWeight: 500 }}>Prototype Dataset Active</span>
          </div>
          <p style={{ margin: '8px 0 0', fontSize: '14px', color: 'var(--color-text-secondary)' }}>
            Serving a static local dataset sourced from `data/` directory. This is not production data and contains incomplete validations.
          </p>
        </div>

        <div style={{ padding: '24px', backgroundColor: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: '8px' }}>
          <h2 style={{ margin: '0 0 16px', fontSize: '16px', fontWeight: 600, color: 'var(--color-text-primary)' }}>NLP Model & Inference Engine</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: 'var(--color-border)' }}></div>
            <span style={{ fontSize: '14px', color: 'var(--color-text-secondary)', fontWeight: 500 }}>Not yet deployed</span>
          </div>
          <p style={{ margin: '8px 0 0', fontSize: '14px', color: 'var(--color-text-secondary)' }}>
            There is currently no trained RiskForge NLP model connected. All model insights shown in the UI are strictly deterministic derivatives of the provided prototype data labels.
          </p>
        </div>

        <div style={{ padding: '24px', backgroundColor: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: '8px' }}>
          <h2 style={{ margin: '0 0 16px', fontSize: '16px', fontWeight: 600, color: 'var(--color-text-primary)' }}>Backend API Adapter</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: 'var(--color-border)' }}></div>
            <span style={{ fontSize: '14px', color: 'var(--color-text-secondary)', fontWeight: 500 }}>Offline</span>
          </div>
          <p style={{ margin: '8px 0 0', fontSize: '14px', color: 'var(--color-text-secondary)' }}>
            Routing via local DevAdapters. The production REST API is not currently connected.
          </p>
        </div>

      </div>
    </div>
  );
};

export default SystemStatusPage;
