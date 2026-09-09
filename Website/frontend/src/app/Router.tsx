import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import AppShell from './AppShell';

import OverviewPage from '../features/overview/OverviewPage';
import IncidentQueue from '../features/incidents/IncidentQueue';
import IncidentDetail from '../features/incidents/IncidentDetail';
import EscalationsPage from '../features/escalations/EscalationsPage';
import AssetIntelligencePage from '../features/assets/AssetIntelligencePage';
import AssetDetailPage from '../features/assets/AssetDetailPage';
import SafetyBarriersPage from '../features/barriers/SafetyBarriersPage';
import AnalyticsPage from '../features/analytics/AnalyticsPage';
import SystemStatusPage from '../features/system/SystemStatusPage';

const Router: React.FC = () => (
  <BrowserRouter>
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/"             element={<Navigate to="/overview" replace />} />
        <Route path="/overview"     element={<OverviewPage />} />
        <Route path="/incidents"    element={<IncidentQueue />} />
        <Route path="/incidents/:logId" element={<IncidentDetail />} />
        <Route path="/escalations"  element={<EscalationsPage />} />
        <Route path="/assets"       element={<AssetIntelligencePage />} />
        <Route path="/assets/:assetId" element={<AssetDetailPage />} />
        <Route path="/barriers"     element={<SafetyBarriersPage />} />
        <Route path="/analytics"    element={<AnalyticsPage />} />
        <Route path="/system-status" element={<SystemStatusPage />} />
        
        <Route path="*" element={
          <div style={{ padding: '32px', backgroundColor: 'var(--color-critical-bg)', border: '1px solid var(--color-critical-border)', borderRadius: '8px' }}>
            <h2 style={{ margin: '0 0 8px', color: 'var(--color-critical)', fontSize: '20px', fontWeight: 700 }}>Page Not Found</h2>
            <p style={{ margin: 0, color: 'var(--color-text-secondary)', fontSize: '14px' }}>The requested route does not exist.</p>
          </div>
        } />
      </Route>
    </Routes>
  </BrowserRouter>
);

export default Router;
