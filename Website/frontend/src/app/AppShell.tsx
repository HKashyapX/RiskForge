import React from 'react';
import { Outlet, NavLink, useLocation, Link } from 'react-router-dom';
import { Shield, FileText, AlertTriangle, Database, ShieldAlert, LayoutDashboard, CheckCircle2, Activity } from 'lucide-react';
import RiskForgeAssistant from '../components/assistant/RiskForgeAssistant';

const navItems = [
  { path: '/overview',    label: 'Overview',        icon: <LayoutDashboard size={16} /> },
  { path: '/incidents',   label: 'Incidents',       icon: <FileText size={16} /> },
  { path: '/escalations', label: 'Escalations',     icon: <AlertTriangle size={16} /> },
  { path: '/assets',      label: 'Assets',          icon: <Database size={16} /> },
  { path: '/barriers',    label: 'Safety Barriers', icon: <ShieldAlert size={16} /> },
  { path: '/analytics',   label: 'Analytics',       icon: <Activity size={16} /> },
];

const AppShell: React.FC = () => {
  const location = useLocation();

  const crumb = location.pathname.split('/').filter(Boolean)[0] ?? 'overview';
  const label = navItems.find(n => n.path === `/${crumb}`)?.label ?? crumb;

  return (
    <div style={{ minHeight: '100vh', backgroundColor: 'var(--color-background)', display: 'flex', flexDirection: 'column', fontFamily: 'var(--font-family-sans)' }}>
      
      {/* ── Top Navigation Bar ── */}
      <header style={{
        position: 'sticky', top: 0, zIndex: 50,
        backgroundColor: 'var(--color-surface)',
        borderBottom: '1px solid var(--color-border)',
        boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 24px', height: '56px',
      }}>
        {/* Logo + Nav */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '32px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Shield size={22} color="var(--color-brand)" />
            <span style={{ fontWeight: 700, fontSize: '17px', color: 'var(--color-text-primary)', letterSpacing: '-0.3px' }}>
              RiskForge
            </span>
          </div>

          <nav style={{ display: 'flex', alignItems: 'center', gap: '2px', borderLeft: '1px solid var(--color-border)', paddingLeft: '24px' }}>
            {navItems.map(item => (
              <NavLink
                key={item.path}
                to={item.path}
                style={({ isActive }) => ({
                  display: 'flex', alignItems: 'center', gap: '6px',
                  padding: '6px 12px', borderRadius: '6px',
                  fontSize: '14px', fontWeight: 500, textDecoration: 'none',
                  transition: 'background 0.15s, color 0.15s',
                  backgroundColor: isActive ? 'var(--color-brand-light)' : 'transparent',
                  color: isActive ? 'var(--color-brand)' : 'var(--color-text-secondary)',
                })}
              >
                {item.icon}
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>

        {/* Status */}
        <Link to="/system-status" style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--color-text-muted)', textDecoration: 'none' }}>
          <CheckCircle2 size={15} color="var(--color-safe)" />
          Prototype Dataset
        </Link>
      </header>

      {/* ── Breadcrumb Bar ── */}
      <div style={{
        backgroundColor: 'var(--color-surface)',
        borderBottom: '1px solid var(--color-border)',
        padding: '8px 24px',
        display: 'flex', alignItems: 'center', gap: '8px',
        fontSize: '13px', color: 'var(--color-text-muted)',
      }}>
        <span>Home</span>
        <span>/</span>
        <span style={{ color: 'var(--color-text-primary)', fontWeight: 500 }}>{label.charAt(0).toUpperCase() + label.slice(1)}</span>
      </div>

      {/* ── Page Content ── */}
      <main style={{ flex: 1, padding: '32px 24px', maxWidth: '1440px', margin: '0 auto', width: '100%' }}>
        <Outlet />
      </main>

      <RiskForgeAssistant />
    </div>
  );
};

export default AppShell;
