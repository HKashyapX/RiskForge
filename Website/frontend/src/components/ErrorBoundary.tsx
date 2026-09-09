import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
  children?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error:', error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div style={{
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: 'var(--color-background)',
          fontFamily: 'var(--font-family-sans)',
          padding: '24px'
        }}>
          <div style={{
            maxWidth: '500px',
            width: '100%',
            backgroundColor: 'var(--color-surface)',
            border: '1px solid var(--color-critical-border)',
            borderRadius: '12px',
            boxShadow: '0 12px 32px rgba(0,0,0,0.1)',
            padding: '32px',
            textAlign: 'center'
          }}>
            <div style={{
              width: '64px',
              height: '64px',
              backgroundColor: 'var(--color-critical-bg)',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 24px'
            }}>
              <AlertTriangle size={32} color="var(--color-critical)" />
            </div>
            
            <h1 style={{ margin: '0 0 12px', fontSize: '24px', fontWeight: 700, color: 'var(--color-text-primary)' }}>
              RiskForge
            </h1>
            
            <h2 style={{ margin: '0 0 16px', fontSize: '18px', fontWeight: 600, color: 'var(--color-text-primary)' }}>
              Something went wrong while loading this view.
            </h2>
            
            <p style={{ margin: '0 0 32px', fontSize: '14px', color: 'var(--color-text-secondary)', lineHeight: 1.5 }}>
              The application encountered an unexpected error. This is a prototype environment using local development data.
            </p>
            
            <button
              onClick={() => window.location.reload()}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '12px 24px',
                backgroundColor: 'var(--color-text-primary)',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                fontSize: '14px',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'opacity 0.2s'
              }}
            >
              <RefreshCw size={16} />
              Try Again
            </button>

            {import.meta.env.DEV && this.state.error && (
              <div style={{ marginTop: '32px', textAlign: 'left' }}>
                <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--color-text-muted)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Development Details
                </div>
                <div style={{ padding: '12px', backgroundColor: 'var(--color-bg-secondary)', borderRadius: '6px', fontSize: '12px', color: 'var(--color-critical)', fontFamily: 'var(--font-family-mono)', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                  {this.state.error.toString()}
                </div>
              </div>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
