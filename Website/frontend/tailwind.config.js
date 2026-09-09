/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: 'var(--color-background)',
        surface: 'var(--color-surface)',
        'surface-hover': 'var(--color-surface-hover)',
        'text-primary': 'var(--color-text-primary)',
        'text-secondary': 'var(--color-text-secondary)',
        'text-muted': 'var(--color-text-muted)',
        border: 'var(--color-border)',
        brand: 'var(--color-brand)',
        'brand-light': 'var(--color-brand-light)',
        safe: 'var(--color-safe)',
        'safe-bg': 'var(--color-safe-bg)',
        warning: 'var(--color-warning)',
        'warning-bg': 'var(--color-warning-bg)',
        critical: 'var(--color-critical)',
        'critical-bg': 'var(--color-critical-bg)',
        info: 'var(--color-info)',
        'info-bg': 'var(--color-info-bg)',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
