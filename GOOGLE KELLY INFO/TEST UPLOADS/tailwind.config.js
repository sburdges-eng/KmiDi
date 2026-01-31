/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html",
  ],
  theme: {
    extend: {
      colors: {
        'ableton-bg': '#1a1a1a',
        'ableton-surface': '#2a2a2a',
        'ableton-border': '#3a3a3a',
        'ableton-accent': '#ff5500',
        'ableton-text': '#ffffff',
        'ableton-text-dim': '#888888',
        'ableton-green': '#00ff00',
        'ableton-yellow': '#ffff00',
        'ableton-red': '#ff0000',
        'emotion-grief': '#4a6fa5',
        'emotion-joy': '#ffd700',
        'emotion-anger': '#dc143c',
        'emotion-fear': '#8b008b',
        'emotion-love': '#ff69b4',
      },
      fontFamily: {
        mono: ['SF Mono', 'Monaco', 'Inconsolata', 'Fira Mono', 'Droid Sans Mono', 'Source Code Pro', 'monospace'],
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { boxShadow: '0 0 5px var(--ableton-accent)' },
          '50%': { boxShadow: '0 0 20px var(--ableton-accent)' },
        },
      },
    },
  },
  plugins: [],
}
