/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    "./KmiDi_FINAL/engine/src/**/*.{js,ts,jsx,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        // Background tokens (dark-first design)
        'bg-primary': 'rgb(32, 32, 36)',
        'bg-secondary': 'rgb(42, 42, 46)',
        'bg-tertiary': 'rgb(52, 52, 56)',
        
        // Text tokens
        'text-primary': 'rgb(224, 224, 228)',
        'text-secondary': 'rgb(160, 160, 164)',
        'text-tertiary': 'rgb(112, 112, 116)',
        
        // Accent tokens
        'accent-primary': 'rgb(0, 122, 255)',
        'accent-secondary': 'rgb(88, 166, 255)',
        'accent-success': 'rgb(52, 199, 89)',
        'accent-warning': 'rgb(255, 149, 0)',
        'accent-error': 'rgb(255, 59, 48)',
        
        // Border tokens
        'border-light': 'rgb(64, 64, 68)',
        'border-medium': 'rgb(84, 84, 88)',
        
        // Emotion colors (muted, per spec)
        'emotion-valence': 'rgb(120, 120, 140)',
        'emotion-arousal': 'rgb(140, 120, 120)',
        'emotion-dominance': 'rgb(120, 140, 120)',
        'emotion-complexity': 'rgb(140, 140, 120)',
        
        // API status colors
        'status-online': 'rgb(76, 175, 80)',
        'status-offline': 'rgb(244, 67, 54)',
        'status-checking': 'rgb(255, 152, 0)',
      },
      spacing: {
        // 4pt baseline grid
        'unit': '4px',
        '1': '4px',   // 1 unit
        '2': '8px',   // 2 units
        '3': '12px',  // 3 units
        '4': '16px',  // 4 units
        '6': '24px',  // 6 units
        '8': '32px',  // 8 units
        '12': '48px', // 12 units
      },
      fontSize: {
        // Minimum 11pt as per spec
        'xs': ['11px', '16px'],
        'sm': ['13px', '18px'],
        'base': ['16px', '24px'],
        'lg': ['18px', '28px'],
        'xl': ['20px', '28px'],
      },
      minHeight: {
        // Minimum 44pt touch targets
        'touch': '44px',
      },
      minWidth: {
        // Minimum 44pt touch targets
        'touch': '44px',
      }
    },
  },
  plugins: [],
}
