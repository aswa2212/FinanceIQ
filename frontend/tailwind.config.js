/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Light Theme
        light: {
          bg:           "#FAFBFC", // very soft gray-white
          surface:      "#FFFFFF", // pure white
          "surface-sub":"#F5F7FA", // subtle gray
          border:       "#E8ECF1", // soft borders
          primary:      "#0F172A", // dark slate text
          secondary:    "#64748B", // gray text
          muted:        "#94A3B8", // light gray
        },
        
        // Dark Theme (no purple)
        dark: {
          bg:           "#0A0E1A", // deep navy
          surface:      "#151B2E", // dark blue-gray
          "surface-sub":"#1E2842", // lighter dark
          border:       "#2D3748", // dark borders
          primary:      "#F1F5F9", // light text
          secondary:    "#94A3B8", // gray text
          muted:        "#64748B", // muted text
        },

        // Trading Colors (theme-independent)
        accent:       "#1E40AF", // deep professional blue
        "accent-light":"#EFF6FF",
        "accent-dim": "rgba(30,64,175,0.1)",
        bull:         "#00C896", // teal-green
        "bull-dim":   "rgba(0,200,150,0.1)",
        "bull-light": "#E6F9F5",
        bear:         "#EF4444", // red
        "bear-dim":   "rgba(239,68,68,0.1)",
        "bear-light": "#FEE2E2",
        warning:      "#F59E0B",
        "warning-dim":"rgba(245,158,11,0.1)",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "Consolas", "monospace"],
      },
      fontSize: {
        "2xs": ["10px", "14px"],
        "3xs": ["9px", "13px"],
      },
      boxShadow: {
        glow: "0 4px 16px rgba(30,64,175,0.15)",
        "card": "0 1px 3px rgba(15,23,42,0.08)",
        "card-hover": "0 4px 12px rgba(15,23,42,0.12)",
        "bull-glow": "0 2px 8px rgba(0,200,150,0.15)",
        "bear-glow": "0 2px 8px rgba(239,68,68,0.15)",
        "soft": "0 1px 2px rgba(15,23,42,0.05)",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideIn: {
          "0%": { opacity: "0", transform: "translateX(-12px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-4px)" },
        },
      },
      animation: {
        fadeIn: "fadeIn 0.3s ease-out",
        slideIn: "slideIn 0.3s ease-out",
        float: "float 3s ease-in-out infinite",
      },
      backgroundImage: {
        'gradient-accent': 'linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%)',
        'gradient-bull': 'linear-gradient(135deg, #00C896 0%, #00E5A8 100%)',
        'gradient-bear': 'linear-gradient(135deg, #EF4444 0%, #F87171 100%)',
      },
    },
  },
  plugins: [],
}
