/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Surfaces & ink — RGB triplet format lets Tailwind opacity modifiers work at runtime
        canvas:   "rgb(var(--canvas-rgb) / <alpha-value>)",
        surface:  "rgb(var(--surface-rgb) / <alpha-value>)",
        ink:      "rgb(var(--ink-rgb) / <alpha-value>)",
        muted:    "rgb(var(--muted-rgb) / <alpha-value>)",
        line:     "rgb(var(--line-rgb) / <alpha-value>)",
        // Pathways
        forward:  "rgb(var(--forward-rgb) / <alpha-value>)",
        gradient: "rgb(var(--gradient-rgb) / <alpha-value>)",
        neutral:  "rgb(var(--neutral-rgb) / <alpha-value>)",
        // Semantic
        positive: "rgb(var(--positive-rgb) / <alpha-value>)",
        warning:  "rgb(var(--warning-rgb) / <alpha-value>)",
        critical: "rgb(var(--critical-rgb) / <alpha-value>)",
        // Dark sidebar shades
        navy: {
          DEFAULT: "#17232B",
          800: "#1C2A33",
          700: "#243440",
          600: "#33454F",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        display: ["'DM Sans'", "Inter", "sans-serif"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
      letterSpacing: {
        label: "0.14em",
      },
      borderRadius: {
        sm: "4px",
        DEFAULT: "6px",
        md: "8px",
        lg: "10px",
      },
      boxShadow: {
        card: "0 1px 2px rgba(23, 35, 43, 0.04)",
        panel: "0 1px 3px rgba(23, 35, 43, 0.05), 0 1px 1px rgba(23,35,43,0.03)",
      },
      maxWidth: {
        content: "1180px",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "fade-in": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        "draw": {
          "0%": { strokeDashoffset: "1" },
          "100%": { strokeDashoffset: "0" },
        },
        "blur-fade-in": {
          "0%": { opacity: "0", filter: "blur(12px)", transform: "translateY(12px)" },
          "100%": { opacity: "1", filter: "blur(0)", transform: "translateY(0)" },
        },
        "spin-slow": {
          "0%": { transform: "rotate(0deg)" },
          "100%": { transform: "rotate(360deg)" },
        },
        "shimmer": {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.5s ease-out both",
        "fade-in": "fade-in 0.6s ease-out both",
        "blur-fade-in": "blur-fade-in 0.5s ease-out both",
        "spin-slow": "spin-slow 6s linear infinite",
        "shimmer": "shimmer 2.5s linear infinite",
      },
    },
  },
  plugins: [],
};
