/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Surfaces & ink
        canvas: "#F5F3EE",
        surface: "#FCFBF8",
        ink: "#17232B",
        muted: "#687278",
        line: "#DCD9D1",
        // Pathways
        forward: "#557A95",
        gradient: "#9A7358",
        neutral: "#7A8589",
        // Semantic
        positive: "#547A67",
        warning: "#B08354",
        critical: "#A65C52",
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
      },
      animation: {
        "fade-up": "fade-up 0.5s ease-out both",
        "fade-in": "fade-in 0.6s ease-out both",
      },
    },
  },
  plugins: [],
};
