/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // F1 broadcast palette — carbon dark + the trademark red.
        carbon: {
          DEFAULT: "#0b0b0d",
          800: "#15151a",
          700: "#1d1d24",
          600: "#26262f",
        },
        line: "#2c2c36",
        ink: { DEFAULT: "#ecedf0", muted: "#9a9aa6" },
        f1: { DEFAULT: "#e2231a", dark: "#b51a13", glow: "#ff3b30" },
        // Tyre-compound accents (used in legends/badges).
        soft: "#e2231a",
        medium: "#f3c700",
        hard: "#dcdce2",
      },
      fontFamily: {
        sans: ["'Titillium Web'", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "monospace"],
      },
      boxShadow: {
        card: "0 1px 0 0 rgba(255,255,255,0.03) inset, 0 8px 24px -12px rgba(0,0,0,0.6)",
      },
      borderRadius: { xl2: "0.875rem" },
    },
  },
  plugins: [],
};
