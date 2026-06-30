/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // --- Pit-wall palette (from the design handoff) ---------------------
        surface: {
          DEFAULT: "#0b0f17", // cards
          page: "#080b11", // page background
          rail: "#0a0e15", // header + control rail
          inset: "#10151e", // inset controls
          inset2: "#11161f", // tooltips / timeline track
          strip: "#090d14", // timeline strip
        },
        line: {
          DEFAULT: "#1b222c", // primary borders
          card: "#232c38",
          ctl: "#1f2733",
          hover: "#3a4654",
          faint: "#12171f",
        },
        grid: "#1a212b",
        ink: {
          DEFAULT: "#f2f6fa",
          soft: "#dbe3ec",
          muted: "#aab3c0",
          dim: "#6b7585",
          faint: "#5c6675",
          fainter: "#3f4856",
        },
        accent: { DEFAULT: "#2dd4bf", ink: "#06231f", dark: "#14b8a6" },
        // Tyre compounds (also the logo bars / mean marker stay red+yellow).
        soft: "#e2231a",
        medium: "#f6c700",
        hard: "#d8dde3",

        // --- Aliases so pre-migration components keep working ---------------
        // `carbon` (old near-black) -> the navy surfaces; `f1` (old red) -> the
        // teal accent. These get removed once every view uses the names above.
        carbon: {
          DEFAULT: "#080b11",
          800: "#0b0f17",
          700: "#10151e",
          600: "#141b24",
        },
        f1: { DEFAULT: "#2dd4bf", dark: "#14b8a6", glow: "#2dd4bf" },
      },
      fontFamily: {
        sans: ["'IBM Plex Sans'", "system-ui", "sans-serif"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "monospace"],
      },
      boxShadow: {
        card: "0 1px 0 0 rgba(255,255,255,0.02) inset, 0 10px 30px -16px rgba(0,0,0,0.7)",
        glow: "0 0 14px -2px rgba(45,212,191,0.5)",
      },
      borderRadius: { xl2: "0.875rem" },
      keyframes: {
        f1pulse: { "0%,100%": { opacity: "1" }, "50%": { opacity: "0.35" } },
      },
      animation: { f1pulse: "f1pulse 1.8s ease-in-out infinite" },
    },
  },
  plugins: [],
};
