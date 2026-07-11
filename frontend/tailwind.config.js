/** @type {import('tailwindcss').Config} */

// V2 "broadcast ceremony" system (design handoff: black/gold, Archivo Black,
// ambient motion). The ACCENT is a runtime CSS variable — Settings switches it
// (gold default / cyan / violet) without a rebuild; the `<alpha-value>` form
// keeps Tailwind's `/50`-style alpha modifiers working. Two deliberate
// deviations from the handoff, both correctness:
//   1. Tyre-compound colours are DOMAIN-FIXED (Pirelli red/yellow/white) and
//      never follow the accent — the handoff mapped Medium to accent gold,
//      which would repaint tyres cyan when the theme switches.
//   2. Text tokens keep v1's measured WCAG-AA floor; the handoff's dims
//      (#5a6272/#3f4756 on #06070a) fail the 4.5:1 check we fixed once already.
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // --- V2 surfaces (black, not navy) -----------------------------------
        surface: {
          DEFAULT: "#0b0d12", // cards
          page: "#06070a", // page background
          rail: "#090a0f", // sidebar / top bar
          inset: "#0d0f15", // inset blocks (hero ceremony, controls)
          inset2: "#11141b", // tooltips / timeline track
          strip: "#08090e", // timeline strip
        },
        line: {
          DEFAULT: "#1c212b", // primary borders
          card: "#232a35",
          ctl: "#1e242e",
          hover: "#3a4654",
          faint: "#14171f",
        },
        grid: "#161a22",
        ink: {
          DEFAULT: "#eef1f5",
          soft: "#dbe0e8",
          muted: "#aab3c0",
          // dim/faint carry real labels at small sizes — kept above the WCAG AA
          // 4.5:1 contrast floor (v1 measured ~6.2:1 / ~5.2:1; the handoff's
          // #5a6272/#3f4756 fail on this surface).
          dim: "#8994a4",
          faint: "#7d8698",
          fainter: "#3f4756", // decorative only (never body/label text)
        },
        // --- Theme accent: runtime CSS variable (see index.css + settings) ---
        accent: {
          DEFAULT: "rgb(var(--accent) / <alpha-value>)",
          ink: "#161109", // text on solid accent fills
        },
        // Podium tiers (P1 uses the accent; silver/bronze fixed)
        silver: "#c0c0c0",
        bronze: "#c98a4c",
        // --- Tyre compounds: domain-fixed, NEVER themed ----------------------
        soft: "#ff2b2b",
        medium: "#f6c700",
        hard: "#d8dde3",

        // --- Aliases so pre-port views keep compiling (mapped onto V2) -------
        carbon: {
          DEFAULT: "#06070a",
          800: "#0b0d12",
          700: "#0d0f15",
          600: "#141821",
        },
        f1: {
          DEFAULT: "rgb(var(--accent) / <alpha-value>)",
          dark: "rgb(var(--accent) / 0.85)",
          glow: "rgb(var(--accent) / 1)",
        },
      },
      fontFamily: {
        display: ["'Archivo Black'", "system-ui", "sans-serif"],
        sans: ["'Space Grotesk'", "system-ui", "sans-serif"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "monospace"],
      },
      boxShadow: {
        card: "0 1px 0 0 rgba(255,255,255,0.02) inset, 0 10px 30px -16px rgba(0,0,0,0.75)",
        glow: "0 0 14px -2px rgb(var(--accent) / 0.5)",
      },
      borderRadius: { xl2: "0.875rem" },
      keyframes: {
        f1pulse: { "0%,100%": { opacity: "1" }, "50%": { opacity: "0.3" } },
        fadein: {
          "0%": { opacity: "0", transform: "translateY(10px) scale(.99)" },
          "100%": { opacity: "1", transform: "translateY(0) scale(1)" },
        },
        rise: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        streak: {
          "0%": { transform: "scaleX(0)" },
          "100%": { transform: "scaleX(1)" },
        },
        gridmove: {
          "0%": { backgroundPosition: "0 0" },
          "100%": { backgroundPosition: "28px 28px" },
        },
        sweep: {
          "0%,100%": { transform: "translateX(-120%)" },
          "50%": { transform: "translateX(240%)" },
        },
        breathe: {
          "0%,100%": { boxShadow: "0 0 6px -2px currentColor" },
          "50%": { boxShadow: "0 0 18px -2px currentColor" },
        },
      },
      animation: {
        f1pulse: "f1pulse 1.7s ease-in-out infinite",
        fadein: "fadein 300ms ease-out",
        rise: "rise 450ms ease-out both",
        streak: "streak 800ms ease-out both",
        gridmove: "gridmove 7s linear infinite",
        sweep: "sweep 5s ease-in-out infinite",
        breathe: "breathe 3.2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
