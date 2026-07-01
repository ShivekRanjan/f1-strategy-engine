// Presentation helpers — the TS equivalents of the old Streamlit formatters,
// so the UI speaks the same plain-English language.

/** Seconds -> H:MM:SS (or M:SS when under an hour) — human race time. */
export function clock(seconds: number): string {
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return h ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
}

/** A single lap time, e.g. "90.4 s". */
export function secs(x: number, dp = 1): string {
  return `${x.toFixed(dp)} s`;
}

/** Probability -> %, without rounding a real small chance down to "0%". */
export function pct(p: number): string {
  if (p === 0) return "0%";
  if (p < 0.1) return `${(p * 100).toFixed(1)}%`;
  return `${Math.round(p * 100)}%`;
}

/** "M → H  (pit lap 18)" */
export function fmtPlan(compounds: string[], pitLaps: number[]): string {
  const base = compounds.join(" → ");
  return pitLaps.length ? `${base}  (pit lap ${pitLaps.join(", ")})` : base;
}

/** Plain-English version of the paired win-probability column. */
export function beatsPick(rank: number, prob: number): string {
  return rank === 1 ? "★ our pick" : `wins ${Math.round(prob * 100)}% of races`;
}

/** Gap text: "2.1s behind" / "1.4s ahead" (negative = ahead). */
export function gapText(g: number): string {
  return `${Math.abs(g).toFixed(1)}s ${g < 0 ? "ahead" : "behind"}`;
}

export const COMPOUND_COLOR: Record<string, string> = {
  SOFT: "#e2231a",
  MEDIUM: "#f3c700",
  HARD: "#dcdce2",
  INTERMEDIATE: "#43b02a",
  WET: "#1f6feb",
};

export function compoundColor(c: string): string {
  return COMPOUND_COLOR[c?.toUpperCase()] ?? "#9a9aa6";
}

// Team liveries (substring-matched so "Red Bull Racing", "RB", etc. all resolve).
export const TEAM_COLOR: Record<string, string> = {
  mercedes: "#27f4d2",
  ferrari: "#e8002d",
  "red bull": "#3671c6",
  mclaren: "#ff8000",
  alpine: "#0093cc",
  aston: "#229971",
  williams: "#64c4ff",
  haas: "#b6babd",
  sauber: "#52e252",
  audi: "#52e252",
  "racing bulls": "#6692ff",
  alphatauri: "#6692ff",
};

export function teamColor(team: string): string {
  const t = (team || "").toLowerCase();
  for (const key in TEAM_COLOR) if (t.includes(key)) return TEAM_COLOR[key];
  return "#9a9aa6";
}
