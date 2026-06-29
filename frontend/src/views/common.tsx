import type { ReactNode } from "react";
import { api } from "../api/client";
import { Callout, ErrorNote, Spinner } from "../components/ui";
import { useAsync } from "../lib/useAsync";

/** Tracks that have raced under 2026 regs (for default selection + the 🆕 hint). */
const PREFERRED_2026 = [
  "Japanese Grand Prix",
  "Miami Grand Prix",
  "Monaco Grand Prix",
  "Australian Grand Prix",
];

export function pickDefaultTrack(tracks: string[]): string {
  for (const p of PREFERRED_2026) if (tracks.includes(p)) return p;
  if (tracks.includes("Spanish Grand Prix")) return "Spanish Grand Prix";
  return tracks[0];
}

/** Fetch the circuit list once, then hand it to children. Keeps every view's
 *  inner component mounted only with real data (no undefined-track juggling). */
export function TracksGate({ children }: { children: (tracks: string[]) => ReactNode }) {
  const s = useAsync(() => api.tracks(), []);
  if (s.loading) return <Spinner label="Loading circuits…" />;
  if (s.error) return <ErrorNote error={s.error} />;
  if (!s.data?.tracks?.length) return <Callout>No circuits found.</Callout>;
  return <>{children(s.data.tracks)}</>;
}

export function ViewIntro({ children }: { children: ReactNode }) {
  return <p className="mb-5 max-w-3xl text-sm text-ink-muted">{children}</p>;
}
