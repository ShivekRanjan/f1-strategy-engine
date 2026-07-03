import type { ReactNode } from "react";
import { compoundColor } from "../lib/format";

// --- Card -------------------------------------------------------------------
export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-xl2 border border-line bg-carbon-800 shadow-card ${className}`}
    >
      {children}
    </div>
  );
}

export function SectionTitle({ children }: { children: ReactNode }) {
  return (
    <h3 className="mb-3 text-xs font-700 uppercase tracking-[0.14em] text-ink-muted">
      {children}
    </h3>
  );
}

// --- Metric -----------------------------------------------------------------
export function Metric({
  label,
  value,
  sub,
  accent = false,
  title,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  accent?: boolean;
  title?: string;
}) {
  return (
    <Card className={`p-4 ${accent ? "border-l-2 border-l-accent" : ""}`}>
      <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-ink-faint" title={title}>
        {label}
      </div>
      <div className={`nums mt-1 font-mono text-2xl ${accent ? "text-accent" : "text-ink"}`}>{value}</div>
      {sub != null && <div className="nums mt-1 font-mono text-[11px] text-ink-muted">{sub}</div>}
    </Card>
  );
}

// --- Badge / CompoundPill ---------------------------------------------------
export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "red" | "green" | "amber";
}) {
  const tones: Record<string, string> = {
    neutral: "bg-surface-inset2 text-ink-dim",
    red: "bg-soft/15 text-soft",
    green: "bg-emerald-500/15 text-emerald-400",
    amber: "bg-amber-400/15 text-amber-300",
  };
  return (
    <span className={`rounded-full px-2 py-0.5 text-[11px] font-600 ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function CompoundPill({ compound }: { compound: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="inline-block h-2.5 w-2.5 rounded-full"
        style={{ background: compoundColor(compound) }}
      />
      {compound}
    </span>
  );
}

// --- Callout ----------------------------------------------------------------
export function Callout({
  children,
  tone = "info",
}: {
  children: ReactNode;
  tone?: "info" | "success" | "warn";
}) {
  const tones: Record<string, string> = {
    info: "border-l-sky-500 bg-sky-500/5 text-ink",
    success: "border-l-f1 bg-f1/5 text-ink",
    warn: "border-l-amber-400 bg-amber-400/5 text-amber-100",
  };
  return (
    <div className={`rounded-md border border-line border-l-2 px-3 py-2 text-sm ${tones[tone]}`}>
      {children}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-ink-muted">
      <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-line border-t-f1" />
      {label}
    </div>
  );
}

// --- Skeletons ----------------------------------------------------------------
// For the heavy cards (Monte-Carlo, model training): reserve the final height so
// the layout doesn't shift when results land. Spinner stays for quick loads.
export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded-md bg-surface-inset ${className}`} />;
}

export function CardSkeleton({ label, height = 240 }: { label?: string; height?: number }) {
  return (
    <Card className="p-4">
      {label && <Spinner label={label} />}
      <div className="mt-3 space-y-2.5">
        <Skeleton className="h-4 w-1/3" />
        <Skeleton className="h-4 w-2/3" />
        <div style={{ height: Math.max(0, height - 60) }}>
          <Skeleton className="h-full w-full" />
        </div>
      </div>
    </Card>
  );
}

export function ErrorNote({ error }: { error: string }) {
  // A failed fetch (API down / CORS) reads very differently from a real engine
  // error returned with a message — don't mislabel the latter as "unreachable".
  const isNetwork = /failed to fetch|load failed|networkerror|fetch/i.test(error);
  return (
    <Callout tone="warn">
      {isNetwork ? (
        <>
          <span className="font-600">Couldn’t reach the engine.</span> {error}
          <div className="mt-1 text-xs text-ink-muted">
            The API may be waking from sleep (free hosting takes ~30–60s) — refresh in a moment.
            Running locally? <code className="text-ink">uvicorn f1se.api:app</code>
          </div>
        </>
      ) : (
        <>
          <span className="font-600">Engine error:</span> {error}
        </>
      )}
    </Callout>
  );
}
