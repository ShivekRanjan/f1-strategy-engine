import { useState } from "react";
import StrategyView from "./views/StrategyView";
import UndercutView from "./views/UndercutView";
import OutcomeView from "./views/OutcomeView";
import LiveView from "./views/LiveView";

const REPO = "https://github.com/ShivekRanjan/f1-strategy-engine";

const TABS = [
  { id: "strategy", label: "Strategy", el: <StrategyView /> },
  { id: "undercut", label: "Undercut", el: <UndercutView /> },
  { id: "outcome", label: "Outcome", el: <OutcomeView /> },
  { id: "live", label: "Live Race", el: <LiveView /> },
] as const;

export default function App() {
  const [tab, setTab] = useState<(typeof TABS)[number]["id"]>("strategy");

  return (
    <div className="min-h-screen">
      {/* Header — pit-wall bar */}
      <header className="sticky top-0 z-20 border-b border-line bg-surface-rail/95 backdrop-blur">
        <div className="flex h-[60px] items-center justify-between px-5">
          <div className="flex items-center gap-3">
            <Logo />
            <div className="leading-tight">
              <div className="text-[15px] font-700 text-ink">
                F1<span className="text-accent">SE</span>
                <span className="px-1 font-500 text-ink-faint">/</span>
                Strategy Engine
              </div>
              <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">
                Tyre degradation · Monte-Carlo pit optimiser
              </div>
            </div>
          </div>
          <div className="hidden items-center gap-4 font-mono text-[11px] sm:flex">
            <span className="flex items-center gap-2 text-accent">
              <span className="h-[7px] w-[7px] animate-f1pulse rounded-full bg-accent" />
              LIVE MODEL
            </span>
            <span className="text-ink-faint">2023–26 · CRN</span>
            <a
              href={REPO}
              target="_blank"
              rel="noreferrer"
              className="rounded-md border border-line-ctl px-2.5 py-1.5 text-ink-dim transition hover:border-line-hover hover:text-ink-soft"
            >
              repo ↗
            </a>
          </div>
        </div>
      </header>

      {/* Tab nav */}
      <nav className="sticky top-[60px] z-10 border-b border-line bg-surface-page/90 backdrop-blur">
        <div className="mx-auto flex max-w-[1180px] gap-1 overflow-x-auto px-5">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`relative whitespace-nowrap px-4 py-3 font-mono text-[12px] uppercase tracking-[0.08em] transition ${
                tab === t.id ? "text-ink" : "text-ink-dim hover:text-ink-muted"
              }`}
            >
              {t.label}
              {tab === t.id && (
                <span className="absolute inset-x-3 -bottom-px h-0.5 rounded-full bg-accent" />
              )}
            </button>
          ))}
        </div>
      </nav>

      <main className="mx-auto max-w-[1180px] px-5 py-6">{TABS.find((t) => t.id === tab)!.el}</main>
    </div>
  );
}

function Logo() {
  return (
    <div className="flex items-center gap-[3px]">
      <span className="h-6 w-[11px] rounded-sm bg-accent shadow-glow" />
      <span className="h-6 w-1.5 rounded-sm bg-soft opacity-90" />
      <span className="h-6 w-1.5 rounded-sm bg-medium opacity-90" />
    </div>
  );
}
