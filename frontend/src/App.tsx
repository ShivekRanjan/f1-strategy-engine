import { useState } from "react";
import StrategyView from "./views/StrategyView";
import UndercutView from "./views/UndercutView";
import OutcomeView from "./views/OutcomeView";
import LiveView from "./views/LiveView";

const TABS = [
  { id: "strategy", label: "Strategy", icon: "🏁", el: <StrategyView /> },
  { id: "undercut", label: "Undercut", icon: "🆚", el: <UndercutView /> },
  { id: "outcome", label: "Outcome", icon: "🏆", el: <OutcomeView /> },
  { id: "live", label: "Live Race", icon: "🔴", el: <LiveView /> },
] as const;

export default function App() {
  const [tab, setTab] = useState<(typeof TABS)[number]["id"]>("strategy");

  return (
    <div className="mx-auto max-w-6xl px-4 pb-16">
      <header className="flex flex-col gap-2 border-b border-line py-6 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="flex items-center gap-3">
            <span className="h-7 w-1.5 rounded-full bg-f1 shadow-[0_0_16px_2px_rgba(226,35,26,0.5)]" />
            <h1 className="text-2xl font-900 uppercase tracking-tight text-ink">
              F1 Strategy Engine
            </h1>
          </div>
          <p className="mt-1 text-sm text-ink-muted">
            Not <em>who will win</em> — <em>what should the team do</em>. Pit strategy,
            outcomes &amp; live calls, with quantified uncertainty.
          </p>
        </div>
        <a
          href="https://github.com/ShivekRanjan/f1-strategy-engine"
          target="_blank"
          rel="noreferrer"
          className="text-xs text-ink-muted underline-offset-4 hover:text-ink hover:underline"
        >
          source · methodology ↗
        </a>
      </header>

      <nav className="mt-5 flex gap-1 overflow-x-auto border-b border-line">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`relative whitespace-nowrap px-4 py-2.5 text-sm font-600 transition ${
              tab === t.id ? "text-ink" : "text-ink-muted hover:text-ink"
            }`}
          >
            <span className="mr-1.5">{t.icon}</span>
            {t.label}
            {tab === t.id && (
              <span className="absolute inset-x-2 -bottom-px h-0.5 rounded-full bg-f1" />
            )}
          </button>
        ))}
      </nav>

      <main className="mt-6">{TABS.find((t) => t.id === tab)!.el}</main>
    </div>
  );
}
