import { useState } from "react";
import StrategyView from "./views/StrategyView";
import RaceHubView from "./views/RaceHubView";
import UndercutView from "./views/UndercutView";
import OutcomeView from "./views/OutcomeView";
import StandingsView from "./views/StandingsView";
import ProfilesView from "./views/ProfilesView";
import NewsView from "./views/NewsView";
import CalendarView from "./views/CalendarView";
import LiveView from "./views/LiveView";

const REPO = "https://github.com/ShivekRanjan/f1-strategy-engine";

const TABS = [
  { id: "strategy", label: "Strategy", group: "Strategy", el: <StrategyView /> },
  { id: "undercut", label: "Undercut", group: "Strategy", el: <UndercutView /> },
  { id: "calendar", label: "Calendar", group: "Race weekend", el: <CalendarView /> },
  { id: "racehub", label: "Race Hub", group: "Race weekend", el: <RaceHubView /> },
  { id: "live", label: "Live Race", group: "Race weekend", el: <LiveView /> },
  { id: "standings", label: "Standings", group: "Championship", el: <StandingsView /> },
  { id: "profiles", label: "Drivers & Teams", group: "Championship", el: <ProfilesView /> },
  { id: "outcome", label: "Outcome", group: "Championship", el: <OutcomeView /> },
  { id: "news", label: "News", group: "Paddock", el: <NewsView /> },
] as const;

type TabId = (typeof TABS)[number]["id"];
const GROUP_ORDER = ["Strategy", "Race weekend", "Championship", "Paddock"] as const;

export default function App() {
  const [tab, setTab] = useState<TabId>("strategy");
  const active = TABS.find((t) => t.id === tab)!;

  return (
    <div className="lg:flex">
      {/* ---- Sidebar (desktop) ------------------------------------------- */}
      <aside className="sticky top-0 z-20 hidden h-screen w-60 shrink-0 flex-col border-r border-line bg-surface-rail lg:flex">
        <Brand />
        <nav className="flex-1 overflow-y-auto px-3 py-4">
          {GROUP_ORDER.map((g) => (
            <div key={g} className="mb-5">
              <div className="mb-1.5 px-2 font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
                {g}
              </div>
              {TABS.filter((t) => t.group === g).map((t) => (
                <NavItem key={t.id} label={t.label} active={tab === t.id} onClick={() => setTab(t.id)} />
              ))}
            </div>
          ))}
        </nav>
        <Footer />
      </aside>

      {/* ---- Mobile top bar ---------------------------------------------- */}
      <header className="sticky top-0 z-20 border-b border-line bg-surface-rail/95 backdrop-blur lg:hidden">
        <div className="flex h-[56px] items-center justify-between px-4">
          <Brand compact />
          <a href={REPO} target="_blank" rel="noreferrer" className="font-mono text-[11px] text-ink-dim">
            repo ↗
          </a>
        </div>
        <div className="flex gap-1 overflow-x-auto border-t border-line px-3 pb-2 pt-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`whitespace-nowrap rounded-md px-3 py-1.5 font-mono text-[12px] transition ${
                tab === t.id ? "bg-accent/10 text-accent" : "text-ink-dim hover:text-ink-muted"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </header>

      {/* ---- Main content ------------------------------------------------- */}
      <div className="min-w-0 flex-1">
        <div className="hidden items-center justify-between border-b border-line px-6 py-3 lg:flex">
          <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-ink-muted">
            <span className="text-ink-faint">{active.group}</span>
            <span className="px-2 text-ink-faint">/</span>
            <span className="text-ink">{active.label}</span>
          </div>
          <div className="flex items-center gap-4 font-mono text-[11px]">
            <span className="flex items-center gap-2 text-accent">
              <span className="h-[7px] w-[7px] animate-f1pulse rounded-full bg-accent" />
              LIVE MODEL
            </span>
            <span className="text-ink-faint">2023–26 · CRN</span>
          </div>
        </div>
        <main className="mx-auto max-w-[1180px] px-5 py-6">{active.el}</main>
      </div>
    </div>
  );
}

function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`flex items-center gap-3 ${compact ? "" : "border-b border-line px-4 py-4"}`}>
      <Logo />
      <div className="leading-tight">
        <div className="text-[15px] font-700 text-ink">
          F1<span className="text-accent">SE</span>
          <span className="px-1 font-500 text-ink-faint">/</span>
          <span className="text-ink-soft">F1 OS</span>
        </div>
        <div className="font-mono text-[9px] uppercase tracking-[0.14em] text-ink-faint">
          Strategy · standings · live
        </div>
      </div>
    </div>
  );
}

function NavItem({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`relative flex w-full items-center rounded-md px-3 py-2 text-left text-[13px] transition ${
        active ? "bg-accent/10 text-accent" : "text-ink-dim hover:bg-surface-inset/60 hover:text-ink-soft"
      }`}
    >
      {active && <span className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-accent" />}
      {label}
    </button>
  );
}

function Footer() {
  return (
    <div className="border-t border-line px-4 py-3">
      <a
        href={REPO}
        target="_blank"
        rel="noreferrer"
        className="flex items-center justify-between font-mono text-[11px] text-ink-dim transition hover:text-ink-soft"
      >
        <span>github ↗</span>
        <span className="flex items-center gap-1.5 text-accent">
          <span className="h-[6px] w-[6px] animate-f1pulse rounded-full bg-accent" />
          live
        </span>
      </a>
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
