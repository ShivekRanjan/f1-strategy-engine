import { useEffect, useState } from "react";
import { api } from "../api/client";
import { Badge, Card, ErrorNote, SectionTitle, Spinner } from "../components/ui";
import { useAsync } from "../lib/useAsync";
import type { CalendarRound, CalendarResp } from "../api/types";
import { ViewIntro } from "./common";

/** Ticking clock (1 Hz) for the live countdown. */
function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [active]);
  return now;
}

function countdown(target: string, now: number): string {
  const ms = new Date(target).getTime() - now;
  if (ms <= 0) return "under way";
  const s = Math.floor(ms / 1000);
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (d > 0) return `${d}d ${h}h ${m}m`;
  if (h > 0) return `${h}h ${m}m ${sec}s`;
  return `${m}m ${sec}s`;
}

const fmtDay = (iso: string | null) =>
  iso ? new Date(iso + "T00:00:00").toLocaleDateString(undefined, { month: "short", day: "numeric" }) : "";
const fmtSession = (iso: string) =>
  new Date(iso).toLocaleString(undefined, {
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
  });

export default function CalendarView() {
  const seasons = useAsync(() => api.allSeasons(), []);
  const [season, setSeason] = useState<number | null>(null);
  useEffect(() => {
    if (seasons.data?.seasons?.length) setSeason(seasons.data.seasons.at(-1)!);
  }, [seasons.data]);

  const cal = useAsync(
    () => (season == null ? Promise.resolve(null) : api.calendar(season)),
    [season],
  );

  return (
    <div className="space-y-5">
      <ViewIntro>
        The full season calendar — every round, circuit and session time, with the next race counted
        down live. Real-time timing streams only while a session is running; between sessions, the{" "}
        <strong>Live Race</strong> tab replays any completed race lap by lap.
      </ViewIntro>

      {(seasons.data?.seasons?.length ?? 0) > 1 && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="mr-1 font-mono text-[11px] uppercase tracking-[0.12em] text-ink-faint">Season</span>
          {seasons.data!.seasons.slice().reverse().map((y) => (
            <button
              key={y}
              onClick={() => setSeason(y)}
              className={`rounded-md border px-3 py-1.5 font-mono text-[12px] transition ${
                y === season
                  ? "border-accent/60 bg-accent/10 text-accent"
                  : "border-line-ctl text-ink-dim hover:border-line-hover hover:text-ink-soft"
              }`}
            >
              {y}
            </button>
          ))}
        </div>
      )}

      {cal.error && <ErrorNote error={cal.error} />}
      {season != null && !cal.data && !cal.error && <Spinner label="Loading the calendar…" />}
      {cal.data && <Body cal={cal.data} />}
    </div>
  );
}

function Body({ cal }: { cal: CalendarResp }) {
  const nextRound = cal.rounds.find((r) => r.round === cal.next_round) ?? null;
  return (
    <>
      {nextRound && cal.next_session && <NextRaceCard round={nextRound} nextSessionIso={cal.next_session.date} nextSessionName={cal.next_session.name} />}
      <Card className="p-4">
        <SectionTitle>
          {cal.season} calendar · {cal.rounds.filter((r) => r.done).length}/{cal.rounds.length} run
        </SectionTitle>
        <div className="divide-y divide-line">
          {cal.rounds.map((r) => (
            <RoundRow key={r.round} r={r} isNext={r.round === cal.next_round} />
          ))}
        </div>
      </Card>
    </>
  );
}

function NextRaceCard({
  round,
  nextSessionIso,
  nextSessionName,
}: {
  round: CalendarRound;
  nextSessionIso: string;
  nextSessionName: string;
}) {
  const now = useNow(true);
  const isSprint = round.format?.includes("sprint");
  return (
    <Card className="border-l-2 border-l-accent p-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.12em] text-ink-faint">
            Next up · Round {round.round}
          </div>
          <div className="mt-1 flex items-center gap-2">
            <span className="text-2xl font-700 text-ink">{round.event_name}</span>
            {isSprint && <Badge tone="amber">Sprint</Badge>}
          </div>
          <div className="text-sm text-ink-muted">
            {round.location}, {round.country}
          </div>
        </div>
        <div className="text-right">
          <div className="font-mono text-[10px] uppercase tracking-[0.12em] text-ink-faint">
            {nextSessionName} in
          </div>
          <div className="nums font-mono text-3xl text-accent">{countdown(nextSessionIso, now)}</div>
          <div className="font-mono text-[11px] text-ink-muted">{fmtSession(nextSessionIso)}</div>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
        {round.sessions.map((s) => {
          const upcoming = new Date(s.date).getTime() > now;
          return (
            <div
              key={s.name}
              className={`rounded-md border px-3 py-2 ${
                upcoming ? "border-line-card bg-surface-inset" : "border-line bg-transparent opacity-60"
              }`}
            >
              <div className="font-mono text-[10px] uppercase tracking-[0.1em] text-ink-faint">{s.name}</div>
              <div className="text-sm text-ink">{fmtSession(s.date)}</div>
              {upcoming && <div className="font-mono text-[11px] text-accent">in {countdown(s.date, now)}</div>}
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function RoundRow({ r, isNext }: { r: CalendarRound; isNext: boolean }) {
  return (
    <div className={`flex items-center gap-3 py-2.5 ${r.done ? "opacity-55" : ""}`}>
      <span className="w-7 text-center font-mono text-[12px] text-ink-faint">{r.round}</span>
      <span className="w-14 font-mono text-[12px] text-ink-muted">{fmtDay(r.event_date)}</span>
      <span className={`font-600 ${isNext ? "text-accent" : "text-ink"}`}>{r.event_name}</span>
      <span className="text-xs text-ink-muted">{r.country}</span>
      {r.format?.includes("sprint") && <Badge tone="amber">Sprint</Badge>}
      <span className="ml-auto">
        {r.done ? (
          <span className="font-mono text-[11px] text-ink-faint">✓ done</span>
        ) : isNext ? (
          <Badge tone="red">next</Badge>
        ) : (
          <span className="font-mono text-[11px] text-ink-faint">upcoming</span>
        )}
      </span>
    </div>
  );
}
