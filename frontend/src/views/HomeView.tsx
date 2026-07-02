import { api } from "../api/client";
import { Badge, Card, CardSkeleton, ErrorNote, SectionTitle, Skeleton } from "../components/ui";
import { pct, teamColor, timeAgo } from "../lib/format";
import { countdown, fmtSession, useNow } from "../lib/time";
import { useAsync } from "../lib/useAsync";
import type { CalendarResp, NewsResp, StandingsResp, UpcomingResp } from "../api/types";

/** The OS home: what's next, what the model expects, where the title stands,
 *  and what the paddock is talking about — each block deep-links to its section. */
export default function HomeView() {
  return (
    <div className="space-y-5">
      <p className="max-w-3xl text-sm text-ink-muted">
        One screen for the season: the next race with the model's podium call, the live title race,
        and the latest headlines. Everything links into its full section — start anywhere.
      </p>
      <NextRaceHero />
      <div className="grid gap-5 lg:grid-cols-[1.25fr_1fr]">
        <TitleRace />
        <Headlines />
      </div>
      <ExploreStrip />
    </div>
  );
}

// --- Next race + the model's call --------------------------------------------
function NextRaceHero() {
  const cal = useAsync(async () => {
    const seasons = await api.allSeasons();
    const latest = seasons.seasons.at(-1);
    return latest ? api.calendar(latest) : null;
  }, []);
  const up = useAsync(() => api.predictUpcoming(), []);

  if (cal.error) return <ErrorNote error={cal.error} />;
  if (!cal.data) return <CardSkeleton label="Finding the next race…" height={200} />;

  const round = cal.data.rounds.find((r) => r.round === cal.data!.next_round);
  const next = cal.data.next_session;
  if (!round || !next) return <SeasonOver cal={cal.data} />;

  return (
    <Card className="border-l-2 border-l-accent p-4">
      <Hero round={round} next={next} up={up.data ?? null} upErr={!!up.error} />
    </Card>
  );
}

function Hero({
  round,
  next,
  up,
  upErr,
}: {
  round: NonNullable<CalendarResp["rounds"][number]>;
  next: NonNullable<CalendarResp["next_session"]>;
  up: UpcomingResp | null;
  upErr: boolean;
}) {
  const now = useNow();
  const podium = up && up.next_round === round.round ? up.predictions.slice(0, 3) : null;
  return (
    <>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-ink-faint">
            Next race · Round {round.round}
          </div>
          <div className="mt-1 flex items-center gap-2">
            <span className="text-2xl font-700 text-ink">{round.event_name}</span>
            {round.format?.includes("sprint") && <Badge tone="amber">Sprint</Badge>}
          </div>
          <div className="text-sm text-ink-muted">
            {round.location}, {round.country}
          </div>
        </div>
        <div className="text-right">
          <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-ink-faint">
            {next.name} in
          </div>
          <div className="nums font-mono text-3xl text-accent">{countdown(next.date, now)}</div>
          <div className="font-mono text-[11px] text-ink-muted">{fmtSession(next.date)}</div>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <span className="font-mono text-[11px] uppercase tracking-[0.12em] text-ink-faint">
          🔮 Predicted podium
        </span>
        {podium ? (
          podium.map((p, i) => (
            <span key={p.driver} className="rounded-lg border border-line-card bg-surface-inset px-3 py-1.5">
              <span className="mr-1.5 font-mono text-[11px] text-ink-faint">P{i + 1}</span>
              <span className="font-700 text-ink">{p.driver}</span>{" "}
              <span className="nums font-mono text-[12px] text-accent">{pct(p.podium_prob)}</span>
            </span>
          ))
        ) : upErr ? (
          <span className="text-sm text-ink-muted">prediction unavailable</span>
        ) : (
          <Skeleton className="h-8 w-64" />
        )}
        <span className="ml-auto flex gap-3">
          <a href="#/outcome" className="font-mono text-[11px] text-accent hover:opacity-80">
            tune the grid →
          </a>
          <a href="#/calendar" className="font-mono text-[11px] text-accent hover:opacity-80">
            full schedule →
          </a>
        </span>
      </div>
    </>
  );
}

function SeasonOver({ cal }: { cal: CalendarResp }) {
  return (
    <Card className="p-4">
      <SectionTitle>Season complete</SectionTitle>
      <p className="text-sm text-ink-muted">
        All {cal.rounds.length} rounds of {cal.season} have run —{" "}
        <a href="#/standings" className="text-accent">see the final standings</a> or{" "}
        <a href="#/racehub" className="text-accent">relive any race in the Race Hub</a>.
      </p>
    </Card>
  );
}

// --- Title race snapshot ------------------------------------------------------
function TitleRace() {
  const s = useAsync(() => api.standings(), []);
  if (s.error) return <ErrorNote error={s.error} />;
  if (!s.data) return <CardSkeleton label="Tallying the championship…" height={280} />;
  return <TitleRaceBody data={s.data} />;
}

function TitleRaceBody({ data }: { data: StandingsResp }) {
  const top = data.drivers.slice(0, 5);
  const maxPts = Math.max(1, ...top.map((d) => d.points));
  return (
    <Card className="p-4">
      <div className="mb-3 flex items-baseline justify-between">
        <SectionTitle>
          Title race — {data.season}{" "}
          {data.ongoing && (
            <Badge tone="red">
              {data.races_done} of {data.total_races} races
            </Badge>
          )}
        </SectionTitle>
        <a href="#/standings" className="font-mono text-[11px] text-accent hover:opacity-80">
          full standings →
        </a>
      </div>
      <div className="space-y-2">
        {top.map((d) => (
          <div key={d.driver} className="flex items-center gap-3">
            <span className="w-5 text-right font-mono text-[12px] text-ink-faint">{d.pos}</span>
            <span className="inline-block h-3.5 w-1 rounded-sm" style={{ background: teamColor(d.team) }} />
            <span className="w-12 font-700 text-ink">{d.driver}</span>
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-inset">
              <div
                className="h-full rounded-full bg-accent/70"
                style={{ width: `${(d.points / maxPts) * 100}%` }}
              />
            </div>
            <span className="nums w-12 text-right font-mono text-[12px] text-ink">
              {d.points.toFixed(0)}
            </span>
            {data.ongoing && (
              <span className="nums w-12 text-right font-mono text-[12px] text-accent">
                {d.win_prob != null ? pct(d.win_prob) : "—"}
              </span>
            )}
          </div>
        ))}
      </div>
      {data.ongoing && (
        <p className="mt-2 text-right font-mono text-[11px] text-ink-faint">points · title odds</p>
      )}
    </Card>
  );
}

// --- Headlines ----------------------------------------------------------------
function Headlines() {
  const n = useAsync(() => api.news(6), []);
  if (n.error) return <ErrorNote error={n.error} />;
  if (!n.data) return <CardSkeleton label="Fetching headlines…" height={280} />;
  return <HeadlinesBody data={n.data} />;
}

function HeadlinesBody({ data }: { data: NewsResp }) {
  if (!data.items.length) return null;
  return (
    <Card className="p-4">
      <div className="mb-2 flex items-baseline justify-between">
        <SectionTitle>Paddock news</SectionTitle>
        <a href="#/news" className="font-mono text-[11px] text-accent hover:opacity-80">
          all headlines →
        </a>
      </div>
      <div className="divide-y divide-line/60">
        {data.items.slice(0, 5).map((it) => (
          <a
            key={it.link}
            href={it.link}
            target="_blank"
            rel="noreferrer noopener"
            className="group block py-2"
          >
            <div className="flex items-center gap-2 font-mono text-[11px]">
              <span className="text-accent">{it.source}</span>
              <span className="text-ink-faint">{timeAgo(it.ts)}</span>
            </div>
            <div className="text-sm font-600 text-ink group-hover:text-accent">{it.title}</div>
          </a>
        ))}
      </div>
    </Card>
  );
}

// --- Explore the toolkit --------------------------------------------------------
const TOOLS = [
  {
    href: "#/strategy",
    title: "Strategy optimiser",
    blurb: "Monte-Carlo search over 1,000+ pit plans — with a track-temp control.",
  },
  {
    href: "#/racehub",
    title: "Race Hub",
    blurb: "Any race: prediction vs result, strategy call, tyre curves, pace trace.",
  },
  {
    href: "#/live",
    title: "Live replay",
    blurb: "Replay lap by lap; the engine re-optimises from every state.",
  },
] as const;

function ExploreStrip() {
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      {TOOLS.map((t) => (
        <a
          key={t.href}
          href={t.href}
          className="group rounded-xl2 border border-line bg-carbon-800 p-4 shadow-card transition hover:border-accent/40"
        >
          <div className="mb-1 font-700 text-ink group-hover:text-accent">{t.title} →</div>
          <div className="text-sm text-ink-muted">{t.blurb}</div>
        </a>
      ))}
    </div>
  );
}
