import { useState } from "react";
import { api } from "../api/client";
import { Column, DataTable } from "../components/DataTable";
import { Badge, Callout, Card, CardSkeleton, ErrorNote, SectionTitle } from "../components/ui";
import { pct } from "../lib/format";
import { teamColor } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import type { ConstructorStanding, DriverStanding, StandingsResp } from "../api/types";
import { ViewIntro } from "./common";

export default function StandingsView() {
  const [season, setSeason] = useState<number | null>(null);
  const s = useAsync(() => api.standings(season ?? undefined), [season]);

  return (
    <div className="space-y-5">
      <ViewIntro>
        Live drivers’ and constructors’ championships. For the season in progress, each driver also
        carries a <strong>title-win probability</strong> — from the same Monte-Carlo season simulator
        the Outcome tab uses, so it reflects how much racing is still left, not a naive points
        extrapolation.
      </ViewIntro>
      {s.error && <ErrorNote error={s.error} />}
      {!s.data && !s.error && (
        <CardSkeleton label="Tallying the championship…" height={420} />
      )}
      {s.data && <Body data={s.data} season={season} setSeason={setSeason} />}
    </div>
  );
}

function Body({
  data,
  season,
  setSeason,
}: {
  data: StandingsResp;
  season: number | null;
  setSeason: (y: number | null) => void;
}) {
  const active = season ?? data.latest;
  return (
    <>
      {/* Season selector */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="mr-1 font-mono text-[11px] uppercase tracking-[0.12em] text-ink-faint">
          Season
        </span>
        {data.seasons
          .slice()
          .reverse()
          .map((y) => (
            <button
              key={y}
              onClick={() => setSeason(y === data.latest ? null : y)}
              className={`rounded-md border px-3 py-1.5 font-mono text-[12px] transition ${
                y === active
                  ? "border-accent/60 bg-accent/10 text-accent"
                  : "border-line-ctl text-ink-dim hover:border-line-hover hover:text-ink-soft"
              }`}
            >
              {y}
            </button>
          ))}
        {data.ongoing && (
          <Badge tone="red">
            live · {data.races_done} of {data.total_races} races
          </Badge>
        )}
      </div>

      {data.ongoing && <LeaderStrip data={data} />}

      <div className="grid gap-5 lg:grid-cols-[1.35fr_1fr]">
        <DriversCard data={data} />
        <ConstructorsCard rows={data.constructors} />
      </div>
    </>
  );
}

// --- Championship leader hero (ongoing season only) --------------------------
function LeaderStrip({ data }: { data: StandingsResp }) {
  const leader = data.drivers[0];
  const second = data.drivers[1];
  const gap = second ? leader.points - second.points : 0;
  return (
    <Card className="flex flex-wrap items-center gap-x-8 gap-y-3 border-l-2 border-l-accent p-4">
      <div>
        <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-ink-faint">
          Championship leader
        </div>
        <div className="mt-1 flex items-baseline gap-2">
          <span
            className="inline-block h-4 w-1 rounded-sm"
            style={{ background: teamColor(leader.team) }}
          />
          <span className="text-2xl font-700 text-ink">{leader.driver}</span>
          <span className="text-sm text-ink-muted">{leader.team}</span>
        </div>
      </div>
      <Stat label="Points" value={leader.points.toFixed(0)} />
      <Stat label="Lead" value={second ? `+${gap.toFixed(0)}` : "—"} sub={second ? `over ${second.driver}` : ""} />
      <Stat label="Wins" value={String(leader.wins)} />
      {leader.win_prob != null && (
        <Stat label="Title odds" value={pct(leader.win_prob)} accent />
      )}
    </Card>
  );
}

function Stat({
  label,
  value,
  sub,
  accent = false,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: boolean;
}) {
  return (
    <div>
      <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-ink-faint">{label}</div>
      <div className={`nums mt-1 font-mono text-2xl ${accent ? "text-accent" : "text-ink"}`}>
        {value}
      </div>
      {sub && <div className="nums font-mono text-[11px] text-ink-muted">{sub}</div>}
    </div>
  );
}

function TeamCell({ team }: { team: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-ink-muted">
      <span className="inline-block h-3.5 w-1 rounded-sm" style={{ background: teamColor(team) }} />
      {team}
    </span>
  );
}

// --- Drivers standings ------------------------------------------------------
function DriversCard({ data }: { data: StandingsResp }) {
  const maxProb = Math.max(0.01, ...data.drivers.map((d) => d.win_prob ?? 0));
  const cols: Column<DriverStanding>[] = [
    { key: "pos", header: "#", align: "right", render: (d) => <span className="text-ink-muted">{d.pos}</span> },
    { key: "driver", header: "Driver", render: (d) => <span className="font-700">{d.driver}</span> },
    { key: "team", header: "Team", render: (d) => <TeamCell team={d.team} /> },
    { key: "wins", header: "W", align: "right", render: (d) => (d.wins ? d.wins : <span className="text-ink-faint">—</span>) },
    { key: "pod", header: "Pod", align: "right", render: (d) => (d.podiums ? d.podiums : <span className="text-ink-faint">—</span>) },
    { key: "points", header: "Points", align: "right", render: (d) => <span className="font-600">{d.points.toFixed(0)}</span> },
  ];
  if (data.ongoing) {
    cols.push({
      key: "title",
      header: "Title odds",
      align: "right",
      render: (d) =>
        d.win_prob != null ? (
          <span className="nums inline-flex items-center justify-end gap-2">
            <span
              className="inline-block h-1.5 rounded-full bg-accent align-middle"
              style={{ width: `${Math.max(3, (d.win_prob / maxProb) * 56)}px` }}
            />
            {pct(d.win_prob)}
          </span>
        ) : (
          <span className="text-ink-faint">—</span>
        ),
    });
  }
  return (
    <Card className="p-4">
      <SectionTitle>Drivers’ championship — {data.season}</SectionTitle>
      <DataTable columns={cols} rows={data.drivers} getKey={(d) => d.driver} highlightFirst />
      {data.ongoing && (
        <Callout>
          Title odds run <strong>{data.total_races - data.races_done}</strong> remaining races
          through the season simulator, bootstrapping each driver’s current-season form — so a big
          early lead reads as “very likely”, not a false 100%.
        </Callout>
      )}
    </Card>
  );
}

// --- Constructors standings -------------------------------------------------
function ConstructorsCard({ rows }: { rows: ConstructorStanding[] }) {
  const cols: Column<ConstructorStanding>[] = [
    { key: "pos", header: "#", align: "right", render: (c) => <span className="text-ink-muted">{c.pos}</span> },
    { key: "team", header: "Constructor", render: (c) => <TeamCell team={c.team} /> },
    { key: "wins", header: "W", align: "right", render: (c) => (c.wins ? c.wins : <span className="text-ink-faint">—</span>) },
    { key: "points", header: "Points", align: "right", render: (c) => <span className="font-600">{c.points.toFixed(0)}</span> },
  ];
  return (
    <Card className="p-4">
      <SectionTitle>Constructors’ championship</SectionTitle>
      <DataTable columns={cols} rows={rows} getKey={(c) => c.team} highlightFirst />
    </Card>
  );
}
