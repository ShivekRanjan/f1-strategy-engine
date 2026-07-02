import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import { Column, DataTable } from "../components/DataTable";
import { Combobox, Field } from "../components/controls";
import { Callout, Card, ErrorNote, Metric, SectionTitle, Spinner } from "../components/ui";
import { teamColor } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import type {
  ConstructorProfile,
  DriverProfile,
  RecentResult,
  TeammateH2H,
} from "../api/types";
import { ViewIntro } from "./common";

type Mode = "drivers" | "constructors";

export default function ProfilesView() {
  const [mode, setMode] = useState<Mode>("drivers");
  return (
    <div className="space-y-5">
      <ViewIntro>
        Per-season and aggregate records for every driver and constructor, plus the classic{" "}
        <strong>teammate head-to-head</strong> (who out-qualified and out-raced whom). Totals cover{" "}
        <strong>this dataset’s window (2023–26)</strong> — they’re not all-time career figures.
      </ViewIntro>
      <div className="flex gap-2">
        {(["drivers", "constructors"] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`rounded-md border px-4 py-1.5 font-mono text-[12px] uppercase tracking-[0.08em] transition ${
              mode === m
                ? "border-accent/60 bg-accent/10 text-accent"
                : "border-line-ctl text-ink-dim hover:border-line-hover hover:text-ink-soft"
            }`}
          >
            {m}
          </button>
        ))}
      </div>
      {mode === "drivers" ? <Drivers /> : <Constructors />}
    </div>
  );
}

// ============================ Drivers =======================================
function Drivers() {
  const idx = useAsync(() => api.driversIndex(), []);
  const [code, setCode] = useState<string | null>(null);
  useEffect(() => {
    const ds = idx.data?.drivers;
    if (ds?.length) setCode((prev) => prev ?? ds[0].driver);
  }, [idx.data]);

  if (idx.error) return <ErrorNote error={idx.error} />;
  if (!idx.data) return <Spinner label="Loading drivers…" />;

  return (
    <>
      <Card className="p-4">
        <Field label="Driver">
          <Combobox
            value={code ?? ""}
            options={idx.data.drivers.map((d) => d.driver)}
            onChange={(v) => setCode(v)}
            getLabel={(v) => {
              const d = idx.data!.drivers.find((x) => x.driver === v);
              return d ? `${d.driver} — ${d.team}` : String(v);
            }}
            placeholder="Search drivers…"
          />
        </Field>
      </Card>
      {code && <DriverCard key={code} code={code} />}
    </>
  );
}

function DriverCard({ code }: { code: string }) {
  const p = useAsync(() => api.driverProfile(code), [code]);
  if (p.error) return <ErrorNote error={p.error} />;
  if (!p.data) return <Spinner label="Loading profile…" />;
  return <DriverBody p={p.data} />;
}

function DriverBody({ p }: { p: DriverProfile }) {
  const c = p.career;
  return (
    <div className="space-y-5">
      {/* Header */}
      <Card className="flex flex-wrap items-center gap-x-6 gap-y-2 p-4">
        <span className="inline-block h-8 w-1.5 rounded-sm" style={{ background: teamColor(p.team) }} />
        <div>
          <div className="text-2xl font-700 text-ink">{p.driver}</div>
          <div className="text-sm text-ink-muted">{p.team}</div>
        </div>
        <div className="ml-auto font-mono text-[11px] text-ink-faint">
          {p.seasons[0]}–{p.seasons.at(-1)} · {p.seasons.length} season{p.seasons.length === 1 ? "" : "s"}
        </div>
      </Card>

      {/* Window totals (2023–26 — not all-time career) */}
      <div>
        <div className="mb-2 flex items-baseline justify-between">
          <SectionTitle>
            {p.seasons[0]}–{String(p.seasons.at(-1)).slice(2)} totals
          </SectionTitle>
          <span className="font-mono text-[11px] uppercase tracking-[0.1em] text-ink-faint">
            within dataset · not all-time career
          </span>
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
          <Metric label="Races" value={c.races} />
          <Metric label="Wins" value={c.wins} accent={c.wins > 0} />
          <Metric label="Podiums" value={c.podiums} />
          <Metric label="Points" value={(c.points ?? 0).toFixed(0)} />
          <Metric label="Best" value={c.best ? `P${c.best}` : "—"} />
          <Metric label="Avg finish" value={c.avg_finish ? `P${c.avg_finish.toFixed(1)}` : "—"} />
          <Metric label="DNFs" value={c.dnf} />
        </div>
      </div>

      <div className="grid gap-5 lg:grid-cols-[1.4fr_1fr]">
        <PointsBySeason
          data={p.by_season.map((s) => ({ label: String(s.season), points: s.points ?? 0 }))}
        />
        <TeammateCard driver={p.driver} season={p.h2h_season} h2h={p.teammate_h2h} />
      </div>

      <SeasonTable rows={p.by_season} />
      <RecentTable rows={p.recent} />
    </div>
  );
}

function TeammateCard({
  driver,
  season,
  h2h,
}: {
  driver: string;
  season: number;
  h2h: TeammateH2H[];
}) {
  return (
    <Card className="p-4">
      <SectionTitle>Teammate head-to-head · {season}</SectionTitle>
      {h2h.length === 0 ? (
        <p className="text-sm text-ink-muted">No teammate data for {season}.</p>
      ) : (
        <div className="space-y-4">
          {h2h.map((h) => (
            <div key={h.teammate}>
              <div className="mb-1.5 flex items-center justify-between text-sm">
                <span className="font-600 text-ink">{driver}</span>
                <span className="text-ink-faint">vs</span>
                <span className="font-600 text-ink">{h.teammate}</span>
              </div>
              <ScoreBar label="Qualifying" self={h.quali_ahead} total={h.quali_races} />
              <ScoreBar label="Race" self={h.race_ahead} total={h.race_races} />
              <div className="mt-1 flex justify-between font-mono text-[11px] text-ink-muted">
                <span>{(h.pts_self ?? 0).toFixed(0)} pts</span>
                <span>points</span>
                <span>{(h.pts_mate ?? 0).toFixed(0)} pts</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

function ScoreBar({ label, self, total }: { label: string; self: number; total: number }) {
  const pct = total ? (self / total) * 100 : 50;
  return (
    <div className="mb-1.5">
      <div className="mb-0.5 flex justify-between font-mono text-[11px] uppercase tracking-[0.1em] text-ink-faint">
        <span>
          {label} {self}–{total - self}
        </span>
      </div>
      <div className="flex h-2 overflow-hidden rounded-full bg-soft/25">
        <div className="h-full rounded-l-full bg-accent" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function SeasonTable({ rows }: { rows: DriverProfile["by_season"] }) {
  const cols: Column<DriverProfile["by_season"][number]>[] = [
    { key: "season", header: "Season", render: (s) => <span className="font-600">{s.season}</span> },
    {
      key: "team",
      header: "Team",
      render: (s) => (
        <span className="inline-flex items-center gap-2 text-ink-muted">
          <span className="inline-block h-3.5 w-1 rounded-sm" style={{ background: teamColor(s.team) }} />
          {s.team}
        </span>
      ),
    },
    { key: "races", header: "R", align: "right", render: (s) => s.races },
    { key: "wins", header: "W", align: "right", render: (s) => s.wins || <span className="text-ink-faint">—</span> },
    { key: "pod", header: "Pod", align: "right", render: (s) => s.podiums || <span className="text-ink-faint">—</span> },
    { key: "pts", header: "Points", align: "right", render: (s) => <span className="font-600">{(s.points ?? 0).toFixed(0)}</span> },
    { key: "grid", header: "Avg grid", align: "right", render: (s) => (s.avg_grid ? s.avg_grid.toFixed(1) : "—") },
    { key: "fin", header: "Avg fin", align: "right", render: (s) => (s.avg_finish ? s.avg_finish.toFixed(1) : "—") },
  ];
  return (
    <Card className="p-4">
      <SectionTitle>Season by season</SectionTitle>
      <DataTable columns={cols} rows={rows} getKey={(s) => `${s.season}-${s.team}`} />
    </Card>
  );
}

function RecentTable({ rows }: { rows: RecentResult[] }) {
  const cols: Column<RecentResult>[] = [
    { key: "race", header: "Race", render: (r) => <span className="font-600">{r.event_name}</span> },
    { key: "season", header: "Season", align: "right", render: (r) => <span className="text-ink-muted">{r.season}</span> },
    { key: "grid", header: "Grid", align: "right", render: (r) => r.grid ?? "—" },
    {
      key: "pos",
      header: "Finish",
      align: "right",
      render: (r) => (r.position == null ? <span className="text-soft">DNF</span> : `P${r.position}`),
    },
    { key: "pts", header: "Pts", align: "right", render: (r) => (r.points ? r.points.toFixed(0) : "0") },
  ];
  return (
    <Card className="p-4">
      <SectionTitle>Recent results</SectionTitle>
      <DataTable columns={cols} rows={rows} getKey={(r) => `${r.season}-${r.round}`} />
    </Card>
  );
}

// ========================== Constructors ====================================
function Constructors() {
  const idx = useAsync(() => api.constructorsIndex(), []);
  const [team, setTeam] = useState<string | null>(null);
  useEffect(() => {
    const cs = idx.data?.constructors;
    if (cs?.length) setTeam((prev) => prev ?? cs[0].team);
  }, [idx.data]);

  if (idx.error) return <ErrorNote error={idx.error} />;
  if (!idx.data) return <Spinner label="Loading constructors…" />;

  return (
    <>
      <Card className="p-4">
        <Field label="Constructor">
          <Combobox
            value={team ?? ""}
            options={idx.data.constructors.map((c) => c.team)}
            onChange={(v) => setTeam(v)}
            placeholder="Search teams…"
          />
        </Field>
      </Card>
      {team && <ConstructorCard key={team} team={team} />}
    </>
  );
}

function ConstructorCard({ team }: { team: string }) {
  const p = useAsync(() => api.constructorProfile(team), [team]);
  if (p.error) return <ErrorNote error={p.error} />;
  if (!p.data) return <Spinner label="Loading profile…" />;
  return <ConstructorBody p={p.data} />;
}

function ConstructorBody({ p }: { p: ConstructorProfile }) {
  const c = p.career;
  const drvCols: Column<ConstructorProfile["drivers"][number]>[] = [
    { key: "driver", header: "Driver", render: (d) => <span className="font-700">{d.driver}</span> },
    { key: "seasons", header: "Seasons", render: (d) => <span className="text-ink-muted">{d.seasons.join(", ")}</span> },
    { key: "wins", header: "W", align: "right", render: (d) => d.wins || <span className="text-ink-faint">—</span> },
    { key: "pts", header: "Points", align: "right", render: (d) => <span className="font-600">{(d.points ?? 0).toFixed(0)}</span> },
  ];
  const seaCols: Column<ConstructorProfile["by_season"][number]>[] = [
    { key: "season", header: "Season", render: (s) => <span className="font-600">{s.season}</span> },
    { key: "drivers", header: "Drivers", render: (s) => <span className="text-ink-muted">{s.drivers.join(", ")}</span> },
    { key: "races", header: "R", align: "right", render: (s) => s.races },
    { key: "wins", header: "W", align: "right", render: (s) => s.wins || <span className="text-ink-faint">—</span> },
    { key: "pod", header: "Pod", align: "right", render: (s) => s.podiums || <span className="text-ink-faint">—</span> },
    { key: "pts", header: "Points", align: "right", render: (s) => <span className="font-600">{(s.points ?? 0).toFixed(0)}</span> },
  ];
  return (
    <div className="space-y-5">
      <Card className="flex flex-wrap items-center gap-x-6 gap-y-2 p-4">
        <span className="inline-block h-8 w-1.5 rounded-sm" style={{ background: teamColor(p.team) }} />
        <div className="text-2xl font-700 text-ink">{p.team}</div>
        <div className="ml-auto font-mono text-[11px] text-ink-faint">
          {p.seasons[0]}–{p.seasons.at(-1)}
        </div>
      </Card>

      <div>
        <div className="mb-2 flex items-baseline justify-between">
          <SectionTitle>
            {p.seasons[0]}–{String(p.seasons.at(-1)).slice(2)} totals
          </SectionTitle>
          <span className="font-mono text-[11px] uppercase tracking-[0.1em] text-ink-faint">
            within dataset · not all-time
          </span>
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
          <Metric label="Race entries" value={c.races} />
          <Metric label="Wins" value={c.wins} accent={c.wins > 0} />
          <Metric label="Podiums" value={c.podiums} />
          <Metric label="Points" value={(c.points ?? 0).toFixed(0)} />
          <Metric label="Best" value={c.best ? `P${c.best}` : "—"} />
        </div>
      </div>

      <PointsBySeason data={p.by_season.map((s) => ({ label: String(s.season), points: s.points ?? 0 }))} />

      <Card className="p-4">
        <SectionTitle>Season by season</SectionTitle>
        <DataTable columns={seaCols} rows={p.by_season} getKey={(s) => s.season} />
      </Card>

      <Card className="p-4">
        <SectionTitle>Drivers who raced for {p.team}</SectionTitle>
        <DataTable columns={drvCols} rows={p.drivers} getKey={(d) => d.driver} highlightFirst />
      </Card>
      <Callout>Race entries count each car entered per race (two per weekend for a full-season team).</Callout>
    </div>
  );
}

// ============================ shared ========================================
function PointsBySeason({ data }: { data: { label: string; points: number }[] }) {
  return (
    <Card className="p-4">
      <SectionTitle>Points by season</SectionTitle>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 12, right: 12, bottom: 4, left: -12 }}>
          <XAxis dataKey="label" stroke="#9a9aa6" fontSize={11} />
          <YAxis stroke="#9a9aa6" fontSize={11} />
          <Bar dataKey="points" radius={[3, 3, 0, 0]}>
            {data.map((_, i) => (
              <Cell key={i} fill="#2dd4bf" />
            ))}
            <LabelList dataKey="points" position="top" formatter={(v: number) => v.toFixed(0)} fill="#ecedf0" fontSize={11} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Card>
  );
}
