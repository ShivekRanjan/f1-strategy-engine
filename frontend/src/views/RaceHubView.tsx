import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import { Column, DataTable } from "../components/DataTable";
import { Field, Select } from "../components/controls";
import { Badge, Callout, Card, ErrorNote, SectionTitle, Spinner } from "../components/ui";
import { beatsPick, clock, compoundColor, fmtPlan, pct, teamColor } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import type {
  DegradationResp,
  LapHistory,
  RaceCardResp,
  RecommendResp,
} from "../api/types";
import { TracksGate, ViewIntro, pickDefaultTrack } from "./common";

export default function RaceHubView() {
  return <TracksGate>{() => <Inner />}</TracksGate>;
}

function Inner() {
  // Season-first nav: pick a season, then only that season's circuits show.
  const [season, setSeason] = useState<number | null>(null);
  const [track, setTrack] = useState<string | null>(null);

  const seasons = useAsync(() => api.allSeasons(), []);
  useEffect(() => {
    if (seasons.data?.seasons?.length) setSeason(seasons.data.seasons.at(-1)!);
  }, [seasons.data]);

  const circuits = useAsync(
    () => (season == null ? Promise.resolve(null) : api.circuits(season)),
    [season],
  );
  useEffect(() => {
    const cs = circuits.data?.circuits;
    if (cs?.length) setTrack((prev) => (prev && cs.includes(prev) ? prev : pickDefaultTrack(cs)));
  }, [circuits.data]);

  return (
    <div className="space-y-5">
      <ViewIntro>
        One race, the whole picture — what actually happened, what the model{" "}
        <strong>predicted before it</strong>, the strategy the engine would have called, and the tyre
        behaviour behind it. Everything the other tabs compute, unified per race.
      </ViewIntro>

      <Card className="p-4">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <Field label="Season">
            <Select
              value={season ?? 0}
              options={seasons.data?.seasons ?? []}
              onChange={(v) => {
                setSeason(Number(v));
                setTrack(null);
              }}
              getLabel={(y) => (Number(y) >= 2026 ? `${y} · new regs` : String(y))}
            />
          </Field>
          <Field label="Circuit">
            <Select
              value={track ?? ""}
              options={circuits.data?.circuits ?? []}
              onChange={(v) => setTrack(String(v))}
            />
          </Field>
        </div>
      </Card>

      {season != null && track && <RaceHub key={`${track}|${season}`} track={track} season={season} />}
    </div>
  );
}

function RaceHub({ track, season }: { track: string; season: number }) {
  const info = useAsync(() => api.raceInfo(track), [track]);
  const card = useAsync(() => api.raceCard(season, track), [track, season]);

  return (
    <div className="space-y-5">
      {/* Race header strip */}
      <Card className="flex flex-wrap items-center gap-x-8 gap-y-3 p-4">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.12em] text-ink-faint">
            {season}
            {card.data ? ` · Round ${card.data.round}` : ""}
          </div>
          <div className="mt-0.5 text-xl font-700 text-ink">{track}</div>
        </div>
        {info.data && (
          <>
            <HeadStat label="Race laps" value={String(info.data.total_laps)} />
            <HeadStat label="Pit loss" value={`${info.data.pit_loss_s.toFixed(1)} s`} />
            <HeadStat label="SC / lap" value={pct(info.data.sc_prob_per_lap)} />
            {!info.data.well_sampled && <Badge tone="amber">sparse tyre data</Badge>}
          </>
        )}
      </Card>

      {card.error && <ErrorNote error={card.error} />}
      {!card.data && !card.error && <Spinner label="Loading the race card…" />}
      {card.data && <PodiumCompare card={card.data} />}
      {card.data && <ResultTable card={card.data} />}

      <StrategyCall track={track} season={season} />
      <Degradation track={track} season={season} />
      <PaceReplay track={track} season={season} />
    </div>
  );
}

function HeadStat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="font-mono text-[10px] uppercase tracking-[0.12em] text-ink-faint">{label}</div>
      <div className="nums mt-0.5 font-mono text-lg text-ink">{value}</div>
    </div>
  );
}

// --- Predicted (pre-race) vs actual podium — the star of the hub -------------
function PodiumCompare({ card }: { card: RaceCardResp }) {
  const pred = card.prediction;
  const predTop3 = pred?.predictions.slice(0, 3) ?? [];
  const actual = card.actual_podium;
  const predSet = new Set(predTop3.map((p) => p.driver));

  return (
    <Card className="border-l-2 border-l-accent p-4">
      <SectionTitle>
        Predicted vs actual podium
        {pred && (
          <span className="ml-2">
            <Badge tone={pred.hit_at_3 >= 2 ? "green" : pred.hit_at_3 === 1 ? "amber" : "red"}>
              {pred.hit_at_3} / 3 correct
            </Badge>
          </span>
        )}
      </SectionTitle>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div>
          <div className="mb-2 font-mono text-[11px] uppercase tracking-[0.1em] text-ink-faint">
            Model predicted · before the race
          </div>
          {pred ? (
            <ol className="space-y-1.5">
              {predTop3.map((p, i) => (
                <PodiumRow
                  key={p.driver}
                  place={i + 1}
                  driver={p.driver}
                  team={p.team}
                  right={pct(p.podium_prob)}
                  hit={p.actual}
                />
              ))}
            </ol>
          ) : (
            <p className="text-sm text-ink-muted">No prior-season data to predict this race.</p>
          )}
        </div>
        <div>
          <div className="mb-2 font-mono text-[11px] uppercase tracking-[0.1em] text-ink-faint">
            Actual result
          </div>
          <ol className="space-y-1.5">
            {actual.map((d, i) => (
              <PodiumRow key={d} place={i + 1} driver={d} team="" hit={predSet.has(d)} />
            ))}
          </ol>
        </div>
      </div>
      {pred && (
        <Callout>
          A genuine <em>forward</em> test: the podium model is trained only on seasons{" "}
          <em>before</em> {card.season} (ROC-AUC {pred.auc.toFixed(2)} over that test year), then
          applied here. A ✓ marks a driver the model had in its top-3.
        </Callout>
      )}
    </Card>
  );
}

function PodiumRow({
  place,
  driver,
  team,
  right,
  hit,
}: {
  place: number;
  driver: string;
  team: string;
  right?: string;
  hit?: boolean;
}) {
  return (
    <li className="flex items-center gap-2 rounded-md border border-line-card bg-surface-inset px-3 py-1.5">
      <span className="font-mono text-[11px] text-ink-faint">P{place}</span>
      {team && (
        <span className="inline-block h-3.5 w-1 rounded-sm" style={{ background: teamColor(team) }} />
      )}
      <span className="font-700 text-ink">{driver}</span>
      {team && <span className="text-xs text-ink-muted">{team}</span>}
      <span className="ml-auto flex items-center gap-2">
        {right && <span className="nums font-mono text-[12px] text-accent">{right}</span>}
        {hit && <span title="in the model's top-3">✓</span>}
      </span>
    </li>
  );
}

// --- Full finishing order ---------------------------------------------------
function ResultTable({ card }: { card: RaceCardResp }) {
  const cols: Column<RaceCardResp["result"][number]>[] = [
    {
      key: "pos",
      header: "Pos",
      align: "right",
      render: (r) => (r.pos == null ? <span className="text-ink-faint">DNF</span> : r.pos),
    },
    { key: "driver", header: "Driver", render: (r) => <span className="font-700">{r.driver}</span> },
    {
      key: "team",
      header: "Team",
      render: (r) => (
        <span className="inline-flex items-center gap-2 text-ink-muted">
          <span className="inline-block h-3.5 w-1 rounded-sm" style={{ background: teamColor(r.team) }} />
          {r.team}
        </span>
      ),
    },
    { key: "grid", header: "Grid", align: "right", render: (r) => r.grid ?? "—" },
    {
      key: "gained",
      header: "+/−",
      align: "right",
      render: (r) =>
        r.gained == null ? (
          <span className="text-ink-faint">—</span>
        ) : (
          <span className={r.gained > 0 ? "text-emerald-400" : r.gained < 0 ? "text-soft" : "text-ink-muted"}>
            {r.gained > 0 ? `+${r.gained}` : r.gained}
          </span>
        ),
    },
    { key: "points", header: "Pts", align: "right", render: (r) => (r.points ? r.points.toFixed(0) : <span className="text-ink-faint">0</span>) },
    { key: "status", header: "Status", render: (r) => <span className="text-xs text-ink-muted">{r.status}</span> },
  ];
  return (
    <Card className="p-4">
      <SectionTitle>Finishing order</SectionTitle>
      <div className="max-h-96 overflow-y-auto">
        <DataTable columns={cols} rows={card.result} getKey={(r) => r.driver} highlightFirst />
      </div>
    </Card>
  );
}

// --- The engine's strategy call for this circuit ----------------------------
function StrategyCall({ track, season }: { track: string; season: number }) {
  const rec = useAsync(
    () => api.recommend({ track, season, n_runs: 1500, top_k: 4 }),
    [track, season],
  );
  return (
    <Card className="p-4">
      <SectionTitle>Optimal strategy — what the engine would call</SectionTitle>
      {rec.error && <ErrorNote error={rec.error} />}
      {!rec.data && !rec.error && <Spinner label="Optimising strategy (Monte-Carlo)…" />}
      {rec.data && <StrategyBody rec={rec.data} />}
    </Card>
  );
}

function StrategyBody({ rec }: { rec: RecommendResp }) {
  const cols: Column<RecommendResp["shortlist"][number]>[] = [
    { key: "rank", header: "#", render: (r) => r.rank },
    { key: "plan", header: "Plan", render: (r) => fmtPlan(r.compounds, r.pit_laps) },
    { key: "time", header: "Race time", align: "right", render: (r) => clock(r.mean_s) },
    {
      key: "cmp",
      header: "Compares",
      render: (r) => (
        <span className={r.rank === 1 ? "text-f1" : "text-ink-muted"}>
          {beatsPick(r.rank, r.win_prob_vs_best)}
        </span>
      ),
    },
  ];
  return (
    <>
      <div className="mb-1 text-lg font-700 text-ink">{fmtPlan(rec.best.compounds, rec.best.pit_laps)}</div>
      <div className="mb-3 text-xs text-ink-muted">
        evaluated {rec.n_evaluated} strategies · median race {clock(rec.best.p50_s)}
      </div>
      <DataTable columns={cols} rows={rec.shortlist} highlightFirst getKey={(r) => r.rank} />
    </>
  );
}

// --- Tyre degradation curves ------------------------------------------------
function Degradation({ track, season }: { track: string; season: number }) {
  const deg = useAsync(() => api.degradation(track, season, true), [track, season]);
  return (
    <Card className="p-4">
      <SectionTitle>Tyre degradation — pace lost vs a fresh tyre</SectionTitle>
      {deg.error && <ErrorNote error={deg.error} />}
      {!deg.data && !deg.error && <Spinner label="Loading degradation curves…" />}
      {deg.data && <DegChart deg={deg.data} />}
    </Card>
  );
}

function DegChart({ deg }: { deg: DegradationResp }) {
  const comps = Object.keys(deg.compounds);
  const maxAge = Math.max(...comps.map((c) => deg.compounds[c].max_age));
  const data: Record<string, number | null>[] = [];
  for (let age = 0; age <= maxAge; age++) {
    const row: Record<string, number | null> = { age };
    for (const c of comps) {
      const cv = deg.compounds[c];
      row[c] = age <= cv.max_age ? cv.cliff[age] : null;
    }
    data.push(row);
  }
  return (
    <>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -12 }}>
          <CartesianGrid stroke="#2c2c36" vertical={false} />
          <XAxis dataKey="age" stroke="#9a9aa6" fontSize={11} label={{ value: "tyre age (laps)", position: "insideBottom", offset: -2, fill: "#9a9aa6", fontSize: 11 }} />
          <YAxis stroke="#9a9aa6" fontSize={11} tickFormatter={(v) => `+${v.toFixed(1)}`} unit="s" width={52} />
          <Tooltip
            labelFormatter={(v) => `age ${v} laps`}
            formatter={(v: number, n: string) => [`+${v.toFixed(2)} s`, n]}
          />
          {comps.map((c) => (
            <Line key={c} type="monotone" dataKey={c} stroke={compoundColor(c)} strokeWidth={1.8} dot={false} isAnimationActive={false} connectNulls />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-1 flex gap-4">
        {comps.map((c) => (
          <span key={c} className="flex items-center gap-1.5 text-xs text-ink-muted">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: compoundColor(c) }} />
            {c}
          </span>
        ))}
      </div>
    </>
  );
}

// --- Lap-pace replay (per driver) -------------------------------------------
function PaceReplay({ track, season }: { track: string; season: number }) {
  const [driver, setDriver] = useState<string | null>(null);
  const drivers = useAsync(() => api.drivers(track, season), [track, season]);
  useEffect(() => {
    const ds = drivers.data?.drivers;
    if (ds?.length) setDriver((prev) => (prev && ds.includes(prev) ? prev : ds.includes("VER") ? "VER" : ds[0]));
  }, [drivers.data]);

  return (
    <Card className="p-4">
      <SectionTitle>Lap-pace trace — how the race unfolded</SectionTitle>
      <div className="mb-3 max-w-xs">
        <Field label="Driver">
          <Select
            value={driver ?? ""}
            options={drivers.data?.drivers ?? []}
            onChange={(v) => setDriver(String(v))}
          />
        </Field>
      </div>
      {driver && <PaceTrace key={driver} track={track} season={season} driver={driver} />}
    </Card>
  );
}

function PaceTrace({ track, season, driver }: { track: string; season: number; driver: string }) {
  const laps = useAsync(() => api.laps(track, season, driver), [track, season, driver]);
  if (laps.error) return <ErrorNote error={laps.error} />;
  if (!laps.data) return <Spinner label="Loading laps…" />;
  return <PaceTraceInner hist={laps.data} />;
}

function PaceTraceInner({ hist }: { hist: LapHistory }) {
  const series = hist.laps
    .filter((l) => l.lap_time_fuel_corr_s != null)
    .map((l) => ({ lap: l.lap, t: l.lap_time_fuel_corr_s as number, compound: l.compound }));
  // Pit laps = where the compound changes between consecutive laps.
  const pits: number[] = [];
  for (let i = 1; i < hist.laps.length; i++) {
    if (hist.laps[i].compound !== hist.laps[i - 1].compound) pits.push(hist.laps[i].lap);
  }
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={series} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
        <CartesianGrid stroke="#2c2c36" vertical={false} />
        <XAxis dataKey="lap" stroke="#9a9aa6" fontSize={11} type="number" domain={["dataMin", "dataMax"]} />
        <YAxis stroke="#9a9aa6" fontSize={11} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(0)} width={40} />
        <Tooltip
          labelFormatter={(v) => `lap ${v}`}
          formatter={(v: number) => [`${v.toFixed(2)} s`, "fuel-corr"]}
        />
        {pits.map((p) => (
          <ReferenceLine key={p} x={p} stroke="#2dd4bf" strokeDasharray="3 3" strokeOpacity={0.6} />
        ))}
        <Line type="monotone" dataKey="t" stroke="#ecedf0" strokeWidth={1.6} dot={false} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
