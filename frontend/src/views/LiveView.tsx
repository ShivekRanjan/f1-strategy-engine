import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import { Column, DataTable } from "../components/DataTable";
import { Field, Select, Slider } from "../components/controls";
import { Badge, Callout, Card, ErrorNote, Metric, SectionTitle, Spinner } from "../components/ui";
import { beatsPick, clock, secs } from "../lib/format";
import { useAsync, useDebounced } from "../lib/useAsync";
import type { LapHistory, LiveRecommendation, Nowcast } from "../api/types";
import { TracksGate, ViewIntro, pickDefaultTrack } from "./common";

export default function LiveView() {
  return <TracksGate>{(tracks) => <Inner tracks={tracks} />}</TracksGate>;
}

function Inner({ tracks: _tracks }: { tracks: string[] }) {
  // Season-first navigation: pick a season, then only that season's circuits show.
  const [season, setSeason] = useState<number | null>(null);
  const [track, setTrack] = useState<string | null>(null);
  const [driver, setDriver] = useState<string | null>(null);

  const seasons = useAsync(() => api.allSeasons(), []);
  useEffect(() => {
    if (seasons.data?.seasons?.length) setSeason(seasons.data.seasons.at(-1)!);
  }, [seasons.data]);

  const circuits = useAsync(
    () => (season == null ? Promise.resolve(null) : api.circuits(season)),
    [season],
  );
  // When the season changes, default the circuit to a sensible one in that season.
  useEffect(() => {
    const cs = circuits.data?.circuits;
    if (cs?.length) setTrack((prev) => (prev && cs.includes(prev) ? prev : pickDefaultTrack(cs)));
  }, [circuits.data]);

  const drivers = useAsync(
    () => (season == null || track == null ? Promise.resolve(null) : api.drivers(track, season)),
    [track, season],
  );
  useEffect(() => {
    const ds = drivers.data?.drivers;
    if (ds?.length) setDriver(ds.includes("VER") ? "VER" : ds[0]);
  }, [drivers.data]);

  return (
    <div className="space-y-5">
      <ViewIntro>
        Replay any race lap by lap — the engine re-optimises the remaining strategy from the current
        state each lap (the same call a live-timing feed would drive on race day).
      </ViewIntro>
      <Card className="p-4">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <Field label="Season">
            <Select
              value={season ?? 0}
              options={seasons.data?.seasons ?? []}
              onChange={(v) => {
                setSeason(Number(v));
                setTrack(null); // reset so the new season's default circuit is picked
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
          <Field label="Driver">
            <Select
              value={driver ?? ""}
              options={drivers.data?.drivers ?? []}
              onChange={(v) => setDriver(String(v))}
            />
          </Field>
        </div>
      </Card>
      {season != null && track && driver && (
        <Replay key={`${track}|${season}|${driver}`} track={track} season={season} driver={driver} />
      )}
    </div>
  );
}

function Replay({ track, season, driver }: { track: string; season: number; driver: string }) {
  const laps = useAsync(() => api.laps(track, season, driver), [track, season, driver]);
  if (laps.error) return <ErrorNote error={laps.error} />;
  if (!laps.data) return <Spinner label="Loading laps…" />;
  return <ReplayInner hist={laps.data} track={track} season={season} driver={driver} />;
}

function ReplayInner({
  hist,
  track,
  season,
  driver,
}: {
  hist: LapHistory;
  track: string;
  season: number;
  driver: string;
}) {
  const [cur, setCur] = useState(Math.min(hist.lap_max, Math.max(2, Math.floor(hist.lap_max / 3))));
  const curD = useDebounced(cur, 200);
  const live = useAsync(
    () => api.live({ track, season, driver, current_lap: curD, n_runs: 2000 }),
    [track, season, driver, curD],
  );

  const series = hist.laps
    .filter((l) => l.lap_time_fuel_corr_s != null)
    .map((l) => ({ lap: l.lap, t: l.lap_time_fuel_corr_s as number }));
  const nowcast = live.data?.nowcast ?? null;

  return (
    <div className="space-y-5">
      <Card className="p-4">
        <Field label={`Current lap · drag to play the race`}>
          <Slider
            value={cur}
            min={hist.lap_min}
            max={hist.lap_max}
            onChange={setCur}
            display={`lap ${cur} of ${hist.total_laps}`}
          />
        </Field>
      </Card>

      {live.error && <ErrorNote error={live.error} />}
      {!live.data && !live.error && <Spinner label="Re-optimising from current state…" />}

      {live.data && (
        <>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Metric label="Lap" value={`${live.data.state.current_lap} / ${live.data.state.total_laps}`} />
            <Metric label="Current tyre" value={live.data.state.current_compound} />
            <Metric label="Tyre age" value={`${live.data.state.tyre_age} laps`} />
            <Metric label="Laps remaining" value={live.data.state.laps_remaining} />
          </div>

          <div className="grid grid-cols-1 gap-5 lg:grid-cols-5">
            <Card className="p-4 lg:col-span-3">
              <SectionTitle>Lap pace (fuel-corrected)</SectionTitle>
              <PaceChart series={series} cur={cur} nowcast={nowcast} />
            </Card>
            <div className="space-y-5 lg:col-span-2">
              {live.data.recommendation ? (
                <RecCard rec={live.data.recommendation} usedCount={live.data.state.compounds_used.length} />
              ) : (
                <Callout>Race effectively over — nothing left to optimise.</Callout>
              )}
              <NowcastCard nowcast={nowcast} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function RecCard({ rec, usedCount }: { rec: LiveRecommendation; usedCount: number }) {
  const cols: Column<(typeof rec.shortlist)[number]>[] = [
    { key: "rank", header: "#", render: (r) => r.rank },
    { key: "plan", header: "Plan", render: (r) => r.plan },
    { key: "left", header: "Time left", align: "right", render: (r) => clock(r.mean_remaining_s) },
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
    <Card className="border-l-2 border-l-f1 p-4">
      <SectionTitle>Recommended from here</SectionTitle>
      <div className="mb-1 text-lg font-700 text-ink">{rec.best_plan}</div>
      <div className="mb-3 text-xs text-ink-muted">
        {usedCount} compound{usedCount === 1 ? "" : "s"} used · evaluated {rec.n_evaluated} remaining plans
      </div>
      <DataTable columns={cols} rows={rec.shortlist} highlightFirst getKey={(r) => r.rank} />
    </Card>
  );
}

function NowcastCard({ nowcast }: { nowcast: Nowcast | null }) {
  if (!nowcast) return null;
  return (
    <Card className="p-4">
      <SectionTitle>🔮 Next-lap nowcast · LSTM</SectionTitle>
      {!nowcast.ok ? (
        <p className="text-sm text-ink-muted">{nowcast.reason}</p>
      ) : (
        <>
          <div className="flex items-end justify-between">
            <Metric
              label="Predicted next lap"
              value={secs(nowcast.predicted_s!)}
              sub={`${nowcast.delta_s! >= 0 ? "+" : ""}${nowcast.delta_s!.toFixed(2)} s vs last lap`}
            />
            <Badge tone={nowcast.delta_s! > 0.05 ? "red" : nowcast.delta_s! < -0.05 ? "green" : "neutral"}>
              {nowcast.delta_s! > 0.05 ? "🔻 fading" : nowcast.delta_s! < -0.05 ? "🔺 improving" : "➡ steady"}
            </Badge>
          </div>
          <p className="mt-2 text-xs text-ink-muted">
            From the recent lap <em>sequence</em> if you stay out. The dumb baseline just repeats the
            last lap ({secs(nowcast.last_s!)}); on held-out 2025 this LSTM beat that by ~8.5%.
          </p>
        </>
      )}
    </Card>
  );
}

function PaceChart({
  series,
  cur,
  nowcast,
}: {
  series: { lap: number; t: number }[];
  cur: number;
  nowcast: Nowcast | null;
}) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={series} margin={{ top: 8, right: 12, bottom: 4, left: -12 }}>
        <CartesianGrid stroke="#2c2c36" vertical={false} />
        <XAxis dataKey="lap" stroke="#9a9aa6" fontSize={11} type="number" domain={["dataMin", "dataMax"]} />
        <YAxis stroke="#9a9aa6" fontSize={11} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(0)} />
        <Tooltip
          labelFormatter={(v) => `lap ${v}`}
          formatter={(v: number) => [`${v.toFixed(2)} s`, "fuel-corr"]}
        />
        <ReferenceLine x={cur} stroke="#e2231a" strokeDasharray="4 3" />
        <Line type="monotone" dataKey="t" stroke="#ecedf0" strokeWidth={1.6} dot={false} isAnimationActive={false} />
        {nowcast?.ok && (
          <ReferenceDot x={cur + 1} y={nowcast.predicted_s!} r={4} fill="#38bdf8" stroke="#0b0b0d" />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}
