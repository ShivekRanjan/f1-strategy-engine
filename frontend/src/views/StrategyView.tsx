import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import { Column, DataTable } from "../components/DataTable";
import { Field, Segmented, Select, Slider } from "../components/controls";
import { Badge, Callout, Card, ErrorNote, Metric, SectionTitle, Spinner } from "../components/ui";
import { beatsPick, clock, fmtPlan } from "../lib/format";
import { useAsync, useDebounced } from "../lib/useAsync";
import type { ShortlistRow } from "../api/types";
import { TracksGate, ViewIntro, pickDefaultTrack } from "./common";

const OBJECTIVES = [
  { value: "mean", label: "Expected" },
  { value: "median", label: "Median" },
  { value: "p85", label: "Risk-averse" },
];

export default function StrategyView() {
  return <TracksGate>{(tracks) => <Inner tracks={tracks} />}</TracksGate>;
}

function Inner({ tracks }: { tracks: string[] }) {
  const [track, setTrack] = useState(() => pickDefaultTrack(tracks));
  const [season, setSeason] = useState<number | null>(null);
  const [objective, setObjective] = useState("mean");
  const [maxStops, setMaxStops] = useState(2);

  const seasons = useAsync(() => api.seasons(track), [track]);
  // When the circuit changes, default Season to its most recent year (2026 if present).
  useEffect(() => {
    if (seasons.data?.seasons?.length) setSeason(seasons.data.seasons.at(-1)!);
  }, [seasons.data]);

  const stops = useDebounced(maxStops, 200);
  const rec = useAsync(
    () =>
      season == null
        ? Promise.resolve(null)
        : api.recommend({ track, season, objective, max_stops: stops, n_runs: 2000, top_k: 5 }),
    [track, season, objective, stops],
  );
  const best = rec.data?.best;
  const sim = useAsync(
    () =>
      best && season != null
        ? api.simulate({ track, compounds: best.compounds, pit_laps: best.pit_laps, season, n_runs: 4000 })
        : Promise.resolve(null),
    [track, season, best?.compounds.join(), best?.pit_laps.join()],
  );

  const is2026 = season != null && season >= 2026;

  return (
    <div className="space-y-5">
      <ViewIntro>
        Searches ~1,000+ pit strategies via Monte Carlo (per-circuit safety cars) and ranks them
        with the honest spread — a <em>typical</em> race vs a <em>bad-luck</em> one.
      </ViewIntro>

      <Card className="p-4">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Field label="Circuit">
            <Select value={track} options={tracks} onChange={setTrack} />
          </Field>
          <Field label="Season">
            <Select
              value={season ?? 0}
              options={seasons.data?.seasons ?? []}
              onChange={(v) => setSeason(Number(v))}
              getLabel={(y) => (Number(y) >= 2026 ? `${y} · new regs` : String(y))}
            />
          </Field>
          <Field label="Objective">
            <div className="pt-0.5">
              <Segmented value={objective} options={OBJECTIVES} onChange={setObjective} />
            </div>
          </Field>
          <Field label={`Max stops · ${maxStops}`}>
            <Slider value={maxStops} min={1} max={3} onChange={setMaxStops} display={`${maxStops} stop${maxStops > 1 ? "s" : ""}`} />
          </Field>
        </div>
        {is2026 && (
          <div className="mt-3">
            <Badge tone="red">🆕 2026 mode</Badge>{" "}
            <span className="text-xs text-ink-muted">
              New-regulation cars — degradation blends {season} data with the pre-2026 prior, so
              early-season numbers are necessarily uncertain.
            </span>
          </div>
        )}
      </Card>

      {rec.error && <ErrorNote error={rec.error} />}
      {!rec.data && !rec.error && <Spinner label="Searching strategies…" />}

      {rec.data && best && (
        <>
          {!rec.data.well_sampled && (
            <Callout tone="warn">
              Limited data for this circuit — a compound lacks fitted per-track pace, so its
              predictions are rougher (still realistic).
            </Callout>
          )}

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <Metric label="Race distance" value={`${rec.data.total_laps} laps`} />
            <Metric
              label="Safety-car chance"
              value={`${Math.round((1 - (1 - rec.data.sc_prob_per_lap) ** rec.data.total_laps) * 100)}%`}
              title="Probability of at least one safety car this race (calibrated per circuit)."
            />
            <Metric label="Pit loss" value={`${rec.data.pit_loss_s.toFixed(1)} s`} />
          </div>

          <Card className="border-l-2 border-l-f1 p-4">
            <SectionTitle>Recommended strategy</SectionTitle>
            <div className="text-lg font-700 text-ink">{fmtPlan(best.compounds, best.pit_laps)}</div>
            <div className="nums mt-1 text-sm text-ink-muted">
              expected <b className="text-ink">{clock(best.mean_s)}</b> · typical {clock(best.p50_s)} ·
              bad luck {clock(best.p90_s)}
            </div>
            <div className="mt-1 text-xs text-ink-muted">
              searched {rec.data.n_evaluated} strategies · objective: {objective}
            </div>
            <ClearCutNote shortlist={rec.data.shortlist} />
          </Card>

          <div className="grid grid-cols-1 gap-5 lg:grid-cols-5">
            <Card className="p-4 lg:col-span-3">
              <SectionTitle>Closest alternatives</SectionTitle>
              <ShortlistTable rows={rec.data.shortlist} />
            </Card>
            <Card className="p-4 lg:col-span-2">
              <SectionTitle>Outcome distribution</SectionTitle>
              {sim.data ? <DistChart sim={sim.data} /> : <Spinner label="Simulating…" />}
            </Card>
          </div>
        </>
      )}
    </div>
  );
}

function ClearCutNote({ shortlist }: { shortlist: ShortlistRow[] }) {
  if (shortlist.length < 2) return null;
  const runnerUp = shortlist[1].win_prob_vs_best;
  const spread = Math.max(...shortlist.map((r) => r.mean_s)) - Math.min(...shortlist.map((r) => r.mean_s));
  return (
    <div className="mt-3">
      {runnerUp >= 0.3 ? (
        <Callout>
          The top plans are near-tied (within ~{spread.toFixed(0)}s) — the call is robust; safety-car
          timing matters more than which you pick.
        </Callout>
      ) : (
        <Callout tone="success">
          The pick is clear-cut — the runner-up wins only {Math.round(runnerUp * 100)}% of simulated
          races.
        </Callout>
      )}
    </div>
  );
}

function ShortlistTable({ rows }: { rows: ShortlistRow[] }) {
  const cols: Column<ShortlistRow>[] = [
    { key: "rank", header: "#", render: (r) => r.rank },
    { key: "plan", header: "Plan", render: (r) => fmtPlan(r.compounds, r.pit_laps) },
    { key: "exp", header: "Expected", align: "right", render: (r) => clock(r.mean_s) },
    { key: "typ", header: "Typical", align: "right", render: (r) => clock(r.p50_s) },
    { key: "bad", header: "Bad luck", align: "right", render: (r) => clock(r.p90_s) },
    {
      key: "cmp",
      header: "How it compares",
      render: (r) => (
        <span className={r.rank === 1 ? "text-f1" : "text-ink-muted"}>
          {beatsPick(r.rank, r.win_prob_vs_best)}
        </span>
      ),
    },
  ];
  return <DataTable columns={cols} rows={rows} highlightFirst getKey={(r) => r.rank} />;
}

function DistChart({ sim }: { sim: import("../api/types").SimulateResp }) {
  const edges = sim.hist_edges.map((e) => e / 60);
  const data = sim.hist_counts.map((c, i) => ({
    x: (edges[i] + edges[i + 1]) / 2,
    c,
  }));
  return (
    <>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -16 }}>
          <CartesianGrid stroke="#2c2c36" vertical={false} />
          <XAxis
            dataKey="x"
            tickFormatter={(v) => v.toFixed(0)}
            stroke="#9a9aa6"
            fontSize={11}
            label={{ value: "race time (min)", position: "insideBottom", offset: -2, fill: "#9a9aa6", fontSize: 11 }}
          />
          <YAxis stroke="#9a9aa6" fontSize={11} />
          <Tooltip
            cursor={{ fill: "#ffffff10" }}
            labelFormatter={(v) => `${Number(v).toFixed(1)} min`}
            formatter={(v) => [v, "races"]}
          />
          <ReferenceLine x={sim.p50_s / 60} stroke="#ecedf0" strokeDasharray="4 3" />
          <Bar dataKey="c" radius={[2, 2, 0, 0]}>
            {data.map((_, i) => (
              <Cell key={i} fill="#e2231a" />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="nums mt-1 text-xs text-ink-muted">
        P(safety car) = {Math.round(sim.p_safety_car * 100)}% · spread (p90−p10) ={" "}
        {(sim.p90_s - sim.p10_s).toFixed(0)}s
      </div>
    </>
  );
}
