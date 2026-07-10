import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import { Field, Segmented, Select, Slider } from "../components/controls";
import { Callout, Card, CardSkeleton, ErrorNote, SectionTitle, Spinner } from "../components/ui";
import { beatsPick, clock, compoundColor, trackSearchText } from "../lib/format";
import { useAsync, useDebounced } from "../lib/useAsync";
import type {
  DegradationResp,
  RecommendResp,
  SimulateResp,
  StrategySummary,
  TrackInfo,
} from "../api/types";
import { pickDefaultTrack } from "./common";

const COMPS = ["SOFT", "MEDIUM", "HARD"] as const;
const OBJECTIVES = [
  { value: "mean", label: "Fastest" },
  { value: "median", label: "Median" },
  { value: "p85", label: "Risk-averse" },
];

export default function StrategyView() {
  const ti = useAsync(() => api.tracksInfo(), []);
  if (ti.loading) return <Spinner label="Loading circuits…" />;
  if (ti.error) return <ErrorNote error={ti.error} />;
  if (!ti.data?.tracks?.length) return <Callout>No circuits found.</Callout>;
  return <Dashboard tracks={ti.data.tracks} />;
}

function Dashboard({ tracks }: { tracks: TrackInfo[] }) {
  const [track, setTrack] = useState(() => pickDefaultTrack(tracks.map((t) => t.track)));
  const [season, setSeason] = useState<number | null>(null);
  const [objective, setObjective] = useState("mean");
  const [maxStops, setMaxStops] = useState(2);
  const [cliff, setCliff] = useState(true);
  const [trackTemp, setTrackTemp] = useState(35);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const seasons = useAsync(() => api.seasons(track), [track]);
  useEffect(() => {
    if (seasons.data?.seasons?.length) setSeason(seasons.data.seasons.at(-1)!);
  }, [seasons.data]);

  const stops = useDebounced(maxStops, 200);
  const temp = useDebounced(trackTemp, 250);
  const rec = useAsync(
    () =>
      season == null
        ? Promise.resolve(null)
        : api.recommend({ track, season, objective, max_stops: stops, use_cliff: cliff,
                          track_temp: temp, n_runs: 2000, top_k: 7 }),
    [track, season, objective, stops, cliff, temp],
  );
  const best = rec.data?.best;
  const sim = useAsync(
    () =>
      best && season != null
        ? api.simulate({ track, compounds: best.compounds, pit_laps: best.pit_laps, season, use_cliff: cliff, n_runs: 4000 })
        : Promise.resolve(null),
    [track, season, cliff, best?.compounds.join(), best?.pit_laps.join()],
  );
  const deg = useAsync(
    () => (season == null ? Promise.resolve(null) : api.degradation(track, season, cliff)),
    [track, season, cliff],
  );

  return (
    <div className="grid gap-5 lg:grid-cols-[236px_1fr]">
      <Rail
        tracks={tracks}
        track={track}
        setTrack={setTrack}
        seasons={seasons.data?.seasons ?? []}
        season={season}
        setSeason={setSeason}
        objective={objective}
        setObjective={setObjective}
        maxStops={maxStops}
        setMaxStops={setMaxStops}
        cliff={cliff}
        setCliff={setCliff}
        trackTemp={trackTemp}
        setTrackTemp={setTrackTemp}
        showAdvanced={showAdvanced}
        setShowAdvanced={setShowAdvanced}
        info={rec.data}
      />
      <div className="space-y-5">
        {rec.error && <ErrorNote error={rec.error} />}
        {!rec.data && !rec.error && <CardSkeleton label="Searching strategies…" height={380} />}
        {rec.data && best && (
          <>
            {rec.loading && (
              <div className="flex items-center gap-2 rounded-lg border border-accent/30 bg-accent/5 px-3 py-2">
                <Spinner label="Recalculating for the new selection… (showing the previous result meanwhile)" />
              </div>
            )}
            <div className={rec.loading ? "opacity-40 transition-opacity" : "transition-opacity"}>
              <div className="space-y-5">
                <RecommendationBanner rec={rec.data} best={best} />
                <div className="grid gap-5 xl:grid-cols-2">
                  <DegradationCard deg={deg.data} loading={deg.loading} cliff={cliff} />
                  <OutcomeCard sim={sim.data} />
                </div>
                <ShortlistCard rec={rec.data} />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- //
function Rail(props: {
  tracks: TrackInfo[];
  track: string;
  setTrack: (t: string) => void;
  seasons: number[];
  season: number | null;
  setSeason: (s: number) => void;
  objective: string;
  setObjective: (o: string) => void;
  maxStops: number;
  setMaxStops: (n: number) => void;
  cliff: boolean;
  setCliff: (b: boolean) => void;
  trackTemp: number;
  setTrackTemp: (n: number) => void;
  showAdvanced: boolean;
  setShowAdvanced: (b: boolean) => void;
  info: RecommendResp | null | undefined;
}) {
  const p = props;
  const pSC = p.info ? 1 - (1 - p.info.sc_prob_per_lap) ** p.info.total_laps : null;
  // Filter the circuit list as you type — 24+ circuits is too many to scroll.
  const [circuitQ, setCircuitQ] = useState("");
  const q = circuitQ.trim().toLowerCase();
  const shown = q
    ? p.tracks.filter(
        (t) =>
          trackSearchText(t.track).includes(q) || shortName(t.track).toLowerCase().includes(q),
      )
    : p.tracks;
  return (
    <Card className="h-fit space-y-5 p-4">
      <div>
        <SectionTitle>Circuit</SectionTitle>
        <input
          value={circuitQ}
          onChange={(e) => setCircuitQ(e.target.value)}
          placeholder="Search circuits…"
          className="mb-2 w-full rounded-lg border border-line bg-carbon-700 px-3 py-2 text-sm text-ink outline-none transition placeholder:text-ink-dim focus:border-f1/60 focus:ring-1 focus:ring-f1/40"
        />
        <div className="max-h-60 space-y-1.5 overflow-y-auto pr-1">
          {shown.length === 0 && (
            <p className="px-1 py-2 text-sm text-ink-muted">No circuit matches “{circuitQ}”.</p>
          )}
          {shown.map((t) => {
            const active = t.track === p.track;
            return (
              <button
                key={t.track}
                onClick={() => p.setTrack(t.track)}
                className={`block w-full rounded-lg border px-3 py-2 text-left transition ${
                  active
                    ? "border-accent bg-accent/10 text-ink"
                    : "border-line-ctl bg-surface-inset text-ink-muted hover:border-line-hover"
                }`}
              >
                <div className="text-[13px] font-600">{shortName(t.track)}</div>
                <div className="font-mono text-[11px] text-ink-dim">
                  {t.total_laps} laps{!t.well_sampled && " · limited data"}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      <Field label="Season">
        <Select
          value={p.season ?? 0}
          options={p.seasons}
          onChange={(v) => p.setSeason(Number(v))}
          getLabel={(y) => (Number(y) >= 2026 ? `${y} · new regs` : String(y))}
        />
      </Field>

      <div>
        <SectionTitle>Objective</SectionTitle>
        <Segmented value={p.objective} options={OBJECTIVES} onChange={p.setObjective} />
        {/* Immediate feedback: say what the choice *does*, in outcome terms. */}
        <p className="mt-1.5 text-[11px] leading-snug text-ink-muted">
          {p.objective === "mean" && "Optimises the average race — fastest overall, exposed to bad luck."}
          {p.objective === "median" && "Optimises the typical race — ignores freak outcomes either way."}
          {p.objective === "p85" && "Protects the bad-luck tail — trades a little pace for safety."}
        </p>
      </div>

      <Field label={`Max stops · ${p.maxStops}`}>
        <Slider value={p.maxStops} min={1} max={3} onChange={p.setMaxStops}
                display={`${p.maxStops} stop${p.maxStops > 1 ? "s" : ""}`} />
      </Field>

      <Field label={`Track temp · ${p.trackTemp}°C`}>
        <Slider value={p.trackTemp} min={15} max={55} onChange={p.setTrackTemp}
                display={p.trackTemp < 30 ? `${p.trackTemp}°C · cool → less wear`
                  : p.trackTemp > 42 ? `${p.trackTemp}°C · hot → more wear`
                  : `${p.trackTemp}°C`} />
      </Field>

      {/* Progressive disclosure (Hick's law): expert knobs and model internals
          stay one click away, so the first-time reader faces four decisions,
          not seven. State is preserved either way. */}
      <div className="border-t border-line pt-2">
        <button
          onClick={() => p.setShowAdvanced(!p.showAdvanced)}
          className="flex w-full items-center justify-between py-1 font-mono text-[11px] uppercase tracking-[0.12em] text-ink-dim transition hover:text-ink-soft"
        >
          <span>Advanced · model assumptions</span>
          <span>{p.showAdvanced ? "−" : "+"}</span>
        </button>

        {p.showAdvanced && (
          <div className="mt-2 space-y-3">
            <button
              onClick={() => p.setCliff(!p.cliff)}
              className={`flex w-full items-center justify-between rounded-lg border px-3 py-2.5 text-left transition ${
                p.cliff ? "border-accent/55 bg-surface-inset" : "border-line-ctl bg-surface-inset"
              }`}
            >
              <span>
                <span className="block text-[12.5px] font-600 text-ink-soft">Cliff prior</span>
                <span className="font-mono text-[11px] text-ink-dim">domain assumption</span>
              </span>
              <span className={`rounded px-1.5 py-0.5 font-mono text-[11px] font-600 ${
                p.cliff ? "bg-accent/20 text-accent" : "bg-surface-inset2 text-ink-dim"
              }`}>
                {p.cliff ? "ON" : "OFF"}
              </span>
            </button>

            {p.info && (
              <div className="space-y-1 font-mono text-[11px] text-ink-faint">
                <div>pit loss · {p.info.pit_loss_s.toFixed(1)}s (measured)</div>
                <div>P(safety car) · {pSC != null ? Math.round(pSC * 100) : "–"}%</div>
                <div>searched {p.info.n_evaluated} strategies</div>
              </div>
            )}
          </div>
        )}
      </div>

    </Card>
  );
}

// --------------------------------------------------------------------------- //
function RecommendationBanner({ rec, best }: { rec: RecommendResp; best: StrategySummary }) {
  const runnerUp = rec.shortlist[1]?.win_prob_vs_best ?? 0;
  const confidence = Math.round((1 - runnerUp) * 100);
  const softLaps = stintLaps(best, rec.total_laps).find((s) => s.compound === "SOFT")?.laps;
  return (
    <Card className="overflow-hidden border-line-card">
      <div className="flex flex-wrap">
        <div className="min-w-[280px] flex-1 border-line p-5 sm:border-r">
          <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-accent">
            ◆ Recommended plan <span className="text-ink-dim">· {rec.track}</span>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            {best.compounds.map((c, i) => (
              <span key={i} className="flex items-center">
                <CompoundChip compound={c} />
                {i < best.compounds.length - 1 && <span className="px-1.5 text-ink-fainter">→</span>}
              </span>
            ))}
          </div>
          <div className="mt-3 font-mono text-[12px] text-ink-muted">
            {best.pit_laps.length ? `pit lap ${best.pit_laps.join(", ")} · ` : ""}
            {best.pit_laps.length}-stop{softLaps ? ` · ${softLaps} laps on soft` : ""}
          </div>
        </div>
        <div className="grid flex-1 grid-cols-2">
          <KpiCell label="Expected race" value={clock(best.mean_s)} />
          <KpiCell label="Typical (p50)" value={clock(best.p50_s)} />
          <KpiCell label="Bad luck (p90)" value={clock(best.p90_s)} />
          <KpiCell label="Pick confidence" value={`${confidence}%`} accent />
        </div>
      </div>
      <StintTimeline best={best} total={rec.total_laps} />
    </Card>
  );
}

function KpiCell({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="border-b border-l border-line p-4">
      <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-ink-faint">{label}</div>
      <div className={`nums mt-1 font-mono text-[22px] ${accent ? "text-accent" : "text-ink"}`}>{value}</div>
    </div>
  );
}

function StintTimeline({ best, total }: { best: StrategySummary; total: number }) {
  const stints = stintLaps(best, total);
  return (
    <div className="border-t border-line bg-surface-strip p-5">
      <div className="mb-2 flex justify-between font-mono text-[11px] uppercase tracking-wide text-ink-faint">
        <span>Stint plan</span>
        <span>lap 1 → {total}</span>
      </div>
      <div className="relative flex h-8 overflow-hidden rounded-md bg-surface-inset2">
        {stints.map((s, i) => {
          const light = s.compound !== "SOFT";
          return (
            <div
              key={i}
              className="flex items-center justify-center font-mono text-[11px] font-600"
              style={{
                width: `${(s.laps / total) * 100}%`,
                background: `linear-gradient(180deg, ${compoundColor(s.compound)}E6, ${compoundColor(s.compound)}B0)`,
                color: light ? "#1a1500" : "#fff",
                borderRight: i < stints.length - 1 ? "2px solid #090d14" : undefined,
              }}
            >
              {s.compound[0]} · {s.laps}
            </div>
          );
        })}
      </div>
      <div className="relative mt-1 h-4">
        {best.pit_laps.map((lap, i) => (
          <span
            key={i}
            className="absolute font-mono text-[11px] text-accent"
            style={{ left: `${(lap / total) * 100}%`, transform: "translateX(-50%)" }}
          >
            ▲ L{lap}
          </span>
        ))}
      </div>
    </div>
  );
}

function CompoundChip({ compound }: { compound: string }) {
  const c = compoundColor(compound);
  return (
    <span
      className="flex items-center gap-2 rounded-lg px-3 py-2 text-[15px] font-700 text-ink"
      style={{ background: `${c}22`, border: `1px solid ${c}55` }}
    >
      <span className="h-2.5 w-2.5 rounded-full" style={{ background: c, boxShadow: `0 0 8px ${c}` }} />
      {compound}
    </span>
  );
}

// --------------------------------------------------------------------------- //
function DegradationCard({ deg, loading, cliff }: { deg: DegradationResp | null | undefined; loading: boolean; cliff: boolean }) {
  return (
    <Card className="p-4">
      <div className="flex items-start justify-between">
        <div>
          <SectionTitle>Tyre degradation model</SectionTitle>
          <div className="-mt-2 mb-3 font-mono text-[11px] text-ink-faint">
            pace loss vs fresh · dashed = linear · solid = +cliff
          </div>
        </div>
        <div className="flex gap-3 font-mono text-[11px]">
          {COMPS.map((c) => (
            <span key={c} className="flex items-center gap-1 text-ink-muted">
              <span className="inline-block h-[3px] w-3.5 rounded" style={{ background: compoundColor(c) }} />
              {c}
            </span>
          ))}
        </div>
      </div>
      {!deg ? (
        <div className="h-[260px]">{loading && <Spinner label="…" />}</div>
      ) : (
        <DegradationChart deg={deg} cliff={cliff} />
      )}
    </Card>
  );
}

function DegradationChart({ deg, cliff }: { deg: DegradationResp; cliff: boolean }) {
  const comps = deg.compounds;
  const maxAge = Math.max(...Object.values(comps).map((c) => c.max_age));
  const data: Record<string, number | null>[] = [];
  for (let age = 0; age <= maxAge; age++) {
    const row: Record<string, number | null> = { age };
    for (const name of COMPS) {
      const c = comps[name];
      row[`${name}_lin`] = c && age <= c.max_age ? c.linear[age] : null;
      row[`${name}_cliff`] = c && age <= c.max_age ? c.cliff[age] : null;
    }
    data.push(row);
  }
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 6, right: 10, bottom: 2, left: -20 }}>
        <CartesianGrid stroke="#1a212b" vertical={false} />
        <XAxis dataKey="age" stroke="#9a9aa6" fontSize={11} tickCount={8} />
        <YAxis stroke="#9a9aa6" fontSize={11} tickFormatter={(v) => `+${v.toFixed(0)}`} />
        <Tooltip
          labelFormatter={(v) => `tyre age ${v}`}
          formatter={(val: number, name: string) => [`+${val.toFixed(2)}s`, name.replace("_cliff", "").replace("_lin", "")]}
        />
        {COMPS.map((c) => (
          <Line key={`${c}_lin`} dataKey={`${c}_lin`} stroke={compoundColor(c)} strokeWidth={1.4}
                strokeDasharray="5 5" strokeOpacity={0.5} dot={false} isAnimationActive={false} connectNulls={false} />
        ))}
        {cliff &&
          COMPS.map((c) => (
            <Line key={`${c}_cliff`} dataKey={`${c}_cliff`} stroke={compoundColor(c)} strokeWidth={2.4}
                  dot={false} isAnimationActive={false} connectNulls={false} />
          ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

// --------------------------------------------------------------------------- //
function OutcomeCard({ sim }: { sim: SimulateResp | null | undefined }) {
  return (
    <Card className="p-4">
      <SectionTitle>Monte-Carlo outcome</SectionTitle>
      <div className="-mt-2 mb-3 font-mono text-[11px] text-ink-faint">
        race-time distribution · stochastic safety car
      </div>
      {!sim ? <div className="h-[260px]"><Spinner label="Simulating…" /></div> : <OutcomeChart sim={sim} />}
    </Card>
  );
}

function OutcomeChart({ sim }: { sim: SimulateResp }) {
  const edges = sim.hist_edges.map((e) => e / 60);
  const data = sim.hist_counts.map((c, i) => ({ x: (edges[i] + edges[i + 1]) / 2, c }));
  return (
    <>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 6, right: 10, bottom: 2, left: -22 }}>
          <CartesianGrid stroke="#1a212b" vertical={false} />
          <XAxis dataKey="x" tickFormatter={(v) => v.toFixed(0)} stroke="#9a9aa6" fontSize={11} />
          <YAxis stroke="#9a9aa6" fontSize={11} />
          <Tooltip labelFormatter={(v) => `${Number(v).toFixed(1)} min`} formatter={(v) => [v, "races"]} />
          <ReferenceLine x={sim.p50_s / 60} stroke="#e2231a" strokeWidth={1.6} />
          <Bar dataKey="c" radius={[1, 1, 0, 0]}>
            {data.map((_, i) => <Cell key={i} fill="#2dd4bf" fillOpacity={0.82} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="nums mt-1 font-mono text-[11px] text-ink-faint">
        P(safety car) {Math.round(sim.p_safety_car * 100)}% · spread (p90−p10) {(sim.p90_s - sim.p10_s).toFixed(0)}s · red line = median
      </div>
    </>
  );
}

// --------------------------------------------------------------------------- //
function ShortlistCard({ rec }: { rec: RecommendResp }) {
  return (
    <Card className="p-4">
      <SectionTitle>Ranked shortlist</SectionTitle>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse font-mono text-[12.5px]">
          <thead>
            <tr className="border-b border-line text-[11px] uppercase tracking-[0.1em] text-ink-faint">
              <th className="px-3 py-2 text-left">#</th>
              <th className="px-3 py-2 text-left">Strategy</th>
              <th className="px-3 py-2 text-center">Stops</th>
              <th className="px-3 py-2 text-right">vs pick</th>
              <th className="px-3 py-2 text-right">p50</th>
              <th className="px-3 py-2 text-right">p90</th>
              <th className="px-3 py-2 text-right">P(beat #1)</th>
            </tr>
          </thead>
          <tbody>
            {rec.shortlist.map((r) => (
              <tr key={r.rank} className={`border-b border-line/60 ${r.rank === 1 ? "bg-accent/[0.06]" : ""}`}>
                <td className={`px-3 py-2.5 ${r.rank === 1 ? "text-accent" : "text-ink-dim"}`}>{r.rank}</td>
                <td className="px-3 py-2.5">
                  <span className="mr-2 inline-flex gap-1 align-middle">
                    {r.compounds.map((c, i) => (
                      <span key={i} className="h-2 w-2 rounded-full" style={{ background: compoundColor(c) }} />
                    ))}
                  </span>
                  <span className="text-ink-soft">{r.compounds.map((c) => c[0]).join("–")}</span>
                  {r.pit_laps.length > 0 && <span className="text-ink-dim"> @{r.pit_laps.join(",")}</span>}
                </td>
                <td className="px-3 py-2.5 text-center text-ink-muted">{r.pit_laps.length}</td>
                <td className="px-3 py-2.5 text-right text-ink">
                  {r.rank === 1 ? (
                    <span className="text-accent">{clock(r.mean_s)}</span>
                  ) : (
                    /* Anchoring: a delta from the pick is instantly comparable;
                       a second absolute H:MM:SS has to be mentally subtracted. */
                    `+${(r.mean_s - rec.shortlist[0].mean_s).toFixed(1)}s`
                  )}
                </td>
                <td className="px-3 py-2.5 text-right text-ink-muted">{clock(r.p50_s)}</td>
                <td className="px-3 py-2.5 text-right text-ink-muted">{clock(r.p90_s)}</td>
                <td className="px-3 py-2.5 text-right text-ink-muted">
                  {r.rank === 1 ? "—" : beatsPick(r.rank, r.win_prob_vs_best).replace("wins ", "")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

// --------------------------------------------------------------------------- //
interface Stint {
  compound: string;
  laps: number;
}
function stintLaps(best: StrategySummary, total: number): Stint[] {
  const bounds = [0, ...best.pit_laps, total];
  return best.compounds.map((compound, i) => ({
    compound,
    laps: Math.max(bounds[i + 1] - bounds[i], 0),
  }));
}

function shortName(track: string): string {
  return track.replace(" Grand Prix", " GP");
}
