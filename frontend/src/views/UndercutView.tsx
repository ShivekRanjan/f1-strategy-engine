import { useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import { Combobox, Field, Select, Slider } from "../components/controls";
import { Callout, Card, ErrorNote, Metric, SectionTitle, Spinner } from "../components/ui";
import { gapText, trackSearchText } from "../lib/format";
import { useAsync, useDebounced } from "../lib/useAsync";
import type { UndercutTrajectory } from "../api/types";
import { TracksGate, ViewIntro, pickDefaultTrack } from "./common";

const COMPS = ["SOFT", "MEDIUM", "HARD"] as const;

export default function UndercutView() {
  return <TracksGate>{(tracks) => <Inner tracks={tracks} />}</TracksGate>;
}

function Inner({ tracks }: { tracks: string[] }) {
  const [track, setTrack] = useState(() => pickDefaultTrack(tracks));
  const info = useAsync(() => api.raceInfo(track), [track]);
  return (
    <div className="space-y-5">
      <ViewIntro>
        Should you pit <em>now</em> to jump a rival, or hold and cover? A two-car cumulative-time
        model of the undercut — fresh-tyre pace vs the gap and pit loss, judged at the crossover
        once both have stopped.
      </ViewIntro>
      <Card className="p-4">
        <Field label="Circuit">
          <div className="max-w-sm">
            <Combobox
              value={track}
              options={tracks}
              onChange={setTrack}
              getSearchText={trackSearchText}
              placeholder="Search circuits…"
            />
          </div>
        </Field>
      </Card>
      {info.error && <ErrorNote error={info.error} />}
      {!info.data && !info.error && <Spinner label="Loading circuit…" />}
      {info.data && <Panel key={track} track={track} total={info.data.total_laps} />}
    </div>
  );
}

function Panel({ track, total }: { track: string; total: number }) {
  const [cur, setCur] = useState(Math.min(Math.floor(total / 3), total - 5));
  const [gap, setGap] = useState(2);
  const [yc, setYc] = useState("MEDIUM");
  const [ya, setYa] = useState(Math.min(15, total));
  const [ynew, setYnew] = useState("SOFT");
  const [rc, setRc] = useState("HARD");
  const [ra, setRa] = useState(Math.min(15, total));
  const [rnew, setRnew] = useState("MEDIUM");
  const [rpit, setRpit] = useState(Math.min(cur + 8, total - 1));

  const d = useDebounced({ cur, gap, yc, ya, ynew, rc, ra, rnew, rpit }, 200);
  const res = useAsync(
    () =>
      api.undercut({
        track,
        current_lap: d.cur,
        gap_s: d.gap,
        your_compound: d.yc,
        your_age: d.ya,
        your_new_compound: d.ynew,
        rival_compound: d.rc,
        rival_age: d.ra,
        rival_new_compound: d.rnew,
        rival_pit_lap: Math.max(d.cur + 1, d.rpit),
        n_runs: 2000,
      }),
    [track, JSON.stringify(d)],
  );

  return (
    <>
      <Card className="p-4">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <Field label={`Current lap · ${cur}`}>
            <Slider value={cur} min={2} max={total - 4} onChange={setCur} display={`lap ${cur} of ${total}`} />
          </Field>
          <Field label="Gap to rival (+ = you're behind)">
            <Slider value={gap} min={-5} max={25} step={0.5} onChange={setGap} display={`${gap.toFixed(1)} s`} />
          </Field>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-6 sm:grid-cols-2">
          <DriverPanel title="You" accent>
            <Field label="Current tyre">
              <Select value={yc} options={COMPS} onChange={setYc} />
            </Field>
            <Field label={`Tyre age · ${ya}`}>
              <Slider value={ya} min={1} max={total} onChange={setYa} display={`${ya} laps`} />
            </Field>
            <Field label="Pit to">
              <Select value={ynew} options={COMPS} onChange={setYnew} />
            </Field>
          </DriverPanel>
          <DriverPanel title="Rival">
            <Field label="Current tyre">
              <Select value={rc} options={COMPS} onChange={setRc} />
            </Field>
            <Field label={`Tyre age · ${ra}`}>
              <Slider value={ra} min={1} max={total} onChange={setRa} display={`${ra} laps`} />
            </Field>
            <Field label="Pit to">
              <Select value={rnew} options={COMPS} onChange={setRnew} />
            </Field>
            <Field label={`Rival's expected pit lap · ${rpit}`}>
              <Slider value={rpit} min={cur + 1} max={total - 1} onChange={setRpit} display={`lap ${rpit}`} />
            </Field>
          </DriverPanel>
        </div>
      </Card>

      {res.error && <ErrorNote error={res.error} />}
      {!res.data && !res.error && <Spinner label="Modelling the crossover…" />}
      {res.data && (
        <>
          <Callout tone={res.data.undercut_works ? "success" : "info"}>
            <span className="text-base font-700">{res.data.verdict}</span>
          </Callout>
          {res.data.trajectory && <CrossoverChart t={res.data.trajectory} />}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Metric
              label="Pit now (undercut)"
              accent={res.data.undercut_works}
              value={gapText(res.data.undercut.final_gap_s)}
              sub={`ends ahead ${Math.round(res.data.undercut.p_ahead * 100)}% of races`}
            />
            <Metric
              label="Cover (pit with rival)"
              accent={!res.data.undercut_works}
              value={gapText(res.data.cover.final_gap_s)}
              sub={`ends ahead ${Math.round(res.data.cover.p_ahead * 100)}% of races`}
            />
          </div>
          <p className="nums text-xs text-ink-muted">
            Undercutting nets <b className="text-ink">{res.data.undercut_gain_s >= 0 ? "+" : ""}
            {res.data.undercut_gain_s.toFixed(1)}s</b> vs covering, measured at the crossover.
            Positive = the undercut is faster.
          </p>
        </>
      )}
    </>
  );
}

/** The undercut, made spatial: your gap to the rival lap by lap, one line per
 *  option. Up = you ahead (people map up to winning), the zero line = the rival,
 *  and the moment a line crosses it IS the crossover — no mental simulation
 *  needed. Solid vs dashed is a second encoding on top of colour. */
function CrossoverChart({ t }: { t: UndercutTrajectory }) {
  const data = t.laps.map((lap, i) => ({
    lap,
    // Negate so "ahead of the rival" plots upward; the axis labels say so.
    undercut: -t.undercut[i],
    cover: -t.cover[i],
  }));
  return (
    <Card className="p-4">
      <SectionTitle>The crossover, lap by lap</SectionTitle>
      <div className="relative">
        <span className="pointer-events-none absolute left-12 top-1 font-mono text-[11px] uppercase tracking-[0.1em] text-accent/80">
          ▲ you ahead
        </span>
        <span className="pointer-events-none absolute bottom-8 left-12 font-mono text-[11px] uppercase tracking-[0.1em] text-ink-faint">
          ▼ rival ahead
        </span>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data} margin={{ top: 14, right: 42, bottom: 4, left: -8 }}>
            <CartesianGrid stroke="#1a212b" vertical={false} />
            <XAxis dataKey="lap" stroke="#9a9aa6" fontSize={11} type="number"
                   domain={["dataMin", "dataMax"]} tickCount={8} />
            <YAxis stroke="#9a9aa6" fontSize={11} width={44}
                   tickFormatter={(v: number) => `${v > 0 ? "+" : ""}${v.toFixed(0)}s`} />
            <Tooltip
              labelFormatter={(v) => `lap ${v}`}
              formatter={(v: number, name: string) => [
                `${Math.abs(v).toFixed(1)}s ${v >= 0 ? "ahead" : "behind"}`,
                name,
              ]}
            />
            <ReferenceLine y={0} stroke="#8994a4" strokeWidth={1.5}
                           label={{ value: "rival", position: "right", fill: "#8994a4", fontSize: 11 }} />
            {/* No marker for your own stop: it's always "now" (the chart's left
                edge) and the teal line's pit-loss dive already shows it. */}
            <ReferenceLine x={t.rival_pit_lap} stroke="#8994a4" strokeDasharray="2 3" strokeOpacity={0.6}
                           label={{ value: "rival pits", position: "insideTopRight", fill: "#8994a4", fontSize: 10 }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line name="Pit now (undercut)" type="monotone" dataKey="undercut"
                  stroke="#2dd4bf" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line name="Cover (pit with rival)" type="monotone" dataKey="cover"
                  stroke="#f6c700" strokeWidth={2} strokeDasharray="6 4" dot={false}
                  isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-1 text-xs text-ink-muted">
        Mean paths from the same pace model the simulator uses. Free-air caveat: this is the
        <em> clock</em> crossover — it doesn't model whether you can physically pass.
      </p>
    </Card>
  );
}

function DriverPanel({ title, accent, children }: { title: string; accent?: boolean; children: React.ReactNode }) {
  return (
    <div className={`rounded-lg border border-line p-3 ${accent ? "border-l-2 border-l-f1" : ""}`}>
      <SectionTitle>{title}</SectionTitle>
      <div className="space-y-3">{children}</div>
    </div>
  );
}
