import { useState } from "react";
import { api } from "../api/client";
import { Field, Select, Slider } from "../components/controls";
import { Callout, Card, ErrorNote, Metric, SectionTitle, Spinner } from "../components/ui";
import { gapText } from "../lib/format";
import { useAsync, useDebounced } from "../lib/useAsync";
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
            <Select value={track} options={tracks} onChange={setTrack} />
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

function DriverPanel({ title, accent, children }: { title: string; accent?: boolean; children: React.ReactNode }) {
  return (
    <div className={`rounded-lg border border-line p-3 ${accent ? "border-l-2 border-l-f1" : ""}`}>
      <SectionTitle>{title}</SectionTitle>
      <div className="space-y-3">{children}</div>
    </div>
  );
}
