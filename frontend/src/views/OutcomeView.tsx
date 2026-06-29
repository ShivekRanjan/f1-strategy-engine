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
import { Field, Slider } from "../components/controls";
import { Badge, Callout, Card, ErrorNote, Metric, SectionTitle, Spinner } from "../components/ui";
import { pct } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import type { OutcomeResp, PodiumPred } from "../api/types";
import { ViewIntro } from "./common";

export default function OutcomeView() {
  const o = useAsync(() => api.outcome(), []);
  return (
    <div className="space-y-5">
      <ViewIntro>
        Title odds and per-race podium probabilities — forward-tested (never a shuffled split). The
        championship sim <em>bootstraps driver strength</em>, so a few-race leader doesn’t show a
        dishonest 100%.
      </ViewIntro>
      {o.error && <ErrorNote error={o.error} />}
      {!o.data && !o.error && <Spinner label="Training the podium model & simulating the title…" />}
      {o.data && <Body o={o.data} />}
    </div>
  );
}

function Body({ o }: { o: OutcomeResp }) {
  const champ = o.championship.slice(0, 8);
  const data = [...champ].reverse().map((c) => ({ driver: c.driver, p: c.win_prob * 100 }));
  return (
    <>
      <Card className="p-4">
        <SectionTitle>
          Championship projection — {o.test_year}{" "}
          {o.ongoing ? (
            <Badge tone="red">live · {o.done} of {o.full} races</Badge>
          ) : (
            <Badge>full season</Badge>
          )}
        </SectionTitle>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart layout="vertical" data={data} margin={{ left: 8, right: 44, top: 4, bottom: 4 }}>
            <XAxis type="number" domain={[0, 100]} stroke="#9a9aa6" fontSize={11} unit="%" />
            <YAxis type="category" dataKey="driver" stroke="#ecedf0" fontSize={12} width={48} />
            <Bar dataKey="p" radius={[0, 3, 3, 0]}>
              {data.map((_, i) => (
                <Cell key={i} fill="#e2231a" />
              ))}
              <LabelList
                dataKey="p"
                position="right"
                formatter={(v: number) => (v >= 0.5 ? `${v.toFixed(0)}%` : `${v.toFixed(1)}%`)}
                fill="#ecedf0"
                fontSize={11}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        {o.ongoing && (
          <p className="mt-1 text-xs text-ink-muted">
            Current points + the remaining races simulated using <em>this</em> season’s form. Each
            run bootstraps driver strength from the few races so far, so the odds honestly reflect
            how little evidence exists yet.
          </p>
        )}
      </Card>

      <PodiumSection o={o} />
    </>
  );
}

function PodiumSection({ o }: { o: OutcomeResp }) {
  const rounds = o.rounds;
  const [idx, setIdx] = useState(0);
  useEffect(() => setIdx(0), [o.test_year]);
  if (!rounds.length) return null;
  const round = rounds[Math.min(idx, rounds.length - 1)];

  const cols: Column<PodiumPred>[] = [
    { key: "driver", header: "Driver", render: (p) => <span className="font-600">{p.driver}</span> },
    { key: "team", header: "Team", render: (p) => <span className="text-ink-muted">{p.team}</span> },
    { key: "grid", header: "Grid", align: "right", render: (p) => p.grid },
    { key: "prob", header: "Podium prob", align: "right", render: (p) => pct(p.podium_prob) },
    { key: "actual", header: "Result", align: "center", render: (p) => (p.actual ? "🏆" : "") },
  ];

  return (
    <Card className="p-4">
      <SectionTitle>Podium predictor — forward-tested on {o.test_year}</SectionTitle>
      <div className="mb-3 max-w-md">
        <Field label={`Round · ${round.round} — ${round.event_name}`}>
          <Slider
            value={idx}
            min={0}
            max={rounds.length - 1}
            onChange={setIdx}
            display={`${round.event_name} (R${round.round})`}
          />
        </Field>
      </div>
      <DataTable columns={cols} rows={round.predictions} getKey={(p) => p.driver} />

      <div className="mt-4 grid grid-cols-3 gap-3">
        <Metric label="ROC-AUC" value={o.model_metrics.auc.toFixed(3)} title="Ranks podium vs non-podium drivers (1.0 = perfect)." />
        <Metric label="Model precision@3" value={pct(o.model_metrics.model_precision_at_3)} />
        <Metric label="Grid baseline @3" value={pct(o.model_metrics.grid_baseline_precision_at_3)} />
      </div>
      <Callout>
        Grid position is itself the strongest podium signal; the model’s value is the{" "}
        <em>calibrated probability</em> per driver, not reshuffling the grid’s top 3.
      </Callout>
    </Card>
  );
}
