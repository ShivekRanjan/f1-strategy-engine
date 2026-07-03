import { Card, SectionTitle } from "../components/ui";
import { ViewIntro } from "./common";

const REPO = "https://github.com/ShivekRanjan/f1-strategy-engine";

/** The story behind the app, in plain English — what it does, how it was
 *  validated, and what it honestly can't do. The receipts live in the repo. */
export default function AboutView() {
  return (
    <div className="max-w-3xl space-y-5">
      <ViewIntro>
        F1SE is a Formula 1 <strong>decision engine</strong> that grew into an F1 OS. The question it
        answers isn't <em>"who will win?"</em> — it's <em>"what should the team do?"</em> Every
        number is either calibrated from data or explicitly labelled as an assumption.
      </ViewIntro>

      <Card className="p-5">
        <SectionTitle>What's under the hood</SectionTitle>
        <ul className="list-disc space-y-2 pl-5 text-sm text-ink-soft">
          <li>
            <strong>Tyre degradation model</strong> — fitted per circuit and compound on fuel-corrected
            laps from 2023–26, with the 2026 regulation reset handled by blending new-era data with
            the old-era prior (shrinkage) and recency-weighting for mid-season car upgrades.
          </li>
          <li>
            <strong>Monte-Carlo race simulator</strong> — thousands of race runs per question, with
            per-circuit safety-car risk and pit loss <em>measured</em> from 78 races, not assumed.
            Results are distributions, never single numbers.
          </li>
          <li>
            <strong>Labelled priors</strong> for what data can't show: the tyre "cliff" (censored out
            of race data because teams pit first), track-position cost per stop, and a{" "}
            <strong>thermal prior</strong> — track temperature shifts degradation, which fixed a real
            over-stopping bias found by backtesting (stop-count match went 4/8 → 7/8).
          </li>
          <li>
            <strong>An LSTM next-lap forecaster</strong> — the one place deep learning beat the simple
            baseline on a leakage-safe split (+8.5% vs persistence), so it's the one place it's used.
          </li>
          <li>
            <strong>Podium & championship models</strong> — grid + form only, always validated{" "}
            <em>forward in time</em> (trained on earlier seasons, tested on later ones — never a
            shuffled split), with title odds that bootstrap uncertainty so an early leader doesn't
            show a dishonest 100%.
          </li>
        </ul>
      </Card>

      <Card className="p-5">
        <SectionTitle>The honesty policy</SectionTitle>
        <p className="mb-2 text-sm text-ink-soft">
          Sophisticated models were adopted <strong>only when they beat a simpler baseline</strong>{" "}
          on a leakage-safe split. Most didn't — XGBoost lost to a straight line, a fitted tyre
          cliff was worse than a labelled prior — and those results are documented, not deleted.
          The Race Hub shows every prediction <em>next to what actually happened</em>, misses
          included.
        </p>
        <p className="text-sm text-ink-soft">
          Known limits, stated plainly: totals cover the <strong>2023–26 dataset window</strong>, not
          all-time careers; real-time timing only exists while a session runs, so between sessions
          the app counts down and replays; and one honest backtest miss (Canada — cold but
          tyre-punishing) remains documented rather than patched over.
        </p>
      </Card>

      <Card className="p-5">
        <SectionTitle>Stack & receipts</SectionTitle>
        <p className="text-sm text-ink-soft">
          Python 3.12 · FastF1 · pandas · scikit-learn · PyTorch (training only — inference is a
          28 KB numpy export) · FastAPI · React + Vite + TypeScript + Tailwind. 136 no-network
          tests; CI runs the Python suite and the frontend build.
        </p>
        <div className="mt-3 flex flex-wrap gap-3">
          <a
            href={REPO}
            target="_blank"
            rel="noreferrer"
            className="rounded-md border border-line-ctl px-3 py-1.5 font-mono text-[12px] text-ink-dim transition hover:border-line-hover hover:text-ink-soft"
          >
            Source on GitHub ↗
          </a>
          <a
            href={`${REPO}/blob/main/docs/METHODOLOGY.md`}
            target="_blank"
            rel="noreferrer"
            className="rounded-md border border-line-ctl px-3 py-1.5 font-mono text-[12px] text-ink-dim transition hover:border-line-hover hover:text-ink-soft"
          >
            Full methodology — the receipts ↗
          </a>
        </div>
      </Card>
    </div>
  );
}
