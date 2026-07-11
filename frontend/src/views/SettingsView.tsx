import { Callout, Card, SectionTitle } from "../components/ui";
import { type Accent, type Density, useSettings } from "../lib/useSettings";
import { ViewIntro } from "./common";

const ACCENTS: { id: Accent; hex: string; label: string }[] = [
  { id: "gold", hex: "#d4af37", label: "Podium gold" },
  { id: "cyan", hex: "#22e0ff", label: "Cool cyan" },
  { id: "violet", hex: "#a855f7", label: "Night violet" },
];

/** V2 Settings — the app's one page of personal preference. Everything applies
 *  instantly (no save step) and persists locally; nothing here changes the
 *  models or the data, only how the OS looks and moves. */
export default function SettingsView() {
  const [settings, update] = useSettings();

  return (
    <div className="max-w-2xl space-y-5">
      <ViewIntro>
        Make the OS yours. Choices apply <strong>instantly</strong> and are remembered on this
        device — they change how the app looks and moves, never what the models say.
      </ViewIntro>

      {/* Accent */}
      <Card className="p-5">
        <SectionTitle>Accent colour</SectionTitle>
        <div className="flex gap-4">
          {ACCENTS.map((a) => (
            <button
              key={a.id}
              onClick={() => update({ accent: a.id })}
              aria-pressed={settings.accent === a.id}
              className="group flex flex-col items-center gap-2"
            >
              <span
                className={`h-10 w-10 rounded-full transition ${
                  settings.accent === a.id
                    ? "ring-2 ring-white ring-offset-2 ring-offset-surface"
                    : "group-hover:scale-110"
                }`}
                style={{ background: a.hex }}
              />
              <span className={`text-[12px] ${settings.accent === a.id ? "text-ink" : "text-ink-dim"}`}>
                {a.label}
              </span>
            </button>
          ))}
        </div>
        <p className="mt-3 text-xs text-ink-muted">
          Re-themes highlights, glows, charts' pick colour — everywhere, immediately. Tyre-compound
          colours never change: soft is red, medium is yellow, hard is white, always.
        </p>
      </Card>

      {/* Motion */}
      <Card className="p-5">
        <SectionTitle>Ambient motion</SectionTitle>
        <div className="flex items-center justify-between gap-4">
          <p className="text-sm text-ink-soft">
            The decorative layer — drifting grid, light sweeps, breathing glows. Functional
            feedback (loading states, hover) always stays on.
          </p>
          <button
            role="switch"
            aria-checked={settings.motion}
            onClick={() => update({ motion: !settings.motion })}
            className={`relative h-[26px] w-[46px] shrink-0 rounded-full transition ${
              settings.motion ? "bg-accent" : "bg-surface-inset2 border border-line-ctl"
            }`}
          >
            <span
              className={`absolute top-[3px] h-5 w-5 rounded-full bg-white transition-all ${
                settings.motion ? "left-[23px]" : "left-[3px]"
              }`}
            />
          </button>
        </div>
        <p className="mt-2 text-xs text-ink-muted">
          Also switched off automatically when your system asks for reduced motion.
        </p>
      </Card>

      {/* Density */}
      <Card className="p-5">
        <SectionTitle>Density</SectionTitle>
        <div className="inline-flex rounded-lg border border-line bg-surface-inset p-0.5">
          {(["comfortable", "compact"] as Density[]).map((d) => (
            <button
              key={d}
              onClick={() => update({ density: d })}
              className={`rounded-md px-4 py-1.5 text-sm font-600 capitalize transition ${
                settings.density === d ? "bg-accent text-accent-ink" : "text-ink-muted hover:text-ink"
              }`}
            >
              {d}
            </button>
          ))}
        </div>
        <p className="mt-2 text-xs text-ink-muted">
          Compact tightens paddings and table rows across data-heavy views.
        </p>
      </Card>

      <Callout>
        Settings live in your browser only — clearing site data resets them to podium gold,
        motion on, comfortable.
      </Callout>
    </div>
  );
}
