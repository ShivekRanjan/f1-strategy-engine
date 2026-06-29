import type { ReactNode } from "react";

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] uppercase tracking-wide text-ink-muted">
        {label}
      </span>
      {children}
    </label>
  );
}

const controlCls =
  "w-full rounded-lg border border-line bg-carbon-700 px-3 py-2 text-sm text-ink " +
  "outline-none transition focus:border-f1/60 focus:ring-1 focus:ring-f1/40";

export function Select<T extends string | number>({
  value,
  options,
  onChange,
  getLabel = (v) => String(v),
}: {
  value: T;
  options: readonly T[];
  onChange: (v: T) => void;
  getLabel?: (v: T) => string;
}) {
  return (
    <select
      className={controlCls}
      value={String(value)}
      onChange={(e) => {
        const raw = e.target.value;
        const match = options.find((o) => String(o) === raw);
        onChange((match ?? raw) as T);
      }}
    >
      {options.map((o) => (
        <option key={String(o)} value={String(o)}>
          {getLabel(o)}
        </option>
      ))}
    </select>
  );
}

export function Slider({
  value,
  min,
  max,
  step = 1,
  onChange,
  display,
}: {
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  display?: ReactNode;
}) {
  return (
    <div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-carbon-600 accent-f1"
      />
      <div className="nums mt-1 text-sm font-600 text-ink">{display ?? value}</div>
    </div>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex rounded-lg border border-line bg-carbon-700 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          className={`rounded-md px-3 py-1.5 text-sm transition ${
            value === o.value ? "bg-f1 text-white" : "text-ink-muted hover:text-ink"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
