import { useEffect, useMemo, useRef, useState } from "react";
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

/** Searchable select for long lists (24+ circuits, 30+ drivers): type to
 *  filter, ↑/↓ to move, Enter to pick, Esc to close. Falls back to showing the
 *  current value when closed, so it reads like a Select. */
export function Combobox<T extends string>({
  value,
  options,
  onChange,
  getLabel = (v) => String(v),
  placeholder = "Type to search…",
}: {
  value: T | "";
  options: readonly T[];
  onChange: (v: T) => void;
  getLabel?: (v: T) => string;
  placeholder?: string;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [hi, setHi] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? options.filter((o) => getLabel(o).toLowerCase().includes(q)) : [...options];
  }, [options, query, getLabel]);

  // Close on outside click.
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, []);

  // Keep the highlighted row in view while arrowing.
  useEffect(() => {
    listRef.current?.children[hi]?.scrollIntoView({ block: "nearest" });
  }, [hi]);

  const pick = (o: T) => {
    onChange(o);
    setOpen(false);
    setQuery("");
  };

  return (
    <div ref={rootRef} className="relative">
      <input
        className={controlCls}
        value={open ? query : value ? getLabel(value as T) : ""}
        placeholder={placeholder}
        onFocus={() => {
          setOpen(true);
          setQuery("");
          // Highlight the current value, not row 0 — a stray Enter then re-picks
          // the same option instead of silently switching to the first one.
          const cur = options.findIndex((o) => o === value);
          setHi(cur >= 0 ? cur : 0);
        }}
        onChange={(e) => {
          setQuery(e.target.value);
          setHi(0);
          if (!open) setOpen(true);
        }}
        onKeyDown={(e) => {
          if (!open && (e.key === "ArrowDown" || e.key === "Enter")) return setOpen(true);
          if (e.key === "ArrowDown") { e.preventDefault(); setHi((h) => Math.min(h + 1, filtered.length - 1)); }
          else if (e.key === "ArrowUp") { e.preventDefault(); setHi((h) => Math.max(h - 1, 0)); }
          else if (e.key === "Enter") { e.preventDefault(); if (filtered[hi]) pick(filtered[hi]); }
          else if (e.key === "Escape") { setOpen(false); (e.target as HTMLInputElement).blur(); }
        }}
        role="combobox"
        aria-expanded={open}
        aria-autocomplete="list"
      />
      {open && (
        <ul
          ref={listRef}
          role="listbox"
          className="absolute z-30 mt-1 max-h-64 w-full overflow-y-auto rounded-lg border border-line bg-surface-rail py-1 shadow-card"
        >
          {filtered.length === 0 && (
            <li className="px-3 py-2 text-sm text-ink-muted">No match.</li>
          )}
          {filtered.map((o, i) => (
            <li
              key={String(o)}
              role="option"
              aria-selected={o === value}
              onMouseDown={(e) => { e.preventDefault(); pick(o); }}
              onMouseEnter={() => setHi(i)}
              className={`cursor-pointer px-3 py-1.5 text-sm ${
                i === hi ? "bg-accent/10 text-accent" : "text-ink-soft"
              } ${o === value ? "font-600" : ""}`}
            >
              {getLabel(o)}
            </li>
          ))}
        </ul>
      )}
    </div>
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
          className={`rounded-md px-3 py-1.5 text-sm font-600 transition ${
            value === o.value ? "bg-accent text-accent-ink" : "text-ink-muted hover:text-ink"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
