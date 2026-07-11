import { useEffect, useState } from "react";

/** V2 user settings — the one global theme state (design handoff: Settings).
 *  Persisted to localStorage; accent + motion are applied as attributes on
 *  <html> so pure CSS (the --accent variable, the motion gate) picks them up
 *  with no re-render of the tree. A custom event keeps every mounted hook
 *  instance in sync when any one of them writes. */

export type Accent = "gold" | "cyan" | "violet";
export type Density = "comfortable" | "compact";

export interface Settings {
  accent: Accent;
  motion: boolean;
  density: Density;
}

const KEY = "f1se:settings";
const EVT = "f1se:settings-changed";

const DEFAULTS: Settings = { accent: "gold", motion: true, density: "comfortable" };

function load(): Settings {
  try {
    return { ...DEFAULTS, ...JSON.parse(window.localStorage.getItem(KEY) ?? "{}") };
  } catch {
    return DEFAULTS;
  }
}

function applyToDom(s: Settings) {
  const el = document.documentElement;
  el.setAttribute("data-accent", s.accent);
  el.setAttribute("data-motion", s.motion ? "on" : "off");
  el.setAttribute("data-density", s.density);
}

// Apply persisted settings immediately at module load (before first paint of
// any accent-coloured element), not on first hook mount.
if (typeof document !== "undefined") applyToDom(load());

export function useSettings(): [Settings, (patch: Partial<Settings>) => void] {
  const [settings, setSettings] = useState<Settings>(load);

  useEffect(() => {
    const onChange = () => setSettings(load());
    window.addEventListener(EVT, onChange);
    return () => window.removeEventListener(EVT, onChange);
  }, []);

  const update = (patch: Partial<Settings>) => {
    const next = { ...load(), ...patch };
    try {
      window.localStorage.setItem(KEY, JSON.stringify(next));
    } catch {
      /* private mode — settings just don't persist */
    }
    applyToDom(next);
    window.dispatchEvent(new Event(EVT));
  };

  return [settings, update];
}
