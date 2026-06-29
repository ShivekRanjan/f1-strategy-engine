import { useEffect, useRef, useState } from "react";

export interface AsyncState<T> {
  data?: T;
  error?: string;
  loading: boolean;
}

/** Run an async fn whenever deps change; track loading/error/data, ignore stale. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({ loading: true });
  useEffect(() => {
    let alive = true;
    setState((s) => ({ ...s, loading: true, error: undefined }));
    fn()
      .then((d) => alive && setState({ data: d, loading: false }))
      .catch((e) => alive && setState({ error: String(e?.message ?? e), loading: false }));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

/** Debounce a fast-changing value (e.g. a slider) before firing expensive calls. */
export function useDebounced<T>(value: T, ms = 250): T {
  const [v, setV] = useState(value);
  const t = useRef<number>();
  useEffect(() => {
    window.clearTimeout(t.current);
    t.current = window.setTimeout(() => setV(value), ms);
    return () => window.clearTimeout(t.current);
  }, [value, ms]);
  return v;
}
