import { useRef } from "react";

const KEY = "f1se:lastVisit";

/** Epoch seconds of the PREVIOUS visit (0 on first ever visit), then records
 *  now as the latest. Read once per mount so the cue stays stable while the
 *  user looks at the page. Powers the "new since you were here" markers —
 *  offloading "what have I already seen?" from memory to the interface. */
export function useLastVisit(): number {
  const prev = useRef<number | null>(null);
  if (prev.current === null) {
    let stored = 0;
    try {
      stored = Number(window.localStorage.getItem(KEY)) || 0;
      window.localStorage.setItem(KEY, String(Math.floor(Date.now() / 1000)));
    } catch {
      /* storage unavailable (private mode etc.) — cue simply never shows */
    }
    prev.current = stored;
  }
  return prev.current;
}
