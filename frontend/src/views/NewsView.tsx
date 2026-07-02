import { useMemo, useState } from "react";
import { api } from "../api/client";
import { Callout, Card, ErrorNote, Spinner } from "../components/ui";
import { timeAgo } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import type { NewsItem, NewsResp } from "../api/types";
import { ViewIntro } from "./common";

export default function NewsView() {
  const n = useAsync(() => api.news(40), []);
  return (
    <div className="space-y-5">
      <ViewIntro>
        The latest F1 headlines, aggregated from across the paddock — <strong>The Race</strong>,
        Autosport, Motorsport.com, RaceFans and Formula1.com. Headlines and links only; click through
        to read the full story at the source.
      </ViewIntro>
      {n.error && <ErrorNote error={n.error} />}
      {!n.data && !n.error && <Spinner label="Fetching the latest headlines…" />}
      {n.data && <Body data={n.data} />}
    </div>
  );
}

function Body({ data }: { data: NewsResp }) {
  const [source, setSource] = useState<string | null>(null);
  const items = useMemo(
    () => (source ? data.items.filter((i) => i.source === source) : data.items),
    [data.items, source],
  );

  if (!data.items.length) {
    return (
      <Callout tone="warn">
        No headlines available right now — the news sources may be unreachable. They’ll appear once
        the feeds respond.
      </Callout>
    );
  }

  return (
    <>
      {/* Source filter chips */}
      <div className="flex flex-wrap items-center gap-2">
        <Chip active={source === null} onClick={() => setSource(null)}>
          All
        </Chip>
        {data.sources.map((s) => (
          <Chip key={s} active={source === s} onClick={() => setSource(s)}>
            {s}
          </Chip>
        ))}
        <span className="ml-auto font-mono text-[11px] text-ink-faint">
          updated {timeAgo(data.fetched_at)}
        </span>
      </div>

      <Card className="divide-y divide-line p-0">
        {items.map((it) => (
          <NewsRow key={it.link} it={it} />
        ))}
      </Card>
    </>
  );
}

function NewsRow({ it }: { it: NewsItem }) {
  return (
    <a
      href={it.link}
      target="_blank"
      rel="noreferrer noopener"
      className="group block px-4 py-3 transition hover:bg-surface-inset/60"
    >
      <div className="mb-1 flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.12em]">
        <span className="rounded-full bg-accent/12 px-2 py-0.5 text-accent">{it.source}</span>
        <span className="text-ink-faint">{timeAgo(it.ts)}</span>
        <span className="ml-auto text-ink-faint opacity-0 transition group-hover:opacity-100">
          read ↗
        </span>
      </div>
      <div className="font-600 text-ink group-hover:text-accent">{it.title}</div>
      {it.summary && <p className="mt-1 line-clamp-2 text-sm text-ink-muted">{it.summary}</p>}
    </a>
  );
}

function Chip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-md border px-3 py-1 font-mono text-[11px] transition ${
        active
          ? "border-accent/60 bg-accent/10 text-accent"
          : "border-line-ctl text-ink-dim hover:border-line-hover hover:text-ink-soft"
      }`}
    >
      {children}
    </button>
  );
}
