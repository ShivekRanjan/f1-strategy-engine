import type { ReactNode } from "react";

export interface Column<R> {
  key: string;
  header: ReactNode;
  render: (row: R) => ReactNode;
  align?: "left" | "right" | "center";
  className?: string;
}

/** Minimal, dark-themed table. Highlights the first row (rank 1 / our pick). */
export function DataTable<R>({
  columns,
  rows,
  highlightFirst = false,
  getKey,
}: {
  columns: Column<R>[];
  rows: R[];
  highlightFirst?: boolean;
  getKey: (row: R, i: number) => string | number;
}) {
  const alignCls = { left: "text-left", right: "text-right", center: "text-center" };
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-line">
            {columns.map((c) => (
              <th
                key={c.key}
                className={`px-3 py-2 text-[11px] font-600 uppercase tracking-wide text-ink-muted ${alignCls[c.align ?? "left"]}`}
              >
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={getKey(row, i)}
              className={`border-b border-line/60 ${
                highlightFirst && i === 0 ? "bg-f1/[0.06]" : ""
              }`}
            >
              {columns.map((c) => (
                <td
                  key={c.key}
                  className={`nums px-3 py-2 text-ink ${alignCls[c.align ?? "left"]} ${c.className ?? ""}`}
                >
                  {c.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
