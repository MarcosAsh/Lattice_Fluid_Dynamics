export function DocTable({
  headers,
  rows,
}: {
  headers: string[];
  rows: (string | React.ReactNode)[][];
}) {
  return (
    <div className="border border-ctp-surface1 rounded-lg bg-ctp-mantle overflow-x-auto mb-4">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-ctp-surface1">
            {headers.map((h) => (
              <th
                key={h}
                className="py-2.5 px-3 text-ctp-subtext0 font-medium text-xs sticky top-0 bg-ctp-mantle"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="text-ctp-subtext1 text-xs">
          {rows.map((row, i) => (
            <tr
              key={i}
              className="border-b border-ctp-surface0 last:border-0"
            >
              {row.map((cell, j) => (
                <td key={j} className="py-2 px-3 font-mono">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
