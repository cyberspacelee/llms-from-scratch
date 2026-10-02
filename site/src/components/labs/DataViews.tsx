import { useEffect, useRef, useState, type ReactNode } from 'react'

export function TraceTable({ rows, active = -1, label }: { rows: [string, string][]; active?: number; label: string }) {
  return <dl aria-label={label} className="mt-4 space-y-2">{rows.map(([name, value], index) => <div key={name} className={`rounded-lg border px-3 py-2 ${active === index ? 'border-accent bg-accent-soft' : 'border-rule bg-paper'}`}><dt className="text-sm">{name}</dt><dd className="mt-1 font-mono text-xs wrap-anywhere">{value}</dd></div>)}</dl>
}

export function TokenSequence({ tokens, label }: { tokens: { label: string; detail?: string; state?: 'past' | 'current' | 'future' }[]; label: string }) {
  return <ol aria-label={label} className="mt-4 flex list-none flex-wrap gap-2 p-0">{tokens.map((token, index) => <li key={index} className={`min-w-14 rounded-lg border px-2 py-2 text-center font-mono text-xs ${token.state === 'current' ? 'border-accent2 bg-accent2-soft text-accent2' : token.state === 'future' ? 'border-rule bg-sunken text-muted' : 'border-accent bg-accent-soft text-accent-strong'}`}>{token.label}{token.detail && <span className="mt-1 block">{token.detail}</span>}</li>)}</ol>
}

/** Data cells have a readable text value; tone is a second channel, never the only one. */
export function MatrixGrid({ values, rowLabels, columnLabels, label, activeRow, allowed }: {
  values: ReactNode[][]; rowLabels: ReactNode[]; columnLabels: ReactNode[]; label: string; activeRow?: number; allowed?: boolean[][]
}) {
  const viewport = useRef<HTMLDivElement>(null)
  const [scrolls, setScrolls] = useState(false)
  useEffect(() => {
    const element = viewport.current
    if (!element) return
    const measure = () => setScrolls(element.scrollWidth > element.clientWidth + 1)
    const observer = new ResizeObserver(measure)
    observer.observe(element)
    measure()
    return () => observer.disconnect()
  }, [values])
  return <div ref={viewport} className="diagram-viewport mt-4" tabIndex={scrolls ? 0 : undefined} role={scrolls ? 'region' : undefined} aria-label={scrolls ? `${label}；左右滚动查看` : undefined}><table className="w-full border-separate border-spacing-1 text-center text-xs"><caption className="mb-2 text-left text-sm text-muted">{label}</caption><thead><tr><th scope="col" className="min-w-12">行 / 列</th>{columnLabels.map((name, index) => <th scope="col" key={index} className="min-w-12 px-2 py-2">{name}</th>)}</tr></thead><tbody>{values.map((row, i) => <tr key={i} className={activeRow === i ? 'font-semibold' : ''}><th scope="row" className="min-w-12 px-2">{rowLabels[i]}</th>{row.map((value, j) => <td key={j} className={`min-w-12 rounded-sm border px-2 py-2 font-mono ${allowed?.[i][j] === false ? 'border-rule bg-sunken text-muted' : 'border-accent bg-accent-soft text-ink'} ${activeRow === i ? 'outline-1 outline-info' : ''}`}>{value}</td>)}</tr>)}</tbody></table></div>
}
