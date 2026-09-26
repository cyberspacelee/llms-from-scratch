import { useEffect, useId, useRef, useState, type ReactNode } from 'react'

/** The shared frame of every lab: title, hint, body. */
export function LabFrame({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  const id = useId()
  return (
    <section aria-labelledby={id} className="not-prose my-8 rounded-lg border border-rule bg-raised">
      <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 border-b border-rule px-4 py-3">
        <h3 id={id} className="m-0 text-base font-semibold">
          <span className="mr-2 text-xs font-semibold text-accent">实验</span>
          {title}
        </h3>
        {hint && <p className="m-0 text-xs text-muted">{hint}</p>}
      </header>
      <div className="p-4">{children}</div>
    </section>
  )
}

export function Controls({ children }: { children: ReactNode }) {
  return <div className="grid grid-cols-[repeat(auto-fit,minmax(11rem,1fr))] items-end gap-x-6 gap-y-2">{children}</div>
}

type RangeProps = {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (value: number) => void
  format?: (value: number) => string
}

export function Range({ label, value, min, max, step = 1, onChange, format = String }: RangeProps) {
  return (
    <label className="block text-sm">
      <span className="flex justify-between gap-2">
        {label}
        <output className="font-mono text-xs font-semibold text-accent2">{format(value)}</output>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="block h-8 w-full accent-accent"
      />
    </label>
  )
}

export function Button({ children, onClick, primary = false, label }: {
  children: ReactNode
  onClick: () => void
  primary?: boolean
  label?: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      className={
        primary
          ? 'h-9 rounded-lg border border-accent bg-accent px-4 text-sm text-paper hover:bg-accent-strong'
          : 'h-9 rounded-lg border border-rule-strong bg-paper px-4 text-sm text-ink hover:border-accent hover:text-accent-strong'
      }
    >
      {children}
    </button>
  )
}

export function Readout({ children }: { children: ReactNode }) {
  return (
    <p aria-live="polite" className="mt-4 mb-0 rounded-lg bg-sunken px-4 py-3 font-mono text-xs leading-relaxed wrap-anywhere">
      {children}
    </p>
  )
}

/** Classes for SVG drawings inside labs, so every lab draws with the same pens. */
export const pen = {
  canvas: 'mx-auto mt-4 block h-auto w-full max-w-176 overflow-visible font-sans text-[13px] [&_text]:fill-ink',
  axis: 'fill-none stroke-rule-strong',
  grid: 'fill-none stroke-rule',
  guide: 'fill-none stroke-info [stroke-dasharray:4_4] stroke-[1.5]',
  a: 'fill-none stroke-accent stroke-3',
  b: 'fill-none stroke-accent2 stroke-3 [stroke-dasharray:7_5]',
  c: 'fill-none stroke-info stroke-3',
  muted: 'fill-muted!',
  mono: 'font-mono text-[12px]',
  textA: 'fill-accent! font-semibold',
  textB: 'fill-accent2! font-semibold',
}

/** Width of an element, for labs that switch to a stacked layout on narrow screens. */
export function useWidth<T extends Element>(fallback = 640) {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(fallback)
  useEffect(() => {
    const element = ref.current
    if (!element) return
    const observer = new ResizeObserver(([entry]) => setWidth(Math.round(entry.contentRect.width)))
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  return [ref, width] as const
}

export const fmt = (value: number, digits = 3) => {
  const rounded = Math.abs(value) < 0.5 * 10 ** -digits ? 0 : value
  return rounded.toFixed(digits)
}

export function Arrow({ id, className }: { id: string; className: string }) {
  return (
    <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0L10 5L0 10Z" className={className} />
    </marker>
  )
}
