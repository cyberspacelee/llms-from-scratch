/**
 * The diagram palette. Every primitive takes one of these tones, so all figures
 * share the same colors and follow the light/dark theme through the site tokens.
 */
export type Tone = 'neutral' | 'accent' | 'accent2' | 'info' | 'muted' | 'ghost' | 'strong'
export type TextTone = 'ink' | 'muted' | 'accent' | 'accent2' | 'info' | 'paper'

/** Shape fill + stroke, and the text color that reads on that fill. */
export const shape: Record<Tone, { box: string; text: string; sub: string }> = {
  neutral: { box: 'fill-raised stroke-rule-strong', text: 'fill-ink', sub: 'fill-muted' },
  accent: { box: 'fill-accent-soft stroke-accent', text: 'fill-ink', sub: 'fill-muted' },
  accent2: { box: 'fill-accent2-soft stroke-accent2', text: 'fill-ink', sub: 'fill-muted' },
  info: { box: 'fill-info-soft stroke-info', text: 'fill-ink', sub: 'fill-muted' },
  muted: { box: 'fill-sunken stroke-rule', text: 'fill-muted', sub: 'fill-muted' },
  ghost: { box: 'fill-none stroke-rule-strong [stroke-dasharray:5_4]', text: 'fill-muted', sub: 'fill-muted' },
  strong: { box: 'fill-accent stroke-accent', text: 'fill-paper', sub: 'fill-paper' },
}

/** Line/arrow color per tone. */
export const line: Record<Tone, { stroke: string; fill: string }> = {
  neutral: { stroke: 'stroke-ink', fill: 'fill-ink' },
  accent: { stroke: 'stroke-accent', fill: 'fill-accent' },
  accent2: { stroke: 'stroke-accent2', fill: 'fill-accent2' },
  info: { stroke: 'stroke-info', fill: 'fill-info' },
  muted: { stroke: 'stroke-muted', fill: 'fill-muted' },
  ghost: { stroke: 'stroke-rule-strong', fill: 'fill-rule-strong' },
  strong: { stroke: 'stroke-accent', fill: 'fill-accent' },
}

export const text: Record<TextTone, string> = {
  ink: 'fill-ink',
  muted: 'fill-muted',
  accent: 'fill-accent',
  accent2: 'fill-accent2',
  info: 'fill-info',
  paper: 'fill-paper',
}

/** Font sizes in SVG user units. Diagrams are drawn at ~1 unit = 1 CSS px (width ≈ 720). */
export const size = { xs: 11, sm: 12.5, md: 14, lg: 16 } as const
export type Size = keyof typeof size

/** Split a label on "\n" into lines, centred around `y`. */
export function lines(label: string | undefined, y: number, fontSize: number, gap = 1.3) {
  if (!label) return []
  const parts = String(label).split('\n')
  const step = fontSize * gap
  const top = y - ((parts.length - 1) * step) / 2
  return parts.map((value, index) => ({ value, y: top + index * step }))
}
