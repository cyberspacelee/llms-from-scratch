/**
 * Inline TeX in diagram labels: "策略 $\pi_\theta$" renders the $…$ parts with KaTeX at build time.
 * Labels without a $ stay plain SVG text.
 *
 * Inside SVG the math is MathML: WebKit paints the positioned spans of KaTeX's HTML output
 * without the viewBox scale, so sub- and superscripts drift away from their labels.
 */
import katex from 'katex'

const math = /\$([^$]+)\$/g

export const hasTex = (label: string) => /\$[^$]+\$/.test(label)

const escape = (value: string) => value.replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[ch]!)

/** The label as HTML: plain parts escaped, $…$ parts typeset as MathML, or KaTeX HTML outside SVG. */
export function texHtml(label: string, output: 'mathml' | 'html' = 'mathml') {
  let html = '', last = 0
  for (const match of label.matchAll(math)) {
    html += escape(label.slice(last, match.index))
    html += katex.renderToString(match[1], { throwOnError: true, strict: 'error', output })
    last = match.index + match[0].length
  }
  return html + escape(label.slice(last))
}

/** Roughly what the reader sees, for width estimates: TeX commands count as one glyph, markup as none. */
export const visible = (label: string) =>
  label.replace(math, (_, tex: string) => tex.replace(/\\[a-zA-Z]+/g, 'x').replace(/[{}_^\\]/g, ''))

/** Estimated width in user units: CJK glyphs ~1em, Latin ~0.6em, at the 13-unit label size. */
export const labelWidth = (label: string) => [...visible(label)].reduce((sum, ch) => sum + (/[⺀-￿]/.test(ch) ? 13 : 7.8), 0)
