import { useEffect, useRef, useState, type SVGProps } from 'react'

/** Keep diagram text readable after viewBox and nested transforms are applied. */
export default function SvgCanvas({ children, ...props }: SVGProps<SVGSVGElement>) {
  const svg = useRef<SVGSVGElement>(null)
  const frame = useRef<HTMLDivElement>(null)
  const [minimum, setMinimum] = useState(0)
  const [scrolls, setScrolls] = useState(false)
  useEffect(() => {
    const drawing = svg.current, viewport = frame.current
    if (!drawing || !viewport) return
    const measure = () => {
      const width = drawing.getBoundingClientRect().width
      const sizes = [...drawing.querySelectorAll('text')].filter(text => text.textContent?.trim()).map(text => {
        const matrix = text.getScreenCTM()
        return matrix ? parseFloat(getComputedStyle(text).fontSize) * Math.hypot(matrix.a, matrix.b) : Infinity
      })
      const smallest = Math.min(...sizes)
      if (width > 0 && smallest > 0 && Number.isFinite(smallest)) {
        // Recover intrinsic width instead of repeatedly multiplying an already expanded SVG.
        const needed = Math.ceil(width * 12 / smallest)
        setMinimum(current => Math.abs(current - needed) > 1 ? needed : current)
        setScrolls(needed > viewport.clientWidth + 1)
      }
    }
    const observer = new ResizeObserver(measure)
    observer.observe(viewport)
    measure()
    return () => observer.disconnect()
  }, [children])
  return <div ref={frame} className="diagram-viewport" tabIndex={scrolls ? 0 : undefined} role={scrolls ? 'region' : undefined} aria-label={scrolls ? `${props['aria-label'] ?? '实验图'}；左右滚动查看` : undefined}>
    <svg {...props} ref={svg} style={{ ...props.style, minWidth: minimum || undefined }}>{children}</svg>
  </div>
}
