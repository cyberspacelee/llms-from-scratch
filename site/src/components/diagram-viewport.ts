// Figures keep their smallest label at ≥10 CSS px; narrower screens scroll the figure instead of shrinking it.
const registeredViewports = new WeakSet<Element>()
const registeredSvgs = new WeakSet<Element>()

const measureOverflow = (viewport: HTMLElement) => {
  const scrolls = viewport.scrollWidth > viewport.clientWidth + 1
  if (scrolls) {
    viewport.tabIndex = 0
    viewport.setAttribute('role', 'region')
    viewport.setAttribute('aria-label', `${viewport.dataset.diagramLabel ?? '流程图'}；左右滚动查看`)
  } else {
    viewport.removeAttribute('tabindex')
    viewport.removeAttribute('role')
    viewport.removeAttribute('aria-label')
  }
}

const register = () => {
  document.querySelectorAll<HTMLElement>('[data-diagram-label], .mermaid[data-rendered]').forEach(viewport => {
    // A rendered Mermaid can sit inside FigureShell: its outer frame owns the scroll region.
    const owner = viewport.closest<HTMLElement>('[data-diagram-label]') ?? viewport
    if (!registeredViewports.has(owner)) {
      registeredViewports.add(owner)
      const observer = new ResizeObserver(() => measureOverflow(owner))
      observer.observe(owner)
      for (const child of owner.children) observer.observe(child)
      owner.addEventListener('load', () => measureOverflow(owner), true)
      measureOverflow(owner)
    }
    viewport.querySelectorAll<SVGSVGElement>('svg').forEach(svg => {
      if (registeredSvgs.has(svg)) return
      registeredSvgs.add(svg)
      const requestedMinimum = parseFloat(getComputedStyle(svg).minWidth) || 0
      const measure = () => {
        const width = svg.getBoundingClientRect().width
        const sizes = [...svg.querySelectorAll<SVGTextElement>('text')].filter(text => text.textContent?.trim()).map(text => {
          const matrix = text.getScreenCTM()
          return matrix ? parseFloat(getComputedStyle(text).fontSize) * Math.hypot(matrix.a, matrix.b) : Infinity
        })
        // Matplotlib path glyphs retain a 100-unit font transformed by their enclosing group.
        svg.querySelectorAll<SVGGElement>('g[id*="-text_"] > g[transform*="scale"]').forEach(group => {
          const matrix = group.getScreenCTM()
          if (matrix) sizes.push(100 * Math.hypot(matrix.a, matrix.b))
        })
        svg.querySelectorAll<HTMLElement>('foreignObject span').forEach(text => {
          const object = text.closest('foreignObject') as SVGForeignObjectElement | null
          const matrix = object?.getScreenCTM()
          if (matrix) sizes.push(parseFloat(getComputedStyle(text).fontSize) * Math.hypot(matrix.a, matrix.b))
        })
        const smallest = Math.min(...sizes)
        if (width > 0 && smallest > 0 && Number.isFinite(smallest)) {
          svg.style.minWidth = `${Math.max(requestedMinimum, Math.ceil(width * 10 / smallest))}px`
        }
        measureOverflow(owner)
      }
      const observer = new ResizeObserver(measure)
      observer.observe(owner)
      observer.observe(svg)
      measure()
    })
  })
}
register()
new MutationObserver(register).observe(document.body, { childList: true, subtree: true })

export {}
