/** Page behavior shared by every page: theme, sidebar drawer, section tracking, code copy, diagrams. */

const root = document.documentElement

// Theme: cycle system → light → dark; "system" follows the OS live.
type Choice = 'system' | 'light' | 'dark'
const choices: Choice[] = ['system', 'light', 'dark']
const choiceLabel: Record<Choice, string> = { system: '跟随系统', light: '浅色', dark: '深色' }
const systemDark = matchMedia('(prefers-color-scheme: dark)')

function readChoice(): Choice {
  try {
    const stored = localStorage.getItem('theme')
    return choices.includes(stored as Choice) ? (stored as Choice) : 'system'
  } catch {
    return 'system'
  }
}

function applyTheme(choice: Choice) {
  const dark = choice === 'dark' || (choice === 'system' && systemDark.matches)
  const changed = root.dataset.theme !== (dark ? 'dark' : 'light')
  root.dataset.theme = dark ? 'dark' : 'light'
  root.dataset.themeChoice = choice
  const button = document.querySelector('[data-theme-toggle]')
  button?.setAttribute('aria-label', `配色：${choiceLabel[choice]}。点击切换`)
  if (changed) document.dispatchEvent(new CustomEvent('themechange'))
}

applyTheme(readChoice())

document.querySelector('[data-theme-toggle]')?.addEventListener('click', () => {
  const next = choices[(choices.indexOf(readChoice()) + 1) % choices.length]
  try {
    if (next === 'system') localStorage.removeItem('theme')
    else localStorage.setItem('theme', next)
  } catch {}
  applyTheme(next)
})

systemDark.addEventListener('change', () => {
  if (readChoice() === 'system') applyTheme('system')
})

// Sidebar drawer on narrow screens.
const sidebar = document.getElementById('sidebar')
const navToggle = document.querySelector<HTMLButtonElement>('[data-nav-toggle]')
const scrim = document.querySelector<HTMLElement>('[data-nav-scrim]')

function setNav(open: boolean) {
  if (!sidebar || !navToggle) return
  sidebar.toggleAttribute('data-open', open)
  scrim?.classList.toggle('hidden', !open)
  navToggle.setAttribute('aria-expanded', String(open))
  navToggle.setAttribute('aria-label', open ? '关闭章节目录' : '打开章节目录')
  if (open) sidebar.querySelector<HTMLElement>('[aria-current="page"]')?.focus()
}

navToggle?.addEventListener('click', () => setNav(!sidebar?.hasAttribute('data-open')))
scrim?.addEventListener('click', () => setNav(false))
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && sidebar?.hasAttribute('data-open')) {
    setNav(false)
    navToggle?.focus()
  }
})
sidebar?.addEventListener('click', (event) => {
  if (event.target instanceof Element && event.target.closest('a')) setNav(false)
})
matchMedia('(min-width: 64rem)').addEventListener('change', () => setNav(false))

// Reading progress and the current section in the sidebar.
const progress = document.querySelector<HTMLElement>('[data-progress]')
const sectionLinks = [...document.querySelectorAll<HTMLAnchorElement>('[data-section-link]')]
const sections = sectionLinks
  .map((link) => document.getElementById(decodeURIComponent(link.hash.slice(1))))
  .filter((section): section is HTMLElement => section !== null)

let frame = 0
function track() {
  frame = 0
  const scrollable = document.documentElement.scrollHeight - innerHeight
  progress?.style.setProperty('transform', `scaleX(${scrollable > 0 ? Math.min(1, scrollY / scrollable) : 0})`)
  const line = Math.min(innerHeight * 0.3, 240)
  let active = -1
  sections.forEach((section, index) => {
    if (section.getBoundingClientRect().top <= line) active = index
  })
  sectionLinks.forEach((link, index) => {
    if (index === active) link.setAttribute('aria-current', 'true')
    else link.removeAttribute('aria-current')
  })
}

if (progress || sections.length) {
  addEventListener('scroll', () => { frame ||= requestAnimationFrame(track) }, { passive: true })
  addEventListener('resize', () => { frame ||= requestAnimationFrame(track) })
  track()
}

// Copy buttons on code blocks.
document.querySelectorAll<HTMLPreElement>('pre.astro-code').forEach((pre) => {
  const button = document.createElement('button')
  button.type = 'button'
  button.textContent = '复制'
  button.className =
    'absolute top-2 right-2 rounded-sm border border-rule bg-raised px-2 py-0.5 font-sans text-xs text-muted opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100 pointer-coarse:opacity-100'
  pre.classList.add('group')
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(pre.querySelector('code')?.innerText ?? pre.innerText)
      button.textContent = '已复制'
    } catch {
      button.textContent = '无法复制'
    }
    setTimeout(() => { button.textContent = '复制' }, 1600)
  })
  pre.append(button)
})

// Mermaid diagrams: render from source, redraw in the new colors when the theme changes.
const diagrams = [...document.querySelectorAll<HTMLElement>('.mermaid')]
for (const diagram of diagrams) diagram.dataset.source = diagram.textContent ?? ''

async function renderDiagrams() {
  if (!diagrams.length) return
  const { default: mermaid } = await import('mermaid')
  const style = getComputedStyle(root)
  // Mermaid only parses hex/rgb, and the tokens are oklch: resolve each through a 1×1 canvas.
  const context = document.createElement('canvas').getContext('2d', { willReadFrequently: true })
  const color = (name: string) => {
    const value = style.getPropertyValue(name).trim()
    if (!context) return value
    context.clearRect(0, 0, 1, 1)
    context.fillStyle = value
    context.fillRect(0, 0, 1, 1)
    const [r, g, b] = context.getImageData(0, 0, 1, 1).data
    return `#${[r, g, b].map((channel) => channel.toString(16).padStart(2, '0')).join('')}`
  }
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict',
    theme: 'base',
    fontFamily: style.fontFamily,
    themeVariables: {
      darkMode: root.dataset.theme === 'dark',
      background: color('--raised'),
      primaryColor: color('--accent-soft'),
      primaryTextColor: color('--ink'),
      primaryBorderColor: color('--accent'),
      secondaryColor: color('--info-soft'),
      tertiaryColor: color('--sunken'),
      lineColor: color('--muted'),
      textColor: color('--ink'),
      noteBkgColor: color('--accent2-soft'),
      noteTextColor: color('--ink'),
      noteBorderColor: color('--accent2'),
      actorBkg: color('--accent-soft'),
      actorBorder: color('--accent'),
      actorTextColor: color('--ink'),
      signalColor: color('--muted'),
      signalTextColor: color('--ink'),
      labelBoxBkgColor: color('--sunken'),
      labelBoxBorderColor: color('--rule-strong'),
      loopTextColor: color('--ink'),
      fontSize: '15px',
    },
    // Natural size keeps labels readable; wide diagrams scroll inside their frame.
    flowchart: { htmlLabels: true, useMaxWidth: false },
    sequence: { useMaxWidth: false },
    state: { useMaxWidth: false },
  })
  for (const [index, diagram] of diagrams.entries()) {
    try {
      const { svg } = await mermaid.render(`mermaid-${index}-${Date.now()}`, diagram.dataset.source ?? '')
      diagram.innerHTML = svg
      diagram.dataset.rendered = ''
      delete diagram.dataset.failed
    } catch (error) {
      diagram.textContent = `图无法绘制：${error instanceof Error ? error.message : String(error)}`
      diagram.dataset.failed = ''
    }
  }
}

if (diagrams.length) {
  const observer = new IntersectionObserver((entries) => {
    if (entries.some((entry) => entry.isIntersecting)) {
      observer.disconnect()
      renderDiagrams()
    }
  }, { rootMargin: '400px' })
  diagrams.forEach((diagram) => observer.observe(diagram))
  document.addEventListener('themechange', () => {
    if (diagrams.some((diagram) => 'rendered' in diagram.dataset)) renderDiagrams()
  })
}
