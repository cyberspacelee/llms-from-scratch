// Fails the build when a page links to a file or #anchor that the build did not produce.
import { readdir, readFile, stat } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const dist = fileURLToPath(new URL('../dist/', import.meta.url))
const base = '/llms-from-scratch/'

async function htmlFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true })
  const nested = await Promise.all(entries.map((entry) => {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) return htmlFiles(full)
    return entry.name.endsWith('.html') ? [full] : []
  }))
  return nested.flat()
}

const decode = (value) => value.replace(/&amp;/g, '&').replace(/&quot;/g, '"').replace(/&#39;/g, "'")
const pages = new Map()
for (const file of await htmlFiles(dist)) {
  const html = await readFile(file, 'utf8')
  const ids = new Set([...html.matchAll(/\sid="([^"]*)"/g)].map((match) => decode(match[1])))
  pages.set(file, { html, ids })
}

async function resolveTarget(fromFile, rawPath) {
  const sitePath = rawPath.startsWith('/')
    ? rawPath
    : path.posix.join(path.posix.dirname('/' + path.relative(dist, fromFile).split(path.sep).join('/')).replace(/^\/?/, base), rawPath)
  if (!sitePath.startsWith(base)) return { error: `outside the site base: ${sitePath}` }
  let target = path.join(dist, decodeURIComponent(sitePath.slice(base.length)))
  try {
    if ((await stat(target)).isDirectory()) target = path.join(target, 'index.html')
  } catch {
    return { error: `missing file for ${sitePath}` }
  }
  try {
    await stat(target)
  } catch {
    return { error: `missing file for ${sitePath}` }
  }
  return { target }
}

const problems = []
let checked = 0
for (const [file, { html }] of pages) {
  const page = '/' + path.relative(dist, file)
  for (const match of html.matchAll(/\s(?:href|src)="([^"]+)"/g)) {
    const link = decode(match[1])
    if (/^(?:[a-z]+:|\/\/|data:|mailto:)/i.test(link)) continue
    checked += 1
    const [rawPath, anchor] = link.split('#')
    let target = file
    if (rawPath) {
      const resolved = await resolveTarget(file, rawPath)
      if (resolved.error) {
        problems.push(`${page}: ${link} → ${resolved.error}`)
        continue
      }
      target = resolved.target
    }
    if (anchor && target.endsWith('.html')) {
      const ids = pages.get(target)?.ids
      if (!ids?.has(decodeURIComponent(anchor))) problems.push(`${page}: ${link} → no element with id "${decodeURIComponent(anchor)}"`)
    }
  }
}

if (problems.length) {
  console.error(`check-links: ${problems.length} broken of ${checked} internal links`)
  for (const problem of problems) console.error(`  ${problem}`)
  process.exit(1)
}
console.log(`check-links: ${checked} internal links across ${pages.size} pages resolve`)
