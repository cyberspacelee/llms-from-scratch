import { getCollection, type CollectionEntry } from 'astro:content'
import { getTrack, tracks, type Track } from './tracks'
import { learningPaths, routeFor } from './curriculum'

export type Lesson = {
  entry: CollectionEntry<'lessons'>
  track: Track
  /** Path under the site base, without leading or trailing slash: "math" or "math/probability". */
  path: string
  /** M0 for a track introduction, M3 for its third chapter. */
  code: string
  isIntro: boolean
  title: string
  question: string
  description: string
}

const base = import.meta.env.BASE_URL.replace(/\/?$/, '/')

/** Absolute URL for a path under the site base. */
export function url(path = ''): string {
  return path ? `${base}${path.replace(/^\/|\/$/g, '')}/` : base
}

function toLesson(entry: CollectionEntry<'lessons'>): Lesson {
  const [trackId, name] = entry.id.split('/')
  const track = getTrack(trackId)
  const isIntro = name === 'index'
  if (isIntro !== (entry.data.order === 0)) {
    throw new Error(`${entry.id}: only index.mdx may use order 0`)
  }
  return {
    entry,
    track,
    path: isIntro ? track.id : `${track.id}/${name}`,
    code: `${track.letter}${entry.data.order}`,
    isIntro,
    title: entry.data.title,
    question: entry.data.question,
    description: entry.data.description,
  }
}

/** Every lesson in reading order: tracks in site order, chapters by `order`. */
export function getLessons(): Promise<Lesson[]> {
  return getCollection('lessons').then((entries) => {
    const lessons = entries.map(toLesson)
    const rank = (lesson: Lesson) => tracks.indexOf(lesson.track) * 1000 + lesson.entry.data.order
    lessons.sort((a, b) => rank(a) - rank(b))
    for (const [index, lesson] of lessons.entries()) {
      const previous = lessons[index - 1]
      if (previous && previous.track === lesson.track && rank(previous) === rank(lesson)) {
        throw new Error(`${previous.entry.id} and ${lesson.entry.id} share order ${lesson.entry.data.order}`)
      }
    }
    const paths = new Set(lessons.map(lesson => lesson.path))
    for (const lesson of lessons) {
      for (const prerequisite of lesson.entry.data.prerequisites) {
        if (!paths.has(prerequisite) || prerequisite === lesson.path) {
          throw new Error(`${lesson.path}: invalid prerequisite ${prerequisite}`)
        }
      }
    }
    const routed = new Set<string>()
    for (const route of learningPaths) {
      const routePaths = new Set<string>(route.steps), completed = new Set<string>()
      for (const step of route.steps) {
        if (routed.has(step)) throw new Error(`${step}: primary navigation belongs to multiple routes`)
        routed.add(step)
        if (!paths.has(step)) throw new Error(`${route.id}: missing lesson ${step}`)
        for (const prerequisite of lessons.find(lesson => lesson.path === step)!.entry.data.prerequisites) {
          if (routePaths.has(prerequisite) && !completed.has(prerequisite)) {
            throw new Error(`${route.id}: ${step} comes before prerequisite ${prerequisite}`)
          }
        }
        completed.add(step)
      }
    }
    const visited = new Set<string>(), active = new Set<string>()
    const visit = (lesson: Lesson) => {
      if (active.has(lesson.path)) throw new Error(`Cyclic prerequisites at ${lesson.path}`)
      if (visited.has(lesson.path)) return
      active.add(lesson.path)
      for (const path of lesson.entry.data.prerequisites) visit(lessons.find(item => item.path === path)!)
      active.delete(lesson.path)
      visited.add(lesson.path)
    }
    lessons.forEach(visit)
    return lessons
  })
}

export async function getLessonContext(lesson: Lesson) {
  const lessons = await getLessons()
  const route = routeFor(lesson.path)
  const sequence = route
    ? route.steps.map(path => lessons.find(item => item.path === path)!)
    : lessons.filter(item => item.track === lesson.track && !item.entry.data.optional)
  const index = sequence.indexOf(lesson)
  return {
    route,
    previous: index > 0 ? sequence[index - 1] : undefined,
    next: index >= 0 ? sequence[index + 1] : undefined,
    prerequisites: lesson.entry.data.prerequisites.map(path => lessons.find(item => item.path === path)!),
  }
}

export async function getTrackLessons(trackId: string): Promise<Lesson[]> {
  return (await getLessons()).filter((lesson) => lesson.track.id === trackId)
}

/** Minutes to read, at roughly 400 CJK characters or 200 Latin words a minute. */
export function readingMinutes(body: string): number {
  const prose = body
    .replace(/```[\s\S]*?```/g, '')
    .replace(/\$\$[\s\S]*?\$\$/g, ' ')
    .replace(/<[^>]+>/g, '')
    .replace(/\]\([^)]*\)/g, ']')
  const cjk = prose.match(/[㐀-鿿]/g)?.length ?? 0
  const words = prose.replace(/[㐀-鿿]/g, ' ').match(/[A-Za-z0-9]+/g)?.length ?? 0
  return Math.max(1, Math.round(cjk / 400 + words / 200))
}

export const repoUrl = 'https://github.com/cyberspacelee/llms-from-scratch'
