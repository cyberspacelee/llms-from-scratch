import { getCollection, type CollectionEntry } from 'astro:content'
import { getTrack, tracks, type Track } from './tracks'

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

let cache: Promise<Lesson[]> | undefined

/** Every lesson in reading order: tracks in site order, chapters by `order`. */
export function getLessons(): Promise<Lesson[]> {
  cache ??= getCollection('lessons').then((entries) => {
    const lessons = entries.map(toLesson)
    const rank = (lesson: Lesson) => tracks.indexOf(lesson.track) * 1000 + lesson.entry.data.order
    lessons.sort((a, b) => rank(a) - rank(b))
    for (const [index, lesson] of lessons.entries()) {
      const previous = lessons[index - 1]
      if (previous && previous.track === lesson.track && rank(previous) === rank(lesson)) {
        throw new Error(`${previous.entry.id} and ${lesson.entry.id} share order ${lesson.entry.data.order}`)
      }
    }
    return lessons
  })
  return cache
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
