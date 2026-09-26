import { defineCollection } from 'astro:content'
import { glob } from 'astro/loaders'
import { z } from 'astro/zod'

// A lesson's track is its folder; `index.mdx` is the track's introduction.
const lessons = defineCollection({
  loader: glob({
    pattern: '*/*.mdx',
    base: './src/content/lessons',
    generateId: ({ entry }) => entry.replace(/\.mdx$/, ''),
  }),
  schema: z.object({
    title: z.string(),
    question: z.string(),
    description: z.string(),
    order: z.number().int().nonnegative(),
    /** Repository paths of the scripts that verify this lesson. */
    code: z.array(z.string()).default([]),
  }),
})

export const collections = { lessons }
