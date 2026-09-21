import { defineCollection } from 'astro:content'
import { glob } from 'astro/loaders'
import { z } from 'astro/zod'

const lessons = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/lessons' }),
  schema: z.object({
    title: z.string(),
    section: z.string(),
    chapter: z.number(),
    order: z.number(),
    description: z.string().optional(),
  }),
})

export const collections = { lessons }
