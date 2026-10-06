import { defineCollection } from 'astro:content'
import { glob } from 'astro/loaders'
import { z } from 'astro/zod'

// A chapter's part is its folder; `index.mdx` (order 0) is the part's introduction.
const book = defineCollection({
  loader: glob({
    pattern: '*/*.mdx',
    base: './src/content/book',
    generateId: ({ entry }) => entry.replace(/\.mdx$/, ''),
  }),
  schema: z.object({
    title: z.string(),
    /** One or two sentences: the page lead and meta description. */
    description: z.string(),
    order: z.number().int().nonnegative(),
    /** Repository paths of the code this chapter builds or runs. */
    code: z.array(z.string()).default([]),
    /** Draft chapters render a notice and stay out of the reading-time count. */
    draft: z.boolean().default(false),
  }),
})

export const collections = { book }
