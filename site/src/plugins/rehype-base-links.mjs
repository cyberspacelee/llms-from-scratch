/** Lessons link to other pages root-relative ("/math/calculus/"); prefix the deploy base so links survive any base path. */
export function rehypeBaseLinks({ base }) {
  const prefix = base.replace(/\/$/, '')
  return (tree) => {
    const visit = (node) => {
      if (node.type === 'element' && node.tagName === 'a') {
        const href = node.properties?.href
        if (typeof href === 'string' && href.startsWith('/') && !href.startsWith('//')) {
          node.properties.href = `${prefix}${href}`
        }
      }
      for (const child of node.children ?? []) visit(child)
    }
    visit(tree)
  }
}
