function classList(node) {
  const value = node.properties?.className
  if (!value) return []
  return (Array.isArray(value) ? value : [value]).map(String)
}

function textOnly(node) {
  if (node.type === 'text') return node.value
  if (node.type !== 'element') return ''
  return (node.children ?? []).map(textOnly).join('')
}

function diagramSource(pre) {
  const lines = []
  const walk = (node) => {
    if (node.type !== 'element') return
    if (classList(node).includes('line')) {
      lines.push(textOnly(node))
      return
    }
    for (const child of node.children ?? []) walk(child)
  }
  walk(pre)
  const source = (lines.length > 0 ? lines.join('\n') : textOnly(pre)).trim()
  return source.length > 0 ? source : null
}

function isMermaidBlock(node) {
  if (node.type !== 'element' || node.tagName !== 'pre') return false
  const props = node.properties ?? {}
  const language = props.dataLanguage ?? props['data-language']
  if (language === 'mermaid') return true
  return (node.children ?? []).some((child) => child.type === 'element' && child.tagName === 'code' && classList(child).includes('language-mermaid'))
}

/** Shiki turns mermaid fences into highlighted spans, which the browser cannot draw. Emit the source as a plain `.mermaid` node instead. */
export function rehypeMermaidBlocks() {
  return (tree) => {
    const visit = (node) => {
      const children = node.children
      if (!Array.isArray(children)) return
      for (let index = 0; index < children.length; index += 1) {
        const child = children[index]
        if (isMermaidBlock(child)) {
          const source = diagramSource(child)
          if (source) {
            children[index] = {
              type: 'element',
              tagName: 'div',
              properties: { className: ['mermaid'] },
              children: [{ type: 'text', value: `${source}\n` }],
            }
          }
          continue
        }
        visit(child)
      }
    }
    visit(tree)
  }
}
