/** Keep table semantics and let wide grids scroll inside a region. */
export function rehypeTableScroll() {
  return (tree) => {
    const visit = (node) => {
      const children = node.children
      if (!Array.isArray(children)) return
      for (let index = 0; index < children.length; index += 1) {
        const child = children[index]
        if (child.type === 'element' && child.tagName === 'table') {
          children[index] = {
            type: 'element',
            tagName: 'div',
            properties: {
              className: ['table-scroll'],
              tabIndex: 0,
              role: 'region',
              ariaLabel: '可横向滚动的表格',
            },
            children: [child],
          }
          continue
        }
        visit(child)
      }
    }
    visit(tree)
  }
}
