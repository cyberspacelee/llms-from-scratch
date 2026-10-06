/**
 * Code-fence meta for the book:
 *   ```python title="model.py" showLineNumbers {3-5}
 * `title` wraps the block in a titled frame, `showLineNumbers` numbers the lines,
 * `{…}` highlights lines (handled by transformerMetaHighlight).
 */
export function transformerBookMeta() {
  return {
    name: 'book-meta',
    pre(node) {
      const meta = this.options.meta?.__raw ?? ''
      if (/\bshowLineNumbers\b/.test(meta)) node.properties['data-line-numbers'] = ''
      node.properties['data-language'] = this.options.lang
    },
    root(root) {
      const meta = this.options.meta?.__raw ?? ''
      const title = meta.match(/\btitle=(?:"([^"]*)"|'([^']*)')/)
      if (!title) return
      const pre = root.children.find((child) => child.type === 'element' && child.tagName === 'pre')
      if (!pre) return
      root.children = [{
        type: 'element',
        tagName: 'div',
        properties: { className: ['code-titled'] },
        children: [
          { type: 'element', tagName: 'div', properties: { className: ['code-title'] }, children: [{ type: 'text', value: title[1] ?? title[2] }] },
          pre,
        ],
      }]
    },
  }
}
