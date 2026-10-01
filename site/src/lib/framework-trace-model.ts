export function headCoordinate(offset: number) {
  if (!Number.isInteger(offset) || offset < 0 || offset > 7) throw new RangeError('Offset must be in 0..7')
  const t = Math.floor(offset / 4), c = offset % 4, head = Math.floor(c / 2), channel = c % 2
  return { t, c, head, channel, original: [0, t, c], split: [0, t, head, channel], permuted: [0, head, t, channel], value: [0, 1, 2, 0, 3, 4, 5, 0][offset], offset }
}

export function compileTrace(step: number, broken: boolean) {
  if (!Number.isInteger(step) || step < 0 || step > 4) throw new RangeError('Step must be in 0..4')
  const calls = broken ? [[1, 2], [4, 5], [1, 2, 3], [1, 2]] : [[-2, 0, 2], [-2, 0, 2], [-2, 0, 2, 1], [1, 0, -1]]
  const shapes = new Set<number>()
  return calls.slice(0, step).map((x, index) => {
    const hit = shapes.has(x.length); shapes.add(x.length)
    return { index, x, shape: x.length, hit, graphs: shapes.size, output: x.map(v => broken ? (2 * v + 1) * 3 : v * v + 3 * v), gradient: x.map(v => broken ? 6 : 2 * v + 3), broken, route: hit ? 'guard 命中 → 复用' : 'guard 未命中 → 捕获/编译' }
  })
}

export const moduleParameters = [
  { name: 'embedding.weight', shape: '(4,2)', count: 8, owner: 'embedding', identity: 'E' },
  { name: 'norm.weight', shape: '(2,)', count: 2, owner: 'norm', identity: 'gamma' },
  { name: 'norm.bias', shape: '(2,)', count: 2, owner: 'norm', identity: 'beta' },
  { name: 'head.weight', shape: '(4,2)', count: 8, owner: 'head', identity: 'W' },
]
