export function storageSelection(mode: string, cell: number, value: number) {
  const source = [0, 1, 2, 3, 4, 5]
  const indices = [1, 2, 4, 5]
  const selected = indices.map(i => source[i])
  selected[cell] = value
  if (mode === 'view' || mode === 'assign') source[indices[cell]] = value
  return { source, selected, shared: mode === 'view', writesBack: mode === 'view' || mode === 'assign' }
}

export function axisCalculation(mode: string, axis: number, keep: boolean) {
  const input = [1, 2, 3, 4, 5, 6]
  if (mode === 'feature') return { values: input.map((v, i) => v + [10, 20, 30][i % 3]), shape: '(2,3)', valid: true }
  if (mode === 'sample') return { values: input.map((v, i) => v + [100, 200][Math.floor(i / 3)]), shape: '(2,3)', valid: true }
  if (mode === 'invalid') return { values: [], shape: '不兼容', valid: false }
  const values = axis === 1 ? [2, 5] : [2.5, 3.5, 4.5]
  return { values, shape: axis === 1 ? keep ? '(2,1)' : '(2,)' : keep ? '(1,3)' : '(3,)', valid: true }
}

export function contraction(row: number, output: number) {
  const x = [[1, 2, 3], [4, 5, 6]]
  const w = [[1, 0, -1], [2, 1, 0]]
  const products = x[row].map((v, k) => v * w[output][k])
  return { x, w, products, value: products.reduce((a, b) => a + b, 0) }
}
