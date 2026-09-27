export function tensorLayout(transposed: boolean, row: number, col: number) {
  const rows = transposed ? 3 : 2, cols = transposed ? 2 : 3
  if (!Number.isInteger(row) || !Number.isInteger(col) || row < 0 || row >= rows || col < 0 || col >= cols) throw new RangeError('Coordinate outside tensor')
  const stride = transposed ? [1, 3] : [3, 1]
  return { rows, cols, stride, offset: row * stride[0] + col * stride[1],
    traversal: Array.from({ length: 6 }, (_, index) => Math.floor(index / cols) * stride[0] + index % cols * stride[1]) }
}

export function branchGradient(x: number, seed: number, previous: number) {
  if (![x, seed, previous].every(Number.isFinite)) throw new RangeError('Finite inputs required')
  return { square: x * x, linear: 3 * x, output: x * x + 3 * x,
    squareGradient: 2 * x * seed, linearGradient: 3 * seed,
    gradient: (2 * x + 3) * seed, accumulated: previous + (2 * x + 3) * seed }
}
