export function tensorLayout(transposed: boolean, row: number, col: number) {
  const rows = transposed ? 3 : 2, cols = transposed ? 2 : 3
  if (!Number.isInteger(row) || !Number.isInteger(col) || row < 0 || row >= rows || col < 0 || col >= cols) throw new RangeError('Coordinate outside tensor')
  const stride = transposed ? [1, 3] : [3, 1]
  return { rows, cols, stride, offset: row * stride[0] + col * stride[1],
    traversal: Array.from({ length: 6 }, (_, index) => Math.floor(index / cols) * stride[0] + index % cols * stride[1]) }
}

/** F4: the vector loss, a shared bias, and a scalar reverse seed. */
export function vectorBranch(w0: number, w1: number, bias: number, seed: number) {
  if (![w0, w1, bias, seed].every(Number.isFinite)) throw new RangeError('Finite inputs required')
  const u = [2 * w0 + bias, -w1 + bias]
  const square = u[0] ** 2, product = u[0] * u[1], sum = square + product
  const loss = 0.5 * (sum - 2) ** 2, upstream = (sum - 2) * seed
  const branches = [2 * u[0] * upstream, u[1] * upstream, u[0] * upstream]
  const gu = [branches[0] + branches[1], branches[2]]
  return { u, square, product, sum, loss, upstream, branches, gu, gw: [2 * gu[0], -gu[1]], gb: gu[0] + gu[1] }
}
