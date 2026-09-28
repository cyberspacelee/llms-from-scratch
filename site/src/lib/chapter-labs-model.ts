export type Vec2 = [number, number]

export function rotate([x, y]: Vec2, degrees: number): Vec2 {
  const a = degrees * Math.PI / 180
  return [Math.cos(a) * x - Math.sin(a) * y, Math.sin(a) * x + Math.cos(a) * y]
}

export function spectralTransform(x: Vec2, degrees: number, values: Vec2) {
  const coordinates = rotate(x, -degrees)
  const scaled: Vec2 = [coordinates[0] * values[0], coordinates[1] * values[1]]
  return { coordinates, scaled, output: rotate(scaled, degrees) }
}

// Orthogonal rank-one terms give a fixed matrix with singular values 5, 2, 1.
export function lowRank(rank: number) {
  const u = [[0.8, 0.6, 0], [-0.6, 0.8, 0], [0, 0, 1]]
  const v = [[1, 0, 0], [0, 0.8, 0.6], [0, -0.6, 0.8]]
  const values = [5, 2, 1]
  const matrix = (r: number) => Array.from({ length: 3 }, (_, i) =>
    Array.from({ length: 3 }, (_, j) => values.slice(0, r).reduce((sum, s, k) => sum + s * u[k][i] * v[k][j], 0)))
  const full = matrix(3), approximation = matrix(rank)
  const residual = full.map((row, i) => row.map((x, j) => x - approximation[i][j]))
  return { full, approximation, residual, error: Math.sqrt(values.slice(rank).reduce((sum, s) => sum + s * s, 0)), energy: values.slice(0, rank).reduce((sum, s) => sum + s * s, 0) / 30 }
}

export function conditioning(exponent: number, delta: number) {
  const smallest = 10 ** -exponent
  return { condition: 1 / smallest, inputError: delta, outputError: delta / smallest }
}

// A two-coordinate, single-head attention calculation, after RoPE with omega = 1.
export function cachedAttention(past: number, length: number, row: number, wrongMask = false) {
  const position = past + row
  const q = rotate([1, 0.5], position * 180 / Math.PI)
  const keys = Array.from({ length: past + length }, (_, j) => rotate([1, (j % 3 - 1) * 0.5], j * 180 / Math.PI))
  const values: Vec2[] = keys.map((_, j) => [j + 1, j % 2 ? -1 : 1])
  const allowed = keys.map((_, j) => j <= (wrongMask ? row : position))
  const scores = keys.map(([a, b]) => (q[0] * a + q[1] * b) / Math.SQRT2)
  const maximum = Math.max(...scores.filter((_, j) => allowed[j]))
  const weights = scores.map((s, j) => allowed[j] ? Math.exp(s - maximum) : 0)
  const sum = weights.reduce((a, b) => a + b, 0)
  const probabilities = weights.map(x => x / sum)
  const output = values.reduce<Vec2>((acc, v, j) => [acc[0] + probabilities[j] * v[0], acc[1] + probabilities[j] * v[1]], [0, 0])
  return { position, q, keys, values, allowed, scores, probabilities, output }
}

export function decoderLedger(kvHeads: number, length: number, past: number) {
  const width = 12, heads = 4, headWidth = 4, ffWidth = 20, layers = 2, vocab = 8
  const attention = 2 * width * (heads + kvHeads) * headWidth
  return {
    parameters: vocab * width + layers * (attention + 3 * width * ffWidth + 2 * width) + width,
    cacheElements: 2 * layers * kvHeads * (past + length) * headWidth,
    scoreElements: heads * length * (past + length),
  }
}

export function budgetLoss(modelSize: number, budget: number, alpha: number, beta: number) {
  const dataSize = budget / modelSize
  const modelTerm = 4 / modelSize ** alpha, dataTerm = 2 / dataSize ** beta
  const optimalSize = ((alpha * 4 / (beta * 2)) * budget ** beta) ** (1 / (alpha + beta))
  return { dataSize, modelTerm, dataTerm, loss: 0.5 + modelTerm + dataTerm, optimalSize }
}
