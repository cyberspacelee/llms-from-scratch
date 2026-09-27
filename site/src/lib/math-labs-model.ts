/** Teaching models behind the math-track labs; each mirrors a worked example in the text. */

/** M7: softmax cross-entropy with a temperature on the logits, and its gradient on the raw logits. */
export function softmaxLoss(logits: readonly number[], temperature: number, target: number) {
  if (!logits.length || !logits.every(Number.isFinite) || !(temperature > 0) || !Number.isInteger(target) || target < 0 || target >= logits.length)
    throw new RangeError('Finite logits, positive temperature and a valid target required')
  const scaled = logits.map(z => z / temperature)
  const top = Math.max(...scaled)
  const exps = scaled.map(z => Math.exp(z - top))
  const total = exps.reduce((a, b) => a + b, 0)
  const probs = exps.map(e => e / total)
  const loss = top + Math.log(total) - scaled[target]
  // d/dz_c of -log softmax(z / tau)_target = (p_c - 1[c = target]) / tau
  const gradient = probs.map((p, c) => (p - (c === target ? 1 : 0)) / temperature)
  const entropy = -probs.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0)
  return { probs, loss, gradient, entropy }
}

/** M4: gradient descent on L = (x1^2 + 4 x2^2) / 2 from a start point. */
export function descend(eta: number, start: readonly [number, number], steps: number) {
  if (!(eta > 0) || !Number.isInteger(steps) || steps < 0 || !start.every(Number.isFinite)) throw new RangeError('Positive step size and steps required')
  const loss = ([a, b]: readonly number[]) => 0.5 * (a * a + 4 * b * b)
  const path: [number, number][] = [[start[0], start[1]]]
  for (let t = 0; t < steps; t++) {
    const [a, b] = path[path.length - 1]
    path.push([a - eta * a, b - eta * 4 * b])
  }
  const factors = [1 - eta, 1 - 4 * eta] as const
  const verdict: 'converge' | 'oscillate' | 'diverge' =
    Math.max(Math.abs(factors[0]), Math.abs(factors[1])) < 1 ? (factors[1] < 0 ? 'oscillate' : 'converge') : 'diverge'
  return { path, losses: path.map(loss), factors, verdict }
}

/** M6: a = x^2, b = 3a, loss = a + b; the shared node a collects two backward contributions. */
export function sharedGraph(x: number, keepDirect: boolean, keepThroughB: boolean) {
  if (!Number.isFinite(x)) throw new RangeError('Finite x required')
  const a = x * x, b = 3 * a, loss = a + b
  const direct = keepDirect ? 1 : 0
  const throughB = keepThroughB ? 3 : 0
  const gradA = direct + throughB
  return { a, b, loss, direct, throughB, gradA, gradX: gradA * 2 * x, exact: 8 * x }
}
