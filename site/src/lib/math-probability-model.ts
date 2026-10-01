export const eventSets = { positive: [0, 3], negative: [1, 2], D: [0, 1, 2], firstTwo: [0, 1], empty: [] } as const
export type EventName = keyof typeof eventSets

export function eventProbability(a: EventName, b: EventName) {
  const selectedA: readonly number[] = eventSets[a], selectedB: readonly number[] = eventSets[b]
  const intersection = selectedA.filter(i => selectedB.includes(i))
  const pA = selectedA.length / 4, pB = selectedB.length / 4, joint = intersection.length / 4
  return { selectedA, selectedB, intersection, pA, pB, joint, conditional: pB ? joint / pB : null, independent: joint === pA * pB }
}

export function bayesPaths(priorD: number) {
  if (!Number.isFinite(priorD) || priorD < 0 || priorD > 1) throw new RangeError('Prior must be a probability')
  const jointD = priorD / 3, jointE = 1 - priorD, positive = jointD + jointE
  return { jointD, jointE, positive, posteriorD: jointD / positive, posteriorE: jointE / positive }
}

export function sequencePath(step: number) {
  if (!Number.isInteger(step) || step < 0 || step > 3) throw new RangeError('Step must be in 0..3')
  const probabilities = [0.6, 0.7, 0.6]
  const selected = probabilities.slice(0, step)
  return { probabilities, joint: selected.reduce((a, p) => a * p, 1), nll: selected.reduce((a, p) => a - Math.log(p), 0) }
}

export function batchDistribution(size: number, mode: 'independent' | 'copied' | 'complete') {
  if (!Number.isInteger(size) || size < 1 || size > 64) throw new RangeError('Batch size must be in 1..64')
  let rows: { mean: number; probability: number }[]
  if (mode === 'complete') rows = [{ mean: 2, probability: 1 }]
  else if (mode === 'copied') rows = [{ mean: 1, probability: 0.75 }, { mean: 5, probability: 0.25 }]
  else {
    let probability = 0.75 ** size
    rows = Array.from({ length: size + 1 }, (_, k) => {
      const row = { mean: 1 + 4 * k / size, probability }
      probability *= (size - k) / (k + 1) / 3
      return row
    })
  }
  const mean = rows.reduce((s, row) => s + row.mean * row.probability, 0)
  const variance = rows.reduce((s, row) => s + (row.mean - mean) ** 2 * row.probability, 0)
  return { rows, mean, variance, standardError: Math.sqrt(variance), closeProbability: rows.filter(row => Math.abs(row.mean - 2) < 1 - 1e-12).reduce((s, row) => s + row.probability, 0) }
}

export function informationCost(q: number, source: 'all' | 'D' | 'E', bits: boolean) {
  if (!Number.isFinite(q) || q < 0 || q > 1) throw new RangeError('q must be a probability')
  const p = source === 'D' ? [1 / 3, 2 / 3] : source === 'E' ? [1, 0] : [0.5, 0.5]
  const model = [q, 1 - q], scale = bits ? Math.log(2) : 1
  const entropyTerms = p.map(value => value ? -value * Math.log(value) / scale : 0)
  const crossTerms = p.map((value, i) => value ? -value * Math.log(model[i]) / scale : 0)
  const entropy = entropyTerms.reduce((a, b) => a + b, 0), crossEntropy = crossTerms.reduce((a, b) => a + b, 0)
  const hD = -(Math.log(1 / 3) / 3 + 2 * Math.log(2 / 3) / 3) / scale
  return { p, model, entropyTerms, crossTerms, entropy, crossEntropy, kl: crossEntropy - entropy, conditionalEntropy: 0.75 * hD, hD, information: p.map(value => -Math.log(value) / scale) }
}
