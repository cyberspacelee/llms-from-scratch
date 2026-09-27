/**
 * The filtering order of code/principles/generation.py: repetition penalty on raw logits,
 * then temperature, top-k, top-p (keeping the crossing token), and min-p.
 */
export type SamplingOptions = {
  temperature: number
  topK: number
  topP: number
  minP: number
  penalty: number
  history: readonly number[]
}

const softmax = (scores: number[]) => {
  const top = Math.max(...scores)
  const exps = scores.map(s => (s === -Infinity ? 0 : Math.exp(s - top)))
  const total = exps.reduce((a, b) => a + b, 0)
  return exps.map(e => e / total)
}

/** Indices sorted by descending score; equal scores keep index order (a stable sort). */
const descending = (scores: number[]) => scores.map((s, i) => [s, i] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1]).map(([, i]) => i)

export function sampleDistribution(logits: readonly number[], options: SamplingOptions) {
  const { temperature, topK, topP, minP, penalty, history } = options
  if (!logits.length || !logits.every(Number.isFinite)) throw new RangeError('Finite logits required')
  if (!(temperature > 0) || !(topP > 0 && topP <= 1) || !(minP >= 0 && minP <= 1) || !(penalty > 0)
      || !Number.isInteger(topK) || topK < 1 || topK > logits.length) throw new RangeError('Invalid sampling options')
  const scores = [...logits]
  for (const id of new Set(history)) {
    if (!Number.isInteger(id) || id < 0 || id >= logits.length) throw new RangeError('History ID outside vocabulary')
    scores[id] = scores[id] > 0 ? scores[id] / penalty : scores[id] * penalty
  }
  const stages: { name: string; kept: boolean[] }[] = []
  let current = scores.map(s => s / temperature)
  const byK = descending(current)
  current = current.map((s, i) => (byK.indexOf(i) < topK ? s : -Infinity))
  stages.push({ name: 'top-k', kept: current.map(s => s > -Infinity) })
  const order = descending(current)
  const ordered = softmax(current)
  let before = 0
  for (const i of order) {
    if (before >= topP) current[i] = -Infinity
    before += ordered[i]
  }
  stages.push({ name: 'top-p', kept: current.map(s => s > -Infinity) })
  const probs = softmax(current)
  const top = Math.max(...probs)
  const kept = probs.map(p => (p < minP * top ? 0 : p))
  stages.push({ name: 'min-p', kept: kept.map(p => p > 0) })
  const total = kept.reduce((a, b) => a + b, 0)
  const final = kept.map(p => p / total)
  const entropy = -final.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0)
  return { penalized: scores, final, stages, support: final.filter(p => p > 0).length, entropy }
}

/** Probability of each token under the raw logits, for the before/after comparison. */
export const rawProbabilities = (logits: readonly number[]) => softmax([...logits])
