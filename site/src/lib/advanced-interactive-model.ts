const dot = (a: number[], b: number[]) => a.reduce((sum, v, i) => sum + v * b[i], 0)
const mv = (m: number[][], v: number[]) => m.map(row => dot(row, v))
const softmax = (x: number[]) => { const m = Math.max(...x), e = x.map(v => Math.exp(v - m)), total = e.reduce((s, v) => s + v, 0); return e.map(v => v / total) }

export function moeDispatch(crowded: boolean, capacity: number, drop: boolean) {
  const inputs = [[1, 2], [2, -1], [-1, 3]]
  const logits = crowded ? inputs.map(() => [4, 3, 1, 0]) : [[4, 3, 1, 0], [0, 1, 4, 3], [4, 0, 3, 1]]
  const counts = [0, 0, 0, 0]
  const routes = inputs.flatMap((x, token) => {
    const experts = logits[token].map((v, expert) => ({ v, expert })).sort((a, b) => b.v - a.v || a.expert - b.expert).slice(0, 2)
    const weights = softmax(experts.map(e => e.v))
    return experts.map((e, i) => { const slot = counts[e.expert]++; const overflow = slot >= capacity;
      return { token, expert: e.expert, slot, weight: weights[i], input: x,
        expertOutput: x.map(v => v * (e.expert + 1)), dropped: overflow && drop, overflow } })
  })
  const output = inputs.map(() => [0, 0]), reference = inputs.map(() => [0, 0])
  for (const r of routes) for (let i = 0; i < 2; i++) {
    reference[r.token][i] += r.weight * r.expertOutput[i]
    if (!r.dropped) output[r.token][i] += r.weight * r.expertOutput[i]
  }
  return { inputs, routes, counts, output, reference }
}

export function mlaPaths(head: number) {
  const latent = [[1, 0], [0, 1], [1, 1]]
  const uk = head === 0 ? [[1, 1], [0, 1]] : [[1, 0], [1, 1]]
  const uv = head === 0 ? [[1, 0], [0, 2]] : [[0, 1], [1, 1]]
  const query = head === 0 ? [2, 1] : [1, 2]
  const projected = uk[0].map((_, c) => query.reduce((s, v, r) => s + uk[r][c] * v, 0))
  const explicitKeys = latent.map(c => mv(uk, c)), values = latent.map(c => mv(uv, c))
  const explicitScores = explicitKeys.map(k => dot(query, k)), absorbedScores = latent.map(c => dot(projected, c))
  const positional = head === 0 ? [-1, 0, 1] : [0, -1, 0]
  const weights = softmax(explicitScores.map((s, i) => (s + positional[i]) / 2))
  const mixture = latent[0].map((_, d) => latent.reduce((s, c, j) => s + weights[j] * c[d], 0))
  const explicit = values[0].map((_, d) => values.reduce((s, v, j) => s + weights[j] * v[d], 0))
  const absorbed = mv(uv, mixture)
  return { latent, uk, uv, query, projected, explicitKeys, values, explicitScores, absorbedScores, positional, weights, mixture, explicit, absorbed }
}

export function dpoLedger(lambda: number, includeEos: boolean) {
  const probabilities = [[0.55, 0.8], [0.3, 0.6], [0.4, 0.75], [0.4, 0.5]]
  const logs = probabilities.map(row => row.slice(0, includeEos ? 2 : 1).reduce((s, p) => s + Math.log(p), 0))
  const difference = logs[0] - logs[1] - logs[2] + logs[3], margin = lambda * difference
  const loss = Math.max(0, -margin) + Math.log1p(Math.exp(-Math.abs(margin)))
  return { probabilities, logs, difference, margin, loss, gradient: -lambda / (1 + Math.exp(margin)) }
}

export function frequencyCache(query: number, key: number, scale: number) {
  const omega = Math.PI / 4
  const oldAngle = key * omega - query * omega / scale
  const newAngle = (key - query) * omega / scale
  return { oldAngle, newAngle, mixedScore: Math.cos(oldAngle), correctScore: Math.cos(newAngle) }
}

export function grpoGroup(equal: boolean, ratio: number, epsilon: number, selected: number) {
  const rewards = equal ? [0, 0, 0, 0] : [1, 0, 0, 1]
  const mean = rewards.reduce((s, v) => s + v, 0) / 4
  const std = Math.sqrt(rewards.reduce((s, v) => s + (v - mean) ** 2, 0) / 4)
  const advantages = rewards.map(r => std === 0 ? 0 : (r - mean) / std)
  const advantage = advantages[selected], clipped = Math.max(1 - epsilon, Math.min(1 + epsilon, ratio))
  const raw = ratio * advantage, restricted = clipped * advantage, objective = Math.min(raw, restricted)
  const stopped = (advantage > 0 && ratio > 1 + epsilon) || (advantage < 0 && ratio < 1 - epsilon)
  return { rewards, mean, std, advantages, advantage, ratio, clipped, raw, restricted, objective,
    lossDerivativeLogRatio: stopped ? 0 : -ratio * advantage }
}

export function stateRecurrence(selective: boolean, retention: number) {
  const inputs = [1, 2, -1, 3]
  let state = 0, cumulativeA = 1, cumulativeB = 0
  return inputs.map((input, step) => {
    const a = selective && input < 0 ? 1 : retention, b = selective && input < 0 ? 0 : input
    const previous = state, decayed = a * previous
    state = decayed + b
    cumulativeA = a * cumulativeA; cumulativeB = a * cumulativeB + b
    return { input, step, a, b, previous, decayed, state, cumulativeA, cumulativeB }
  })
}

export function hybridBoundary(boundary: number, full: boolean, window: boolean, recursive: boolean) {
  // The checkpoint/window layout is a teaching fixture, not an SGLang allocation trace.
  const fullAvailable = Array.from({ length: 8 }, (_, i) => i)
  const windowAvailable = [2, 3, 4, 5]
  const checkpoints = [0, 4, 8]
  const requiredWindow = Array.from({ length: Math.min(3, boundary) }, (_, i) => Math.max(0, boundary - 3) + i)
  const votes = [!full || fullAvailable.slice(0, boundary).length === boundary,
    !window || requiredWindow.every(p => windowAvailable.includes(p)),
    !recursive || checkpoints.includes(boundary)]
  return { fullAvailable, windowAvailable, checkpoints, requiredWindow, votes, valid: votes.every(Boolean) }
}

export function sparseSelection(selected: number[]) {
  if (new Set(selected).size !== selected.length || selected.some(i => i < 0 || i > 7)) throw new Error('Invalid selected keys')
  const weights = [1, 1, 1, 2, 1, 1, 1, 4], values = [0, 10, 20, 30, 40, 50, 60, 70]
  const denominator = selected.reduce((s, j) => s + weights[j], 0)
  const probabilities = weights.map((w, j) => selected.includes(j) && denominator ? w / denominator : 0)
  const output = denominator ? dot(probabilities, values) : null
  const full = dot(weights, values) / 12, discarded = 1 - denominator / 12
  return { weights, values, denominator, probabilities, output, full, discarded,
    error: output === null ? null : output - full, bound: 2 * 70 * discarded }
}

export function multimodalPatch(patch: number, swapped: boolean) {
  const indices = [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]
  const pixels = indices[patch].map(i => ((swapped ? (i + 8) % 16 : i) / 15))
  const mean = pixels.reduce((s, v) => s + v, 0) / 4
  return { indices: indices[patch], pixels, mean, position: patch + 1,
    targets: [{ prediction: 6, target: 'ANSWER' }, { prediction: 7, target: 'EOS' }] }
}
