const sum = (values: number[]) => values.reduce((a, b) => a + b, 0)
const softmax = (values: number[]) => {
  const m = Math.max(...values), e = values.map(x => Math.exp(x - m)), z = sum(e)
  return e.map(x => x / z)
}
export function packing(isolated: boolean) {
  const stream = [1, 2, 5, 3, 4, 5], docs = [0, 0, 0, 1, 1, 1]
  const valid = stream.slice(0, -1).map((_, i) => !isolated || docs[i] === docs[i + 1])
  const allowed = valid.map((_, i) => valid.map((__, j) => j <= i && (!isolated || docs[i] === docs[j])))
  return { stream, docs, valid, allowed, positions: [0, 1, 2, isolated ? 0 : 3, isolated ? 1 : 4], count: valid.filter(Boolean).length }
}
export function adamTrajectory(steps: number, resetAt = -1) {
  let w = [1, -2], m = [0, 0], v = [0, 0], k = 0
  const rows = [{ w: [...w], m: [...m], v: [...v], g: [0, 0], mh: [0, 0], vh: [0, 0], k }]
  for (let t = 0; t < steps; t++) {
    if (t === resetAt) { m = [0, 0]; v = [0, 0]; k = 0 }
    k++
    const g = [w[0] - 0.5, w[1] + 1.75]
    m = m.map((x, i) => 0.9 * x + 0.1 * g[i])
    v = v.map((x, i) => 0.99 * x + 0.01 * g[i] ** 2)
    const mh = m.map(x => x / (1 - 0.9 ** k)), vh = v.map(x => x / (1 - 0.99 ** k))
    w = w.map((x, i) => 0.98 * x - 0.1 * mh[i] / (Math.sqrt(vh[i]) + 1e-8))
    rows.push({ w: [...w], m: [...m], v: [...v], g, mh, vh, k })
  }
  return rows
}

type Snapshot = { w: number; m: number; v: number; step: number; rng: number; order: number[]; cursor: number }
const fresh = (): Snapshot => ({ w: 0, m: 0, v: 0, step: 0, rng: 29, order: [0, 1, 2], cursor: 0 })
function checkpointStep(state: Snapshot) {
  if (state.cursor === 3) {
    state.order = [0, 1, 2]
    for (let i = 2; i > 0; i--) {
      state.rng = (1664525 * state.rng + 1013904223) >>> 0
      const j = Math.floor(state.rng / 4294967296 * (i + 1))
      ;[state.order[i], state.order[j]] = [state.order[j], state.order[i]]
    }
    state.cursor = 0
  }
  const document = state.order[state.cursor++], target = [1, -1, 2][document]
  const loss = 0.5 * (state.w - target) ** 2, gradient = state.w - target
  state.step++
  state.m = 0.9 * state.m + 0.1 * gradient
  state.v = 0.99 * state.v + 0.01 * gradient ** 2
  state.w = 0.999 * state.w - 0.1 * (state.m / (1 - 0.9 ** state.step)) / (Math.sqrt(state.v / (1 - 0.99 ** state.step)) + 1e-8)
  return { ...state, order: [...state.order], document, loss, gradient }
}
export function checkpointComparison(omit: 'none' | 'optimizer' | 'rng' | 'cursor') {
  const state = fresh()
  for (let i = 0; i < 5; i++) checkpointStep(state)
  const restored = { ...state, order: [...state.order] }
  if (omit === 'optimizer') { restored.m = 0; restored.v = 0; restored.step = 0 }
  if (omit === 'rng') restored.rng = 1
  if (omit === 'cursor') { restored.cursor = 0; restored.order = [0, 1, 2] }
  const reference = [], resumed = []
  for (let i = 0; i < 7; i++) { reference.push(checkpointStep(state)); resumed.push(checkpointStep(restored)) }
  return { reference, resumed }
}
export function evaluationWindows(stride: number) {
  const windows = []
  for (let start = 1; start < 7; start += stride) {
    const end = Math.min(start + stride, 7), inputStart = Math.max(0, end - 1 - 4)
    windows.push({ inputs: Array.from({ length: end - 1 - inputStart }, (_, i) => inputStart + i), targets: Array.from({ length: end - start }, (_, i) => start + i) })
  }
  return windows
}
export function evaluationLedger(model: 'A' | 'B', included: boolean[]) {
  const counts = [2, 6, 2, 6], means = model === 'A' ? [1, 3, 1, 3] : [0.8, 2.8, 1.2, 2.4]
  const count = sum(counts.map((n, i) => included[i] ? n : 0)), nll = sum(counts.map((n, i) => included[i] ? n * means[i] : 0))
  const selectedMeans = means.filter((_, i) => included[i])
  return { counts, means, count, nll, mean: count ? nll / count : null, documentMean: selectedMeans.length ? sum(selectedMeans) / selectedMeans.length : null }
}
export const chatTokens = ['SYS', '规则', 'USER', 'Q1', 'ASSISTANT', 'OK', 'END', 'USER', 'Q2', 'ASSISTANT', 'OK', 'END']
export function chatMask(onlySecond: boolean, truncated: boolean) {
  const length = truncated ? 5 : 12
  const valid = Array.from({ length: length - 1 }, (_, i) => (onlySecond ? [9, 10] : [4, 5, 9, 10]).includes(i))
  const count = valid.filter(Boolean).length
  // A scalar causal average with a quadratic reply target isolates the gradient path.
  const hidden = Array.from({ length: length - 1 }, (_, i) => (i + 1) / 10)
  const averages = hidden.map((_, i) => sum(hidden.slice(0, i + 1)) / (i + 1))
  const gradient = hidden.map((_, j) => count ? sum(averages.map((u, i) => valid[i] && j <= i ? (u - 1) / ((i + 1) * count) : 0)) : 0)
  return { tokens: chatTokens.slice(0, length), valid, count, averages, gradient }
}
export function loraTrajectory(steps: number, bothZero = false) {
  const x = [1, 2, 3, 4], base = [1, 1, 1, 1, 0, 0]
  let a = bothZero ? [[0, 0, 0, 0], [0, 0, 0, 0]] : [[1, 0, 1, 0], [0, 1, 0, 1]]
  let b = Array.from({ length: 6 }, () => [0, 0])
  const rows = []
  for (let step = 0; step <= steps; step++) {
    const compressed = a.map(row => sum(row.map((v, j) => v * x[j])))
    const delta = b.map(row => sum(row.map((v, j) => v * compressed[j])))
    const logits = base.map((v, i) => v + delta[i]), p = softmax(logits), g = p.map((v, i) => v - (i === 2 ? 1 : 0))
    const gb = g.map(v => compressed.map(z => v * z))
    const ga = a.map((_, j) => x.map(v => v * sum(b.map((row, i) => row[j] * g[i]))))
    const merged = base.map((v, i) => v + sum(x.map((xj, j) => xj * sum(a.map((row, r) => b[i][r] * row[j])))))
    rows.push({ step, compressed, base, delta, logits, merged, loss: -Math.log(p[2]), gradA: Math.sqrt(sum(ga.flat().map(v => v * v))), gradB: Math.sqrt(sum(gb.flat().map(v => v * v))) })
    a = a.map((row, i) => row.map((v, j) => v - 0.01 * ga[i][j]))
    b = b.map((row, i) => row.map((v, j) => v - 0.01 * gb[i][j]))
  }
  return rows
}
const duplicateTexts = ['the cat watches the quiet garden in the rain', 'the cat watches the green garden in the rain', 'the cat watches the quiet garden in the rain', 'a dog sleeps beside the door while a child reads a book', 'the sun warms the street and a bird sings in the tree']
export function duplicateGroups(threshold: number, chain: boolean) {
  const shingles = duplicateTexts.map(text => {
    const words = text.split(' ')
    return new Set(words.slice(0, -2).map((_, i) => words.slice(i, i + 3).join(' ')))
  })
  const similarities = shingles.map((a, i) => shingles.map((b, j) => chain ? (i === j ? 1 : Math.abs(i - j) === 1 && i < 3 && j < 3 ? 0.6 : 0) : [...a].filter(s => b.has(s)).length / new Set([...a, ...b]).size))
  const group = [0, 1, 2, 3, 4]
  const edges: [number, number][] = []
  for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) if (similarities[i][j] >= threshold) {
    edges.push([i, j]); const old = group[j], replacement = group[i]
    group.forEach((v, k) => { if (v === old) group[k] = replacement })
  }
  return { similarities, group, edges }
}
export function ddpLedger(wrong: boolean, microSteps: number) {
  const counts = [[3, 2], [5, 4]], gradients = [-1, -3]
  const localSums = counts.map((row, r) => sum(row.slice(0, microSteps)) * gradients[r])
  const local = localSums.map((s, r) => wrong ? s / sum(counts[r]) : 2 * s / 14)
  const gradient = sum(local) / 2
  return { counts, localSums, local, gradient, theta: -0.1 * gradient, reference: -32 / 14 }
}
export function stateAllocation(ranks: number) {
  return [16, 4 + 12 / ranks, 2 + 14 / ranks, 16 / ranks]
}
